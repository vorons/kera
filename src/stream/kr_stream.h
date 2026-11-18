// Streaming a MoE model's routed experts from the SSD (spec Phases 2-5, decisions D-008 and D-012).
//
// start() installs llama.cpp's stream hook (hook 3) for one model file. When that model loads, its routed
// experts aren't read; each MoE layer's chosen experts are brought in just before the layer runs:
//   - cache_mode::kera (v2): direct reads into memory Kera manages, within a RAM budget, least recently
//     used out first, plus learned pins and early reads for the next layer;
//   - cache_mode::os (v1): the experts stay mapped from the file and the OS file cache keeps them; Kera
//     asks the OS to start reading each layer's experts (and the next layer's likely ones) early.
// The bytes, kernels and order of operations are the stock ones, so the output is unchanged.
#pragma once

#include "stream/expert_cache.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kera::stream {

enum class cache_mode {
    off = 0,
    os = 1,   // v1
    kera = 2, // v2
};

const char * mode_name(cache_mode m);
bool parse_mode(const std::string & s, cache_mode & m); // "off", "os", "kera"

struct config {
    std::string model_path;              // the model to stream (the first file of a split model)
    cache_mode mode = cache_mode::kera;
    uint64_t budget_bytes = 8ull << 30;  // RAM for cached experts (v2)
    int io_threads = 8;
    uint32_t chunk_bytes = 1u << 20;
    bool direct = true;                  // v2 reads bypass the OS file cache
    // Phase 3: guess the next MoE layer's experts and read them early. Off until Gate 3 passes on the
    // Minisforum: on a 4-core cloud machine it halved the wait for the SSD but cost as much CPU (D-012).
    bool lookahead = false;
    // Phase 4 (learned pins) failed Gate 4 on Nemotron 3 Nano's traces (docs/NEGATIVE_RESULTS.md N-002):
    // pinning only took room from LRU. Off by default; kept so the gate can be re-run on bigger models.
    bool pins = false;                   // pin the experts the usage log says are used most
    double pin_share = 0.3;              // share of the budget for those pins
    bool prompt_pins = false;            // during a reply, keep the experts its prompt used most
    double prompt_pin_share = 0.2;       // share of the budget for those
    std::string usage_file;              // per-model usage log, "" for none
    std::vector<std::vector<std::string>> mirrors; // Phase 5: copies of file i on other drives
};

// Counters since the model loaded; subtract two to get what happened in between.
struct report {
    cache_mode mode = cache_mode::off;
    cache_stats cache;               // v2
    uint64_t predicted = 0;          // experts guessed for the next layer (Phase 3)
    uint64_t predicted_right = 0;    // ... that the layer then used
    uint64_t os_hint_bytes = 0;      // v1: bytes the OS was asked to read early
    uint64_t steps = 0;              // MoE layer evaluations
    uint64_t pinned_experts = 0;
    double callback_s = 0;           // time in Kera's per-layer callback, waiting for reads included
    double guess_s = 0;              // ... of which working out the next layer's guesses
    // Union growth (spec Phase 6 item 4): for MoE layer evaluations of n words at once (n = 1 writes one
    // word; n > 1 checks n - 1 guesses), how many distinct experts they used and how many of those weren't
    // in memory yet (Kera's cache only). Index n, 1..max_batch.
    static constexpr int max_batch = 16;
    uint64_t batch_steps[max_batch + 1] = {};
    uint64_t batch_distinct[max_batch + 1] = {};
    uint64_t batch_missing[max_batch + 1] = {};
};

report operator-(const report & a, const report & b);

class session;

// Stream the experts of cfg.model_path the next time it is loaded in this process. Replaces any earlier
// session for new loads. Returns nullptr (and the reason in *error) if streaming can't be set up.
std::shared_ptr<session> start(const config & cfg, std::string * error = nullptr);

// Loads after this don't stream. Models already loaded keep streaming until freed.
void stop();

class session {
public:
    struct impl;
    explicit session(std::unique_ptr<impl> p);
    ~session();

    const config & cfg() const;
    bool streaming() const;         // a model is loaded and streams through this session
    std::string error() const;      // why the last load didn't stream, or why a read failed
    report snapshot() const;
    uint64_t expert_bytes_total() const; // the streamed experts' size on disk
    size_t moe_layers() const;           // MoE layers whose experts stream

    // Bring these experts of a MoE layer into memory now (one entry per word and choice, as the model's choice tensor
    // holds them): for a caller that changes a layer's choice after the stream hook saw it, as `kera exactness` does
    // when it takes the reference's choice at a near tie. Returns false if a read failed.
    bool bring_in(int layer, const int32_t * experts, size_t n);

    // "per word: 120 MB read from the SSD, 78% of experts already in memory, ..."
    static std::string per_word(const report & delta, uint64_t n_words);

    // Bytes a check of k guesses (k + 1 words at once) is expected to read from the SSD (spec Phase 6 item 2,
    // new_bytes(k) for a MoE model): the experts it is expected to use that aren't in memory, from the measured
    // union growth (or, before any check was seen, RT2's: 4 words use 2.9x one word's experts).
    double predicted_new_bytes(int k) const;
    // "checking 4 words at once used 2.9x one word's experts (...)"; empty before any check was seen
    static std::string union_growth(const report & delta);

    impl * get() { return p_.get(); }

private:
    std::unique_ptr<impl> p_;
};

} // namespace kera::stream
