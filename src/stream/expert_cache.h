// Kera's expert cache (spec Phase 2 "v2", decision D-012): a MoE model's routed experts are read from the
// SSD with direct reads straight into the mirror of the model's files (region.h), kept in RAM within a
// budget, and given back to the OS least recently used first. A MoE layer's chosen experts are locked
// until the next layer asks, so nothing the model is using can be evicted.
//
// Reads run on a pool of threads in three queues: what the model is waiting for, guesses for the next
// layer (lookahead, Phase 3), and pinned experts loading in the background (Phase 4). A guess that turned
// out wrong is dropped before it is read if the layer it was for has already run.
#pragma once

#include "io/kr_os_file.h"
#include "stream/region.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace kera::stream {

// One expert matrix of a layer (gate, up, down or gate_up): expert e's bytes are
// [file_offset + e*stride, file_offset + (e+1)*stride) in its file.
struct part_layout {
    int file = 0;
    uint64_t file_offset = 0;
    uint64_t stride = 0;
};

struct layer_layout {
    int layer = 0; // the model's layer number
    int n_expert = 0;
    std::vector<part_layout> parts;
};

struct cache_options {
    uint64_t budget_bytes = 8ull << 30; // RAM the cached experts may use
    int io_threads = 8;                 // reads in flight
    uint32_t chunk_bytes = 1u << 20;    // the largest single read
    bool direct = true;                 // bypass the OS file cache
};

struct cache_stats {
    uint64_t steps = 0;              // MoE layer evaluations
    uint64_t requests = 0;           // experts asked for (once per layer evaluation)
    uint64_t hits = 0;               // in memory when asked for
    uint64_t late = 0;               // already being read when asked for (an early read that wasn't early enough)
    uint64_t misses = 0;             // not in memory and not being read
    uint64_t prefetched = 0;         // early reads started (lookahead, pins)
    uint64_t prefetch_used = 0;      // ... whose expert was then asked for
    uint64_t prefetch_cancelled = 0; // guesses dropped before being read
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t overflow_steps = 0;     // layer evaluations that needed more than the whole budget
    double io_wait_s = 0;            // time the model spent waiting for reads
    uint64_t used_bytes = 0;         // RAM the cache holds now (resident and being read)
    uint64_t pinned_bytes = 0;
    uint64_t budget_bytes = 0;
};

class expert_cache {
public:
    expert_cache() = default;
    ~expert_cache();
    expert_cache(const expert_cache &) = delete;
    expert_cache & operator=(const expert_cache &) = delete;

    // files[i]: the model's file i; file_base[i]: where it starts in the mirror (a multiple of the page
    // size); mirrors[i]: byte-identical copies of file i on other drives (Phase 5), may be empty.
    bool open(const std::vector<std::string> & files, const std::vector<uint64_t> & file_base,
              const std::vector<std::vector<std::string>> & mirrors, region * mirror,
              const std::vector<layer_layout> & layers, const cache_options & opt, std::string * error);
    void close();

    // A MoE layer is about to use these experts (one entry per token and choice; repeats are fine).
    // Returns once all of them are in memory, false if a read failed. The previous call's experts are
    // unlocked.
    bool require(int layer, const int32_t * experts, size_t n);

    enum class reason { lookahead, pin };

    // Experts that will probably be needed soon: read them early if the budget allows.
    void prefetch(int layer, const int32_t * experts, size_t n, reason why);

    // Keep experts in memory: for good (pin) or until unpin_prompt() (prompt). Absent ones are read in
    // the background. Returns how many bytes are pinned after the call.
    uint64_t pin(int layer, const int32_t * experts, size_t n, bool prompt);
    void unpin_prompt();

    bool is_resident(int layer, int expert) const;
    bool streams(int layer) const;
    uint64_t expert_bytes(int layer) const; // RAM one expert of the layer takes (whole pages)
    int n_expert(int layer) const;
    const std::vector<layer_layout> & layers() const { return layers_; }

    // How often each expert of a layer was asked for since open (the usage log, Phase 4).
    std::vector<uint32_t> usage(int layer) const;

    cache_stats stats() const;
    std::string error_message() const; // why a read failed, if one did

    // Where an expert's part is in the mirror (for tests).
    const uint8_t * part_data(int layer, int expert, int part) const;

private:
    enum : uint8_t { ABSENT = 0, QUEUED = 1, LOADING = 2, RESIDENT = 3 };
    static constexpr int k_short_read = 1000; // a read status: fewer bytes came back than the file holds there
    enum : uint8_t { Q_DEMAND = 0, Q_AHEAD = 1, Q_BACKGROUND = 2 };

    struct expert {
        uint8_t state = ABSENT;
        uint8_t pin = 0;           // 1: pinned, 2: pinned for the current reply
        bool early = false;        // read by prefetch and not asked for yet
        bool failed = false;
        uint32_t lock = 0;         // == epoch_: the current layer uses it
        int32_t prev = -1;         // LRU list of resident, unpinned experts
        int32_t next = -1;
        uint32_t chunks_left = 0;
        uint32_t chunks_started = 0;
        uint32_t chunks_total = 0;
        uint32_t uses = 0;
        uint64_t bytes = 0;        // whole pages
    };

    struct chunk {
        int32_t id;
        int file;
        uint64_t mirror_off;
        uint32_t len;
        uint64_t own_lo; // the part's own bytes in the mirror: the rest of a chunk's first and last page belongs
        uint64_t own_hi; // to a neighbour, and is never written by this read (N-012)
    };

    struct span { // an expert part's pages in the mirror
        uint64_t lo;
        uint64_t hi;
        uint64_t own_lo; // and its own bytes in them
        uint64_t own_hi;
    };

    int id_of(int layer, int e) const;
    void spans_of(int32_t id, std::vector<span> & out) const;
    void lru_remove(int32_t id);
    void lru_push(int32_t id);
    bool make_room(uint64_t bytes, bool force);
    void evict(int32_t id);
    void release_pages(int32_t id);
    void start_load(int32_t id, uint8_t q);
    void promote(int32_t id);
    void cancel_guesses(int li, const std::vector<uint8_t> & wanted);
    void worker();
    int read_chunk(const chunk & c, int copy, uint8_t * bounce, size_t * done);
    int pick_copy(int file);

    std::vector<std::string> files_;
    std::vector<uint64_t> file_base_;
    std::vector<std::vector<os::file>> copies_;   // [file][0 = the file itself, 1.. = mirrors]
    std::vector<std::vector<uint64_t>> inflight_; // bytes being read from each copy
    region * mirror_ = nullptr;
    std::vector<layer_layout> layers_;
    std::vector<int> layer_index_; // model layer -> index in layers_, -1 if not streamed
    std::vector<int32_t> first_id_;
    std::vector<uint64_t> expert_bytes_;
    cache_options opt_;
    uint64_t page_ = 4096;

    mutable std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::vector<std::thread> threads_;
    bool stop_ = false;

    std::vector<expert> experts_;
    std::deque<chunk> queues_[3];
    std::unordered_map<uint64_t, uint32_t> edge_refs_; // pages shared with a neighbouring part
    int32_t lru_head_ = -1;
    int32_t lru_tail_ = -1;
    uint32_t epoch_ = 0;
    uint64_t used_ = 0;
    uint64_t pinned_ = 0;
    std::string error_msg_;
    cache_stats st_;
};

} // namespace kera::stream
