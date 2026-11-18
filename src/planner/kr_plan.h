// Speed planner (spec Phase 1, Appendix C): the best tokens per second physics allows for a model on
// a machine, from the model's real tensor sizes and the machine profile auto-setup saved.
#pragma once

#include "setup/kr_machine.h"

#include <string>
#include <vector>

namespace kera {

// Appendix C, word for word: weights in billions and bits per weight. hit_rate < 0 means "estimate
// from how much of the model fits". Kept for the parity test with reference/planner.py.
double ceiling_tok_s(double total_b, double active_b, double always_b, double bits, double usable_ram_gb,
                     double ram_gb_s, double ssd_gb_s, double hit_rate = -1.0, bool overlap = false);

// What a model file holds, from its GGUF header (no weights are read). Sizes in GB (1e9 bytes).
struct model_sizes {
    std::string file;          // file name only
    std::string arch;
    std::string name;          // general.name, if set
    double file_gb = 0;        // all tensors
    double always_gb = 0;      // read for every word: attention, Mamba, norms, shared experts, dense layers, output
    double embed_gb = 0;       // token embeddings: kept in memory, but each word reads only its own row
    double experts_gb = 0;     // all routed experts
    double active_experts_gb = 0; // routed experts one word reads (experts used / experts, per MoE layer)
    int n_layer = 0;
    int n_embd = 0;            // the width of the model's vectors (embedding_length)
    std::string basename, size_label, finetune; // general.basename, .size_label, .finetune: the name's parts
    bool gguf_any_sampling = false; // the file sets any general.sampling.* value of its own
    int n_moe_layers = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_layer_nextn = 0;     // the last blocks that are the model's own guesser (MTP), not read for each word
    double mtp_gb = 0;         // those blocks' weights
    bool has_mtp = false;      // the file carries an MTP head llama.cpp can guess with
    // the file's own sampling defaults (general.sampling.*, from its publisher's generation_config.json); < 0: absent
    float gguf_temp = -1.0f;
    float gguf_top_p = -1.0f;
    int gguf_top_k = -1;
    float gguf_min_p = -1.0f;
    int gguf_file_type = -1; // general.file_type: 0 all F32, 1 mostly F16, 32 mostly BF16, others quantized; < 0 unknown
    bool quantized() const { return gguf_file_type >= 0 && gguf_file_type != 0 && gguf_file_type != 1 && gguf_file_type != 32; }
    bool moe() const { return experts_gb > 0; }
    double per_word_gb() const { return always_gb + active_experts_gb; }
    double resident_gb() const { return always_gb + embed_gb + mtp_gb; } // what must stay in memory
};

// A split model's parts: "model-00001-of-00004.gguf" -> the paths of all 4; any other name -> just itself.
std::vector<std::string> model_parts(const std::string & path);

// Reads the GGUF header. Returns false (with a reason in *err) if the file isn't a readable GGUF.
bool read_model_sizes(const std::string & path, model_sizes & out, std::string * err = nullptr);

// The share of a word's experts already in memory when an LRU expert cache holds this share of each
// MoE layer's experts. Measured on routing traces (M1 risk test 2: Nemotron 3 Nano, 64 chats and an agent
// session, 38,558 words): words reuse recent experts, so a cache of a quarter of the experts finds about
// 63% of them, not 25%. Between the measured points the curve is interpolated.
double lru_hit_rate(double cache_share);

struct plan_input {
    double usable_ram_gb = 0;   // memory a model may use (machine profile usable_gb)
    double ram_gb_s = 0;        // memory read speed of the chip doing the maths
    double ssd_gb_s = 0;        // the model drive's read speed
    double hit_rate = -1;       // share of expert reads found in memory; < 0 = from the cache size (lru_hit_rate)
    double overlap_residual = 0.15; // the part of the shorter of RAM and SSD time that never hides (Appendix C)
};

struct plan_result {
    bool fits = false;          // everything fits in memory: nothing is read from the SSD
    double in_memory_gb = 0;    // what is kept in memory (all of it, or the always-used part plus a cache)
    double on_ssd_gb = 0;       // what stays on the SSD and is read as needed
    double expert_cache_gb = 0; // memory left for experts once the always-used part is in
    double hit_rate = 1;        // share of expert reads found in memory
    double ram_gb_per_word = 0; // GB read from memory per word
    double ssd_gb_per_word = 0; // GB read from the SSD per word
    double ceiling_tok_s = 0;           // SSD reads and maths one after the other
    double ceiling_overlap_tok_s = 0;   // SSD reads overlapping the maths (section 10.5)
};

plan_result plan(const model_sizes & m, const plan_input & in);

// The planner's inputs from a saved machine profile: usable memory, the graphics chip's memory speed
// (or the CPU's when there is no usable graphics chip), and the fastest measured drive.
plan_input plan_input_from_profile(const machine_profile & p);

// Plain-words lines for the app and `kera plan` (section 4a step 5).
std::vector<std::string> explain_plan(const model_sizes & m, const plan_input & in, const plan_result & r);

} // namespace kera
