// Speed planner (kr_plan.h): Appendix C with the model's real tensor sizes.

#include "planner/kr_plan.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace kera {

double ceiling_tok_s(double total_b, double active_b, double always_b, double bits, double usable_ram_gb,
                     double ram_gb_s, double ssd_gb_s, double hit_rate, bool overlap) {
    const double gb = bits / 8;
    const double always_gb = always_b * gb;
    const double per_token_gb = active_b * gb;
    double from_ssd;
    if (total_b == active_b) { // dense: every weight every token
        from_ssd = std::max(total_b * gb - usable_ram_gb, 0.0);
    } else {
        const double routed_gb = (active_b - always_b) * gb;
        const double pool_gb = (total_b - always_b) * gb;
        const double cache_gb = std::max(usable_ram_gb - always_gb, 0.0);
        const double h = hit_rate >= 0 ? hit_rate : std::min(cache_gb / pool_gb, 1.0);
        from_ssd = routed_gb * (1 - h);
    }
    const double ram_t = per_token_gb / ram_gb_s, ssd_t = from_ssd / ssd_gb_s;
    if (overlap) {
        return 1.0 / (std::max(ram_t, ssd_t) + 0.15 * std::min(ram_t, ssd_t));
    }
    return 1.0 / (ram_t + ssd_t);
}

namespace {

int64_t key_int(const gguf_context * ctx, const std::string & key, int64_t def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(ctx, id);
        default:               return def;
    }
}

// a number stored as a float or an integer; def when absent or of another type
double key_num(const gguf_context * ctx, const char * key, double def) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, id);
        case GGUF_TYPE_UINT32:  return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:   return gguf_get_val_i32(ctx, id);
        default:                return def;
    }
}

// "blk.12.ffn_up_exps.weight" -> 12, otherwise -1
int layer_of(const char * name) {
    if (std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return std::atoi(name + 4);
}

std::string fmt_gb(double gb) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), gb >= 10 ? "%.0f GB" : "%.1f GB", gb);
    return buf;
}

std::string fmt_tok_s(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), v >= 10 ? "%.0f" : "%.1f", v);
    return buf;
}

} // namespace

// "model-00001-of-00004.gguf" -> the paths of all 4 parts; any other name -> just itself
std::vector<std::string> model_parts(const std::string & path) {
    const size_t of = path.rfind("-of-");
    if (of == std::string::npos || of < 6 || path.compare(of - 6, 7, "-00001-") != 0 || path.size() < of + 4 + 5) {
        return { path };
    }
    const int n = std::atoi(path.substr(of + 4, 5).c_str());
    std::vector<std::string> parts;
    for (int i = 1; i <= n; ++i) {
        char num[16];
        std::snprintf(num, sizeof(num), "%05d", i);
        parts.push_back(path.substr(0, of - 5) + num + path.substr(of));
    }
    return parts.empty() ? std::vector<std::string>{ path } : parts;
}

bool read_model_sizes(const std::string & path, model_sizes & out, std::string * err) {
    model_sizes m;
    const size_t slash = path.find_last_of("/\\");
    m.file = slash == std::string::npos ? path : path.substr(slash + 1);
    bool has_output = false;
    double embd = 0;
    std::vector<double> experts_by_layer;
    const auto parts = model_parts(path);
    for (size_t part = 0; part < parts.size(); ++part) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(parts[part].c_str(), params);
    if (ctx == nullptr) {
        if (err) {
            *err = parts.size() > 1 ? "a part of this split model is missing or unreadable: " + parts[part]
                                    : "not a readable GGUF file";
        }
        return false;
    }
    if (part == 0) {
        // read as text only when stored as text: ggml stops the program on a value of another type
        auto key_str = [&](const char * key) -> std::string {
            const int64_t id = gguf_find_key(ctx, key);
            return id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING ? gguf_get_val_str(ctx, id) : "";
        };
        m.arch = key_str("general.architecture");
        m.name = key_str("general.name");
        m.n_layer = (int) key_int(ctx, m.arch + ".block_count", 0);
        m.n_embd = (int) key_int(ctx, m.arch + ".embedding_length", 0);
        m.basename = key_str("general.basename");
        m.size_label = key_str("general.size_label");
        m.finetune = key_str("general.finetune");
        for (int64_t k = 0; k < gguf_get_n_kv(ctx); ++k) {
            if (std::strncmp(gguf_get_key(ctx, k), "general.sampling.", 17) == 0) {
                m.gguf_any_sampling = true;
                break;
            }
        }
        m.n_expert = (int) key_int(ctx, m.arch + ".expert_count", 0);
        m.n_expert_used = (int) key_int(ctx, m.arch + ".expert_used_count", 0);
        m.n_layer_nextn = (int) key_int(ctx, m.arch + ".nextn_predict_layers", 0);
        m.gguf_temp = (float) key_num(ctx, "general.sampling.temp", -1);
        m.gguf_top_p = (float) key_num(ctx, "general.sampling.top_p", -1);
        m.gguf_top_k = (int) key_num(ctx, "general.sampling.top_k", -1);
        m.gguf_min_p = (float) key_num(ctx, "general.sampling.min_p", -1);
        m.gguf_file_type = (int) key_num(ctx, "general.file_type", -1);
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        const double gb = (double) gguf_get_tensor_size(ctx, i) / 1e9;
        m.file_gb += gb;
        // the MTP blocks are only read while guessing (llama.cpp checks for the same tensor)
        if (m.n_layer_nextn > 0 && layer_of(name) >= m.n_layer - m.n_layer_nextn) {
            m.mtp_gb += gb;
            m.has_mtp = m.has_mtp || std::strstr(name, ".nextn.eh_proj.weight") != nullptr;
            continue;
        }
        if (std::strstr(name, "_exps") != nullptr) {
            const int l = layer_of(name);
            if (l >= 0) {
                if ((int) experts_by_layer.size() <= l) {
                    experts_by_layer.resize((size_t) l + 1, 0.0);
                }
                experts_by_layer[(size_t) l] += gb;
            }
            m.experts_gb += gb;
        } else if (std::strncmp(name, "token_embd.", 11) == 0) {
            embd += gb;
        } else {
            has_output = has_output || std::strncmp(name, "output.weight", 13) == 0;
            m.always_gb += gb;
        }
    }
    gguf_free(ctx);
    }

    if (has_output) {
        m.embed_gb = embd;
    } else {
        m.always_gb += embd; // tied embeddings: the output layer reads the whole matrix every word
    }
    for (double e : experts_by_layer) {
        if (e > 0) {
            m.n_moe_layers++;
            if (m.n_expert > 0) {
                m.active_experts_gb += e * m.n_expert_used / m.n_expert;
            }
        }
    }
    out = m;
    return true;
}

double lru_hit_rate(double share) {
    // (cache share, hit rate) from docs/M1_RISK_TESTS.md, RT2
    static const double pts[][2] = { { 0, 0 }, { 0.05, 0.213 }, { 0.10, 0.413 }, { 0.25, 0.626 },
                                     { 0.50, 0.833 }, { 0.75, 0.947 }, { 1.0, 1.0 } };
    share = std::min(std::max(share, 0.0), 1.0);
    for (size_t i = 1; i < sizeof(pts) / sizeof(pts[0]); ++i) {
        if (share <= pts[i][0]) {
            const double t = (share - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]);
            return pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]);
        }
    }
    return 1.0;
}

plan_result plan(const model_sizes & m, const plan_input & in) {
    plan_result r;
    const double total = m.resident_gb() + m.experts_gb;
    r.ram_gb_per_word = m.per_word_gb();
    if (total <= in.usable_ram_gb) {
        r.fits = true;
        r.in_memory_gb = total;
        r.hit_rate = 1;
    } else if (!m.moe()) {
        // dense: every weight every word; what doesn't fit is read from the SSD each word
        r.in_memory_gb = in.usable_ram_gb;
        r.on_ssd_gb = total - in.usable_ram_gb;
        r.ssd_gb_per_word = r.on_ssd_gb;
        r.hit_rate = 0;
    } else {
        r.expert_cache_gb = std::max(in.usable_ram_gb - m.resident_gb(), 0.0);
        r.in_memory_gb = m.resident_gb() + r.expert_cache_gb;
        r.on_ssd_gb = m.experts_gb - r.expert_cache_gb;
        r.hit_rate = in.hit_rate >= 0 ? in.hit_rate : lru_hit_rate(r.expert_cache_gb / m.experts_gb);
        r.ssd_gb_per_word = m.active_experts_gb * (1 - r.hit_rate);
    }
    const double ram_t = in.ram_gb_s > 0 ? r.ram_gb_per_word / in.ram_gb_s : 0;
    const double ssd_t = in.ssd_gb_s > 0 ? r.ssd_gb_per_word / in.ssd_gb_s : (r.ssd_gb_per_word > 0 ? INFINITY : 0);
    r.ceiling_tok_s = ram_t + ssd_t > 0 ? 1.0 / (ram_t + ssd_t) : 0;
    const double ov = std::max(ram_t, ssd_t) + in.overlap_residual * std::min(ram_t, ssd_t);
    r.ceiling_overlap_tok_s = ov > 0 ? 1.0 / ov : 0;
    return r;
}

plan_input plan_input_from_profile(const machine_profile & p) {
    plan_input in;
    in.usable_ram_gb = p.usable_gb;
    in.ram_gb_s = p.ram_gbps.gpu > 0 ? p.ram_gbps.gpu : p.ram_gbps.cpu;
    for (const auto & d : p.drives) {
        if (d.measured) {
            in.ssd_gb_s = std::max(in.ssd_gb_s, d.read_gbps);
        }
    }
    return in;
}

std::vector<std::string> explain_plan(const model_sizes & m, const plan_input & in, const plan_result & r) {
    std::vector<std::string> out;
    std::string what = m.moe() ? "a mixture-of-experts model (" + std::to_string(m.n_expert_used) + " of " +
                                     std::to_string(m.n_expert) + " experts used for each word)"
                               : "a dense model (every word uses every weight)";
    out.push_back(m.file + " is " + fmt_gb(m.resident_gb() + m.experts_gb) + ", " + what + ".");
    if (m.moe()) {
        out.push_back("Each word reads about " + fmt_gb(m.per_word_gb()) + ": " + fmt_gb(m.always_gb) +
                      " that every word needs, plus " + fmt_gb(m.active_experts_gb) + " of experts.");
    } else {
        out.push_back("Each word reads about " + fmt_gb(m.per_word_gb()) + " of weights.");
    }
    if (r.fits) {
        out.push_back("It fits in the " + fmt_gb(in.usable_ram_gb) + " of memory this computer can give a model, "
                      "so nothing is read from the SSD while it runs.");
        out.push_back("Top speed: about " + fmt_tok_s(r.ceiling_tok_s) + " words a second (memory reads at " +
                      fmt_gb(in.ram_gb_s) + "/s). Real speed lands below this.");
        return out;
    }
    if (m.moe()) {
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%.0f%%", 100 * r.hit_rate);
        out.push_back("It doesn't fit in " + fmt_gb(in.usable_ram_gb) + " of memory. The part every word needs (" +
                      fmt_gb(m.resident_gb()) + ") stays in memory with " + fmt_gb(r.expert_cache_gb) +
                      " of experts; the other " + fmt_gb(r.on_ssd_gb) + " of experts are read from the SSD when needed.");
        out.push_back(std::string("About ") + pct + " of the experts a word needs will already be in memory (words "
                      "keep reusing recent experts), so each word reads about " + fmt_gb(r.ssd_gb_per_word) + " from the SSD.");
    } else {
        out.push_back("It doesn't fit in " + fmt_gb(in.usable_ram_gb) + " of memory, so each word reads the other " +
                      fmt_gb(r.ssd_gb_per_word) + " from the SSD.");
    }
    out.push_back("Top speed: about " + fmt_tok_s(r.ceiling_tok_s) + " words a second, or " +
                  fmt_tok_s(r.ceiling_overlap_tok_s) + " if SSD reads overlap with the maths. Real speed lands below this.");
    return out;
}

} // namespace kera
