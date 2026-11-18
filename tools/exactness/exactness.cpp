// `kera exactness MODEL.gguf GOLDEN.json`: checks a model against the reference code's output
// (spec P2 tiny models, section 6 rules) on the CPU.
//
// One pass over the prompt plus the reference's own words (teacher forcing) gives the next-word
// choice at every step. Each step must pick the reference's word, or differ only where the reference's
// top two were within 0.1 percentage points (a near-tie). The log-probabilities of the reference's
// top 5 words must also agree within --tol, which catches subtle errors the choice alone would hide.
// With --gpu the model runs on the graphics chip instead, which section 6 holds to the same rules about words; the
// log-probabilities get a looser limit there (0.25), as a chip rounds differently. If the check fails on the chip, the
// tool looks for near ties inside the model (a MoE router's experts, sparse attention's chosen words): it runs the CPU
// to keep its choices, then the chip again taking the CPU's choice wherever the chip's differed at a near tie, and
// judges that run (N-010). A choice that differs without being a near tie, or a difference left after, fails.
// With --stream os|kera the routed experts are streamed (Phase 2 gate (a)); --stream-budget-mb sets Kera's
// cache small enough to make it evict, and --ubatch N feeds the text N words at a time (1: word by word).
// --self N OUT.json writes a golden file from llama.cpp's own greedy run, one word at a time, from GOLDEN's
// prompt: checking it with --ubatch 512 then shows whether reading many words at once (as a guess check
// does) picks the same words as writing them one by one. --kv f16|q8_0 keeps the model's memory of the text
// at that precision (the server's default is f16; the reference's is 32-bit, the default here; q8_0 turns flash
// attention on, which llama.cpp needs for it).

#include "llama.h"
#include "ggml-backend.h"

#include "stream/kr_stream.h"
#include "app/kera_app.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

// --dump NAME FILE: save one named tensor of the graph (e.g. "nemotron_h_block_out-0") as raw float32,
// to compare a single layer with the reference code when a model doesn't match.
// --dump-all DIR: every float32 tensor the graph computes, in order, as DIR/NNNNN.f32, listed in DIR/index.tsv
// (number, name, operation, shape): run once on the CPU and once on a graphics chip, then
// tools/exactness/first_difference.py shows the first operation where the two part ways (N-010). Also every node's
// memory (DIR/ranges.tsv) and each product's with its inputs' (DIR/products.tsv), and a warning on stderr if a
// product's result was placed over one of its inputs (N-012).
struct dump_target {
    std::string name;
    std::string file;
    bool done = false;
    std::string dir;
    int n = 0;
};

bool dump_all_cb(struct ggml_tensor * t, bool ask, dump_target * d) {
    if (ask) {
        return true; // every node: its memory is listed in ranges.tsv; float32 ones are saved too
    }
    if (FILE * f = std::fopen((d->dir + "/ranges.tsv").c_str(), "a")) {
        const ggml_tensor * b = t->view_src ? t->view_src : t;
        std::fprintf(f, "%s\t%s\t%p\t%p\t%p\n", t->name, ggml_op_desc(t), (const void *) b->buffer, t->data,
                     (const void *) ((const uint8_t *) t->data + ggml_nbytes(t)));
        std::fclose(f);
    }
    // a product's result must never share memory with what it reads (N-012): say so if the allocator placed it there
    if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
        auto buffer_of = [](const ggml_tensor * x) { return x->view_src ? x->view_src->buffer : x->buffer; };
        const auto * lo = (const uint8_t *) t->data;
        const auto * hi = lo + ggml_nbytes(t);
        std::string where;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * s = t->src[j];
            if (s == nullptr || s->data == nullptr) {
                continue;
            }
            const auto * slo = (const uint8_t *) s->data;
            const auto * shi = slo + ggml_nbytes(s);
            char part[160];
            std::snprintf(part, sizeof(part), " src%d %s [%p, %p)", j, s->name, (const void *) slo, (const void *) shi);
            where += part;
            if (buffer_of(s) == buffer_of(t) && slo < hi && lo < shi) {
                std::fprintf(stderr, "kera-overlap: %s (%s) writes [%p, %p), which overlaps its input %d %s [%p, %p)\n",
                             t->name, ggml_op_desc(t), (const void *) lo, (const void *) hi, j, s->name,
                             (const void *) slo, (const void *) shi);
            }
        }
        if (FILE * f = std::fopen((d->dir + "/products.tsv").c_str(), "a")) {
            std::fprintf(f, "%d\t%s\t%s\t[%p, %p)\t%s\n", d->n, t->name, ggml_op_desc(t), (const void *) lo,
                         (const void *) hi, where.c_str());
            std::fclose(f);
        }
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return true;
    }
    std::vector<float> buf((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    char file[64];
    std::snprintf(file, sizeof(file), "%05d.f32", d->n);
    if (FILE * f = std::fopen((d->dir + "/" + file).c_str(), "wb")) {
        std::fwrite(buf.data(), sizeof(float), buf.size(), f);
        std::fclose(f);
    }
    if (FILE * f = std::fopen((d->dir + "/index.tsv").c_str(), "a")) {
        std::fprintf(f, "%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\n", d->n, t->name, ggml_op_desc(t), (long long) t->ne[0],
                     (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        std::fclose(f);
    }
    d->n++;
    return true;
}

bool dump_cb(struct ggml_tensor * t, bool ask, void * user) {
    auto * d = (dump_target *) user;
    if (!d->dir.empty()) {
        return dump_all_cb(t, ask, d);
    }
    if (d->done || d->name != t->name) {
        return false;
    }
    if (ask) {
        return true; // yes, show it to us after it is computed
    }
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        std::vector<float> buf((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        if (FILE * f = std::fopen(d->file.c_str(), "wb")) {
            std::fwrite(buf.data(), sizeof(float), buf.size(), f);
            std::fclose(f);
        }
        std::printf("dumped %s [%lld x %lld] to %s\n", t->name, (long long) t->ne[0], (long long) t->ne[1], d->file.c_str());
        d->done = true;
    }
    return true;
}

// Choices a model makes inside: a MoE router picking each word's experts, sparse attention's indexer picking the
// earlier words each word looks at. On a graphics chip a near tie in one can go the other way and change later words a
// lot (N-010), which the words alone don't show. So when a check on a graphics chip fails, the CPU run's scores and
// choices are kept (pass A), and the graphics chip runs again (pass B), taking the CPU's choice wherever its own
// differed at a near tie (the same idea as feeding in the reference's words). What still differs is a real difference.
struct inner_choices {
    const inner_choices * ref = nullptr;                // pass B: the CPU's, to compare with and take at near ties
    kera::stream::session * stream = nullptr;           // pass B: brings in the experts of a choice taken
    // keyed "router-3#0": the layer's choice in the first batch of words the model computed, "#1" the next
    std::map<std::string, std::vector<float>> scores;   // [rows][n_cand]
    std::map<std::string, int64_t> n_cand;
    std::map<std::string, std::vector<int32_t>> chosen; // [rows][k]
    std::map<std::string, int64_t> k;
    std::map<std::string, std::vector<float>> pending;  // "router-3": the scores seen since its last choice
    std::map<std::string, int64_t> pending_cand;
    std::map<std::string, int> count;                   // "router-3": its choices seen so far
    int forced = 0;                                     // pass B: choices taken from the CPU at a near tie
    int real = 0;                                       // pass B: choices that differ and weren't near ties
    double max_dev = 0.0;                               // pass B: the chip's largest score difference from the CPU's, relative
    std::string first_forced;
    std::string first_real;
};

// "ffn_moe_probs_biased-3" -> "router-3", a score; "ffn_moe_topk-3" -> "router-3", a choice. A choice counts the
// last scores of its kind and layer computed before it (probs, then biased, then masked, as llama.cpp builds them).
bool inner_key(const char * name, bool & is_scores, std::string & key) {
    static const struct { const char * prefix; const char * kind; bool scores; } names[] = {
        { "ffn_moe_probs-", "router-", true },  { "ffn_moe_probs_biased-", "router-", true },
        { "ffn_moe_probs_masked-", "router-", true }, { "ffn_moe_topk-", "router-", false },
        { "indexer_score-", "indexer-", true }, { "top_k-", "indexer-", false },
    };
    for (const auto & n : names) {
        const size_t len = std::strlen(n.prefix);
        if (std::strncmp(name, n.prefix, len) != 0 || name[len] == '\0') {
            continue;
        }
        const char * p = name + len;
        while (*p >= '0' && *p <= '9') {
            ++p;
        }
        if (*p != '\0') {
            continue;
        }
        is_scores = n.scores;
        key = std::string(n.kind) + (name + len);
        return true;
    }
    return false;
}

// a tensor's rows of ne[0] numbers, in order, whatever its strides
template <typename T> std::vector<T> read_rows(const ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<T> out;
    out.reserve((size_t) ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    T v;
                    std::memcpy(&v, raw.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], sizeof(T));
                    out.push_back(v);
                }
            }
        }
    }
    return out;
}

// Two scores close enough that rounding may order them either way: within 0.1% (as section 6's 0.1 percentage
// points for a word's probability). Two masked candidates (-inf) are a tie too: either is ignored.
bool near_tie(float a, float b) {
    if (std::isinf(a) && std::isinf(b)) {
        return a == b;
    }
    return std::fabs((double) a - (double) b) <= 1e-3 * std::max({ 1.0, std::fabs((double) a), std::fabs((double) b) });
}

// How far the chip's score is from the CPU's, relative to the score's size (infinite if one is masked and one isn't)
double score_dev(float cpu, float chip) {
    if (std::isinf(cpu) || std::isinf(chip)) {
        return cpu == chip ? 0.0 : INFINITY;
    }
    return std::fabs((double) chip - (double) cpu) / std::max(1.0, std::fabs((double) cpu));
}


bool inner_cb(struct ggml_tensor * t, bool ask, void * user) {
    auto * ic = (inner_choices *) user;
    bool is_scores = false;
    std::string key;
    if (!inner_key(t->name, is_scores, key)) {
        return !ask;
    }
    const bool f32 = t->type == GGML_TYPE_F32;
    const bool i32 = t->type == GGML_TYPE_I32;
    if (ask) {
        return is_scores ? f32 : i32;
    }
    if (is_scores) {
        if (f32) {
            ic->pending[key] = read_rows<float>(t);
            ic->pending_cand[key] = t->ne[0];
        }
        return true;
    }
    if (!i32) {
        return true;
    }
    // this choice's scores are the last ones of its kind and layer since its previous choice
    const std::string base = key;
    key += "#" + std::to_string(ic->count[base]++);
    if (ic->pending.count(base)) {
        ic->scores[key] = std::move(ic->pending[base]);
        ic->n_cand[key] = ic->pending_cand[base];
        ic->pending.erase(base);
    }
    std::vector<int32_t> got = read_rows<int32_t>(t);
    const int64_t k = t->ne[0];
    if (ic->ref == nullptr) {
        ic->chosen[key] = std::move(got);
        ic->k[key] = k;
        return true;
    }
    // pass B: compare with the CPU's choice, row by row (one row per word)
    const auto rc = ic->ref->chosen.find(key);
    const auto rs = ic->ref->scores.find(key);
    if (rc == ic->ref->chosen.end() || rs == ic->ref->scores.end() || ic->ref->k.at(key) != k ||
        rc->second.size() != got.size() || t->nb[0] != sizeof(int32_t)) {
        return true; // nothing to compare with (a shared indexer layer, another batch shape)
    }
    const int64_t n_cand = ic->ref->n_cand.at(key);
    const int64_t rows = (int64_t) got.size() / std::max<int64_t>(1, k);
    if ((int64_t) rs->second.size() != rows * n_cand) {
        return true;
    }
    // the chip's own scores for this choice, from this run (computed just before it)
    const auto gs = ic->scores.find(key);
    const float * chip_scores = gs != ic->scores.end() && gs->second.size() == rs->second.size() ? gs->second.data() : nullptr;
    if (chip_scores != nullptr) {
        for (size_t i = 0; i < rs->second.size(); ++i) {
            const double d = score_dev(rs->second[i], chip_scores[i]);
            if (std::isfinite(d)) {
                ic->max_dev = std::max(ic->max_dev, d);
            }
        }
    }
    bool taken = false;
    for (int64_t r = 0; r < rows; ++r) {
        const int32_t * g = got.data() + r * k;
        const int32_t * c = rc->second.data() + r * k;
        const float * sc = rs->second.data() + r * n_cand;
        std::vector<int32_t> only_cpu, only_gpu;
        for (int64_t i = 0; i < k; ++i) {
            if (std::find(g, g + k, c[i]) == g + k) {
                only_cpu.push_back(c[i]);
            }
            if (std::find(c, c + k, g[i]) == c + k) {
                only_gpu.push_back(g[i]);
            }
        }
        if (only_cpu.empty()) {
            continue;
        }
        auto score = [&](int32_t e) { return e >= 0 && e < n_cand ? sc[e] : NAN; };
        bool tie = only_cpu.size() == only_gpu.size();
        for (const int32_t a : only_cpu) {
            bool any = false;
            for (const int32_t b : only_gpu) {
                // only a near tie by the CPU's own scores: a chip's rounding (about 4 in 10,000 on Vulkan) can't swap
                // candidates further apart (a looser rule let scores 0.02 apart pass, N-018)
                any = any || near_tie(score(a), score(b));
            }
            tie = tie && any;
        }
        char note[256];
        std::snprintf(note, sizeof(note), "%s, word %lld of batch %d: the CPU chose %d (scored %.6g), the graphics chip %d "
                      "(%.6g by the CPU)", base.c_str(), (long long) r, ic->count[base], only_cpu[0], score(only_cpu[0]),
                      only_gpu.empty() ? -1 : only_gpu[0], only_gpu.empty() ? NAN : score(only_gpu[0]));
        if (!tie) {
            if (ic->real++ == 0) {
                ic->first_real = note;
            }
            continue;
        }
        if (ic->forced++ == 0) {
            ic->first_forced = note;
        }
        // take the CPU's choice for this word, so what follows is computed as the reference did
        const int64_t i1 = r % t->ne[1];
        const int64_t i2 = (r / t->ne[1]) % t->ne[2];
        const int64_t i3 = r / (t->ne[1] * t->ne[2]);
        ggml_backend_tensor_set(t, c, (size_t) (i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]), (size_t) k * sizeof(int32_t));
        std::copy(c, c + k, got.begin() + r * k);
        taken = true;
    }
    // streamed experts: the ones the CPU chose may not be in memory yet
    if (taken && ic->stream != nullptr && base.compare(0, 7, "router-") == 0 &&
        !ic->stream->bring_in(std::atoi(base.c_str() + 7), got.data(), got.size())) {
        if (ic->real++ == 0) {
            ic->first_real = "the experts of " + base + " couldn't be read: " + ic->stream->error();
        }
    }
    return true;
}

} // namespace

int kera_exactness(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: kera exactness MODEL.gguf GOLDEN.json [--tol 0.02] [--gpu] [--dump TENSOR FILE.f32] [--dump-all DIR]\n"
                             "                      [--stream off|os|kera] [--stream-budget-mb N] [--stream-lookahead 0|1] [--ubatch N] [--cpu-moe]\n");
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string golden_path = argv[2];
    double tol = 0.02;
    bool tol_given = false;
    bool gpu = false;
    dump_target dump;
    kera::stream::config sc;
    sc.mode = kera::stream::cache_mode::off;
    sc.model_path = model_path;
    sc.io_threads = 4;
    sc.chunk_bytes = 64 * 1024;
    int ubatch = 0;
    int self_n = 0;
    std::string self_out;
    ggml_type kv_type = GGML_TYPE_F32; // the reference keeps its memory of the text in 32-bit too
    std::string backend;
    bool cpu_moe = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpu-moe") {
            cpu_moe = true; // with --gpu: the routed experts stay in the CPU's memory, as llama.cpp's --cpu-moe
        } else if (a == "--tol" && i + 1 < argc) {
            tol = std::atof(argv[++i]);
            tol_given = true;
        } else if (a == "--gpu") {
            gpu = true;
        } else if (a == "--stream" && i + 1 < argc) {
            if (!kera::stream::parse_mode(argv[++i], sc.mode)) {
                std::fprintf(stderr, "--stream takes off, os or kera\n");
                return 2;
            }
        } else if (a == "--stream-budget-mb" && i + 1 < argc) {
            sc.budget_bytes = (uint64_t) (std::atof(argv[++i]) * 1024 * 1024);
        } else if (a == "--stream-lookahead" && i + 1 < argc) {
            sc.lookahead = std::atoi(argv[++i]) != 0;
        } else if (a == "--ubatch" && i + 1 < argc) {
            ubatch = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--backend" && i + 1 < argc) {
            backend = argv[++i]; // with --gpu: vulkan or hip (the optional ROCm add-on), to check each one
        } else if (a == "--kv" && i + 1 < argc) {
            const std::string t = argv[++i];
            kv_type = t == "f16" ? GGML_TYPE_F16 : t == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        } else if (a == "--self" && i + 2 < argc) {
            self_n = std::max(1, std::atoi(argv[i + 1]));
            self_out = argv[i + 2];
            i += 2;
        } else if (a == "--dump" && i + 2 < argc) {
            dump.name = argv[i + 1];
            dump.file = argv[i + 2];
            i += 2;
        } else if (a == "--dump-all" && i + 1 < argc) {
            dump.dir = argv[++i];
            std::error_code ec;
            std::filesystem::create_directories(dump.dir, ec);
            std::filesystem::remove(dump.dir + "/index.tsv", ec);
        } else {
            std::fprintf(stderr, "Unknown option %s\n", a.c_str());
            return 2;
        }
    }

    // On a graphics chip section 6 judges the words (with its near-tie rule); the log-probabilities are held to a looser
    // limit than on the CPU, since a chip rounds differently: on the Minisforum Vulkan differs from the reference by
    // 0.03-0.12, ROCm by under 0.001 (N-010). Broken maths shows as wrong words or far bigger differences.
    if (gpu && !tol_given) {
        tol = 0.25;
    }

    nlohmann::json g;
    {
        std::ifstream f(golden_path);
        if (!f) {
            std::fprintf(stderr, "Can't read %s\n", golden_path.c_str());
            return 2;
        }
        g = nlohmann::json::parse(f);
    }
    const std::vector<int> prompt = g["prompt"].get<std::vector<int>>();
    const auto & steps = g["steps"];
    const int n_new = self_n > 0 ? self_n : (int) steps.size();

    llama_backend_init();
    ggml_backend_load_all();

    std::shared_ptr<kera::stream::session> stream;
    if (sc.mode != kera::stream::cache_mode::off) {
        std::string err;
        stream = kera::stream::start(sc, &err);
        if (!stream) {
            std::fprintf(stderr, "Can't stream: %s\n", err.c_str());
            return 2;
        }
    }

    std::string device_name;
    if (!backend.empty()) {
        std::string err;
        if (!kera::choose_backend(backend, device_name, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 2;
        }
    }
    if (gpu) {
        bool found = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
            found = found || type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        }
        if (!found) {
            std::printf("SKIP %s: no graphics chip found\n", g.value("design", std::string("?")).c_str());
            llama_backend_free();
            return 0;
        }
    }

    // CPU only: no devices for weights, and nothing offloaded, as section 6's reference run.
    // --gpu: every layer on the graphics chip.
    ggml_backend_dev_t no_devices[] = { nullptr };
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t chosen[] = { nullptr, nullptr };
    if (gpu) {
        mp.n_gpu_layers = 999;
        if (!device_name.empty() && device_name != "none") {
            chosen[0] = ggml_backend_dev_by_name(device_name.c_str());
            if (chosen[0] == nullptr) {
                std::fprintf(stderr, "No %s device found\n", device_name.c_str());
                return 2;
            }
            mp.devices = chosen; // only that back-end's device
        }
    } else {
        mp.devices = no_devices;
        mp.n_gpu_layers = 0;
    }
    // --cpu-moe: llama.cpp's own way to keep the experts on the CPU (no streaming), to tell a back-end's handling of
    // weights in the CPU's memory apart from Kera's (docs/NEGATIVE_RESULTS.md N-012)
    const llama_model_tensor_buft_override cpu_moe_overrides[] = {
        { "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type() }, // LLM_FFN_EXPS_REGEX
        { nullptr, nullptr },
    };
    if (cpu_moe) {
        mp.tensor_buft_overrides = cpu_moe_overrides;
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) {
        std::fprintf(stderr, "Can't load %s\n", model_path.c_str());
        return 2;
    }
    if (stream && !stream->streaming()) {
        std::fprintf(stderr, "The model loaded but doesn't stream: %s\n",
                     stream->error().empty() ? "it has no routed experts" : stream->error().c_str());
        return 2;
    }

    const int n_tokens = (int) prompt.size() + n_new - 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) std::max(256, n_tokens + 16);
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = ubatch > 0 ? (uint32_t) ubatch : cp.n_ctx;
    cp.n_threads = cp.n_threads_batch = (int32_t) std::max(1u, std::thread::hardware_concurrency());
    cp.type_k = kv_type;
    cp.type_v = kv_type;
    // off, as the reference computes attention; llama.cpp needs it on to keep the memory of the text in 8 bits
    cp.flash_attn_type = ggml_is_quantized(kv_type) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = gpu;
    cp.op_offload = gpu;
    if (!dump.name.empty() || !dump.dir.empty()) {
        cp.cb_eval = dump_cb;
        cp.cb_eval_user_data = &dump;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "Can't create a context\n");
        llama_model_free(model);
        return 2;
    }

    if (self_n > 0) {
        // the prompt in one pass, then one word at a time, keeping each step's top 5
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_batch b = llama_batch_init((int) prompt.size(), 0, 1);
        for (size_t i = 0; i < prompt.size(); ++i) {
            b.token[i] = prompt[i];
            b.pos[i] = (llama_pos) i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = i + 1 == prompt.size();
        }
        b.n_tokens = (int) prompt.size();
        nlohmann::json out_steps = nlohmann::json::array();
        std::vector<int> gen;
        int last = b.n_tokens - 1;
        bool ok = llama_decode(ctx, b) == 0;
        for (int s = 0; ok && s < self_n; ++s) {
            const float * lg = llama_get_logits_ith(ctx, last);
            std::vector<int> order((size_t) n_vocab);
            for (int v = 0; v < n_vocab; ++v) {
                order[(size_t) v] = v;
            }
            std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int x, int y) { return lg[x] > lg[y]; });
            double mx = lg[order[0]];
            double sum = 0;
            for (int v = 0; v < n_vocab; ++v) {
                sum += std::exp((double) lg[v] - mx);
            }
            const double lse = mx + std::log(sum);
            nlohmann::json ids = nlohmann::json::array();
            nlohmann::json lps = nlohmann::json::array();
            for (int k = 0; k < 5; ++k) {
                ids.push_back(order[(size_t) k]);
                lps.push_back((double) lg[order[(size_t) k]] - lse);
            }
            const double gap = std::exp((double) lg[order[0]] - lse) - std::exp((double) lg[order[1]] - lse);
            out_steps.push_back({ { "token", order[0] }, { "top_ids", ids }, { "top_logp", lps }, { "gap", gap } });
            gen.push_back(order[0]);
            llama_batch one = llama_batch_init(1, 0, 1);
            one.token[0] = order[0];
            one.pos[0] = (llama_pos) (prompt.size() + (size_t) s);
            one.n_seq_id[0] = 1;
            one.seq_id[0][0] = 0;
            one.logits[0] = 1;
            one.n_tokens = 1;
            ok = llama_decode(ctx, one) == 0;
            llama_batch_free(one);
            last = 0;
        }
        llama_batch_free(b);
        if (!ok) {
            std::fprintf(stderr, "llama_decode failed\n");
            return 2;
        }
        nlohmann::json o = { { "design", "llama.cpp, one word at a time" }, { "prompt", prompt }, { "generated", gen },
                             { "steps", out_steps } };
        std::ofstream(self_out) << o.dump() << "\n";
        std::printf("wrote %s: %d words, one at a time\n", self_out.c_str(), self_n);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        const int tok = i < (int) prompt.size() ? prompt[(size_t) i] : steps[(size_t) (i - (int) prompt.size())]["token"].get<int>();
        batch.token[i] = tok;
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i >= (int) prompt.size() - 1;
    }
    batch.n_tokens = n_tokens;
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed%s%s\n", stream && !stream->error().empty() ? ": " : "",
                     stream ? stream->error().c_str() : "");
        return 2;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    struct step_counts {
        int exact = 0;
        int near_ties = 0;
        int wrong = 0;
        double max_diff = 0.0;
    };
    // each step against the reference: the word picked, and the reference's top 5 log-probabilities
    auto check_steps = [&](llama_context * c, const char * label) {
        step_counts res;
        int & exact = res.exact;
        int & near_ties = res.near_ties;
        int & wrong = res.wrong;
        double & max_diff = res.max_diff;
        std::vector<double> logp((size_t) n_vocab);
        for (int s = 0; s < n_new; ++s) {
            const float * logits = llama_get_logits_ith(c, (int) prompt.size() - 1 + s);
            double mx = -INFINITY;
            int arg = 0;
            for (int v = 0; v < n_vocab; ++v) {
                if (logits[v] > mx) {
                    mx = logits[v];
                    arg = v;
                }
            }
            double sum = 0.0;
            for (int v = 0; v < n_vocab; ++v) {
                sum += std::exp((double) logits[v] - mx);
            }
            const double lse = mx + std::log(sum);
            for (int v = 0; v < n_vocab; ++v) {
                logp[(size_t) v] = (double) logits[v] - lse;
            }

            const auto & st = steps[(size_t) s];
            const int want = st["token"].get<int>();
            const auto ids = st["top_ids"].get<std::vector<int>>();
            const auto ref = st["top_logp"].get<std::vector<double>>();
            double diff = 0.0;
            for (size_t k = 0; k < ids.size(); ++k) {
                diff = std::max(diff, std::fabs(logp[(size_t) ids[k]] - ref[k]));
            }
            max_diff = std::max(max_diff, diff);

            const char * verdict = "ok";
            if (arg == want) {
                exact++;
            } else if (st["gap"].get<double>() <= 0.001) {
                near_ties++;
                verdict = "near-tie";
            } else {
                wrong++;
                verdict = "WRONG";
            }
            if (arg != want || diff > tol) {
                std::printf("  %sstep %2d: picked %d, reference %d (gap %.4f), top-5 log-prob diff %.2e  %s\n",
                            label, s, arg, want, st["gap"].get<double>(), diff, verdict);
            }
        }
        return res;
    };

    const step_counts first = check_steps(ctx, "");
    const int exact = first.exact;
    const int near_ties = first.near_ties;
    const int wrong = first.wrong;
    const double max_diff = first.max_diff;
    bool pass = wrong == 0 && max_diff <= tol;

    // On a graphics chip, a difference may start at a near tie inside the model (inner_choices above): run the CPU to
    // keep its choices, then the chip again, taking the CPU's choice wherever the chip's differed at a near tie.
    std::string inner_note;
    bool inner_pass = false; // passed once the CPU's choices were taken at near ties inside the model
    if (gpu && !pass && dump.name.empty() && dump.dir.empty()) {
        llama_model_params mpc = llama_model_default_params();
        mpc.devices = no_devices;
        mpc.n_gpu_layers = 0;
        inner_choices cpu_ic;
        inner_choices gpu_ic;
        gpu_ic.ref = &cpu_ic;
        gpu_ic.stream = stream.get(); // the chip's model keeps streaming; the CPU's copy below doesn't
        if (stream) {
            kera::stream::stop();
        }
        llama_model * cpu_model = llama_model_load_from_file(model_path.c_str(), mpc);
        llama_context_params cpa = cp;
        cpa.offload_kqv = false;
        cpa.op_offload = false;
        cpa.cb_eval = inner_cb;
        cpa.cb_eval_user_data = &cpu_ic;
        llama_context * cctx = cpu_model != nullptr ? llama_init_from_model(cpu_model, cpa) : nullptr;
        bool ok = cctx != nullptr && llama_decode(cctx, batch) == 0;
        llama_context_params cpb = cp;
        cpb.cb_eval = inner_cb;
        cpb.cb_eval_user_data = &gpu_ic;
        llama_context * gctx = ok ? llama_init_from_model(model, cpb) : nullptr;
        ok = ok && gctx != nullptr && llama_decode(gctx, batch) == 0;
        if (!ok) {
            inner_note = "  inside the model: couldn't run the CPU and the chip again to compare their choices";
        } else if (gpu_ic.real > 0) {
            char dev[64];
            std::snprintf(dev, sizeof(dev), "%.2g%%", 100.0 * gpu_ic.max_dev);
            inner_note = "  inside the model: " + std::to_string(gpu_ic.real) + " choice(s) differ from the CPU's, not at a near "
                         "tie; first: " + gpu_ic.first_real + " (the chip's scores were up to " + dev + " off the CPU's)";
        } else if (gpu_ic.forced == 0) {
            inner_note = "  inside the model: the chip made every choice the CPU made, so no near tie there explains it";
        } else {
            const step_counts again = check_steps(gctx, "with the CPU's choices: ");
            pass = again.wrong == 0 && again.max_diff <= tol;
            inner_pass = pass;
            char buf[768];
            const std::string streamed = stream ? std::string(" (experts streamed, ") + kera::stream::mode_name(sc.mode) + ")" : "";
            std::snprintf(buf, sizeof(buf), "%s %s (graphics chip)%s: %d/%d steps exact, %d near-ties, %d wrong, largest top-5 "
                          "log-prob difference %.6e (limit %.2e), with the CPU's choice taken at %d near tie(s) inside the "
                          "model (first: %s; the chip's scores were within %.2g%% of the CPU's). As the chip chose: %d/%d "
                          "exact, %d wrong, largest difference %.6e", pass ? "PASS" : "FAIL",
                          g.value("design", std::string("?")).c_str(), streamed.c_str(), again.exact, n_new, again.near_ties, again.wrong,
                          again.max_diff, tol, gpu_ic.forced, gpu_ic.first_forced.c_str(), 100.0 * gpu_ic.max_dev,
                          exact, n_new, wrong, max_diff);
            inner_note = buf;
        }
        llama_free(gctx);
        llama_free(cctx);
        llama_model_free(cpu_model);
    }
    const std::string how = stream ? std::string(" (experts streamed, ") + kera::stream::mode_name(sc.mode) + ")" : "";
    if (inner_pass || (!inner_note.empty() && inner_note[0] != ' ')) {
        std::printf("%s\n", inner_note.c_str()); // the verdict once the CPU's choices were taken at near ties
    } else {
        std::printf("%s %s%s%s: %d/%d steps exact, %d near-ties, %d wrong, largest top-5 log-prob difference %.6e (limit %.2e)\n",
                    pass ? "PASS" : "FAIL", g.value("design", std::string("?")).c_str(), gpu ? " (graphics chip)" : "",
                    how.c_str(), exact, n_new, near_ties, wrong, max_diff, tol);
        if (!inner_note.empty()) {
            std::printf("%s\n", inner_note.c_str());
        }
    }
    if (stream && !stream->error().empty()) {
        std::printf("  streaming error: %s\n", stream->error().c_str());
    }
    if (stream) {
        const auto r = stream->snapshot();
        if (sc.mode == kera::stream::cache_mode::kera) {
            std::printf("  streaming: %llu layer evaluations, %llu experts asked for, %llu read from the file (%.1f MB), "
                        "%llu evicted, %llu layer evaluations needed more than the %.1f MB budget\n",
                        (unsigned long long) r.cache.steps, (unsigned long long) r.cache.requests,
                        (unsigned long long) r.cache.misses, (double) r.cache.bytes_read / 1e6,
                        (unsigned long long) r.cache.evictions, (unsigned long long) r.cache.overflow_steps,
                        (double) r.cache.budget_bytes / 1e6);
        } else {
            std::printf("  streaming: %llu layer evaluations, %.1f MB hinted to the OS\n", (unsigned long long) r.steps,
                        (double) r.os_hint_bytes / 1e6);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
