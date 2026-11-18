// The stream session (kr_stream.h): llama.cpp's stream hook (hook 3), the buffer type that puts streamed
// experts in the mirror of the model's files, the callback that brings in each MoE layer's experts just
// before the layer runs, the next-layer guesses (Phase 3), and the usage log and pins (Phase 4).

#include "stream/kr_stream.h"

#include "ggml-backend-impl.h" // to define a buffer type
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace fs = std::filesystem;

namespace kera::stream {

const char * mode_name(cache_mode m) {
    switch (m) {
        case cache_mode::os:   return "os";
        case cache_mode::kera: return "kera";
        default:               return "off";
    }
}

bool parse_mode(const std::string & s, cache_mode & m) {
    if (s == "off") {
        m = cache_mode::off;
    } else if (s == "os" || s == "v1") {
        m = cache_mode::os;
    } else if (s == "kera" || s == "v2") {
        m = cache_mode::kera;
    } else {
        return false;
    }
    return true;
}

report operator-(const report & a, const report & b) {
    report d = a;
    d.cache.steps -= b.cache.steps;
    d.cache.requests -= b.cache.requests;
    d.cache.hits -= b.cache.hits;
    d.cache.late -= b.cache.late;
    d.cache.misses -= b.cache.misses;
    d.cache.prefetched -= b.cache.prefetched;
    d.cache.prefetch_used -= b.cache.prefetch_used;
    d.cache.prefetch_cancelled -= b.cache.prefetch_cancelled;
    d.cache.evictions -= b.cache.evictions;
    d.cache.bytes_read -= b.cache.bytes_read;
    d.cache.overflow_steps -= b.cache.overflow_steps;
    d.cache.io_wait_s -= b.cache.io_wait_s;
    d.predicted -= b.predicted;
    d.predicted_right -= b.predicted_right;
    d.os_hint_bytes -= b.os_hint_bytes;
    d.steps -= b.steps;
    d.callback_s -= b.callback_s;
    d.guess_s -= b.guess_s;
    for (int n = 0; n <= report::max_batch; ++n) {
        d.batch_steps[n] -= b.batch_steps[n];
        d.batch_distinct[n] -= b.batch_distinct[n];
        d.batch_missing[n] -= b.batch_missing[n];
    }
    return d;
}

namespace {

// "blk.12.ffn_down_exps.weight" -> layer 12, part 3 (parts in the order the layer's matrices are kept)
bool parse_expert_name(const char * name, int & layer, int & part) {
    int l = -1;
    int n = 0;
    if (std::sscanf(name, "blk.%d.%n", &l, &n) != 1 || l < 0 || n <= 0) {
        return false;
    }
    static const char * kinds[] = { "ffn_gate_up_exps.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
    for (int k = 0; k < 4; ++k) {
        if (std::strcmp(name + n, kinds[k]) == 0) {
            layer = l;
            part = k;
            return true;
        }
    }
    return false;
}

// "ffn_moe_topk-12" -> 12
// "<prefix>-<layer>": the layer number, if the name has this prefix
bool layer_of_name(const char * name, const char * prefix, int & layer) {
    const size_t n = std::strlen(prefix);
    if (name[0] != prefix[0] || std::strncmp(name, prefix, n) != 0 || name[n] != '-') {
        return false;
    }
    char * end = nullptr;
    const long v = std::strtol(name + n + 1, &end, 10);
    if (end == name + n + 1 || *end != '\0' || v < 0) {
        return false;
    }
    layer = (int) v;
    return true;
}

bool topk_layer(const char * name, int & layer) {
    return layer_of_name(name, "ffn_moe_topk", layer);
}

// The outputs of a streamed layer's expert products, checked for numbers that aren't finite under KERA_STREAM_CHECK
bool expert_out_layer(const char * name, int & layer) {
    return layer_of_name(name, "ffn_moe_gate", layer) || layer_of_name(name, "ffn_moe_up", layer) ||
           layer_of_name(name, "ffn_moe_down", layer) || layer_of_name(name, "ffn_moe_gate_up", layer);
}

// Read what the model just computed, on the queue of the device that computed it (llama_stream_tensor_get).
// KERA_STREAM_READ=plain reads with ggml_backend_tensor_get instead, as before (docs/NEGATIVE_RESULTS.md N-012).
bool read_plain() {
    static const bool plain = [] {
        const char * e = std::getenv("KERA_STREAM_READ");
        return e != nullptr && std::strcmp(e, "plain") == 0;
    }();
    return plain;
}

void read_tensor(const ggml_tensor * t, void * data, size_t size, bool plain) {
    if (plain) {
        ggml_backend_tensor_get(t, data, 0, size);
    } else {
        llama_stream_tensor_get(t, data, 0, size);
    }
}

// KERA_STREAM_CHECK=1: report each read of the chosen experts and look for experts' outputs that aren't finite
bool check_reads() {
    static const bool check = std::getenv("KERA_STREAM_CHECK") != nullptr;
    return check;
}

struct tensor_rec {
    int layer = -1;
    int part = -1;
    int file = -1;
    uint64_t offset = 0;
    size_t nbytes = 0;
    ggml_tensor * t = nullptr;
};

// A MoE layer's router as found in the graph, kept on the CPU to guess the next layer's experts
struct router {
    bool ready = false;
    bool failed = false;
    int64_t n_embd = 0;
    int64_t n_expert = 0;
    std::vector<float> w;        // [n_expert][n_embd]
    std::vector<float> bias;     // added to the logits
    std::vector<float> sel_bias; // added to the scores for choosing only
    std::vector<float> gain;     // the norm weight the router's input was multiplied by
    bool sigmoid = false;
};

std::vector<float> read_floats(const ggml_tensor * t) {
    std::vector<float> out;
    const int64_t n_row = ggml_nrows(t);
    const int64_t ne0 = t->ne[0];
    if (!ggml_is_contiguous(t)) {
        return out;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    out.resize((size_t) (n_row * ne0));
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
        return out;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    if (traits->to_float == nullptr) {
        out.clear();
        return out;
    }
    const size_t row_size = ggml_row_size(t->type, ne0);
    for (int64_t r = 0; r < n_row; ++r) {
        traits->to_float(raw.data() + r * row_size, out.data() + r * ne0, ne0);
    }
    return out;
}

bool is_leaf(const ggml_tensor * t) {
    return t != nullptr && t->op == GGML_OP_NONE;
}

const ggml_tensor * skip_views(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_CONT ||
                 t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)) {
        t = t->view_src ? t->view_src : t->src[0];
    }
    return t;
}

float dot(const float * a, const float * b, int64_t n) {
    float acc[8] = { 0 };
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int k = 0; k < 8; ++k) {
            acc[k] += a[i + k] * b[i + k];
        }
    }
    float s = 0;
    for (; i < n; ++i) {
        s += a[i] * b[i];
    }
    for (float v : acc) {
        s += v;
    }
    return s;
}

// v1: a thread that asks the OS to start reading ranges of the mapped model file. Asking takes the OS a
// few milliseconds per layer (it queues the reads and sets up the pages), which the model's own thread
// shouldn't wait for. The current layer's ranges go first; guesses that fall behind are dropped.
class os_hinter {
public:
    using ranges = std::vector<std::pair<uint8_t *, uint64_t>>;

    ~os_hinter() { stop(); }

    void start() {
        if (!th_.joinable()) {
            quit_ = false;
            th_ = std::thread([this] { run(); });
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(m_);
            quit_ = true;
            now_.clear();
            later_.clear();
        }
        cv_.notify_all();
        if (th_.joinable()) {
            th_.join();
        }
    }

    void push(ranges && r, bool urgent) {
        {
            std::lock_guard<std::mutex> lock(m_);
            if (urgent) {
                now_.push_back(std::move(r));
            } else {
                later_.push_back(std::move(r));
                while (later_.size() > 2) {
                    later_.pop_front(); // guesses this old are for layers that have run already
                }
            }
        }
        cv_.notify_one();
    }

private:
    void run() {
        for (;;) {
            ranges r;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [&] { return quit_ || !now_.empty() || !later_.empty(); });
                if (quit_) {
                    return;
                }
                auto & q = now_.empty() ? later_ : now_;
                r = std::move(q.front());
                q.pop_front();
            }
            issue(r);
        }
    }

    static void issue(const ranges & r) {
#if defined(_WIN32)
        using prefetch_fn = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        static const auto pf = (prefetch_fn) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
        std::vector<WIN32_MEMORY_RANGE_ENTRY> e;
        for (const auto & [p, n] : r) {
            WIN32_MEMORY_RANGE_ENTRY x;
            x.VirtualAddress = p;
            x.NumberOfBytes = (SIZE_T) n;
            e.push_back(x);
        }
        if (pf && !e.empty()) {
            pf(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
        }
#else
        const uintptr_t ps = (uintptr_t) region::page_size();
        for (const auto & [p, n] : r) {
            const uintptr_t lo = (uintptr_t) p / ps * ps;
            const uintptr_t hi = ((uintptr_t) p + n + ps - 1) / ps * ps;
            madvise((void *) lo, hi - lo, MADV_WILLNEED);
        }
#endif
    }

    std::thread th_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<ranges> now_;   // the layer about to run
    std::deque<ranges> later_; // guesses for the next one
    bool quit_ = false;
};

// The usage log: how often each expert of each layer was used, across runs (Phase 4).
// "KERAUSG1", u32 n_layers, then per layer: i32 layer, i32 n_expert, u32 counts[n_expert]
std::map<int, std::vector<uint32_t>> read_usage(const std::string & path) {
    std::map<int, std::vector<uint32_t>> out;
    std::ifstream f(fs::u8path(path), std::ios::binary);
    char magic[8];
    uint32_t n = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, "KERAUSG1", 8) != 0 || !f.read((char *) &n, 4) || n > 100000) {
        return out;
    }
    for (uint32_t i = 0; i < n; ++i) {
        int32_t layer = 0;
        int32_t ne = 0;
        if (!f.read((char *) &layer, 4) || !f.read((char *) &ne, 4) || ne <= 0 || ne > 1 << 20) {
            return {};
        }
        std::vector<uint32_t> v((size_t) ne);
        if (!f.read((char *) v.data(), (std::streamsize) (v.size() * 4))) {
            return {};
        }
        out[layer] = std::move(v);
    }
    return out;
}

void write_usage(const std::string & path, const std::map<int, std::vector<uint32_t>> & u) {
    std::error_code ec;
    fs::create_directories(fs::u8path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(fs::u8path(tmp), std::ios::binary | std::ios::trunc);
        const uint32_t n = (uint32_t) u.size();
        f.write("KERAUSG1", 8);
        f.write((const char *) &n, 4);
        for (const auto & [layer, v] : u) {
            const int32_t l = layer;
            const int32_t ne = (int32_t) v.size();
            f.write((const char *) &l, 4);
            f.write((const char *) &ne, 4);
            f.write((const char *) v.data(), (std::streamsize) (v.size() * 4));
        }
        if (!f) {
            return;
        }
    }
    fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
}

} // namespace

struct session::impl {
    config cfg;
    mutable std::mutex mutex;
    std::string err;

    // the load in progress
    bool dry = false;
    std::map<std::string, tensor_rec> tensors;
    std::vector<std::string> files;

    // the model that streams now
    bool active = false;
    region mirror;
    std::vector<uint64_t> file_base;
    expert_cache cache;
    ggml_backend_buffer_type buft{};
    std::vector<int> layers;                                            // streamed layers, ascending
    std::map<int, std::vector<std::pair<uint8_t *, uint64_t>>> mapped;   // v1: layer -> parts (data, stride)
    os_hinter hinter;                                                   // v1
    std::map<int, int> n_expert;
    uint64_t total_bytes = 0;

    // per-step state (the callback runs on the thread that evaluates the model)
    std::vector<int32_t> ids;
    std::vector<uint8_t> raw;
    report rep;
    std::map<int, std::vector<uint32_t>> uses;          // this run's usage, per layer
    std::map<int, std::vector<uint32_t>> usage_before;  // from the log
    std::map<int, std::vector<uint32_t>> prompt_counts; // token-weighted, this prompt
    bool in_prompt = false;
    std::map<int, router> routers;
    std::map<int, std::vector<int32_t>> guesses;        // layer -> experts guessed for it
    std::vector<float> last_x;                          // the last router input, [n_tok][n_embd]
    int64_t last_tok = 0;
    int last_layer = -1;

    impl() {
        buft.iface = {
            /*.get_name         =*/ [](ggml_backend_buffer_type_t) -> const char * { return "KERA_STREAM"; },
            /*.alloc_buffer     =*/ b_alloc,
            /*.alloc_buffer_n   =*/ b_alloc_n,
            /*.get_alignment    =*/ [](ggml_backend_buffer_type_t) -> size_t { return 64; },
            /*.get_max_size     =*/ nullptr,
            /*.get_alloc_size   =*/ nullptr,
            /*.get_alloc_size_n =*/ [](ggml_backend_buffer_type_t, ggml_tensor **, int) -> size_t { return 0; },
            /*.is_host          =*/ [](ggml_backend_buffer_type_t) -> bool { return true; },
        };
        buft.device = nullptr;
        buft.context = this;
    }

    ~impl() { model_freed(); }

    void fail(const std::string & msg) {
        std::lock_guard<std::mutex> lock(mutex);
        err = msg;
    }

    // ---- hook 3 -------------------------------------------------------------------------------------

    static bool h_begin(void * ud, const char * path, bool no_alloc) {
        auto * s = (impl *) ud;
        std::error_code ec;
        if (s->cfg.mode == cache_mode::off || !fs::equivalent(fs::u8path(path), fs::u8path(s->cfg.model_path), ec)) {
            return false;
        }
        if (no_alloc) {
            s->dry = true; // the memory-fit pass: streamed experts take no memory
            return true;
        }
        if (s->active) {
            s->fail("this model is already loaded and streaming; Kera streams one copy at a time");
            return false;
        }
        s->dry = false;
        s->tensors.clear();
        s->files.clear();
        return true;
    }

    static llama_stream_mode h_mode(void * ud, const char * name, const char * file, uint64_t offset, size_t nbytes,
                                    ggml_backend_buffer_type_t * out) {
        auto * s = (impl *) ud;
        int layer = 0;
        int part = 0;
        if (!parse_expert_name(name, layer, part)) {
            return LLAMA_STREAM_NONE;
        }
        if (s->dry) {
            *out = &s->buft;
            return LLAMA_STREAM_BUFT;
        }
        tensor_rec r;
        r.layer = layer;
        r.part = part;
        const auto it = std::find(s->files.begin(), s->files.end(), std::string(file));
        r.file = (int) (it - s->files.begin());
        if (it == s->files.end()) {
            s->files.push_back(file);
        }
        r.offset = offset;
        r.nbytes = nbytes;
        s->tensors[name] = r;
        if (s->cfg.mode == cache_mode::os) {
            return LLAMA_STREAM_MAP;
        }
        *out = &s->buft;
        return LLAMA_STREAM_BUFT;
    }

    static void h_mapped(void * ud, ggml_tensor * t) {
        auto * s = (impl *) ud;
        const auto it = s->tensors.find(ggml_get_name(t));
        if (it != s->tensors.end()) {
            it->second.t = t;
        }
    }

    static bool h_end(void * ud, const llama_model * model) {
        auto * s = (impl *) ud;
        if (s->dry) {
            s->dry = false;
            return true;
        }
        if (model == nullptr) {
            s->model_freed();
            return true;
        }
        s->finish_load();
        // not streaming after all: llama.cpp fails the load if it placed experts in the mirror, which nothing would
        // then fill (zeros on Linux and macOS, no access on Windows; N-018)
        if (!s->active) {
            std::lock_guard<std::mutex> lock(s->mutex);
            std::fprintf(stderr, "Kera: the model's experts can't be streamed: %s\n",
                         s->err.empty() ? "(no reason given)" : s->err.c_str());
        }
        return s->active;
    }

    static bool h_eval(ggml_tensor * t, bool ask, void * ud) {
        auto * s = (impl *) ud;
        int layer = 0;
        if (s->active && check_reads() && expert_out_layer(t->name, layer) && s->n_expert.count(layer)) {
            if (!ask) {
                s->check_outputs(t);
                if (s->cfg.mode == cache_mode::kera) {
                    s->check_experts(layer, (std::string("after ") + t->name).c_str());
                }
            }
            return true;
        }
        if (!s->active || !topk_layer(t->name, layer) || !s->n_expert.count(layer)) {
            return !ask; // not ours: don't ask for it, and never stop the computation
        }
        if (ask) {
            return true;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = s->on_topk(t, layer);
        s->rep.callback_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return ok;
    }

    // ---- the buffer type for streamed experts (v2) -----------------------------------------------

    static ggml_backend_buffer_t b_alloc(ggml_backend_buffer_type_t buft, size_t size) {
        (void) buft;
        (void) size;
        return nullptr; // only whole models are placed in the mirror (b_alloc_n)
    }

    static ggml_backend_buffer_t b_alloc_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
        auto * s = (impl *) buft->context;
        if (!s->reserve_mirror()) {
            return nullptr;
        }
        static const ggml_backend_buffer_i iface = {
            /*.free_buffer   =*/ [](ggml_backend_buffer_t b) { ((impl *) b->context)->model_freed(); },
            /*.get_base      =*/ [](ggml_backend_buffer_t b) -> void * { return ((impl *) b->context)->mirror.base(); },
            /*.init_tensor   =*/ nullptr,
            /*.memset_tensor =*/ [](ggml_backend_buffer_t, ggml_tensor * t, uint8_t v, size_t off, size_t size) {
                std::memset((uint8_t *) t->data + off, v, size);
            },
            /*.set_tensor    =*/ [](ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t off, size_t size) {
                std::memcpy((uint8_t *) t->data + off, data, size);
            },
            /*.get_tensor    =*/ [](ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t off, size_t size) {
                std::memcpy(data, (const uint8_t *) t->data + off, size);
            },
            /*.set_tensor_2d =*/ nullptr,
            /*.get_tensor_2d =*/ nullptr,
            /*.cpy_tensor    =*/ [](ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) { return false; },
            /*.clear         =*/ [](ggml_backend_buffer_t, uint8_t) {},
            /*.reset         =*/ nullptr,
        };
        // the buffer's size is the RAM budget: what the experts can take, not their size on disk
        ggml_backend_buffer_t b = ggml_backend_buffer_init(buft, iface, s, std::max<uint64_t>(1, s->cfg.budget_bytes));
        for (int i = 0; i < n_tensors; ++i) {
            ggml_tensor * t = tensors[i];
            if (t->view_src != nullptr || t->data != nullptr) {
                continue;
            }
            const auto it = s->tensors.find(ggml_get_name(t));
            if (it == s->tensors.end()) {
                s->fail(std::string("no file position for ") + ggml_get_name(t));
                ggml_backend_buffer_free(b);
                return nullptr;
            }
            it->second.t = t;
            t->buffer = b;
            t->data = s->mirror.base() + s->file_base[(size_t) it->second.file] + it->second.offset;
        }
        for (int i = 0; i < n_tensors; ++i) {
            if (tensors[i]->view_src != nullptr && tensors[i]->buffer == nullptr) {
                ggml_backend_view_init(tensors[i]);
            }
        }
        return b;
    }

    bool reserve_mirror() {
        if (mirror.reserved()) {
            return true;
        }
        constexpr uint64_t align = 1ull << 16; // a multiple of every page size and of Windows' reservation unit
        file_base.clear();
        uint64_t total = 0;
        for (const auto & f : files) {
            std::error_code ec;
            const uint64_t size = (uint64_t) fs::file_size(fs::u8path(f), ec);
            if (ec) {
                fail("can't read the size of " + f);
                return false;
            }
            file_base.push_back(total);
            total += (size + align - 1) / align * align + align;
        }
        if (!mirror.reserve(total)) {
            fail("the OS refused " + std::to_string(total >> 30) + " GB of address space for the model's experts");
            return false;
        }
        return true;
    }

    // ---- after the load ----------------------------------------------------------------------------

    void finish_load() {
        std::map<int, std::vector<tensor_rec *>> by_layer;
        for (auto & [name, r] : tensors) {
            if (r.t == nullptr) {
                continue;
            }
            by_layer[r.layer].push_back(&r);
        }
        std::vector<layer_layout> layouts;
        layers.clear();
        n_expert.clear();
        mapped.clear();
        total_bytes = 0;
        for (auto & [layer, parts] : by_layer) {
            std::sort(parts.begin(), parts.end(), [](const tensor_rec * a, const tensor_rec * b) { return a->part < b->part; });
            layer_layout L;
            L.layer = layer;
            L.n_expert = (int) parts[0]->t->ne[2];
            bool ok = L.n_expert > 0;
            for (const tensor_rec * r : parts) {
                const uint64_t stride = (uint64_t) r->t->nb[2];
                ok = ok && r->t->ne[2] == L.n_expert && stride * (uint64_t) L.n_expert == r->nbytes;
                L.parts.push_back({ r->file, r->offset, stride });
                if (cfg.mode == cache_mode::os) {
                    mapped[layer].push_back({ (uint8_t *) r->t->data, stride });
#if !defined(_WIN32)
                    // llama.cpp maps lazy weights for random access (no read-ahead); experts are read whole
                    const uintptr_t ps = (uintptr_t) region::page_size();
                    const uintptr_t lo = (uintptr_t) r->t->data / ps * ps;
                    madvise((void *) lo, (uintptr_t) r->t->data + r->nbytes - lo, MADV_NORMAL);
#endif
                }
                total_bytes += r->nbytes;
            }
            if (!ok) {
                fail("the experts of layer " + std::to_string(layer) + " aren't laid out one after another");
                return;
            }
            layers.push_back(layer);
            n_expert[layer] = L.n_expert;
            layouts.push_back(std::move(L));
        }
        if (layouts.empty()) {
            fail("it has no routed experts");
            return;
        }
        if (cfg.mode == cache_mode::kera) {
            cache_options opt;
            opt.budget_bytes = cfg.budget_bytes;
            opt.io_threads = cfg.io_threads;
            opt.chunk_bytes = cfg.chunk_bytes;
            opt.direct = cfg.direct;
            std::string e;
            if (!cache.open(files, file_base, cfg.mirrors, &mirror, layouts, opt, &e)) {
                fail(e);
                return;
            }
        }
        rep = report{};
        rep.mode = cfg.mode;
        uses.clear();
        prompt_counts.clear();
        routers.clear();
        guesses.clear();
        last_layer = -1;
        in_prompt = false;
        active = true;
        if (!cfg.usage_file.empty()) {
            usage_before = read_usage(cfg.usage_file);
        }
        if (cfg.pins && cfg.mode == cache_mode::kera) {
            pin_from_usage();
        }
    }

    // Pin the experts the usage log says are used most, up to pin_share of the budget
    void pin_from_usage() {
        struct item { uint32_t count; int layer; int32_t e; };
        std::vector<item> items;
        for (const auto & [layer, v] : usage_before) {
            if (!n_expert.count(layer) || (int) v.size() != n_expert[layer]) {
                continue;
            }
            for (size_t e = 0; e < v.size(); ++e) {
                if (v[e] > 0) {
                    items.push_back({ v[e], layer, (int32_t) e });
                }
            }
        }
        std::sort(items.begin(), items.end(), [](const item & a, const item & b) { return a.count > b.count; });
        const uint64_t limit = (uint64_t) ((double) cfg.budget_bytes * cfg.pin_share);
        uint64_t pinned = 0;
        for (const auto & it : items) {
            const uint64_t b = cache.expert_bytes(it.layer);
            if (pinned + b > limit) {
                break;
            }
            cache.pin(it.layer, &it.e, 1, false);
            pinned += b;
            rep.pinned_experts++;
        }
    }

    void save_usage() {
        if (cfg.usage_file.empty() || uses.empty()) {
            return;
        }
        auto merged = usage_before;
        for (const auto & [layer, v] : uses) {
            auto & m = merged[layer];
            if (m.size() != v.size()) {
                m.assign(v.size(), 0);
            }
            uint64_t total = 0;
            for (size_t e = 0; e < v.size(); ++e) {
                m[e] += v[e];
                total += m[e];
            }
            if (total > (1ull << 30)) {
                for (auto & c : m) {
                    c /= 2; // keep the log weighted towards recent use
                }
            }
        }
        write_usage(cfg.usage_file, merged);
        usage_before = merged;
        uses.clear();
    }

    void model_freed() {
        save_usage();
        hinter.stop();
        if (cfg.mode == cache_mode::kera && active) {
            rep.cache = cache.stats(); // the counts outlive the model, for end-of-run reports
        }
        cache.close();
        mirror.release();
        active = false;
        tensors.clear();
        mapped.clear();
    }

    // ---- each MoE layer ----------------------------------------------------------------------------

    // KERA_STREAM_CHECK: are the expert numbers just read in range and distinct per word, and does a second read
    // a little later, the other way, give the same bytes?
    void check_topk(const ggml_tensor * t, int layer, int ne, int64_t n_used) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::vector<uint8_t> again(raw.size());
        read_tensor(t, again.data(), again.size(), !read_plain());
        int64_t bad = 0;
        int64_t repeated = 0;
        std::vector<uint8_t> seen((size_t) std::max(ne, 1), 0);
        int64_t distinct = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= ne) {
                bad++;
                continue;
            }
            const size_t row = i / (size_t) n_used;
            for (size_t k = row * (size_t) n_used; k < i; ++k) {
                repeated += ids[k] == e ? 1 : 0;
            }
            if (!seen[(size_t) e]) {
                seen[(size_t) e] = 1;
                distinct++;
            }
        }
        // the same hash llama.cpp's expert copy prints (hook 3), over the numbers in the tensor's own order
        uint32_t hash = 2166136261u;
        const size_t n = raw.size() / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            int32_t id = 0;
            std::memcpy(&id, raw.data() + i * sizeof(int32_t), sizeof(id));
            hash = (hash ^ (uint32_t) id) * 16777619u;
        }
        std::fprintf(stderr, "kera-check: %s read (%s) as %lld distinct experts (hash %08x), %lld out of range, %lld "
                     "repeated in a word; a second read %s\n", t->name, read_plain() ? "plain" : "on its queue",
                     (long long) distinct, hash, (long long) bad, (long long) repeated,
                     again == raw ? "matches" : "DIFFERS");
    }

    // KERA_STREAM_CHECK: do the experts this layer uses hold the file's bytes? Read again from the file the ordinary
    // way and compare, once right after they were brought in and once after the layer used them.
    void check_experts(int layer, const char * when) {
        const layer_layout * ll = nullptr;
        for (const auto & l : cache.layers()) {
            ll = l.layer == layer ? &l : ll;
        }
        if (ll == nullptr) {
            return;
        }
        std::vector<uint8_t> seen((size_t) ll->n_expert, 0);
        std::vector<char> buf;
        int64_t bad_experts = 0;
        int64_t bad_bytes = 0;
        std::string first;
        for (const int32_t e : ids) {
            if (e < 0 || e >= ll->n_expert || seen[(size_t) e]) {
                continue;
            }
            seen[(size_t) e] = 1;
            bool bad = false;
            for (size_t p = 0; p < ll->parts.size(); ++p) {
                const part_layout & pl = ll->parts[p];
                const uint8_t * mem = cache.part_data(layer, e, (int) p);
                std::ifstream f(fs::u8path(files[(size_t) pl.file]), std::ios::binary);
                buf.resize((size_t) pl.stride);
                f.seekg((std::streamoff) (pl.file_offset + (uint64_t) e * pl.stride));
                if (mem == nullptr || !f.read(buf.data(), (std::streamsize) buf.size())) {
                    continue;
                }
                for (size_t i = 0; i < buf.size(); ++i) {
                    if ((uint8_t) buf[i] != mem[i]) {
                        if (!bad && first.empty()) {
                            first = "expert " + std::to_string(e) + " part " + std::to_string(p) + " at byte " + std::to_string(i);
                        }
                        bad = true;
                        bad_bytes++;
                    }
                }
            }
            bad_experts += bad ? 1 : 0;
        }
        int64_t checked = 0;
        for (const uint8_t v : seen) {
            checked += v;
        }
        if (bad_experts > 0) {
            std::fprintf(stderr, "kera-check: layer %d %s: %lld of %lld experts differ from the file (%lld bytes; first: %s)\n",
                         layer, when, (long long) bad_experts, (long long) checked, (long long) bad_bytes, first.c_str());
        } else {
            std::fprintf(stderr, "kera-check: layer %d %s: %lld experts match the file\n", layer, when, (long long) checked);
        }
    }

    void check_outputs(const ggml_tensor * t) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return;
        }
        std::vector<float> v((size_t) ggml_nelements(t));
        read_tensor(t, v.data(), v.size() * sizeof(float), read_plain());
        int64_t bad = 0;
        for (const float x : v) {
            bad += std::isfinite(x) ? 0 : 1;
        }
        if (bad > 0) {
            std::fprintf(stderr, "kera-check: %s has %lld of %lld numbers that aren't finite\n", t->name, (long long) bad,
                         (long long) v.size());
        }
    }

    bool on_topk(ggml_tensor * t, int layer) {
        if (t->type != GGML_TYPE_I32 || t->ne[0] <= 0) {
            return true;
        }
        const int64_t n_used = t->ne[0];
        const int64_t n_tok = t->ne[1];
        raw.resize(ggml_nbytes(t));
        read_tensor(t, raw.data(), raw.size(), read_plain());
        ids.clear();
        for (int64_t j = 0; j < n_tok; ++j) {
            const int32_t * row = (const int32_t *) (raw.data() + j * t->nb[1]);
            ids.insert(ids.end(), row, row + n_used);
        }
        rep.steps++;
        const int ne = n_expert[layer];
        if (check_reads()) {
            check_topk(t, layer, ne, n_used);
        }

        // usage, and the prompt's own counts for prompt pins
        auto & u = uses[layer];
        if ((int) u.size() != ne) {
            u.assign((size_t) ne, 0);
        }
        const bool prompt = n_tok > 32;
        if (prompt) {
            auto & pc = prompt_counts[layer];
            if ((int) pc.size() != ne) {
                pc.assign((size_t) ne, 0);
            }
            for (const int32_t e : ids) {
                if (e >= 0 && e < ne) {
                    pc[(size_t) e]++;
                }
            }
            in_prompt = true;
        } else if (in_prompt) {
            in_prompt = false;
            if (cfg.prompt_pins && cfg.mode == cache_mode::kera) {
                pin_prompt();
            }
            prompt_counts.clear();
        }
        {
            std::vector<uint8_t> seen((size_t) ne, 0);
            uint64_t distinct = 0;
            uint64_t missing = 0;
            const bool kera_cache = cfg.mode == cache_mode::kera;
            for (const int32_t e : ids) {
                if (e >= 0 && e < ne && !seen[(size_t) e]) {
                    seen[(size_t) e] = 1;
                    u[(size_t) e]++;
                    distinct++;
                    missing += kera_cache && !cache.is_resident(layer, e) ? 1 : 0;
                }
            }
            // union growth: words written or checked together (prompts are read in bigger batches)
            if (n_tok >= 1 && n_tok <= report::max_batch) {
                rep.batch_steps[n_tok]++;
                rep.batch_distinct[n_tok] += distinct;
                rep.batch_missing[n_tok] += missing;
            }
        }

        // how good was the guess for this layer?
        if (const auto g = guesses.find(layer); g != guesses.end()) {
            std::vector<uint8_t> used((size_t) ne, 0);
            for (const int32_t e : ids) {
                if (e >= 0 && e < ne) {
                    used[(size_t) e] = 1;
                }
            }
            rep.predicted += g->second.size();
            for (const int32_t e : g->second) {
                rep.predicted_right += used[(size_t) e];
            }
            guesses.erase(g);
        }

        // bring this layer's experts in
        if (cfg.mode == cache_mode::kera) {
            if (!cache.require(layer, ids.data(), ids.size())) {
                const std::string why = "reading the experts failed (" + cache.error_message() + "); the model can't continue";
                std::fprintf(stderr, "Kera: %s\n", why.c_str());
                fail(why);
                return false; // stops the computation (llama_decode reports it aborted)
            }
            if (check_reads()) {
                check_experts(layer, "just brought in");
            }
        } else {
            hint_os(layer, ids, /*urgent =*/ true);
        }

        // guess the next MoE layer's experts and start reading them (Phase 3)
        if (cfg.lookahead && n_tok <= 8) {
            const auto t0 = std::chrono::steady_clock::now();
            guess_next(t, layer, (int) n_used, n_tok);
            rep.guess_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        return true;
    }

    void pin_prompt() {
        cache.unpin_prompt();
        struct item { uint32_t count; int layer; int32_t e; };
        std::vector<item> items;
        for (const auto & [layer, v] : prompt_counts) {
            for (size_t e = 0; e < v.size(); ++e) {
                if (v[e] > 0) {
                    items.push_back({ v[e], layer, (int32_t) e });
                }
            }
        }
        std::sort(items.begin(), items.end(), [](const item & a, const item & b) { return a.count > b.count; });
        const uint64_t limit = (uint64_t) ((double) cfg.budget_bytes * cfg.prompt_pin_share);
        uint64_t pinned = 0;
        for (const auto & it : items) {
            const uint64_t b = cache.expert_bytes(it.layer);
            if (pinned + b > limit) {
                break;
            }
            cache.pin(it.layer, &it.e, 1, true);
            pinned += b;
        }
    }

    // v1: ask the OS to start reading these experts' pages from the mapped file (on the hinter's thread)
    void hint_os(int layer, const std::vector<int32_t> & experts, bool urgent) {
        const auto it = mapped.find(layer);
        if (it == mapped.end()) {
            return;
        }
        std::vector<uint8_t> seen((size_t) n_expert[layer], 0);
        os_hinter::ranges r;
        for (const int32_t e : experts) {
            if (e < 0 || e >= n_expert[layer] || seen[(size_t) e]) {
                continue;
            }
            seen[(size_t) e] = 1;
            for (const auto & [data, stride] : it->second) {
                r.push_back({ data + (uint64_t) e * stride, stride });
                rep.os_hint_bytes += stride;
            }
        }
        hinter.start();
        hinter.push(std::move(r), urgent);
    }

    // ---- lookahead (Phase 3) -----------------------------------------------------------------------

    // Find this layer's router behind its chosen experts: top-k <- [+ selection bias] <- [sigmoid | softmax]
    // <- [+ bias] <- router x input. Returns the input tensor, or nullptr if the graph looks different.
    const ggml_tensor * learn_router(const ggml_tensor * topk, int layer) {
        router & r = routers[layer];
        const ggml_tensor * t = skip_views(topk);
        if (t == nullptr || (t->op != GGML_OP_ARGSORT && t->op != GGML_OP_TOP_K)) {
            r.failed = true;
            return nullptr;
        }
        t = skip_views(t->src[0]);
        bool past_activation = false;
        const ggml_tensor * sel_bias = nullptr;
        const ggml_tensor * bias = nullptr;
        bool sigmoid = false;
        for (int depth = 0; t != nullptr && depth < 8; ++depth) {
            if (t->op == GGML_OP_ADD) {
                const ggml_tensor * leaf = is_leaf(t->src[1]) ? t->src[1] : is_leaf(t->src[0]) ? t->src[0] : nullptr;
                if (leaf == nullptr) {
                    break;
                }
                (past_activation ? bias : sel_bias) = leaf;
                t = skip_views(leaf == t->src[1] ? t->src[0] : t->src[1]);
            } else if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID) {
                sigmoid = true;
                past_activation = true;
                t = skip_views(t->src[0]);
            } else if (t->op == GGML_OP_SOFT_MAX) {
                past_activation = true;
                t = skip_views(t->src[0]);
            } else if (t->op == GGML_OP_MUL_MAT) {
                const ggml_tensor * w = t->src[0];
                const ggml_tensor * x = t->src[1];
                char want[64];
                std::snprintf(want, sizeof(want), "blk.%d.ffn_gate_inp.weight", layer);
                if (!is_leaf(w) || std::strcmp(w->name, want) != 0 || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
                    break;
                }
                if (!r.ready && !r.failed) {
                    r.n_embd = w->ne[0];
                    r.n_expert = w->ne[1];
                    r.w = read_floats(w);
                    r.bias = bias ? read_floats(bias) : std::vector<float>();
                    r.sel_bias = sel_bias ? read_floats(sel_bias) : std::vector<float>();
                    r.sigmoid = sigmoid;
                    // the norm weight the input was multiplied by, if the input is (normed x) * weight
                    if (x->op == GGML_OP_MUL) {
                        const ggml_tensor * g = is_leaf(x->src[1]) ? x->src[1] : is_leaf(x->src[0]) ? x->src[0] : nullptr;
                        if (g && g->ne[0] == r.n_embd && ggml_nrows(g) == 1) {
                            r.gain = read_floats(g);
                        }
                    }
                    const bool sizes_ok = (int64_t) r.w.size() == r.n_embd * r.n_expert &&
                                          (r.bias.empty() || (int64_t) r.bias.size() == r.n_expert) &&
                                          (r.sel_bias.empty() || (int64_t) r.sel_bias.size() == r.n_expert);
                    r.ready = sizes_ok;
                    r.failed = !sizes_ok;
                }
                return r.ready ? x : nullptr;
            } else {
                break;
            }
        }
        r.failed = true;
        return nullptr;
    }

    void guess_next(const ggml_tensor * topk, int layer, int n_used, int64_t n_tok) {
        router & cur = routers[layer];
        if (cur.failed) {
            return;
        }
        const ggml_tensor * x = learn_router(topk, layer);
        if (x == nullptr || x->ne[1] != n_tok) {
            return;
        }
        const auto next_it = std::upper_bound(layers.begin(), layers.end(), layer);
        if (next_it == layers.end()) {
            return;
        }
        const int next = *next_it;
        const auto nr_it = routers.find(next);
        if (nr_it == routers.end() || !nr_it->second.ready || nr_it->second.n_embd != cur.n_embd) {
            return; // the next layer's router is learned the first time it runs
        }
        const router & nr = nr_it->second;
        const int64_t n_embd = cur.n_embd;
        last_x.resize((size_t) (n_embd * n_tok));
        ggml_backend_tensor_get(x, last_x.data(), 0, last_x.size() * sizeof(float));

        // the residual changes little from one layer to the next: the next router sees about the same
        // normalised vector, multiplied by its own norm weight instead of this layer's
        std::vector<float> v((size_t) n_embd);
        std::vector<float> score((size_t) nr.n_expert);
        std::vector<int32_t> order((size_t) nr.n_expert);
        std::vector<uint8_t> chosen((size_t) nr.n_expert, 0);
        const bool regain = !cur.gain.empty() && !nr.gain.empty();
        for (int64_t j = 0; j < n_tok; ++j) {
            const float * xj = last_x.data() + j * n_embd;
            for (int64_t i = 0; i < n_embd; ++i) {
                const float g = regain ? cur.gain[(size_t) i] : 0.0f;
                v[(size_t) i] = regain ? (g != 0.0f ? xj[i] / g * nr.gain[(size_t) i] : 0.0f) : xj[i];
            }
            for (int64_t e = 0; e < nr.n_expert; ++e) {
                float s = dot(nr.w.data() + e * n_embd, v.data(), n_embd);
                if (!nr.bias.empty()) {
                    s += nr.bias[(size_t) e];
                }
                if (nr.sigmoid) {
                    s = 1.0f / (1.0f + std::exp(-s));
                }
                if (!nr.sel_bias.empty()) {
                    s += nr.sel_bias[(size_t) e];
                }
                score[(size_t) e] = s;
                order[(size_t) e] = (int32_t) e;
            }
            const int k = std::min<int>(n_used, (int) nr.n_expert);
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int32_t a, int32_t b) { return score[(size_t) a] > score[(size_t) b]; });
            for (int i = 0; i < k; ++i) {
                chosen[(size_t) order[(size_t) i]] = 1;
            }
        }
        std::vector<int32_t> guess;
        for (int32_t e = 0; e < (int32_t) nr.n_expert; ++e) {
            if (chosen[(size_t) e]) {
                guess.push_back(e);
            }
        }
        if (cfg.mode == cache_mode::kera) {
            cache.prefetch(next, guess.data(), guess.size(), expert_cache::reason::lookahead);
        } else {
            hint_os(next, guess, /*urgent =*/ false);
        }
        guesses[next] = std::move(guess);
    }
};

session::session(std::unique_ptr<impl> p) : p_(std::move(p)) {}

session::~session() = default;

const config & session::cfg() const {
    return p_->cfg;
}

bool session::streaming() const {
    return p_->active;
}

std::string session::error() const {
    std::lock_guard<std::mutex> lock(p_->mutex);
    return p_->err;
}

uint64_t session::expert_bytes_total() const {
    return p_->total_bytes;
}

bool session::bring_in(int layer, const int32_t * experts, size_t n) {
    if (!p_->active || p_->cfg.mode != cache_mode::kera) {
        return true; // the OS's file cache maps every expert anyway
    }
    return p_->cache.require(layer, experts, n);
}

size_t session::moe_layers() const {
    return p_->layers.size();
}

report session::snapshot() const {
    report r = p_->rep;
    if (p_->cfg.mode == cache_mode::kera && p_->active) {
        r.cache = p_->cache.stats();
    }
    return r;
}

double session::predicted_new_bytes(int k) const {
    const report r = snapshot();
    const int n = std::max(1, k + 1);
    const size_t layers = std::max<size_t>(1, p_->layers.size());
    // one expert of an average streamed layer
    uint64_t experts = 0;
    for (const auto & [layer, ne] : p_->n_expert) {
        experts += (uint64_t) ne;
    }
    const double expert_bytes = experts > 0 ? (double) p_->total_bytes / (double) experts : 0.0;
    const bool kera_cache = p_->cfg.mode == cache_mode::kera;
    auto per_layer = [&](int m) { // expected experts read per layer for m words at once (0: not seen)
        if (m > report::max_batch || r.batch_steps[m] == 0) {
            return -1.0;
        }
        const uint64_t v = kera_cache ? r.batch_missing[m] : r.batch_distinct[m];
        return (double) v / (double) r.batch_steps[m];
    };
    double read = per_layer(n);
    if (read < 0) {
        // not seen yet: grow one word's reads the way RT2 measured (4 words use 2.9x one word's experts)
        const double one = per_layer(1);
        if (one < 0) {
            return 0.0;
        }
        read = one * std::pow((double) n, std::log(2.9) / std::log(4.0));
    }
    return read * expert_bytes * (double) layers;
}

std::string session::union_growth(const report & d) {
    if (d.batch_steps[1] == 0) {
        return "";
    }
    const double one = (double) d.batch_distinct[1] / (double) d.batch_steps[1];
    std::string out;
    char buf[160];
    for (int n = 2; n <= report::max_batch; ++n) {
        if (d.batch_steps[n] == 0 || one <= 0) {
            continue;
        }
        const double used = (double) d.batch_distinct[n] / (double) d.batch_steps[n];
        std::snprintf(buf, sizeof(buf), "%s%d words %.1fx", out.empty() ? "" : ", ", n, used / one);
        out += buf;
    }
    if (out.empty()) {
        return "";
    }
    return "checking guesses used more experts per layer than writing one word: " + out;
}

std::string session::per_word(const report & d, uint64_t n_words) {
    if (n_words == 0) {
        return "";
    }
    const double w = (double) n_words;
    char buf[512];
    if (d.mode == cache_mode::kera) {
        const double asked = (double) std::max<uint64_t>(1, d.cache.requests);
        const double mb = (double) d.cache.bytes_read / w / 1e6;
        std::snprintf(buf, sizeof(buf),
                      "per word: %.*f MB read from the SSD, %.0f%% of the experts it used were already in memory "
                      "(%.0f%% more were on their way), %.1f ms waiting for the SSD",
                      mb < 10 ? 1 : 0, mb, 100.0 * (double) d.cache.hits / asked,
                      100.0 * (double) d.cache.late / asked, 1000.0 * d.cache.io_wait_s / w);
    } else if (d.mode == cache_mode::os) {
        const double mb = (double) d.os_hint_bytes / w / 1e6;
        std::snprintf(buf, sizeof(buf), "per word: the OS was asked to read %.*f MB early (its file cache keeps what fits)",
                      mb < 10 ? 1 : 0, mb);
    } else {
        return "";
    }
    std::string out = buf;
    if (d.predicted > 0) {
        std::snprintf(buf, sizeof(buf), "; next-layer guesses right %.0f%% of the time",
                      100.0 * (double) d.predicted_right / (double) d.predicted);
        out += buf;
    }
    if (const std::string u = union_growth(d); !u.empty()) {
        out += "; " + u;
    }
    return out;
}

namespace {
std::mutex g_mutex;
std::vector<std::shared_ptr<session>> g_sessions; // kept until the process ends: loaded models point at them
}

std::shared_ptr<session> start(const config & cfg, std::string * error) {
    std::error_code ec;
    if (cfg.mode == cache_mode::off) {
        if (error) *error = "streaming is off";
        return nullptr;
    }
    if (!fs::is_regular_file(fs::u8path(cfg.model_path), ec)) {
        if (error) *error = "no model file at " + cfg.model_path;
        return nullptr;
    }
    auto p = std::make_unique<session::impl>();
    p->cfg = cfg;
    auto s = std::make_shared<session>(std::move(p));
    llama_stream_hook hook = {};
    hook.begin = session::impl::h_begin;
    hook.mode = session::impl::h_mode;
    hook.mapped = session::impl::h_mapped;
    hook.end = session::impl::h_end;
    hook.cb_eval = session::impl::h_eval;
    hook.user_data = s->get();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sessions.push_back(s);
    llama_set_stream_hook(&hook);
    return s;
}

void stop() {
    llama_set_stream_hook(nullptr);
}

} // namespace kera::stream
