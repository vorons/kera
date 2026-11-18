// `kera calibrate --backend vulkan|hip|cpu [--depth N] [--json FILE] -m MODEL [llama.cpp's options]`: one back-end's
// speed on one model, for kera serve's choice of back-end (src/planner/backend_pick.h, docs/DECISIONS.md D-022). It
// reads a prompt from an empty chat and the rest of a long one, writes a few words after a short prompt and after the
// long one, and prints the fitted speeds as one line of JSON (and writes it to FILE). kera serve runs it once per model
// and back-end, each in its own process, so two back-ends never hold the model at once, and passes on the options of
// its own command line that change speed: llama.cpp reads them, and fits the model to the chip's memory as the
// server does, so what is measured is what the server would run.

#include "app/kera_app.h"
#include "planner/backend_pick.h"

#include "kera_version.h"

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace kera {

namespace {

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

// ordinary tokens in a fixed order: the speed doesn't depend on which, but an MoE model routes them like text
std::vector<llama_token> some_tokens(const llama_vocab * vocab, size_t n) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int lo = std::min(1000, n_vocab / 10);
    const int hi = std::max(lo + 1, std::min(n_vocab, 32000));
    std::vector<llama_token> out;
    out.reserve(n);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    while (out.size() < n) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        const llama_token t = (llama_token) (lo + (int) ((x >> 33) % (uint64_t) (hi - lo)));
        if (!llama_vocab_is_control(vocab, t) && !llama_vocab_is_eog(vocab, t)) {
            out.push_back(t);
        }
    }
    return out;
}

// reads tokens [from, to) of `toks` in steps of n_batch; seconds taken (< 0 if it failed)
double read(llama_context * ctx, const std::vector<llama_token> & toks, size_t from, size_t to, int n_batch) {
    const auto t0 = clock_type::now();
    for (size_t i = from; i < to; i += (size_t) n_batch) {
        const int n = (int) std::min(to - i, (size_t) n_batch);
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(toks.data() + i), n)) != 0) {
            return -1;
        }
    }
    llama_synchronize(ctx);
    return seconds_since(t0);
}

// writes n tokens one at a time (the tokens are given, not sampled: only the time counts); seconds taken
double write(llama_context * ctx, const std::vector<llama_token> & toks, size_t from, int n) {
    const auto t0 = clock_type::now();
    for (int i = 0; i < n; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(toks.data() + from + (size_t) i), 1)) != 0) {
            return -1;
        }
        llama_synchronize(ctx);
    }
    return seconds_since(t0);
}

int usage() {
    std::fprintf(stderr,
        "Usage: kera calibrate --backend vulkan|hip|cpu -m MODEL.gguf [--depth 16384] [--json FILE] [llama.cpp's options]\n"
        "\n"
        "Measures one back-end's speed on a model: reading a prompt and writing, in an empty chat and in a long one.\n"
        "Prints one line of JSON. kera serve uses it to choose between Vulkan and the ROCm add-on (gpu.auto_backend).\n");
    return 1;
}

} // namespace

int run_calibrate_cli(int argc, char ** argv) {
    std::string backend, json_path;
    int depth = 16384;
    const int n_write = 32;
    bool has_model = false;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
            backend = argv[++i];
            continue;
        }
        if (a == "--depth" && i + 1 < argc) {
            depth = std::atoi(argv[++i]);
            continue;
        }
        if (a == "--json" && i + 1 < argc) {
            json_path = argv[++i];
            continue;
        }
        if (a == "--help" || a == "-h") {
            return usage();
        }
        has_model = has_model || a == "-m" || a == "--model";
        args.push_back(argv[i]);
    }
    if (backend.empty() || !has_model) {
        return usage();
    }
    std::string device, err;
    if (!choose_backend(backend, device, &err)) {
        std::fprintf(stderr, "Kera: %s\n", err.c_str());
        return 2;
    }
    static char dev_flag[] = "--device";
    static char ngl_flag[] = "-ngl";
    static char zero[] = "0";
    if (!device.empty()) {
        args.push_back(dev_flag);
        args.push_back(device.data());
    }
    if (device == "none") {
        args.push_back(ngl_flag);
        args.push_back(zero);
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return 2;
    }
    // a context just long enough, one chat at a time
    const uint32_t n_ctx_train = common_get_gguf_info(params.model.path).n_ctx_train;
    if (n_ctx_train > 0) {
        depth = std::min(depth, (int) n_ctx_train - n_write - 256);
    }
    depth = std::max(depth, 4096);
    params.n_ctx = (depth + n_write + 256 + 255) / 256 * 256;
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);
    const auto t_load = clock_type::now();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init ? init->model() : nullptr;
    llama_context * ctx = init ? init->context() : nullptr;
    if (model == nullptr || ctx == nullptr) {
        std::fprintf(stderr, "Kera: can't load %s on %s\n", params.model.path.c_str(), backend.c_str());
        return 2;
    }
    const double load_s = seconds_since(t_load);
    // llama.cpp may have made the context shorter to fit the chip's memory
    depth = std::min(depth, (int) llama_n_ctx(ctx) - n_write - 256);
    const int n_batch = (int) llama_n_batch(ctx);
    const size_t first = 2048;
    const std::vector<llama_token> toks = some_tokens(llama_model_get_vocab(model), (size_t) std::max(depth, 1024) + n_write + 600);
    llama_memory_t mem = llama_get_memory(ctx);

    bool ok = depth >= 4096;
    // a warm-up: the chip's programs built and the weights in place
    llama_memory_clear(mem, true);
    ok = ok && read(ctx, toks, 0, 512, n_batch) >= 0 && write(ctx, toks, 512, 8) >= 0;
    // a short prompt, then a few words
    llama_memory_clear(mem, true);
    const double t_short = ok ? read(ctx, toks, 0, 512, n_batch) : -1;
    const double t_write_short = t_short >= 0 ? write(ctx, toks, 512, n_write) : -1;
    // a prompt from an empty chat, the rest of a long one, then a few words deep in it
    llama_memory_clear(mem, true);
    const double t_first = t_write_short >= 0 ? read(ctx, toks, 0, first, n_batch) : -1;
    const double t_rest = t_first >= 0 ? read(ctx, toks, first, (size_t) depth, n_batch) : -1;
    const double t_write_deep = t_rest >= 0 ? write(ctx, toks, (size_t) depth, n_write) : -1;
    ok = t_write_deep > 0;

    nlohmann::json j;
    j["backend"] = backend;
    j["device"] = device;
    j["kera"] = KERA_VERSION;
    j["llama_cpp"] = KERA_LLAMA_CPP_TAG;
    j["n_ctx_train"] = n_ctx_train;
    j["load_s"] = load_s;
    j["ok"] = ok;
    if (ok) {
        const backend_speed s = fit_backend_speed(backend, 0, (double) first, t_first, (double) first,
                                                  (double) depth - (double) first, t_rest, 512, n_write, t_write_short,
                                                  depth, n_write, t_write_deep);
        j["read_s"] = s.read_s;
        j["read_s_per_depth"] = s.read_s_per_depth;
        j["write_s"] = s.write_s;
        j["write_s_per_depth"] = s.write_s_per_depth;
        j["measured"] = { { "depth", depth },
                          { "read_first_tok_s", first / t_first },
                          { "read_rest_tok_s", (depth - (double) first) / t_rest },
                          { "write_short_tok_s", n_write / t_write_short },
                          { "write_deep_tok_s", n_write / t_write_deep } };
        std::fprintf(stderr, "Kera: %s on this model reads %.0f tokens a second from an empty chat and %.0f deeper "
                             "(to %d); writes %.1f a second after a short prompt and %.1f after the long one\n",
                     backend.c_str(), first / t_first, (depth - (double) first) / t_rest, depth, n_write / t_write_short,
                     n_write / t_write_deep);
    } else {
        j["error"] = depth < 4096 ? "the chip's memory leaves too short a context to measure" : "a step failed";
    }
    const std::string line = j.dump();
    std::printf("%s\n", line.c_str());
    if (!json_path.empty()) {
        std::ofstream(std::filesystem::u8path(json_path)) << line << "\n";
    }
    init.reset();
    llama_backend_free();
    return ok ? 0 : 2;
}

} // namespace kera
