// `kera bench --model FILE.gguf`: the section 7 benchmark, writing the P3 bench JSON.
//
// For each bench prompt (bench/prompts/, built into the program) it runs once to warm up, then three
// times, and keeps the median of: the wait for the first word, the prompt reading speed, and the
// writing speed over 256 words. It also records peak memory, the machine profile, the model file's
// hash and every setting. With every Kera feature off (all there is in M0) this measures stock
// llama.cpp, the baseline that later results are compared with. With --stream the model's routed experts
// are streamed (M5) and each prompt also reports, per written word, what came from the SSD.

#include "common/kr_hash.h"
#include "common/kr_paths.h"
#include "io/kr_os_file.h"
#include "setup/kr_machine.h"
#include "stream/kr_stream.h"
#include "tune/kr_settings.h"

#include "kera_version.h"
#include "kera_web_assets.h"

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using nlohmann::json;

namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// The most memory the process has held, in GiB (what the OS reports as its peak working set).
double peak_ram_gib() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double) pmc.PeakWorkingSetSize / 1073741824.0;
    }
    return 0.0;
#else
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return (double) ru.ru_maxrss / 1073741824.0; // bytes
#else
    return (double) ru.ru_maxrss * 1024.0 / 1073741824.0; // KiB
#endif
#endif
}

double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// v rounded to dp decimal places (dividing, so 3.9 stays 3.9 in the JSON)
double round_dp(double v, int dp) {
    const double s = std::pow(10.0, dp);
    return std::round(v * s) / s;
}

std::string base_name(const std::string & path) {
    const size_t s = path.find_last_of("/\\");
    return s == std::string::npos ? path : path.substr(s + 1);
}

struct bench_prompt {
    std::string id;
    json def;
};

// The bench prompts built into the program, in the protocol's order.
std::vector<bench_prompt> builtin_prompts() {
    static const char * order[] = { "short_chat", "code", "long_context", "tool_call" };
    std::vector<bench_prompt> out;
    for (const char * id : order) {
        const std::string path = std::string("bench/prompts/") + id + ".json";
        for (size_t i = 0; i < kera_web_assets_count; ++i) {
            if (path == kera_web_assets[i].path) {
                out.push_back({ id, json::parse(std::string((const char *) kera_web_assets[i].data, kera_web_assets[i].size)) });
            }
        }
    }
    return out;
}

// Render a prompt the way llama.cpp's server does (the model's own chat template, run by its Jinja
// engine). If the template can't take tools or system messages, describe the tools in plain text,
// then fall back to llama.cpp's built-in templates, then to plain "role: text" lines.
std::string render(const common_chat_templates * tmpls, const json & def, const std::string & thinking, std::string * how) {
    std::vector<common_chat_msg> msgs;
    for (const auto & m : def["messages"]) {
        common_chat_msg msg;
        msg.role = m["role"].get<std::string>();
        msg.content = m["content"].get<std::string>();
        msgs.push_back(msg);
    }
    std::vector<common_chat_tool> tools;
    std::string tools_text;
    if (def.contains("tools")) {
        for (const auto & t : def["tools"]) {
            tools.push_back({ t["name"].get<std::string>(), t["description"].get<std::string>(), t["parameters"].dump() });
        }
        tools_text = "\n\nYou can call these tools. To call one, reply with only a JSON object "
                     "{\"name\": ..., \"arguments\": {...}}.\n" + def["tools"].dump(1);
    }
    auto attempt = [&](bool jinja, bool with_tools) -> std::string {
        common_chat_templates_inputs in;
        in.messages = msgs;
        in.use_jinja = jinja;
        in.add_generation_prompt = true;
        if (thinking == "on" || thinking == "off") {
            in.enable_thinking = thinking == "on";
        }
        if (with_tools) {
            in.tools = tools;
        } else if (!tools_text.empty()) {
            in.messages[0].content += tools_text;
        }
        return common_chat_templates_apply(tmpls, in).prompt.str();
    };
    const struct {
        bool jinja, with_tools;
    } ways[] = { { true, true }, { true, false }, { false, false } };
    for (const auto & w : ways) {
        if (w.with_tools && tools.empty()) {
            continue;
        }
        try {
            std::string s = attempt(w.jinja, w.with_tools);
            if (!s.empty()) {
                *how = w.jinja ? "model template" : "llama.cpp built-in template";
                if (!tools.empty() && !w.with_tools) {
                    *how += ", tools as text";
                }
                return s;
            }
        } catch (const std::exception &) {
        }
    }
    std::string s;
    for (size_t i = 0; i < msgs.size(); ++i) {
        s += msgs[i].role + ": " + msgs[i].content + (i == 0 ? tools_text : "") + "\n";
    }
    *how = "plain text";
    return s + "assistant: ";
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> toks((size_t) std::max(n, 0));
    n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, true);
    toks.resize((size_t) std::max(n, 0));
    return toks;
}

// Page faults this process has taken (minor: a page given out or brought in from the OS file cache;
// major: read from the disk). Each costs CPU time, which matters when experts stream in.
void page_faults(uint64_t & minor, uint64_t & major) {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    minor = major = 0;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        minor = pmc.PageFaultCount; // Windows counts both kinds together
    }
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    minor = (uint64_t) ru.ru_minflt;
    major = (uint64_t) ru.ru_majflt;
#endif
}

// Ask the OS to drop the model file from its file cache, so a run starts cold (Linux; macOS has no such call
// for one file, and on Windows the cache can't be dropped for one file without admin rights).
bool drop_file_cache(const std::string & path) {
#if defined(_WIN32)
    (void) path;
    return false;
#else
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
#if defined(POSIX_FADV_DONTNEED)
    const bool ok = posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED) == 0;
#else
    const bool ok = false;
#endif
    std::fclose(f);
    return ok;
#endif
}

struct run_result {
    double ttft_ms = 0;
    double prefill_tok_s = 0;
    double decode_tok_s = 0;
    uint64_t minor_faults = 0;  // while writing
    uint64_t major_faults = 0;
    kera::stream::report stream_write; // what streaming did while it wrote (not while it read the prompt)
};

// One run: read the prompt from an empty context, then write n_gen words greedily. The end-of-text
// word doesn't stop it, so every run writes the same number of words.
run_result run_once(llama_context * ctx, llama_sampler * smpl, const std::vector<llama_token> & prompt, int n_gen, int n_batch,
                    const kera::stream::session * stream) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_sampler_reset(smpl);
    run_result r;
    const auto t0 = clock_type::now();
    for (size_t i = 0; i < prompt.size(); i += (size_t) n_batch) {
        const int n = (int) std::min(prompt.size() - i, (size_t) n_batch);
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(prompt.data() + i), n)) != 0) {
            throw std::runtime_error("reading the prompt failed");
        }
    }
    llama_synchronize(ctx);
    const auto t_read = clock_type::now();
    llama_token tok = llama_sampler_sample(smpl, ctx, -1);
    const auto t1 = clock_type::now();
    const kera::stream::report before = stream ? stream->snapshot() : kera::stream::report{};
    uint64_t min0 = 0, maj0 = 0;
    page_faults(min0, maj0);
    for (int i = 1; i < n_gen; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
            throw std::runtime_error("writing failed");
        }
        tok = llama_sampler_sample(smpl, ctx, -1);
    }
    const auto t2 = clock_type::now();
    uint64_t min1 = 0, maj1 = 0;
    page_faults(min1, maj1);
    r.minor_faults = min1 - min0;
    r.major_faults = maj1 - maj0;
    if (stream) {
        r.stream_write = stream->snapshot() - before;
    }
    r.ttft_ms = ms_since(t0, t1);
    r.prefill_tok_s = (double) prompt.size() / (ms_since(t0, t_read) / 1000.0);
    r.decode_tok_s = n_gen > 1 ? (double) (n_gen - 1) / (ms_since(t1, t2) / 1000.0) : 0.0;
    return r;
}

const char * type_name(ggml_type t) {
    return ggml_type_name(t);
}

void usage() {
    std::fprintf(stderr,
        "Usage: kera bench --model FILE.gguf [options]\n"
        "\n"
        "The section 7 benchmark: wait for the first word, prompt speed and writing speed for each bench\n"
        "prompt (median of 3 runs after a warm-up), saved as JSON.\n"
        "\n"
        "  --only ID[,ID]     only these prompts (short_chat, code, long_context, tool_call)\n"
        "  --runs N           measured runs per prompt (default 3)\n"
        "  --n-gen N          words to write per run (default 256)\n"
        "  --ngl N            layers on the graphics chip (default: all)\n"
        "  --threads N        CPU threads (default: one per performance core)\n"
        "  --batch N          prompt words per step (default 2048); --ubatch N (default 512)\n"
        "  --flash-attn X     auto, on or off (default auto)\n"
        "  --thinking X       default, on or off (for models that think before answering)\n"
        "  --label TEXT       what this run is, e.g. \"stock llama.cpp\"\n"
        "  --set NAME=VALUE   a Kera setting (kera setup --help lists them)\n"
        "  --stream X         off, os or kera: stream the model's routed experts from the SSD (default off)\n"
        "  --stream-budget-gb N   RAM for Kera's expert cache (default 8)\n"
        "  --stream-lookahead N   1 to read the next layer's likely experts early (default 0)\n"
        "  --stream-threads N     reads in flight (default 8)\n"
        "  --load-mode X      auto, mmap or none: how llama.cpp loads the rest of the model (default auto)\n"
        "  --repack N         0 to keep llama.cpp's CPU weight repacking off (default 1)\n"
        "  --warmup N         1 to run each prompt once before measuring (default 1); 0 measures it cold\n"
        "  --cold             drop the model file from the OS file cache before loading (Linux only)\n"
        "  --out FILE         where to save the JSON (default: the Kera folder's bench/)\n"
        "  --markdown         also print rows for docs/BENCHMARKS.md\n");
}

} // namespace

int kera_bench(int argc, char ** argv) {
    std::string model_path, out_path, label = "stock llama.cpp", thinking = "default", flash = "auto";
    std::vector<std::string> only;
    int runs = 3, n_gen = 256, ngl = -1, threads = 0, n_batch = 2048, n_ubatch = 512;
    bool markdown = false;
    std::string load_mode = "auto";
    int repack = 1;
    int warmup = 1;
    bool cold = false;
    kera::stream::config sc;
    sc.mode = kera::stream::cache_mode::off;
    kr_settings_register_builtin();
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(a + " needs a value");
            }
            return argv[++i];
        };
        try {
            if (a == "--model" || a == "-m") {
                model_path = next();
            } else if (a == "--only") {
                std::string v = next();
                for (size_t s = 0, e; s <= v.size(); s = e + 1) {
                    e = v.find(',', s);
                    if (e == std::string::npos) {
                        e = v.size();
                    }
                    if (e > s) {
                        only.push_back(v.substr(s, e - s));
                    }
                }
            } else if (a == "--runs") {
                runs = std::max(1, std::stoi(next()));
            } else if (a == "--n-gen") {
                n_gen = std::max(1, std::stoi(next()));
            } else if (a == "--ngl") {
                ngl = std::stoi(next());
            } else if (a == "--threads") {
                threads = std::stoi(next());
            } else if (a == "--batch") {
                n_batch = std::max(1, std::stoi(next()));
            } else if (a == "--ubatch") {
                n_ubatch = std::max(1, std::stoi(next()));
            } else if (a == "--flash-attn") {
                flash = next();
            } else if (a == "--thinking") {
                thinking = next();
            } else if (a == "--label") {
                label = next();
            } else if (a == "--out") {
                out_path = next();
            } else if (a == "--markdown") {
                markdown = true;
            } else if (a == "--stream") {
                if (!kera::stream::parse_mode(next(), sc.mode)) {
                    throw std::runtime_error("--stream takes off, os or kera");
                }
            } else if (a == "--stream-budget-gb") {
                sc.budget_bytes = (uint64_t) (std::stod(next()) * 1e9);
            } else if (a == "--stream-lookahead") {
                sc.lookahead = std::stoi(next()) != 0;
            } else if (a == "--stream-threads") {
                sc.io_threads = std::max(1, std::stoi(next()));
            } else if (a == "--load-mode") {
                load_mode = next();
            } else if (a == "--repack") {
                repack = std::stoi(next());
            } else if (a == "--warmup") {
                warmup = std::stoi(next());
            } else if (a == "--cold") {
                cold = true;
            } else if (a == "--set") {
                const std::string kv = next();
                const size_t eq = kv.find('=');
                if (eq == std::string::npos || !kr_setting_set(kv.substr(0, eq).c_str(), std::atof(kv.c_str() + eq + 1))) {
                    throw std::runtime_error("unknown setting in " + kv);
                }
            } else if (a == "--help" || a == "-h") {
                usage();
                return 0;
            } else {
                throw std::runtime_error("unknown option " + a);
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr, "%s\n\n", e.what());
            usage();
            return 2;
        }
    }
    if (model_path.empty()) {
        usage();
        return 2;
    }
    if (flash != "auto" && flash != "on" && flash != "off") {
        std::fprintf(stderr, "--flash-attn must be auto, on or off\n");
        return 2;
    }

    std::vector<bench_prompt> prompts;
    for (auto & p : builtin_prompts()) {
        if (only.empty() || std::find(only.begin(), only.end(), p.id) != only.end()) {
            prompts.push_back(std::move(p));
        }
    }
    if (prompts.empty()) {
        std::fprintf(stderr, "No bench prompts match --only\n");
        return 2;
    }

    // what this machine is: the profile auto-setup saved, if any
    kera::machine_profile prof;
    const bool have_profile = kera::load_profile(prof, kera::profile_path());
    if (!have_profile) {
        kera::detect_system(prof);
    }

    std::printf("Hashing the model file...\n");
    int herr = 0;
    const std::string model_hash = kera::model_hash(model_path, &herr);
    if (model_hash.empty()) {
        std::fprintf(stderr, "Can't read %s (error %d)\n", model_path.c_str(), herr);
        return 2;
    }

    llama_backend_init();
    ggml_backend_load_all();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            std::fputs(text, stderr);
        }
    }, nullptr);

    if (cold && !drop_file_cache(model_path)) {
        std::fprintf(stderr, "Couldn't drop the model from the OS file cache; this run isn't cold\n");
    }

    std::shared_ptr<kera::stream::session> stream;
    if (sc.mode != kera::stream::cache_mode::off) {
        sc.model_path = model_path;
        std::string err;
        stream = kera::stream::start(sc, &err);
        if (!stream) {
            std::fprintf(stderr, "Can't stream: %s\n", err.c_str());
            return 2;
        }
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.use_extra_bufts = repack != 0;
    mp.load_mode = load_mode == "mmap" ? LLAMA_LOAD_MODE_MMAP : load_mode == "none" ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_AUTO;
    const auto t_load0 = clock_type::now();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) {
        std::fprintf(stderr, "Can't load %s\n", model_path.c_str());
        return 2;
    }
    if (stream && !stream->streaming()) {
        std::fprintf(stderr, "The model doesn't stream: %s\n", stream->error().empty() ? "it has no routed experts" : stream->error().c_str());
        return 2;
    }
    const double load_s = ms_since(t_load0, clock_type::now()) / 1000.0;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto tmpls = common_chat_templates_init(model, "");
    std::vector<std::vector<llama_token>> toks;
    std::vector<std::string> how;
    int max_tokens = 0;
    for (const auto & p : prompts) {
        std::string h;
        toks.push_back(tokenize(vocab, render(tmpls.get(), p.def, thinking, &h)));
        how.push_back(h);
        max_tokens = std::max(max_tokens, (int) toks.back().size());
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((max_tokens + n_gen + 16 + 255) / 256 * 256);
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) std::min(n_ubatch, n_batch);
    const int n_threads = threads > 0 ? threads : std::max(1, common_cpu_get_num_math());
    cp.n_threads = cp.n_threads_batch = n_threads;
    cp.flash_attn_type = flash == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : flash == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "Can't create a context of %u tokens\n", cp.n_ctx);
        llama_model_free(model);
        return 2;
    }
    llama_sampler * smpl = llama_sampler_init_greedy();

    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    char arch[128] = "";
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    std::printf("Kera %s bench: %s (%s), %d prompt(s), %d run(s) of %d words after a warm-up\n\n",
                KERA_VERSION, base_name(model_path).c_str(), desc, (int) prompts.size(), runs, n_gen);
    std::printf("  %-14s %7s %12s %15s %15s\n", "Prompt", "Words", "First word", "Prompt speed", "Writing speed");

    json results = json::array();
    std::vector<double> md_prefill, md_decode, md_ttft;
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const auto & tk = toks[pi];
        std::vector<run_result> rr;
        try {
            if (warmup) {
                run_once(ctx, smpl, tk, n_gen, n_batch, stream.get()); // warm-up: model and caches in memory
            }
            for (int r = 0; r < runs; ++r) {
                rr.push_back(run_once(ctx, smpl, tk, n_gen, n_batch, stream.get()));
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr, "%s: %s\n", prompts[pi].id.c_str(), e.what());
            continue;
        }
        std::vector<double> ttft, pre, dec;
        json runs_j = json::array();
        kera::stream::report sw; // streaming while writing, summed over the measured runs
        uint64_t minor = 0, major = 0;
        for (const auto & r : rr) {
            minor += r.minor_faults;
            major += r.major_faults;
            sw.callback_s += r.stream_write.callback_s;
            sw.guess_s += r.stream_write.guess_s;
            sw.mode = r.stream_write.mode;
            sw.cache.requests += r.stream_write.cache.requests;
            sw.cache.hits += r.stream_write.cache.hits;
            sw.cache.late += r.stream_write.cache.late;
            sw.cache.misses += r.stream_write.cache.misses;
            sw.cache.bytes_read += r.stream_write.cache.bytes_read;
            sw.cache.io_wait_s += r.stream_write.cache.io_wait_s;
            sw.cache.prefetched += r.stream_write.cache.prefetched;
            sw.cache.prefetch_used += r.stream_write.cache.prefetch_used;
            sw.predicted += r.stream_write.predicted;
            sw.predicted_right += r.stream_write.predicted_right;
            sw.os_hint_bytes += r.stream_write.os_hint_bytes;
            ttft.push_back(r.ttft_ms);
            pre.push_back(r.prefill_tok_s);
            dec.push_back(r.decode_tok_s);
            runs_j.push_back({ { "ttft_ms", round_dp(r.ttft_ms, 1) }, { "prefill_tok_s", round_dp(r.prefill_tok_s, 2) },
                               { "decode_tok_s", round_dp(r.decode_tok_s, 2) } });
        }
        const double m_ttft = median(ttft), m_pre = median(pre), m_dec = median(dec);
        md_ttft.push_back(m_ttft);
        md_prefill.push_back(m_pre);
        md_decode.push_back(m_dec);
        std::printf("  %-14s %7zu %9.0f ms %11.1f tok/s %11.2f tok/s\n", prompts[pi].id.c_str(), tk.size(), m_ttft, m_pre, m_dec);
        const uint64_t n_written = (uint64_t) rr.size() * (uint64_t) std::max(0, n_gen - 1);
        json stream_j = nullptr;
        double ssd_gb_per_tok = -1, expert_hit = -1, prefetch_acc = -1;
        if (n_written > 0) {
            std::printf("    page faults per word while writing: %.0f (%.0f from the disk)\n", (double) minor / (double) n_written,
                        (double) major / (double) n_written);
        }
        if (stream && n_written > 0) {
            std::printf("    %s\n", kera::stream::session::per_word(sw, n_written).c_str());
            std::printf("    Kera's per-layer step: %.1f ms per word, of which %.1f ms guessing the next layer\n",
                        1000.0 * sw.callback_s / (double) n_written, 1000.0 * sw.guess_s / (double) n_written);
            if (sw.mode == kera::stream::cache_mode::kera) {
                ssd_gb_per_tok = (double) sw.cache.bytes_read / 1e9 / (double) n_written;
                expert_hit = sw.cache.requests ? (double) sw.cache.hits / (double) sw.cache.requests : 0.0;
            }
            if (sw.predicted > 0) {
                prefetch_acc = (double) sw.predicted_right / (double) sw.predicted;
            }
            stream_j = {
                { "mode", kera::stream::mode_name(sw.mode) },
                { "experts_asked", sw.cache.requests }, { "in_memory", sw.cache.hits }, { "on_their_way", sw.cache.late },
                { "read_while_waiting", sw.cache.misses }, { "read_early", sw.cache.prefetched },
                { "read_early_used", sw.cache.prefetch_used }, { "gb_read", round_dp((double) sw.cache.bytes_read / 1e9, 3) },
                { "io_wait_ms_per_tok", round_dp(1000.0 * sw.cache.io_wait_s / (double) n_written, 2) },
                { "guessed", sw.predicted }, { "guessed_right", sw.predicted_right },
                { "step_ms_per_tok", round_dp(1000.0 * sw.callback_s / (double) n_written, 2) },
                { "guess_ms_per_tok", round_dp(1000.0 * sw.guess_s / (double) n_written, 2) },
                { "os_hint_gb", round_dp((double) sw.os_hint_bytes / 1e9, 3) },
            };
        }
        results.push_back({
            { "prompt_id", prompts[pi].id },
            { "rendered_with", how[pi] },
            { "n_prompt", tk.size() },
            { "n_gen", n_gen },
            { "ttft_ms", round_dp(m_ttft, 1) },
            { "prefill_tok_s", round_dp(m_pre, 2) },
            { "decode_tok_s", round_dp(m_dec, 2) },
            { "page_faults_per_tok", n_written ? json(round_dp((double) (minor + major) / (double) n_written, 1)) : json(nullptr) },
            { "runs", runs_j },
            // filled in by the milestones that add these parts (streaming, guessing, KV reuse)
            { "ssd_gb_per_tok", ssd_gb_per_tok >= 0 ? json(round_dp(ssd_gb_per_tok, 4)) : json(nullptr) },
            { "expert_hit", expert_hit >= 0 ? json(round_dp(expert_hit, 4)) : json(nullptr) },
            { "prefetch_acc", prefetch_acc >= 0 ? json(round_dp(prefetch_acc, 4)) : json(nullptr) },
            { "stream", stream_j },
            { "kv_hit", nullptr }, { "accept_rate", nullptr }, { "k_histogram", nullptr },
        });
    }

    const double peak = peak_ram_gib();
    std::printf("\nPeak memory %.1f GB. Loading took %.1f s. One word a second is about 45 words of text a minute.\n", peak, load_s);

    json kera_settings = json::object();
    for (size_t i = 0; i < kr_settings_count(); ++i) {
        kr_setting_info info;
        if (kr_settings_get_info(i, &info)) {
            kera_settings[info.name] = info.value;
        }
    }
    json devices = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        devices.push_back(std::string(ggml_backend_dev_name(d)) + ": " + ggml_backend_dev_description(d));
    }
    const std::string prof_json = have_profile ? prof.to_json().dump() : "";
    json out = {
        { "schema", "kera-bench-1" },
        { "label", label },
        { "date", kera::utc_now_iso8601() },
        { "kera", { { "version", KERA_VERSION }, { "commit", KERA_GIT_COMMIT }, { "llama_cpp", KERA_LLAMA_CPP_TAG } } },
        { "machine", {
            { "profile_hash", have_profile ? kera::hash_bytes(prof_json.data(), prof_json.size()).substr(0, 16) : "" },
            { "os", prof.os }, { "chip", prof.chip }, { "ram_gb", prof.ram_gb },
            { "gpu", prof.gpus.empty() ? "" : prof.gpus[0].name }, { "devices", devices },
        } },
        { "model", {
            { "file", base_name(model_path) }, { "hash", model_hash }, { "desc", desc }, { "architecture", arch },
            { "size_gb", round_dp((double) llama_model_size(model) / 1e9, 2) }, { "n_params", llama_model_n_params(model) },
        } },
        { "settings", {
            { "kera", kera_settings },
            { "engine", {
                { "n_gpu_layers", ngl }, { "n_ctx", cp.n_ctx }, { "n_batch", cp.n_batch }, { "n_ubatch", cp.n_ubatch },
                { "n_threads", n_threads }, { "flash_attn", flash }, { "type_k", type_name(cp.type_k) },
                { "type_v", type_name(cp.type_v) }, { "thinking", thinking }, { "sampling", "greedy" },
                { "load_mode", load_mode }, { "repack", repack },
            } },
            { "stream", {
                { "mode", kera::stream::mode_name(sc.mode) },
                { "budget_gb", sc.mode == kera::stream::cache_mode::kera ? json(round_dp((double) sc.budget_bytes / 1e9, 2)) : json(nullptr) },
                { "lookahead", sc.lookahead }, { "io_threads", sc.io_threads },
                { "expert_gb", stream ? json(round_dp((double) stream->expert_bytes_total() / 1e9, 2)) : json(nullptr) },
            } },
        } },
        { "cache_state", warmup ? "warm" : (cold ? "cold" : "as found") },
        { "runs", runs },
        { "load_s", round_dp(load_s, 2) },
        { "peak_ram_gb", round_dp(peak, 2) },
        { "ceiling_tok_s", nullptr }, // the planner's prediction, from M3
        { "results", results },
        { "gates", json::object() },
    };

    if (out_path.empty()) {
        const std::string dir = kera::home_path("bench");
        kera::os::make_dir(dir);
        std::string stem = base_name(model_path);
        if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".gguf") {
            stem.resize(stem.size() - 5);
        }
        std::string stamp = kera::utc_now_iso8601();
        stamp.erase(std::remove(stamp.begin(), stamp.end(), ':'), stamp.end());
        out_path = kera::join(dir, stamp + "_" + stem + ".json");
    }
    {
        std::ofstream f(out_path, std::ios::binary);
        f << out.dump(1) << "\n";
        if (!f) {
            std::fprintf(stderr, "Can't write %s\n", out_path.c_str());
        }
    }
    std::printf("Saved %s\n", out_path.c_str());

    if (markdown && !results.empty()) {
        const std::string machine = prof.chip.empty() ? prof.os : prof.chip;
        std::printf("\n");
        for (size_t i = 0; i < results.size(); ++i) {
            std::printf("| %s | %s | %s | %s | %.1f | %.2f | %.0f | %s prompt, %d words |\n",
                        out["date"].get<std::string>().substr(0, 10).c_str(), machine.c_str(), label.c_str(),
                        base_name(model_path).c_str(), md_prefill[i], md_decode[i], md_ttft[i],
                        results[i]["prompt_id"].get<std::string>().c_str(), results[i]["n_prompt"].get<int>());
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return results.size() == prompts.size() ? 0 : 1;
}
