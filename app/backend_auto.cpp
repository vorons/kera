// kera serve's choice between Vulkan and the optional ROCm add-on (the owner's request of 17 Oct, docs/DECISIONS.md
// D-022; the setting gpu.auto_backend, or --backend auto). On the Minisforum, Vulkan reads short prompts and writes
// faster while ROCm reads long documents twice as fast (docs/BENCHMARKS.md, 16 Oct), so which is faster depends on
// the model and on how it is used. Each back-end is measured once per model (kera calibrate, in its own process, so
// two back-ends never hold the model at once; kept in backends.json), every reply's shape is noted (llama.cpp hook 6:
// how much the chat held, read and wrote; in replies/), and each start takes the back-end with the shorter expected
// time for the model's last 300 replies, or, until it has served 20, for chats with now and then a long document
// (src/planner/backend_pick.h). Only token counts and times are noted, never what was said.

#include "app/kera_app.h"
#include "common/kr_hash.h"
#include "common/kr_paths.h"
#include "planner/backend_pick.h"
#include "spec/kr_guess_memory.h"
#include "tune/kr_settings.h"

#include "kera_version.h"

#include "common.h"
#include "reply-observer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace fs = std::filesystem;

namespace kera {

namespace {

const int k_history = 300;      // the replies weighed
const int k_min_history = 20;   // with fewer, chats with now and then a long document
const int k_depth = 16384;      // how deep kera calibrate reads

std::string kera_exe() {
#if defined(_WIN32)
    return join(exe_dir(), "kera.exe");
#else
    return join(exe_dir(), "kera");
#endif
}

const char * shown(const std::string & backend) {
    return backend == "hip" ? "ROCm" : backend == "vulkan" ? "Vulkan" : backend == "cpu" ? "the CPU" : backend.c_str();
}

#if defined(_WIN32)
std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

// one word of a command line, quoted the way programs split them again (CommandLineToArgvW)
std::wstring quoted(const std::wstring & a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return a;
    }
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (const wchar_t c : a) {
        if (c == L'\\') {
            ++slashes;
            continue;
        }
        out.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        slashes = 0;
        out.push_back(c);
    }
    out.append(2 * slashes, L'\\');
    out.push_back(L'"');
    return out;
}
#else
volatile pid_t g_child = 0;

// stopping kera serve while it measures stops the measurement too (the app stops it with Ctrl+C's signal)
void stop_child(int sig) {
    if (g_child > 0) {
        kill(g_child, SIGKILL);
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}
#endif

// Runs a program with its output going to `log` and waits for it; its exit code, -1 if it didn't start
int run_and_wait(const std::vector<std::string> & argv, const std::string & log) {
#if defined(_WIN32)
    std::wstring cmd;
    for (const auto & a : argv) {
        cmd += (cmd.empty() ? L"" : L" ") + quoted(widen(a));
    }
    SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
    HANDLE out = CreateFileW(widen(log).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    // in a job that ends with kera serve: the app stops kera serve by ending it, which would leave the measurement
    // holding the chip
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job != nullptr) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
    }
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = out;
    si.hStdError = out;
    si.hStdInput = nullptr;
    PROCESS_INFORMATION pi = {};
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                                   nullptr, &si, &pi);
    if (out != INVALID_HANDLE_VALUE) {
        CloseHandle(out);
    }
    if (!ok) {
        if (job != nullptr) {
            CloseHandle(job);
        }
        return -1;
    }
    if (job != nullptr) {
        AssignProcessToJobObject(job, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    if (job != nullptr) {
        CloseHandle(job);
    }
    return (int) code;
#else
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    std::vector<char *> args;
    for (const auto & a : argv) {
        args.push_back(const_cast<char *>(a.c_str()));
    }
    args.push_back(nullptr);
    pid_t child = 0;
    const int rc = posix_spawn(&child, argv[0].c_str(), &fa, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        return -1;
    }
    g_child = child;
    auto old_int = std::signal(SIGINT, stop_child);
    auto old_term = std::signal(SIGTERM, stop_child);
    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    g_child = 0;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

bool has_flag(int argc, char ** argv, const char * name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}

std::string flag_value(int argc, char ** argv, std::initializer_list<const char *> names) {
    for (int i = 1; i + 1 < argc; ++i) {
        for (const char * n : names) {
            if (std::strcmp(argv[i], n) == 0) {
                return argv[i + 1];
            }
        }
    }
    return "";
}

// The options on kera serve's command line that change a back-end's speed, for kera calibrate
std::vector<std::string> speed_options(int argc, char ** argv) {
    static const char * const with_value[] = {
        "-m", "--model", "-ngl", "--gpu-layers", "--n-gpu-layers", "-fa", "--flash-attn", "-ctk", "--cache-type-k",
        "-ctv", "--cache-type-v", "-b", "--batch-size", "-ub", "--ubatch-size", "-t", "--threads", "-tb",
        "--threads-batch", "-ot", "--override-tensor", "-ncmoe", "--n-cpu-moe", "-ts", "--tensor-split", "-sm",
        "--split-mode", "-mg", "--main-gpu", "-fit", "--fit", "-fitt", "--fit-target", "-fitc", "--fit-ctx", "-lm",
        "--load-mode", "--numa",
    };
    static const char * const alone[] = {
        "-cmoe", "--cpu-moe", "-kvo", "--kv-offload", "-nkvo", "--no-kv-offload", "--swa-full", "--op-offload",
        "--no-op-offload", "--repack", "-nr", "--no-repack", "--no-host",
    };
    std::vector<std::string> out;
    for (int i = 1; i < argc; ++i) {
        bool taken = false;
        for (const char * f : with_value) {
            if (std::strcmp(argv[i], f) == 0 && i + 1 < argc) {
                out.push_back(argv[i]);
                out.push_back(argv[++i]);
                taken = true;
                break;
            }
        }
        for (const char * f : alone) {
            if (!taken && std::strcmp(argv[i], f) == 0) {
                out.push_back(argv[i]);
                break;
            }
        }
    }
    return out;
}

// What a measurement fits: the model file, this build, and the options that change speed (not the model's path)
std::string measurement_key(const std::string & model, const std::vector<std::string> & opts) {
    std::string k = guess_memory_key(model) + "|kera " KERA_VERSION "|llama.cpp " KERA_LLAMA_CPP_TAG "|depth " +
                    std::to_string(k_depth);
    for (size_t i = 0; i < opts.size(); ++i) {
        if ((opts[i] == "-m" || opts[i] == "--model") && i + 1 < opts.size()) {
            ++i;
            continue;
        }
        k += "|" + opts[i];
    }
    return k;
}

nlohmann::json load_json(const std::string & path) {
    std::ifstream f(fs::u8path(path));
    if (!f) {
        return nlohmann::json::object();
    }
    try {
        nlohmann::json j = nlohmann::json::parse(f);
        return j.is_object() ? j : nlohmann::json::object();
    } catch (const std::exception &) {
        return nlohmann::json::object();
    }
}

void save_json(const std::string & path, const nlohmann::json & j) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(fs::u8path(tmp), std::ios::trunc);
        f << j.dump(1) << "\n";
        if (!f) {
            return;
        }
    }
    std::error_code ec;
    fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
}

// One back-end measured by kera calibrate in its own process; {"ok": false, ...} if it couldn't be
nlohmann::json measure(const std::string & backend, const std::vector<std::string> & opts) {
    const std::string out = home_path("calibrate-" + backend + ".json");
    const std::string log = home_path("calibrate-" + backend + ".log");
    std::error_code ec;
    fs::remove(fs::u8path(out), ec);
    std::vector<std::string> argv = { kera_exe(), "calibrate", "--backend", backend, "--depth", std::to_string(k_depth),
                                      "--json", out };
    argv.insert(argv.end(), opts.begin(), opts.end());
    std::fprintf(stderr, "Kera: measuring %s on this model, once (a minute or two; the log is %s)\n", shown(backend),
                 log.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = run_and_wait(argv, log);
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    nlohmann::json j = load_json(out);
    fs::remove(fs::u8path(out), ec);
    if (!j.contains("ok")) {
        j = { { "ok", false }, { "error", "kera calibrate stopped with exit code " + std::to_string(rc) } };
    }
    j["took_s"] = took;
    j["time"] = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
    if (!j.value("ok", false)) {
        std::fprintf(stderr, "Kera: couldn't measure %s on this model (%s; see %s)\n", shown(backend),
                     j.value("error", std::string("no reason given")).c_str(), log.c_str());
    }
    return j;
}

backend_speed speed_of(const std::string & backend, const nlohmann::json & j) {
    backend_speed s;
    s.backend = backend;
    if (j.is_object() && j.value("ok", false)) {
        s.read_s = j.value("read_s", 0.0);
        s.read_s_per_depth = j.value("read_s_per_depth", 0.0);
        s.write_s = j.value("write_s", 0.0);
        s.write_s_per_depth = j.value("write_s_per_depth", 0.0);
    }
    return s;
}

std::string replies_path(const std::string & model) {
    const std::string key = guess_memory_key(model);
    return join(home_path("replies"), hash_bytes(key.data(), key.size()).substr(0, 16) + ".tsv");
}

// A past reply, as noted: when, on which back-end, the tokens held, read and written (decode steps), its two times
// and what the measurement predicted for it (-1 if there was none)
struct past_reply {
    long long time = 0;
    std::string backend;
    double held = 0, read = 0, written = 0, t_read = 0, t_write = 0, predicted = -1;
};

// The model's last `keep` replies; a long file is cut down to its last 1000 lines
std::vector<past_reply> past_replies(const std::string & path, size_t keep) {
    std::deque<std::string> lines;
    size_t n_lines = 0;
    {
        std::ifstream f(fs::u8path(path));
        std::string line;
        while (std::getline(f, line)) {
            ++n_lines;
            lines.push_back(line);
            if (lines.size() > 1000) {
                lines.pop_front();
            }
        }
    }
    if (n_lines > 5000) {
        std::string all;
        for (const auto & l : lines) {
            all += l + "\n";
        }
        const std::string tmp = path + ".tmp";
        {
            std::ofstream(fs::u8path(tmp), std::ios::trunc) << all;
        }
        std::error_code ec;
        fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
    }
    std::vector<past_reply> out;
    for (size_t i = lines.size() > keep ? lines.size() - keep : 0; i < lines.size(); ++i) {
        std::istringstream s(lines[i]);
        past_reply r;
        if (s >> r.time >> r.backend >> r.held >> r.read >> r.written >> r.t_read >> r.t_write >> r.predicted) {
            out.push_back(r);
        }
    }
    return out;
}

// Hook 6: notes each reply's shape, and sums its times against the prediction for the summary at the end
class reply_notes : public common_reply_observer {
public:
    reply_notes(std::string path, std::string backend, backend_speed speed)
        : path_(std::move(path)), backend_(std::move(backend)), speed_(speed) {}

    void reply(int32_t n_held, int32_t n_read, int32_t n_written, double t_read_ms, double t_write_ms) override {
        const double steps = n_written > 0 ? n_written - 1 : 0; // the first word comes with the prompt's last batch
        const double predicted = speed_.ok() ? speed_.reply_seconds(n_held, n_read, steps) : -1;
        const double took = (t_read_ms + t_write_ms) / 1000;
        std::lock_guard<std::mutex> lock(mu_);
        ++n_;
        if (predicted > 0) {
            took_ += took;
            predicted_ += predicted;
            ++n_predicted_;
        }
        std::ofstream f(fs::u8path(path_), std::ios::app);
        char line[256];
        std::snprintf(line, sizeof(line), "%lld\t%s\t%d\t%d\t%.0f\t%.4f\t%.4f\t%.4f\n",
                      (long long) std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count(),
                      backend_.c_str(), n_held, n_read, steps, t_read_ms / 1000, t_write_ms / 1000, predicted);
        f << line;
    }

    void summary() const {
        std::lock_guard<std::mutex> lock(mu_);
        if (n_predicted_ > 0 && predicted_ > 0) {
            std::fprintf(stderr, "Kera: over this run, %d replies on %s took %.1f s; the measurement predicted %.1f s (%+.0f%%)\n",
                         n_predicted_, shown(backend_), took_, predicted_, 100 * (took_ - predicted_) / predicted_);
        }
    }

private:
    std::string path_;
    std::string backend_;
    backend_speed speed_;
    mutable std::mutex mu_;
    int n_ = 0;
    int n_predicted_ = 0;
    double took_ = 0;
    double predicted_ = 0;
};

std::unique_ptr<reply_notes> g_notes;

// The back-ends there is a choice between: Vulkan and the ROCm add-on, when it is installed (KERA_AUTO_BACKENDS
// sets the list, for tests)
std::vector<std::string> candidates() {
    if (const char * e = std::getenv("KERA_AUTO_BACKENDS"); e != nullptr && *e != '\0') {
        std::vector<std::string> out;
        std::stringstream s(e);
        std::string b;
        while (std::getline(s, b, ',')) {
            if (!b.empty()) {
                out.push_back(b);
            }
        }
        return out;
    }
    if (!hip_addon_installed()) {
        return {};
    }
    return { "vulkan", "hip" };
}

} // namespace

std::vector<backend_speed> backend_speeds(const std::string & model, int argc, char ** argv) {
    const std::vector<std::string> backends = candidates();
    if (model.empty() || backends.size() < 2 || has_flag(argc, argv, "--device") || has_flag(argc, argv, "-dev")) {
        return {};
    }
    std::vector<backend_speed> speeds;
    // KERA_AUTO_SPEEDS sets each back-end's speeds instead of measuring them, for tests: "a,b,c,d;a,b,c,d" in the
    // order of the list, each read_s, read_s_per_depth, write_s, write_s_per_depth
    if (const char * e = std::getenv("KERA_AUTO_SPEEDS"); e != nullptr && *e != '\0') {
        std::stringstream all(e);
        std::string one;
        for (size_t i = 0; i < backends.size() && std::getline(all, one, ';'); ++i) {
            backend_speed s;
            s.backend = backends[i];
            std::sscanf(one.c_str(), "%lf,%lf,%lf,%lf", &s.read_s, &s.read_s_per_depth, &s.write_s, &s.write_s_per_depth);
            speeds.push_back(s);
        }
        return speeds;
    }
    const std::vector<std::string> opts = speed_options(argc, argv);
    const std::string key = measurement_key(model, opts);
    const std::string file = home_path("backends.json");
    nlohmann::json all = load_json(file);
    if (!all[key].is_object()) {
        all[key] = nlohmann::json::object();
    }
    for (const auto & b : backends) {
        if (!all[key].contains(b)) {
            all[key][b] = measure(b, opts);
            save_json(file, all);
        }
        speeds.push_back(speed_of(b, all[key][b]));
    }
    return speeds;
}

std::string auto_backend(const std::string & model, int argc, char ** argv) {
    const std::vector<backend_speed> speeds = backend_speeds(model, argc, argv);
    if (speeds.empty()) {
        return "";
    }
    const std::string file = home_path("backends.json");

    // how the model is used: its last replies, or with too few of those, chats and now and then a long document
    std::vector<reply_shape> replies;
    for (const auto & r : past_replies(replies_path(model), k_history)) {
        replies.push_back({ r.held, r.read, r.written, 1 });
    }
    const bool own = (int) replies.size() >= k_min_history;
    if (!own) {
        int n_ctx = std::atoi(flag_value(argc, argv, { "-c", "--ctx-size" }).c_str());
        if (n_ctx <= 0) {
            n_ctx = (int) std::min<uint32_t>(common_get_gguf_info(model).n_ctx_train, 1u << 20);
        }
        replies = default_replies(n_ctx > 0 ? n_ctx : 32768);
    }
    const backend_pick p = pick_backend(speeds, replies);
    if (p.backend.empty()) {
        std::fprintf(stderr, "Kera: neither back-end could be measured on this model; leaving the choice to llama.cpp\n");
        return "";
    }
    char why[320];
    if (p.other.empty()) {
        std::snprintf(why, sizeof(why), "the only one that could be measured on this model");
    } else if (own) {
        std::snprintf(why, sizeof(why), "the faster for this model's last %zu replies: %.1f s a reply on average, against "
                      "%.1f s on %s", replies.size(), p.seconds, p.other_seconds, shown(p.other));
    } else {
        std::snprintf(why, sizeof(why), "the faster for chats with now and then a long document, until this model has "
                      "served %d replies: %.1f s a reply on average, against %.1f s on %s", k_min_history, p.seconds,
                      p.other_seconds, shown(p.other));
    }
    std::fprintf(stderr, "Kera: using %s: %s (choose with --backend; delete %s to measure again)\n", shown(p.backend), why,
                 file.c_str());
    return p.backend;
}

void note_replies(const std::string & model, const std::string & backend, int argc, char ** argv) {
    if (model.empty() || candidates().size() < 2) {
        return; // nothing to choose between, so nothing to note
    }
    const std::string b = backend.empty() ? "vulkan" : backend; // llama.cpp's own choice is Vulkan without the add-on
    // this back-end's measurement with these options, if there is one, for the summary at the end
    const nlohmann::json all = load_json(home_path("backends.json"));
    const std::string key = measurement_key(model, speed_options(argc, argv));
    backend_speed s;
    if (all.contains(key) && all[key].is_object() && all[key].contains(b)) {
        s = speed_of(b, all[key][b]);
    }
    std::error_code ec;
    fs::create_directories(fs::u8path(home_path("replies")), ec);
    g_notes = std::make_unique<reply_notes>(replies_path(model), b, s);
    common_set_reply_observer(g_notes.get());
}

void stop_noting_replies() {
    common_set_reply_observer(nullptr);
    if (g_notes) {
        g_notes->summary();
        g_notes.reset();
    }
}

} // namespace kera
