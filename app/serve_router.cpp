// kera serve --backend auto, message by message (the owner's request of 18 Oct, docs/DECISIONS.md D-024). On the
// Minisforum Vulkan writes about a fifth faster and ROCm reads long documents twice as fast, so the best back-end
// depends on the message, not just the model. Kera runs the model on each back-end as its own kera serve on a private
// port (an engine), answers on the asked-for port itself, and sends each request to the engine expected to finish it
// sooner (src/planner/engine_route.h), counting what that engine already holds, so a follow-up stays where its
// document was read. With memory free for two copies both engines stay loaded; otherwise one is loaded at a time and
// the other is loaded in its place, once the loaded one is idle, when that pays. Requests that aren't a prompt
// (/health, /v1/models, /metrics...) go to the loaded engine (the first, with both).

#include "app/kera_app.h"
#include "common/kr_paths.h"
#include "planner/engine_route.h"
#include "planner/kr_plan.h"
#include "setup/kr_machine.h"
#include "tune/kr_settings.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace kera {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char * label(const std::string & backend) {
    return backend == "hip" ? "ROCm" : backend == "vulkan" ? "Vulkan" : backend == "cpu" ? "CPU" : backend.c_str();
}

std::string kera_exe() {
#if defined(_WIN32)
    return join(exe_dir(), "kera.exe");
#else
    return join(exe_dir(), "kera");
#endif
}

// a port on 127.0.0.1 nothing listens on, for an engine
int free_port() {
#if defined(_WIN32)
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int port = 0;
    if (s != INVALID_SOCKET) {
        sockaddr_in a = {};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        int len = sizeof(a);
        if (bind(s, (const sockaddr *) &a, sizeof(a)) == 0 && getsockname(s, (sockaddr *) &a, &len) == 0) {
            port = ntohs(a.sin_port);
        }
        closesocket(s);
    }
    WSACleanup();
    return port;
#else
    const int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
        return 0;
    }
    sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(a);
    int port = 0;
    if (bind(s, (const sockaddr *) &a, sizeof(a)) == 0 && getsockname(s, (sockaddr *) &a, &len) == 0) {
        port = ntohs(a.sin_port);
    }
    close(s);
    return port;
#endif
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
#endif

// One engine: kera serve on one back-end, its output passed on line by line with the back-end's name in front
class engine_process {
public:
    engine_process(std::string backend, int port) : backend_(std::move(backend)), port_(port) {}

    ~engine_process() {
        stop();
        if (reader_.joinable()) {
            reader_.join();
        }
    }

    const std::string & backend() const { return backend_; }
    int port() const { return port_; }

    bool start(const std::vector<std::string> & argv) {
        if (reader_.joinable()) {
            reader_.join(); // the last run's output, which ended with it
        }
#if defined(_WIN32)
        // every engine is in one job that ends with this process, however it ends (the app ends kera serve with
        // TerminateProcess)
        static HANDLE job = [] {
            HANDLE j = CreateJobObjectW(nullptr, nullptr);
            if (j != nullptr) {
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
                li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof(li));
            }
            return j;
        }();
        std::wstring cmd;
        for (const auto & a : argv) {
            cmd += (cmd.empty() ? L"" : L" ") + quoted(widen(a));
        }
        SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
        HANDLE rd = nullptr;
        HANDLE wr = nullptr;
        if (!CreatePipe(&rd, &wr, &sa, 0)) {
            return false;
        }
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = wr;
        si.hStdError = wr;
        si.hStdInput = nullptr;
        PROCESS_INFORMATION pi = {};
        const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED,
                                       nullptr, nullptr, &si, &pi);
        CloseHandle(wr);
        if (!ok) {
            CloseHandle(rd);
            return false;
        }
        if (job != nullptr) {
            AssignProcessToJobObject(job, pi.hProcess);
        }
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        process_ = pi.hProcess;
        reader_ = std::thread([this, rd] {
            char buf[4096];
            DWORD n = 0;
            while (ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n > 0) {
                pass_on(buf, n);
            }
            CloseHandle(rd);
            pass_on("\n", 1);
        });
        return true;
#else
        int fds[2];
        if (pipe(fds) != 0) {
            return false;
        }
        fcntl(fds[0], F_SETFD, FD_CLOEXEC);
        fcntl(fds[1], F_SETFD, FD_CLOEXEC);
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
        posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
        posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
        std::vector<char *> args;
        for (const auto & a : argv) {
            args.push_back(const_cast<char *>(a.c_str()));
        }
        args.push_back(nullptr);
        // in a process group of its own: Ctrl+C in a terminal reaches kera serve, which stops each engine once (a
        // second SIGINT makes llama.cpp's server quit at once, losing chats still being saved)
        posix_spawnattr_t attr;
        posix_spawnattr_init(&attr);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attr, 0);
        pid_t child = 0;
        const int rc = posix_spawn(&child, argv[0].c_str(), &fa, &attr, args.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        posix_spawnattr_destroy(&attr);
        close(fds[1]);
        if (rc != 0) {
            close(fds[0]);
            return false;
        }
        pid_ = child;
        const int rd = fds[0];
        reader_ = std::thread([this, rd] {
            char buf[4096];
            ssize_t n = 0;
            while ((n = read(rd, buf, sizeof(buf))) > 0 || (n < 0 && errno == EINTR)) {
                if (n > 0) {
                    pass_on(buf, (size_t) n);
                }
            }
            close(rd);
            pass_on("\n", 1);
        });
        return true;
#endif
    }

    // whether it has stopped, and its exit code
    bool exited(int * code) {
#if defined(_WIN32)
        if (process_ == nullptr) {
            return true;
        }
        if (WaitForSingleObject(process_, 0) != WAIT_OBJECT_0) {
            return false;
        }
        DWORD c = 0;
        GetExitCodeProcess(process_, &c);
        *code = (int) c;
        return true;
#else
        if (pid_ <= 0) {
            *code = code_;
            return true;
        }
        int status = 0;
        if (waitpid(pid_, &status, WNOHANG) != pid_) {
            return false;
        }
        code_ = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        *code = code_;
        pid_ = 0;
        return true;
#endif
    }

    // Ctrl+C's signal first, so it can finish writing saved chats; then the hard way
    void stop() {
#if defined(_WIN32)
        if (process_ != nullptr) {
            TerminateProcess(process_, 0); // saved chats are renamed into place once complete: none is left half-written
            WaitForSingleObject(process_, 10000);
            CloseHandle(process_);
            process_ = nullptr;
        }
#else
        if (pid_ > 0) {
            kill(pid_, SIGINT);
            int code = 0;
            for (int i = 0; i < 60 && !exited(&code); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            if (pid_ > 0) {
                kill(pid_, SIGKILL);
                waitpid(pid_, nullptr, 0);
                pid_ = 0;
            }
        }
#endif
    }

private:
    void pass_on(const char * p, size_t n) {
        std::lock_guard<std::mutex> lock(line_mu_);
        line_.append(p, n);
        size_t at = 0;
        for (size_t nl; (nl = line_.find('\n', at)) != std::string::npos; at = nl + 1) {
            if (nl > at) {
                std::fprintf(stderr, "[%s] %.*s\n", label(backend_), (int) (nl - at), line_.data() + at);
            }
        }
        line_.erase(0, at);
        std::fflush(stderr);
    }

    std::string backend_;
    int port_;
    std::thread reader_;
    std::mutex line_mu_;
    std::string line_;
#if defined(_WIN32)
    HANDLE process_ = nullptr;
#else
    pid_t pid_ = 0;
    int code_ = 0;
#endif
};

// What a request's answer passes through: the engine's status and headers, then its body as it comes
struct relay {
    std::mutex mu;
    std::condition_variable cv;
    bool have_headers = false;
    bool done = false;
    bool cancel = false;
    httplib::Client * cli = nullptr; // while the request to the engine is under way

    // the chat app went away (or kera serve is stopping): so does the engine's request, which closing the
    // connection to it tells it (with mu held)
    void stop_locked() {
        cancel = true;
        if (cli != nullptr) {
            cli->stop();
        }
    }
    int status = 502;
    httplib::Headers headers;
    std::deque<std::string> chunks;
    std::string tail; // the end of the body, where the token counts are
};

bool hop_by_hop(const std::string & key) {
    std::string k = key;
    std::transform(k.begin(), k.end(), k.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return k == "host" || k == "content-length" || k == "connection" || k == "keep-alive" || k == "transfer-encoding" ||
           k == "accept-encoding" || k == "content-type" || k == "upgrade" || k == "te" || k == "trailer" ||
           k == "proxy-connection";
}

// the last number after `key` in s (-1 if none)
double last_number(const std::string & s, const char * key) {
    const size_t at = s.rfind(key);
    if (at == std::string::npos) {
        return -1;
    }
    size_t i = at + std::strlen(key);
    while (i < s.size() && (s[i] == ' ' || s[i] == ':' || s[i] == '"')) {
        ++i;
    }
    return i < s.size() && std::isdigit((unsigned char) s[i]) ? std::atof(s.c_str() + i) : -1;
}

bool is_prompt_path(const std::string & p) {
    for (const char * x : { "/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/completion",
                            "/v1/messages", "/v1/responses", "/responses", "/infill" }) {
        if (p == x) {
            return true;
        }
    }
    return false;
}

httplib::Server * g_router = nullptr;
std::atomic<bool> g_stopping{ false }; // kera serve is stopping: requests waiting give up

void stop_router(int) {
    g_stopping = true;
    if (g_router != nullptr) {
        g_router->stop();
    }
}

#if defined(_WIN32)
BOOL WINAPI stop_router_console(DWORD) {
    stop_router(0);
    return TRUE;
}
#endif

} // namespace

int serve_each_message(const std::string & model, int argc, char ** argv, int n_llama, char ** llama_args) {
    if (model.empty()) {
        return -1;
    }
    const std::vector<backend_speed> speeds = backend_speeds(model, n_llama, llama_args);
    if (speeds.size() < 2) {
        return -1;
    }
    std::vector<backend_speed> usable;
    for (const auto & s : speeds) {
        if (s.ok()) {
            usable.push_back(s);
        }
    }
    if (usable.size() < 2) {
        return -1; // one couldn't be measured: the per-model choice says which is left
    }

    // room for the model twice in what is free now (the PC's other programs keep theirs): both engines stay loaded;
    // otherwise one at a time, the other loaded in its place when that pays
    model_sizes ms;
    machine_profile prof;
    const bool sized = read_model_sizes(model, ms);
    if (!load_profile(prof, profile_path())) {
        detect_system(prof);
        prof.usable_gb = std::max(0.0, 0.8 * prof.ram_gb - 1.0);
    }
    const double free_gb = available_ram_gb();
    const double one_gb = sized ? ms.file_gb + 2.5 : 0; // its weights, and a little for its chats and working buffers
    // (KERA_AUTO_KEEP=both or one sets it, for tests)
    const char * keep = std::getenv("KERA_AUTO_KEEP");
    const bool both = keep != nullptr && std::strcmp(keep, "both") == 0 ? true
                    : keep != nullptr && std::strcmp(keep, "one") == 0  ? false
                    : sized && free_gb > 0 && 2 * one_gb + 4 <= free_gb && 2 * one_gb <= prof.usable_gb;

    // the address chat apps use, and the command line for each engine
    std::string host = "127.0.0.1";
    int port = 8080;
    std::vector<std::string> base = { kera_exe(), "serve" };
    bool cram_set = false;
    bool store_on = kr_setting_get("kv.store") > 0.5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--host" || a == "--port" || a == "--backend") && i + 1 < argc) {
            if (a == "--host") {
                host = argv[i + 1];
            } else if (a == "--port") {
                port = std::atoi(argv[i + 1]);
            }
            ++i;
            continue;
        }
        cram_set = cram_set || a == "-cram" || a == "--cache-ram";
        store_on = store_on && a != "--no-kv-store";
        base.push_back(a);
    }
    if (both && !cram_set) {
        // each engine's recent chats in RAM: half of what the two models leave (each saves to the SSD too)
        base.push_back("--cache-ram");
        base.push_back(std::to_string((int) std::clamp((free_gb - 2 * one_gb - 4) * 1024 / 2, 512.0, 8192.0)));
    }
#if !defined(_WIN32)
    // an engine whose router is gone stops too (Windows: the job does it)
    setenv("KERA_PARENT_PID", std::to_string((long long) getpid()).c_str(), 1);
#endif

    // the engines: the one suiting chats starts first, so short messages are answered while the other loads
    std::sort(usable.begin(), usable.end(), [](const backend_speed & a, const backend_speed & b) {
        return a.reply_seconds(1000, 100, 300) < b.reply_seconds(1000, 100, 300);
    });
    engine_route_options ro;
    ro.one_at_a_time = !both;
    engine_router router(usable, ro);
    std::vector<std::unique_ptr<engine_process>> engines;
    for (const auto & s : usable) {
        engines.push_back(std::make_unique<engine_process>(s.backend, free_port()));
    }
    if (both) {
        std::fprintf(stderr, "Kera: --backend auto: the model runs on both %s and %s, and each message goes to the one that "
                             "finishes it sooner (%.0f GB free holds two copies)\n",
                     label(usable[0].backend), label(usable[1].backend), free_gb);
    } else {
        std::fprintf(stderr, "Kera: --backend auto: the model runs on %s or %s, one at a time (%.0f GB free doesn't hold two "
                             "copies of %.0f GB): each message goes to the one that finishes it sooner, and the other is "
                             "loaded in its place when that pays\n",
                     label(usable[0].backend), label(usable[1].backend), free_gb, one_gb);
    }

    // each engine's state, guarded by mu with the router
    enum class phase { stopped, loading, ready, failed };
    std::mutex mu;
    std::condition_variable cv;
    std::vector<phase> phases(engines.size(), phase::stopped);
    std::vector<double> since(engines.size(), 0);
    std::vector<bool> timed(engines.size(), false); // its loading time measured
    int load_to = -1;        // one at a time: the engine to load in place of the loaded one
    std::chrono::steady_clock::time_point load_asked{}; // when load_to was set
    bool started = false;    // an engine has been ready
    std::atomic<bool> serving{ true };
    std::atomic<bool> lost{ false }; // every engine stopped by itself

    auto start_engine = [&](size_t i) { // with mu held
        std::vector<std::string> a = base;
        a.insert(a.end(), { "--backend", engines[i]->backend(), "--host", "127.0.0.1", "--port",
                            std::to_string(engines[i]->port()) });
        since[i] = now_s();
        if (!engines[i]->start(a)) {
            std::fprintf(stderr, "Kera: couldn't start the %s engine\n", label(engines[i]->backend()));
            phases[i] = phase::failed;
            router.set_failed((int) i);
            load_to = load_to == (int) i ? -1 : load_to;
            cv.notify_all();
            return false;
        }
        phases[i] = phase::loading;
        return true;
    };
    auto any = [&](phase p) {
        return std::find(phases.begin(), phases.end(), p) != phases.end();
    };
    // watches the engines: ready once /health says so, gone if one stops; loads the next (both: the second once the
    // first is ready, so the two don't read the model from the SSD at once; one at a time: when a message asks)
    auto watch = [&] {
        std::unique_lock<std::mutex> lock(mu);
        start_engine(0);
        while (serving) {
            for (size_t i = 0; i < engines.size(); ++i) {
                if (phases[i] != phase::loading && phases[i] != phase::ready) {
                    continue;
                }
                int code = 0;
                if (engines[i]->exited(&code)) {
                    const bool was_ready = phases[i] == phase::ready;
                    phases[i] = phase::failed;
                    router.set_failed((int) i);
                    std::fprintf(stderr, "Kera: the %s engine %s (exit code %d)\n", label(engines[i]->backend()),
                                 was_ready ? "stopped" : "couldn't load the model", code);
                    if (load_to == (int) i) {
                        load_to = -1;
                    }
                    cv.notify_all();
                    continue;
                }
                if (phases[i] == phase::loading) {
                    lock.unlock();
                    httplib::Client cli("127.0.0.1", engines[i]->port());
                    cli.set_connection_timeout(1, 0);
                    cli.set_read_timeout(2, 0);
                    auto res = cli.Get("/health");
                    lock.lock();
                    if (res && res->status == 200 && phases[i] == phase::loading) {
                        phases[i] = phase::ready;
                        started = true;
                        const double took = now_s() - since[i];
                        router.set_ready((int) i, true);
                        router.set_load_s((int) i, took);
                        timed[i] = true;
                        for (size_t j = 0; j < engines.size(); ++j) {
                            if (!timed[j]) {
                                router.set_load_s((int) j, took); // the same model: about as long, until measured
                            }
                        }
                        if (load_to == (int) i) {
                            load_to = -1;
                        }
                        std::fprintf(stderr, "Kera: the %s engine is ready (loaded in %.1f s)\n", label(engines[i]->backend()), took);
                        cv.notify_all();
                    }
                }
            }
            if (both) {
                // the next engine, once the one before is ready or failed
                for (size_t i = 1; i < engines.size(); ++i) {
                    if (phases[i] == phase::stopped && (phases[i - 1] == phase::ready || phases[i - 1] == phase::failed)) {
                        start_engine(i);
                    }
                }
            } else if (load_to >= 0 && phases[load_to] == phase::stopped) {
                // one at a time: the loaded engine goes once it has finished what it is answering
                int loaded = -1;
                for (size_t i = 0; i < engines.size(); ++i) {
                    loaded = phases[i] == phase::ready ? (int) i : loaded;
                }
                if (loaded < 0 || router.running(loaded) == 0) {
                    if (loaded >= 0) {
                        phases[loaded] = phase::stopped;
                        router.set_ready(loaded, false);
#if defined(_WIN32)
                        // ended at once on Windows: the chats it answered last may not be saved yet
                        (void) store_on;
                        router.forget(loaded);
#else
                        if (!store_on) {
                            router.forget(loaded); // without saved chats, it holds nothing when it comes back
                        }
#endif
                        lock.unlock();
                        engines[loaded]->stop();
                        lock.lock();
                    }
                    start_engine((size_t) load_to);
                }
            }
            if (!both && load_to < 0 && !any(phase::ready) && !any(phase::loading)) {
                // one at a time, and none is loaded (it stopped, or couldn't load or start): another, if one can
                for (size_t i = 0; i < engines.size(); ++i) {
                    if (phases[i] == phase::stopped) {
                        start_engine(i);
                        break;
                    }
                }
            }
            if (!any(phase::ready) && !any(phase::loading)) {
                std::fprintf(stderr, "Kera: no engine is left running\n");
                lost = true;
                lock.unlock();
                stop_router(0);
                return;
            }
            cv.wait_for(lock, std::chrono::milliseconds(250));
        }
    };

    httplib::Server svr;
    g_router = &svr;
    svr.new_task_queue = [] { return new httplib::ThreadPool(16, 256); };
    svr.set_read_timeout(3600, 0);
    svr.set_write_timeout(3600, 0);
    svr.set_payload_max_length(1024ull * 1024 * 1024);
    int last_engine = -1;

    auto handle = [&](const httplib::Request & req, httplib::Response & res) {
        // which engine
        std::vector<route_piece> pieces;
        double max_written = 0;
        if (req.method == "POST" && is_prompt_path(req.path)) {
            pieces = route_pieces(req.body, &max_written);
        }
        int e = -1;
        int64_t id = -1;
        double prompt_bytes = 0;
        {
            std::unique_lock<std::mutex> lock(mu);
            // once an engine has been ready, requests wait while another is loaded in its place
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
            // While one engine is loaded in place of the other, the message that asked for it waits for it. Others
            // don't wait behind it longer than they must: requests that aren't prompts (the model list, health) go
            // to the loaded engine at once, and prompts after 30 s, rather than every user waiting (up to ten
            // minutes, then an error) while the loaded engine finishes a long answer
            auto blocked = [&](bool for_swap) {
                if (!any(phase::ready) && any(phase::loading)) {
                    return true;
                }
                if (load_to < 0) {
                    return false;
                }
                if (for_swap || !any(phase::ready)) {
                    return true;
                }
                return !pieces.empty() && std::chrono::steady_clock::now() - load_asked < std::chrono::seconds(30);
            };
            // a swap waits for the loaded engine to finish what it is answering; with others still sending prompts
            // that may never happen, so after 90 s the message that asked for it is answered where it can be now
            const auto swap_patience = std::chrono::seconds(90);
            auto wait_for_loading = [&](bool for_swap) {
                while (started && serving && !g_stopping && blocked(for_swap) &&
                       std::chrono::steady_clock::now() < deadline && !req.is_connection_closed() &&
                       !(for_swap && load_to >= 0 && phases[load_to] == phase::stopped &&
                         std::chrono::steady_clock::now() - load_asked >= swap_patience)) {
                    cv.wait_for(lock, std::chrono::milliseconds(500));
                }
            };
            if (req.path == "/health" && started && !any(phase::ready)) {
                res.status = 200; // loading the other engine: the server is up
                res.set_content(R"({"status":"ok"})", "application/json");
                return;
            }
            wait_for_loading(false);
            if (!pieces.empty()) {
                for (int tries = 0; tries < 3; ++tries) {
                    const auto c = router.pick(pieces, max_written, now_s());
                    e = c.engine;
                    if (e < 0) {
                        break;
                    }
                    if (c.load) {
                        // one at a time: load it in place of the loaded one, then choose again
                        int o = -1;
                        for (size_t i = 0; i < router.size(); ++i) {
                            o = router.ready((int) i) ? (int) i : o;
                        }
                        if (o < 0) {
                            o = e == 0 ? 1 : 0;
                        }
                        if (c.seconds[o] >= 0 && c.seconds[e] < c.seconds[o]) {
                            std::fprintf(stderr, "Kera: loading %s in place of %s for a message of about %.0f tokens: about "
                                                 "%.1f s with the loading, against %.1f s on %s\n",
                                         label(router.speed(e).backend), label(router.speed(o).backend), c.read, c.seconds[e],
                                         c.seconds[o], label(router.speed(o).backend));
                        } else {
                            std::fprintf(stderr, "Kera: loading %s in place of %s: the messages since the last change "
                                                 "would have been sooner there by more than its loading time\n",
                                         label(router.speed(e).backend), label(router.speed(o).backend));
                        }
                        if (load_to < 0) {
                            load_asked = std::chrono::steady_clock::now();
                        }
                        load_to = e;
                        e = -1;
                        cv.notify_all();
                        wait_for_loading(true);
                        if (load_to >= 0 && phases[load_to] == phase::stopped &&
                            std::chrono::steady_clock::now() - load_asked >= swap_patience) {
                            int r = -1;
                            for (size_t i = 0; i < router.size(); ++i) {
                                r = router.ready((int) i) ? (int) i : r;
                            }
                            if (r >= 0) {
                                std::fprintf(stderr, "Kera: %s is still busy after %d s, so the swap is dropped and the "
                                                     "message goes to it\n", label(router.speed(r).backend),
                                             (int) swap_patience.count());
                                load_to = -1;
                                cv.notify_all();
                                e = r;
                                for (const auto & p : pieces) {
                                    prompt_bytes += p.bytes;
                                }
                                id = router.started(e, pieces, 0, now_s());
                                last_engine = e;
                                break;
                            }
                        }
                        continue;
                    }
                    router.chosen(c);
                    for (const auto & p : pieces) {
                        prompt_bytes += p.bytes;
                    }
                    id = router.started(e, pieces, std::max(0.0, c.seconds[e]), now_s());
                    // say when the choice changes, and for every long prompt
                    const double to_read = c.read - c.held[e];
                    if (e != last_engine || to_read > 4096) {
                        int o = -1; // the next best
                        for (size_t i = 0; i < c.seconds.size(); ++i) {
                            if ((int) i != e && c.seconds[i] > 0 && (o < 0 || c.seconds[i] < c.seconds[o])) {
                                o = (int) i;
                            }
                        }
                        char other[96] = "";
                        if (o >= 0) {
                            std::snprintf(other, sizeof(other), ", against %.1f s on %s", c.seconds[o],
                                          label(router.speed(o).backend));
                        }
                        std::fprintf(stderr, "Kera: a message of about %.0f tokens (%.0f already held there) goes to %s: "
                                             "about %.1f s expected%s\n", c.read, c.held[e], label(router.speed(e).backend),
                                     c.seconds[e], other);
                    }
                    last_engine = e;
                    break;
                }
            } else {
                for (size_t i = 0; i < router.size() && e < 0; ++i) {
                    e = router.ready((int) i) ? (int) i : -1;
                }
                if (e >= 0) {
                    // counted as running too (embeddings, tokenizing...): an engine isn't swapped out under it
                    id = router.started(e, {}, 0, now_s());
                }
            }
        }
        if (e < 0) {
            res.status = 503;
            res.set_content(R"({"error":{"code":503,"message":"Loading model","type":"unavailable_error"}})",
                            "application/json");
            return;
        }

        // the engine's answer, passed on as it comes
        auto r = std::make_shared<relay>();
        httplib::Request q;
        q.method = req.method;
        q.path = req.target.empty() ? req.path : req.target;
        for (const auto & [k, v] : req.headers) {
            if (!hop_by_hop(k)) {
                q.headers.emplace(k, v);
            }
        }
        if (!req.body.empty()) {
            q.body = req.body;
            q.headers.emplace("Content-Type", req.get_header_value("Content-Type", "application/json"));
        }
        const int port_e = engines[e]->port();
        std::thread([r, q, port_e]() mutable {
            httplib::Client cli("127.0.0.1", port_e);
            cli.set_connection_timeout(10, 0);
            cli.set_read_timeout(3600, 0);
            cli.set_write_timeout(3600, 0);
            q.response_handler = [r](const httplib::Response & resp) {
                std::lock_guard<std::mutex> lock(r->mu);
                r->status = resp.status;
                r->headers = resp.headers;
                r->have_headers = true;
                r->cv.notify_all();
                return !r->cancel;
            };
            q.content_receiver = [r](const char * d, size_t n, size_t, size_t) {
                std::lock_guard<std::mutex> lock(r->mu);
                if (r->cancel) {
                    return false;
                }
                r->chunks.emplace_back(d, n);
                r->tail.append(d, n);
                if (r->tail.size() > 16384) {
                    r->tail.erase(0, r->tail.size() - 8192);
                }
                r->cv.notify_all();
                return true;
            };
            {
                std::lock_guard<std::mutex> lock(r->mu);
                if (r->cancel) {
                    r->done = true;
                    r->cv.notify_all();
                    return;
                }
                r->cli = &cli;
            }
            cli.send(q);
            std::lock_guard<std::mutex> lock(r->mu);
            r->cli = nullptr;
            r->done = true;
            r->cv.notify_all();
        }).detach();

        // the status and headers first: they can take as long as reading the prompt, for a non-streamed request
        {
            std::unique_lock<std::mutex> lock(r->mu);
            while (!r->have_headers && !r->done) {
                if (r->cv.wait_for(lock, std::chrono::milliseconds(500)) == std::cv_status::timeout &&
                    (req.is_connection_closed() || g_stopping)) {
                    r->stop_locked();
                }
            }
            if (!r->have_headers) {
                lock.unlock();
                if (id >= 0) {
                    std::lock_guard<std::mutex> l(mu);
                    router.finished(id, 0, 0, 0);
                }
                res.status = 502;
                res.set_content(R"({"error":{"code":502,"message":"the engine didn't answer","type":"server_error"}})",
                                "application/json");
                return;
            }
            res.status = r->status;
            for (const auto & [k, v] : r->headers) {
                if (!hop_by_hop(k)) {
                    res.set_header(k, v);
                }
            }
        }
        // which engine answered, for benchmarks and tests
        res.set_header("X-Kera-Backend", engines[e]->backend());
        res.set_header("X-Kera-Engine", std::to_string(e));
        const std::string type = [&] {
            for (const auto & [k, v] : r->headers) {
                std::string l = k;
                std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return (char) std::tolower(c); });
                if (l == "content-type") {
                    return v;
                }
            }
            return std::string("application/json");
        }();
        res.set_chunked_content_provider(type, [r](size_t, httplib::DataSink & sink) {
            std::unique_lock<std::mutex> lock(r->mu);
            if (!sink.is_writable() || g_stopping) {
                r->stop_locked(); // the chat app went away: so does the engine's request
                return false;
            }
            r->cv.wait_for(lock, std::chrono::milliseconds(500), [&] { return !r->chunks.empty() || r->done; });
            while (!r->chunks.empty()) {
                std::string c = std::move(r->chunks.front());
                r->chunks.pop_front();
                lock.unlock();
                if (!sink.write(c.data(), c.size())) {
                    lock.lock();
                    r->stop_locked();
                    return false;
                }
                lock.lock();
            }
            if (r->done) {
                sink.done();
            }
            return true;
        }, [r, id, prompt_bytes, &mu, &router](bool) {
            std::lock_guard<std::mutex> lock(r->mu);
            if (!r->done) {
                r->stop_locked();
            }
            if (id >= 0) {
                // the reply's token counts teach the router the token size and a reply's length: llama.cpp's timings
                // (every API), else OpenAI's or Anthropic's usage
                double prompt = -1;
                double written = last_number(r->tail, "\"predicted_n\"");
                if (const double n = last_number(r->tail, "\"prompt_n\""); n >= 0) {
                    prompt = n + std::max(0.0, last_number(r->tail, "\"cache_n\""));
                } else if (const double t = last_number(r->tail, "\"prompt_tokens\""); t >= 0) {
                    prompt = t;
                    written = last_number(r->tail, "\"completion_tokens\"");
                } else if (const double i = last_number(r->tail, "\"input_tokens\""); i >= 0) {
                    prompt = i + std::max(0.0, last_number(r->tail, "\"cache_read_input_tokens\"")) +
                             std::max(0.0, last_number(r->tail, "\"cache_creation_input_tokens\""));
                    written = last_number(r->tail, "\"output_tokens\"");
                }
                std::lock_guard<std::mutex> l(mu);
                router.finished(id, prompt_bytes, std::max(0.0, prompt), std::max(0.0, written));
            }
        });
    };
    svr.Get(".*", handle);
    svr.Post(".*", handle);
    svr.Put(".*", handle);
    svr.Delete(".*", handle);
    svr.Options(".*", handle);

#if defined(_WIN32)
    SetConsoleCtrlHandler(stop_router_console, TRUE);
#else
    std::signal(SIGINT, stop_router);
    std::signal(SIGTERM, stop_router);
#endif
    // one server on a port: httplib's defaults (SO_REUSEPORT on Linux and macOS, SO_REUSEADDR on Windows) let a second
    // one listen on the same port, and requests would go to either at random
    svr.set_socket_options([](socket_t sock) {
#if defined(_WIN32)
        BOOL yes = TRUE;
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &yes, sizeof(yes));
#else
        int yes = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    });
    int rc = 0;
    std::thread watcher;
    if (!svr.bind_to_port(host, port)) {
        std::fprintf(stderr, "Kera: couldn't listen on %s:%d\n", host.c_str(), port);
        rc = 1;
    } else {
        std::fprintf(stderr, "Kera: listening on http://%s:%d\n", host.c_str(), port);
        watcher = std::thread(watch);
        svr.listen_after_bind();
    }
    if (rc == 0 && lost) {
        rc = 1; // the engines stopped by themselves
    }
    g_router = nullptr;
    serving = false;
    if (watcher.joinable()) {
        watcher.join();
    }
    std::vector<std::thread> stopping; // at once, not one after the other (each can take a few seconds)
    for (auto & e : engines) {
        stopping.emplace_back([&e] { e->stop(); });
    }
    for (auto & t : stopping) {
        t.join();
    }
    return rc;
}

} // namespace kera
