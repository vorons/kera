// Keeping a model loaded for chat apps (model_runner.h).

#include "app/model_runner.h"

#include "common/kr_paths.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <arpa/inet.h>
#include <csignal>
#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace fs = std::filesystem;

namespace kera {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Bind and close at once (httplib's own bind also listens, and would keep the port)
bool port_free(int port) {
    sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t) port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return false;
    }
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    bool ok = false;
    if (s != INVALID_SOCKET) {
        BOOL excl = TRUE;
        setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &excl, sizeof(excl));
        ok = bind(s, (const sockaddr *) &a, sizeof(a)) == 0;
        closesocket(s);
    }
    WSACleanup();
    return ok;
#else
    const int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
        return false;
    }
    const int one = 1; // as the server does: a just-closed port is free again at once
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    const bool ok = bind(s, (const sockaddr *) &a, sizeof(a)) == 0;
    close(s);
    return ok;
#endif
}

// llama.cpp's server's usual port first, as chat apps often expect it
int pick_port() {
    for (int p = 8080; p < 8100; ++p) {
        if (port_free(p)) {
            return p;
        }
    }
    return 0;
}

std::string kera_exe() {
#if defined(_WIN32)
    return join(exe_dir(), "kera.exe");
#else
    return join(exe_dir(), "kera");
#endif
}

std::string log_path(const std::string & name) {
    return home_path(name);
}

// The last lines of the server's log, to show why it stopped
std::string log_tail(const std::string & name, size_t n) {
    std::ifstream f(fs::u8path(log_path(name)));
    std::deque<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) {
            lines.push_back(line);
            if (lines.size() > n) {
                lines.pop_front();
            }
        }
    }
    std::string out;
    for (const auto & l : lines) {
        out += l + "\n";
    }
    return out;
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
#endif

} // namespace

const char * state_name(model_runner::state s) {
    switch (s) {
        case model_runner::state::loading: return "loading";
        case model_runner::state::ready:   return "ready";
        case model_runner::state::failed:  return "failed";
        default:                           return "stopped";
    }
}

model_runner::~model_runner() {
    stop();
}

std::string model_runner::last_model_file() {
    return home_path("app.json");
}

bool model_runner::start(const std::string & path, std::string * error) {
    std::lock_guard<std::mutex> ops(ops_);
    stop_locked();
    // a start that fails says why on the page, as a model that stopped does
    auto failed = [&](const std::string & why) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = state::failed;
        model_ = path;
        error_ = why;
        if (error) *error = why;
        return false;
    };
    const int port = pick_port();
    if (port == 0) {
        return failed("ports 8080 to 8099 are all in use");
    }
    if (remember_) {
        nlohmann::json j = { { "last_model", path } };
        std::ofstream(fs::u8path(last_model_file())) << j.dump(1) << "\n";
    }
    const std::string port_s = std::to_string(port);
    unsigned long long pid = 0;
    void * handle = nullptr;
#if defined(_WIN32)
    // --cors-origins localhost: the model answers web pages on this computer only (an app's page on localhost:3000),
    // not any website open in the browser (llama.cpp's default lets every page read the answers)
    std::wstring cmd = L"\"" + widen(kera_exe()) + L"\" serve -m \"" + widen(path) + L"\" --port " + widen(port_s) +
                       L" --cors-origins localhost";
    SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
    HANDLE log = CreateFileW(widen(log_path(log_name_)).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = log;
    si.hStdError = log;
    si.hStdInput = nullptr;
    PROCESS_INFORMATION pi = {};
    // in a job that ends with the app, however the app ends: closing its window ends it before any clean-up runs,
    // which would leave the model loaded in a server nobody can see
    static HANDLE job = [] {
        HANDLE j = CreateJobObjectW(nullptr, nullptr);
        if (j != nullptr) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof(li));
        }
        return j;
    }();
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED,
                                   nullptr, nullptr, &si, &pi);
    if (log != INVALID_HANDLE_VALUE) {
        CloseHandle(log);
    }
    if (!ok) {
        return failed("couldn't start kera serve (error " + std::to_string(GetLastError()) + ")");
    }
    if (job != nullptr) {
        AssignProcessToJobObject(job, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    pid = pi.dwProcessId;
    handle = pi.hProcess;
#else
    const std::string exe = kera_exe();
    const std::string log = log_path(log_name_);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    std::vector<char *> argv = { const_cast<char *>(exe.c_str()), const_cast<char *>("serve"), const_cast<char *>("-m"),
                                 const_cast<char *>(path.c_str()), const_cast<char *>("--port"),
                                 const_cast<char *>(port_s.c_str()), const_cast<char *>("--cors-origins"),
                                 const_cast<char *>("localhost"), nullptr };
    // in a process group of its own: Ctrl+C in the app's terminal would also reach it, and with the app's own stop
    // that is two SIGINTs, on which llama.cpp's server quits at once, losing chats still being saved
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    pid_t child = 0;
    const int rc = posix_spawn(&child, exe.c_str(), &fa, &attr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&attr);
    if (rc != 0) {
        return failed("couldn't start kera serve (error " + std::to_string(rc) + ")");
    }
    pid = (unsigned long long) child;
#endif
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = state::loading;
    model_ = path;
    port_ = port;
    error_.clear();
    started_ = now_s();
    ready_after_ = 0;
    pid_ = pid;
    handle_ = handle;
    stopping_ = false;
    watcher_ = std::thread([this, pid, port] { watch(pid, port); });
    return true;
}

void model_runner::watch(unsigned long long pid, int port) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        bool exited = false;
        int code = 0;
#if defined(_WIN32)
        (void) pid;
        if (WaitForSingleObject((HANDLE) handle_, 0) == WAIT_OBJECT_0) {
            DWORD c = 0;
            GetExitCodeProcess((HANDLE) handle_, &c);
            exited = true;
            code = (int) c;
        }
#else
        int status = 0;
        if (waitpid((pid_t) pid, &status, WNOHANG) == (pid_t) pid) {
            exited = true;
            code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        }
#endif
        if (exited) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!stopping_) {
                state_ = state::failed;
                error_ = "The model stopped (exit code " + std::to_string(code) + "). The end of its log:\n" + log_tail(log_name_, 8);
            } else {
                state_ = state::stopped;
            }
            pid_ = 0;
            return;
        }
        bool loading = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            loading = state_ == state::loading && !stopping_;
        }
        if (loading) {
            // llama.cpp's server answers /health; the image engine's has none, and answers /v1/models once loaded
            auto res = cli.Get("/health");
            if (res && res->status == 404) {
                res = cli.Get("/v1/models");
            }
            if (res && res->status == 200) {
                std::lock_guard<std::mutex> lock(mutex_);
                state_ = state::ready;
                ready_after_ = now_s() - started_;
            }
        }
    }
}

void model_runner::stop() {
    std::lock_guard<std::mutex> ops(ops_);
    stop_locked();
}

void model_runner::stop_locked() {
    unsigned long long pid = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pid = pid_;
        stopping_ = true;
    }
    if (pid != 0) {
#if defined(_WIN32)
        // a background process has no console to send Ctrl+C to; saved chats are written as they happen,
        // in files that are only renamed into place once complete, so nothing is left half-written
        TerminateProcess((HANDLE) handle_, 0);
#else
        kill((pid_t) pid, SIGINT); // lets the server finish writing saved chats
        for (int i = 0; i < 60; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            std::lock_guard<std::mutex> lock(mutex_);
            if (pid_ == 0) {
                break;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (pid_ != 0) {
            kill((pid_t) pid, SIGKILL);
        }
#endif
    }
    if (watcher_.joinable()) {
        watcher_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(_WIN32)
    if (handle_) {
        CloseHandle((HANDLE) handle_);
        handle_ = nullptr;
    }
#endif
    pid_ = 0;
    // stopped by request: a failure shown before is dismissed with it
    state_ = state::stopped;
    error_.clear();
}

void model_runner::start_last() {
    std::ifstream f(fs::u8path(last_model_file()));
    if (!f) {
        return;
    }
    try {
        const std::string path = nlohmann::json::parse(f).value("last_model", "");
        std::error_code ec;
        if (!path.empty() && fs::is_regular_file(fs::u8path(path), ec)) {
            start(path);
        }
    } catch (const std::exception &) {
    }
}

model_runner::status model_runner::get() {
    std::lock_guard<std::mutex> lock(mutex_);
    status s;
    s.st = state_;
    s.model = model_;
    s.port = port_;
    s.error = error_;
    s.seconds = state_ == state::loading ? now_s() - started_ : state_ == state::ready ? ready_after_ : 0;
    return s;
}

} // namespace kera
