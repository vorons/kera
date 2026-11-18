#include "common/kr_paths.h"

#include "io/kr_os_file.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <climits>
#else
#include <climits>
#include <unistd.h>
#endif

namespace kera {

char path_sep() {
#if defined(_WIN32)
    return '\\';
#else
    return '/';
#endif
}

std::string join(const std::string & a, const std::string & b) {
    if (a.empty()) {
        return b;
    }
    const char last = a.back();
    if (last == '/' || last == '\\') {
        return a + b;
    }
    return a + path_sep() + b;
}

static std::string env(const char * name) {
#if defined(_WIN32)
    // getenv returns the ANSI code page; read the wide value and convert to UTF-8
    std::wstring wname(name, name + strlen(name));
    const DWORD n = GetEnvironmentVariableW(wname.c_str(), nullptr, 0);
    if (n == 0) {
        return std::string();
    }
    std::wstring w(n, L'\0');
    GetEnvironmentVariableW(wname.c_str(), w.data(), n);
    w.resize(n - 1);
    const int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), len, nullptr, nullptr);
    return s;
#else
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
#endif
}

// mkdir -p
static void make_dirs(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        const bool sep = path[i] == '/' || path[i] == '\\';
        if (sep && cur.size() > 1 && !(cur.size() == 3 && cur[1] == ':')) {
            os::make_dir(cur.substr(0, cur.size() - 1));
        }
    }
    os::make_dir(path);
}

std::string home_dir() {
    std::string dir = env("KERA_HOME");
    if (dir.empty()) {
#if defined(_WIN32)
        dir = join(env("LOCALAPPDATA"), "Kera");
#elif defined(__APPLE__)
        dir = join(env("HOME"), "Library/Application Support/Kera");
#else
        const std::string xdg = env("XDG_DATA_HOME");
        dir = xdg.empty() ? join(env("HOME"), ".local/share/kera") : join(xdg, "kera");
#endif
    }
    make_dirs(dir);
    return dir;
}

std::string home_path(const std::string & name) {
    return join(home_dir(), name);
}

std::string exe_dir() {
    std::string path;
#if defined(_WIN32)
    std::vector<wchar_t> buf(32768);
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD) buf.size());
    const int len = WideCharToMultiByte(CP_UTF8, 0, buf.data(), (int) n, nullptr, 0, nullptr, nullptr);
    path.assign(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf.data(), (int) n, path.data(), len, nullptr, nullptr);
#elif defined(__APPLE__)
    uint32_t size = PATH_MAX;
    std::vector<char> buf(size);
    if (_NSGetExecutablePath(buf.data(), &size) != 0) {
        buf.resize(size);
        _NSGetExecutablePath(buf.data(), &size);
    }
    char real[PATH_MAX];
    path = realpath(buf.data(), real) ? real : buf.data();
#else
    char buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    path = n > 0 ? std::string(buf, (size_t) n) : std::string(".");
#endif
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}

} // namespace kera
