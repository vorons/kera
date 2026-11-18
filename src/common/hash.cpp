// Fast whole-file hashes for model identity (kr_hash.h).

#include "common/kr_hash.h"

#include "common/kr_paths.h"
#include "io/kr_io.h"
#include "io/kr_os_file.h"

#include "hash/xxhash/xxhash.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace kera {

namespace {

std::string to_hex(XXH128_hash_t h) {
    XXH128_canonical_t c;
    XXH128_canonicalFromHash(&c, h);
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    for (unsigned char b : c.digest) {
        out += digits[b >> 4];
        out += digits[b & 15];
    }
    return out;
}

constexpr size_t CHUNK = 16u << 20; // 16 MiB: a multiple of every direct-read alignment

} // namespace

std::string hash_bytes(const void * data, size_t len) {
    return to_hex(XXH3_128bits(data, len));
}

std::string hash_file(const std::string & path, int * err, const std::function<void(double)> & progress) {
    int e = 0;
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(std::filesystem::u8path(path), ec);
    if (ec) {
        if (err) {
            *err = ec.value();
        }
        return "";
    }
    os::file f;
    if ((e = os::open_read(path, true, f)) != 0) {
        if (err) {
            *err = e;
        }
        return "";
    }
    void * bufs[2] = { os::aligned_alloc(CHUNK, KR_IO_ALIGN), os::aligned_alloc(CHUNK, KR_IO_ALIGN) };
    XXH3_state_t * st = XXH3_createState();
    XXH3_128bits_reset(st);

    // read the next chunk while hashing this one
    auto read_at = [&](uint64_t off, void * dst) {
        size_t got = 0;
        const int r = os::pread_full(f, off, CHUNK, dst, &got);
        return std::make_pair(r, got);
    };
    uint64_t off = 0;
    auto pending = std::async(std::launch::async, read_at, off, bufs[0]);
    int cur = 0;
    while (true) {
        const auto [r, got] = pending.get();
        if (r != 0) {
            e = r;
            break;
        }
        const uint64_t next = off + got;
        const bool more = got == CHUNK && next < size;
        if (more) {
            pending = std::async(std::launch::async, read_at, next, bufs[cur ^ 1]);
        }
        XXH3_128bits_update(st, bufs[cur], got);
        off = next;
        if (progress && size > 0) {
            progress((double) off / (double) size);
        }
        if (!more) {
            break;
        }
        cur ^= 1;
    }
    const std::string out = e == 0 && off == size ? to_hex(XXH3_128bits_digest(st)) : "";
    if (e == 0 && off != size) {
        e = EIO;
    }
    XXH3_freeState(st);
    os::aligned_free(bufs[0]);
    os::aligned_free(bufs[1]);
    os::close(f);
    if (err) {
        *err = e;
    }
    return out;
}

namespace {

// When the file's own record last changed (POSIX ctime, Windows ChangeTime). Unlike the modification time, no copy
// tool or `touch` can set it, so a model replaced in place by one of the same size and date still reads as changed,
// and its saved chats aren't continued from the other weights' memory (N-018). "" where the system has none.
std::string change_stamp(const std::filesystem::path & p) {
#ifdef _WIN32
    HANDLE h = CreateFileW(p.wstring().c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE |
                           FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return "";
    }
    FILE_BASIC_INFO bi{};
    const bool ok = GetFileInformationByHandleEx(h, FileBasicInfo, &bi, sizeof(bi)) != 0;
    CloseHandle(h);
    return ok && bi.ChangeTime.QuadPart != 0 ? std::to_string(bi.ChangeTime.QuadPart) : "";
#else
    struct stat st {};
    if (stat(p.c_str(), &st) != 0) {
        return "";
    }
#ifdef __APPLE__
    return std::to_string(st.st_ctimespec.tv_sec) + "." + std::to_string(st.st_ctimespec.tv_nsec);
#else
    return std::to_string(st.st_ctim.tv_sec) + "." + std::to_string(st.st_ctim.tv_nsec);
#endif
#endif
}

} // namespace

std::string model_hash(const std::string & path, int * err, bool known_only) {
    namespace fs = std::filesystem;
    static std::mutex mu;
    std::error_code ec;
    const fs::path p = fs::absolute(fs::u8path(path), ec);
    const uint64_t size = fs::file_size(p, ec);
    if (ec) {
        if (err) {
            *err = ec.value();
        }
        return "";
    }
    const auto mtime = (long long) fs::last_write_time(p, ec).time_since_epoch().count();
    const std::string stamp = change_stamp(p);
    const std::string key = p.u8string();
    const std::string cache_path = home_path("model_hashes.json");

    std::lock_guard<std::mutex> lock(mu);
    nlohmann::json cache = nlohmann::json::object();
    {
        std::ifstream in(fs::u8path(cache_path), std::ios::binary);
        if (in) {
            try {
                cache = nlohmann::json::parse(in);
            } catch (const std::exception &) {
                cache = nlohmann::json::object();
            }
        }
    }
    if (cache.contains(key)) {
        const auto & c = cache[key];
        if (c.value("size", (uint64_t) 0) == size && c.value("mtime", 0LL) == mtime &&
            c.value("changed", std::string()) == stamp && c.contains("hash")) {
            if (err) {
                *err = 0;
            }
            return c["hash"].get<std::string>();
        }
    }
    if (known_only) {
        return ""; // not hashed before (or the file changed since)
    }
    const std::string h = hash_file(key, err);
    if (h.empty()) {
        return "";
    }
    const std::string out = "xxh128:" + h;
    cache[key] = { { "size", size }, { "mtime", mtime }, { "changed", stamp }, { "hash", out } };
    const std::string tmp = cache_path + ".tmp";
    {
        std::ofstream o(fs::u8path(tmp), std::ios::binary);
        o << cache.dump(1) << "\n";
    }
    fs::rename(fs::u8path(tmp), fs::u8path(cache_path), ec);
    return out;
}

} // namespace kera
