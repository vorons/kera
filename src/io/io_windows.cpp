// Windows I/O back-end: unbuffered overlapped ReadFile (FILE_FLAG_NO_BUFFERING) from a thread pool.
// Files are opened as overlapped handles: Windows serialises concurrent reads on a synchronous handle,
// which would hide the drive's parallel speed. IoRing is a later experiment (spec Phase 8, verify).

#if defined(_WIN32)

#include "io/kr_os_file.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <malloc.h>

namespace kera::os {

static std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

static int open_handle(const std::string & path, DWORD access, DWORD disposition, DWORD flags, file & f) {
    HANDLE h = CreateFileW(widen(path).c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, disposition, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | flags, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return (int) GetLastError();
    }
    f.h = h;
    f.valid = true;
    return 0;
}

int open_read(const std::string & path, bool direct, file & f) {
    f.direct = direct;
    return open_handle(path, GENERIC_READ, OPEN_EXISTING, direct ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_SEQUENTIAL_SCAN, f);
}

int create_write(const std::string & path, bool direct, file & f) {
    f.direct = direct;
    return open_handle(path, GENERIC_WRITE, CREATE_ALWAYS, direct ? (FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH) : 0, f);
}

// One overlapped request at a time from this thread; several threads may run this on the same handle.
static int overlapped_io(const file & f, bool write, uint64_t offset, size_t len, void * buf, size_t * done) {
    size_t moved = 0;
    HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (ev == nullptr) {
        return (int) GetLastError();
    }
    int err = 0;
    while (moved < len) {
        const DWORD chunk = (DWORD) ((len - moved) > (1u << 30) ? (1u << 30) : (len - moved));
        OVERLAPPED ov = {};
        ov.Offset = (DWORD) ((offset + moved) & 0xffffffffu);
        ov.OffsetHigh = (DWORD) ((offset + moved) >> 32);
        ov.hEvent = ev;
        DWORD n = 0;
        BOOL ok = write ? WriteFile(f.h, (const char *) buf + moved, chunk, nullptr, &ov)
                        : ReadFile(f.h, (char *) buf + moved, chunk, nullptr, &ov);
        if (!ok) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                if (e != ERROR_HANDLE_EOF) {
                    err = (int) e;
                }
                break;
            }
        }
        if (!GetOverlappedResult(f.h, &ov, &n, TRUE)) {
            const DWORD e = GetLastError();
            if (e != ERROR_HANDLE_EOF) {
                err = (int) e;
            }
            break;
        }
        if (n == 0) {
            break; // end of file
        }
        moved += n;
        if (!write && n < chunk) {
            break; // a short read only happens at the end of the file (and reading on from an unaligned
                   // offset would fail with FILE_FLAG_NO_BUFFERING)
        }
    }
    CloseHandle(ev);
    if (done) *done = moved;
    return err;
}

int pread_full(const file & f, uint64_t offset, size_t len, void * dst, size_t * done) {
    return overlapped_io(f, false, offset, len, dst, done);
}

int pwrite_full(const file & f, uint64_t offset, size_t len, const void * src, size_t * done) {
    return overlapped_io(f, true, offset, len, const_cast<void *>(src), done);
}

int sync(const file & f) {
    return FlushFileBuffers(f.h) ? 0 : (int) GetLastError();
}

void close(file & f) {
    if (f.valid) {
        CloseHandle(f.h);
        f.valid = false;
    }
}

int remove_file(const std::string & path) {
    if (DeleteFileW(widen(path).c_str())) {
        return 0;
    }
    const DWORD e = GetLastError();
    return e == ERROR_FILE_NOT_FOUND ? 0 : (int) e;
}

int make_dir(const std::string & path) {
    if (CreateDirectoryW(widen(path).c_str(), nullptr)) {
        return 0;
    }
    const DWORD e = GetLastError();
    return e == ERROR_ALREADY_EXISTS ? 0 : (int) e;
}

void * aligned_alloc(size_t bytes, size_t alignment) { return _aligned_malloc(bytes, alignment); }
void aligned_free(void * p) { _aligned_free(p); }

const char * io_method_name() { return "overlapped ReadFile thread pool + FILE_FLAG_NO_BUFFERING"; }

} // namespace kera::os

#endif
