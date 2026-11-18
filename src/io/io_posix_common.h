// Internal: the parts io_linux.cpp and io_macos.cpp share.
#pragma once

#include "io/kr_os_file.h"

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kera::os::posix {

inline int pread_full(const file & f, uint64_t offset, size_t len, void * dst, size_t * done) {
    size_t got = 0;
    while (got < len) {
        const ssize_t r = ::pread(f.h, (char *) dst + got, len - got, (off_t) (offset + got));
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (done) *done = got;
            return errno;
        }
        if (r == 0) {
            break; // end of file
        }
        got += (size_t) r;
        if (f.direct && (size_t) r < len - (got - (size_t) r)) {
            break; // a short direct read means the end of the file; reading on from an unaligned offset
                   // would fail with O_DIRECT
        }
    }
    if (done) *done = got;
    return 0;
}

inline int pwrite_full(const file & f, uint64_t offset, size_t len, const void * src, size_t * done) {
    size_t put = 0;
    while (put < len) {
        const ssize_t r = ::pwrite(f.h, (const char *) src + put, len - put, (off_t) (offset + put));
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (done) *done = put;
            return errno;
        }
        put += (size_t) r;
    }
    if (done) *done = put;
    return 0;
}

inline void close(file & f) {
    if (f.valid) {
        ::close(f.h);
        f.valid = false;
    }
}

inline int remove_file(const std::string & path) {
    return ::unlink(path.c_str()) == 0 || errno == ENOENT ? 0 : errno;
}

inline int make_dir(const std::string & path) {
    if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
        return 0;
    }
    return errno;
}

inline void * aligned_alloc(size_t bytes, size_t alignment) {
    void * p = nullptr;
    return ::posix_memalign(&p, alignment, bytes) == 0 ? p : nullptr;
}

} // namespace kera::os::posix
