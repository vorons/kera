// Linux I/O back-end: pread from a thread pool, with O_DIRECT so reads bypass the page cache.

#if defined(__linux__)

#include "io/io_posix_common.h"

namespace kera::os {

static int open_flags(int base, bool direct) {
    return base | O_CLOEXEC | (direct ? O_DIRECT : 0);
}

int open_read(const std::string & path, bool direct, file & f) {
    int fd = ::open(path.c_str(), open_flags(O_RDONLY, direct));
    f.direct = direct;
    if (fd < 0 && direct && errno == EINVAL) {
        // the file system doesn't support O_DIRECT (tmpfs, some network file systems)
        fd = ::open(path.c_str(), open_flags(O_RDONLY, false));
        f.direct = false;
    }
    if (fd < 0) {
        return errno;
    }
    f.h = fd;
    f.valid = true;
    return 0;
}

int create_write(const std::string & path, bool direct, file & f) {
    int fd = ::open(path.c_str(), open_flags(O_WRONLY | O_CREAT | O_TRUNC, direct), 0644);
    f.direct = direct;
    if (fd < 0 && direct && errno == EINVAL) {
        fd = ::open(path.c_str(), open_flags(O_WRONLY | O_CREAT | O_TRUNC, false), 0644);
        f.direct = false;
    }
    if (fd < 0) {
        return errno;
    }
    f.h = fd;
    f.valid = true;
    return 0;
}

int pread_full(const file & f, uint64_t offset, size_t len, void * dst, size_t * done) {
    return posix::pread_full(f, offset, len, dst, done);
}

int pwrite_full(const file & f, uint64_t offset, size_t len, const void * src, size_t * done) {
    return posix::pwrite_full(f, offset, len, src, done);
}

int sync(const file & f) {
    return ::fdatasync(f.h) == 0 ? 0 : errno;
}

void close(file & f) { posix::close(f); }
int remove_file(const std::string & path) { return posix::remove_file(path); }
int make_dir(const std::string & path) { return posix::make_dir(path); }
void * aligned_alloc(size_t bytes, size_t alignment) { return posix::aligned_alloc(bytes, alignment); }
void aligned_free(void * p) { ::free(p); }

const char * io_method_name() { return "pread thread pool + O_DIRECT"; }

} // namespace kera::os

#endif
