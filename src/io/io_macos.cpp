// macOS I/O back-end: pread from a thread pool, with F_NOCACHE so reads bypass the unified buffer cache.
// The speed-test file is written with F_NOCACHE too, or the read-back could come from memory (spec 4a).

#if defined(__APPLE__)

#include "io/io_posix_common.h"

namespace kera::os {

static void set_nocache(int fd, bool direct) {
    if (direct) {
        ::fcntl(fd, F_NOCACHE, 1);
        ::fcntl(fd, F_RDAHEAD, 0);
    }
}

int open_read(const std::string & path, bool direct, file & f) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return errno;
    }
    set_nocache(fd, direct);
    f.h = fd;
    f.direct = direct;
    f.valid = true;
    return 0;
}

int create_write(const std::string & path, bool direct, file & f) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        return errno;
    }
    set_nocache(fd, direct);
    f.h = fd;
    f.direct = direct;
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
    // F_FULLFSYNC asks the drive itself to flush; fsync alone may leave data in its cache
    return ::fcntl(f.h, F_FULLFSYNC) == 0 || ::fsync(f.h) == 0 ? 0 : errno;
}

void close(file & f) { posix::close(f); }
int remove_file(const std::string & path) { return posix::remove_file(path); }
int make_dir(const std::string & path) { return posix::make_dir(path); }
void * aligned_alloc(size_t bytes, size_t alignment) { return posix::aligned_alloc(bytes, alignment); }
void aligned_free(void * p) { ::free(p); }

const char * io_method_name() { return "pread thread pool + F_NOCACHE"; }

} // namespace kera::os

#endif
