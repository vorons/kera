// kr_io: a pool of reader threads, each doing blocking positional reads, so n threads = n reads in
// flight. The per-OS work is in io_linux.cpp, io_macos.cpp and io_windows.cpp.

#include "io/kr_io.h"
#include "io/kr_os_file.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct request {
    int file_id;
    uint64_t offset;
    size_t len;
    void * dst;
    kr_io_done cb;
    void * user;
};

} // namespace

struct kr_io {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv_work;
    std::condition_variable cv_idle;
    std::deque<request> queue;
    int busy = 0;
    bool stop = false;
    std::vector<kera::os::file> files;
    std::atomic<uint64_t> bytes_read{0};

    void worker() {
        for (;;) {
            request r;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_work.wait(lock, [&] { return stop || !queue.empty(); });
                if (queue.empty()) {
                    return; // stop and nothing left to do
                }
                r = queue.front();
                queue.pop_front();
                busy++;
            }
            size_t got = 0;
            kera::os::file f;
            {
                std::lock_guard<std::mutex> lock(mutex);
                f = files[(size_t) r.file_id];
            }
            const int status = kera::os::pread_full(f, r.offset, r.len, r.dst, &got);
            bytes_read += got;
            if (r.cb) {
                r.cb(r.user, status, got);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                busy--;
                if (busy == 0 && queue.empty()) {
                    cv_idle.notify_all();
                }
            }
        }
    }
};

extern "C" kr_io * kr_io_create(int n_threads) {
    if (n_threads < 1) {
        n_threads = 1;
    }
    kr_io * io = new kr_io();
    for (int i = 0; i < n_threads; ++i) {
        io->threads.emplace_back([io] { io->worker(); });
    }
    return io;
}

extern "C" void kr_io_destroy(kr_io * io) {
    if (io == nullptr) {
        return;
    }
    kr_io_wait(io);
    {
        std::lock_guard<std::mutex> lock(io->mutex);
        io->stop = true;
    }
    io->cv_work.notify_all();
    for (auto & t : io->threads) {
        t.join();
    }
    for (auto & f : io->files) {
        kera::os::close(f);
    }
    delete io;
}

extern "C" int kr_io_open(kr_io * io, const char * path, int flags) {
    kera::os::file f;
    const int err = kera::os::open_read(path, (flags & KR_IO_DIRECT) != 0, f);
    if (err != 0) {
        return -(err > 0 ? err : 1);
    }
    std::lock_guard<std::mutex> lock(io->mutex);
    io->files.push_back(f);
    return (int) io->files.size() - 1;
}

extern "C" int kr_io_read(kr_io * io, int file_id, uint64_t offset, size_t len, void * dst, kr_io_done cb, void * user) {
    {
        std::lock_guard<std::mutex> lock(io->mutex);
        if (io->stop || file_id < 0 || (size_t) file_id >= io->files.size()) {
            return -1;
        }
        io->queue.push_back({ file_id, offset, len, dst, cb, user });
    }
    io->cv_work.notify_one();
    return 0;
}

extern "C" void kr_io_wait(kr_io * io) {
    std::unique_lock<std::mutex> lock(io->mutex);
    io->cv_idle.wait(lock, [&] { return io->busy == 0 && io->queue.empty(); });
}

extern "C" uint64_t kr_io_bytes_read(const kr_io * io) {
    return io->bytes_read.load();
}

extern "C" void * kr_io_alloc(size_t bytes) {
    return kera::os::aligned_alloc(bytes, KR_IO_ALIGN);
}

extern "C" void kr_io_free(void * p) {
    kera::os::aligned_free(p);
}

extern "C" const char * kr_io_backend_name(void) {
    return kera::os::io_method_name();
}
