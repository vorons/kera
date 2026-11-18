// Auto-setup step 2, SSD speed (spec 4a): write a test file, read it back with parallel reads that
// bypass the file cache, try several thread counts and read sizes, keep the best, delete the file.

#include "setup/kr_machine.h"

#include "io/kr_io.h"
#include "io/kr_os_file.h"
#include "tune/kr_settings.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

namespace kera {

namespace {

using clock_t_ = std::chrono::steady_clock;

constexpr size_t MiB = 1024 * 1024;

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// One cell of the grid: `inflight` reads of `size` bytes kept in flight until the deadline.
struct read_cell {
    kr_io * io = nullptr;
    int file_id = -1;
    size_t size = 0;
    uint64_t n_blocks = 0;
    clock_t_::time_point deadline;
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> counter{0};
    std::atomic<int> error{0};
    std::mutex mutex;
    std::condition_variable cv;
    int active = 0;

    struct slot {
        read_cell * cell;
        void * buf;
    };
    std::vector<slot> slots;

    void submit(slot * s) {
        const uint64_t block = splitmix64(counter.fetch_add(1)) % n_blocks;
        if (kr_io_read(io, file_id, block * size, size, s->buf, &read_cell::done, s) != 0) {
            finish_slot();
        }
    }

    void finish_slot() {
        std::lock_guard<std::mutex> lock(mutex);
        if (--active == 0) {
            cv.notify_all();
        }
    }

    static void done(void * user, int status, size_t got) {
        slot * s = (slot *) user;
        read_cell * c = s->cell;
        if (status != 0) {
            c->error = status;
            c->finish_slot();
            return;
        }
        // every finished read counts, and the clock stops when the last one lands
        c->bytes += got;
        if (clock_t_::now() < c->deadline) {
            c->submit(s);
        } else {
            c->finish_slot();
        }
    }
};

// Bytes per second reading `size`-byte blocks at random offsets with `inflight` reads at once.
double run_read_cell(const std::string & path, uint64_t file_size, size_t size, int inflight, double seconds, int & error) {
    kr_io * io = kr_io_create(inflight);
    read_cell c;
    c.io = io;
    c.file_id = kr_io_open(io, path.c_str(), KR_IO_DIRECT);
    c.size = size;
    c.n_blocks = std::max<uint64_t>(1, file_size / size);
    if (c.file_id < 0) {
        error = -c.file_id;
        kr_io_destroy(io);
        return 0.0;
    }
    c.slots.resize((size_t) inflight);
    for (auto & s : c.slots) {
        s.cell = &c;
        s.buf = kr_io_alloc(size);
    }

    const auto start = clock_t_::now();
    c.deadline = start + std::chrono::duration_cast<clock_t_::duration>(std::chrono::duration<double>(seconds));
    c.active = inflight;
    for (auto & s : c.slots) {
        c.submit(&s);
    }
    {
        std::unique_lock<std::mutex> lock(c.mutex);
        c.cv.wait(lock, [&] { return c.active == 0; });
    }
    const double elapsed = std::chrono::duration<double>(clock_t_::now() - start).count();
    kr_io_destroy(io);
    for (auto & s : c.slots) {
        kr_io_free(s.buf);
    }
    error = c.error;
    return elapsed > 0 ? (double) c.bytes.load() / elapsed : 0.0;
}

} // namespace

double read_test_file(const std::string & path, int read_mib, int inflight, double seconds, int * error) {
    os::file f;
    int err = os::open_read(path, false, f);
    if (err != 0) {
        if (error) *error = err;
        return 0.0;
    }
    // the size: read until the end, a block at a time
    uint64_t size = 0;
    {
        std::vector<char> probe(1);
        uint64_t lo = 0, hi = 1ull << 44; // binary search for the end of the file
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo + 1) / 2;
            size_t got = 0;
            os::pread_full(f, mid - 1, 1, probe.data(), &got);
            if (got == 1) lo = mid; else hi = mid - 1;
        }
        size = lo;
    }
    os::close(f);
    int e = 0;
    const double bps = run_read_cell(path, size, (size_t) read_mib * MiB, inflight, seconds, e);
    if (error) *error = e;
    return bps / 1e9;
}

void measure_drive(drive_profile & d, const std::string & test_dir, const setup_options & opt) {
    const double test_gb = opt.quick ? 0.25 : kr_setting_get("setup.ssd_test_gb");
    const double cell_s = opt.quick ? 0.15 : kr_setting_get("setup.ssd_cell_seconds");

    uint64_t file_size = (uint64_t) (test_gb * 1e9) / (16 * MiB) * (16 * MiB);
    const double margin_gb = 2.0;
    if (d.free_gb < margin_gb + 0.25) {
        d.note = "There isn't enough free space to test its speed (it needs about " +
                 std::to_string((int) std::ceil(test_gb + margin_gb)) + " GB free).";
        return;
    }
    if (d.free_gb < test_gb + margin_gb) {
        file_size = (uint64_t) ((d.free_gb - margin_gb) * 1e9) / (16 * MiB) * (16 * MiB);
    }

    // the folder is removed again afterwards if the test made it (a folder at the top of each drive tested)
    std::error_code dir_ec;
    const bool made_dir = !std::filesystem::exists(std::filesystem::u8path(test_dir), dir_ec);
    struct remove_made_dir {
        bool made;
        const std::string & dir;
        ~remove_made_dir() {
            std::error_code ec;
            if (made) {
                std::filesystem::remove(std::filesystem::u8path(dir), ec); // only if empty
            }
        }
    } cleanup{ made_dir, test_dir };
    os::make_dir(test_dir);
    const std::string path = test_dir + (test_dir.back() == '/' || test_dir.back() == '\\' ? "" : std::string(1, '/')) +
                             "kera-speed-test.tmp";
    os::remove_file(path); // left over from an interrupted run

    // Write the file with random data (some drives compress zeros), bypassing the cache.
    os::file f;
    int err = os::create_write(path, true, f);
    if (err != 0) {
        d.note = "It couldn't write a test file there (error " + std::to_string(err) + ").";
        return;
    }
    const size_t chunk = 8 * MiB;
    uint64_t * buf = (uint64_t *) os::aligned_alloc(chunk, KR_IO_ALIGN);
    uint64_t seed = 0x6b657261; // "kera"
    const auto w0 = clock_t_::now();
    for (uint64_t off = 0; off < file_size && err == 0; off += chunk) {
        // fresh random data for every chunk: storage that deduplicates or compresses repeated blocks
        // would otherwise read the file back faster than real model weights
        for (size_t i = 0; i < chunk / sizeof(uint64_t); ++i) {
            buf[i] = splitmix64(seed++);
        }
        size_t put = 0;
        err = os::pwrite_full(f, off, chunk, buf, &put);
    }
    if (err == 0) {
        err = os::sync(f);
    }
    const double wsec = std::chrono::duration<double>(clock_t_::now() - w0).count();
    d.direct = f.direct;
    os::close(f);
    os::aligned_free(buf);
    if (err != 0) {
        os::remove_file(path);
        d.note = "Writing the test file failed (error " + std::to_string(err) + ").";
        return;
    }
    d.write_gbps = wsec > 0 ? file_size / wsec / 1e9 : 0.0;

    // Read it back: every read size with every number of reads in flight. Short runs only pick the
    // winner: the best of many short, noisy runs overstates the speed, so the winner (and one read at a
    // time, for comparison with simple tests) is then measured again over a longer run.
    const std::vector<int> sizes_mib = opt.quick ? std::vector<int>{ 1, 16 } : std::vector<int>{ 1, 4, 16 };
    const std::vector<int> inflights = opt.quick ? std::vector<int>{ 1, 8 } : std::vector<int>{ 1, 4, 8, 16, 32 };
    const double confirm_s = opt.quick ? 0.3 : kr_setting_get("setup.ssd_confirm_seconds");
    double best = 0.0;
    int read_err = 0;
    for (int mib : sizes_mib) {
        for (int n : inflights) {
            int e = 0;
            const double bps = run_read_cell(path, file_size, (size_t) mib * MiB, n, cell_s, e);
            if (e != 0) {
                read_err = e;
                continue;
            }
            if (bps > best) {
                best = bps;
                d.best_read_mib = mib;
                d.best_inflight = n;
            }
        }
    }
    if (best > 0) {
        int e = 0;
        best = run_read_cell(path, file_size, (size_t) d.best_read_mib * MiB, d.best_inflight, confirm_s, e);
        if (e != 0) {
            read_err = e;
            best = 0;
        }
        d.single_read_gbps = run_read_cell(path, file_size, MiB, 1, confirm_s, e) / 1e9;
    }
    os::remove_file(path);

    d.read_gbps = best / 1e9;
    d.measured = best > 0;
    if (!d.measured) {
        d.note = "Reading the test file failed (error " + std::to_string(read_err) + ").";
    } else if (!d.direct) {
        d.note = "This drive's file system doesn't allow reads that skip the file cache, so its speed may look higher than it is.";
    } else if (d.read_gbps > 30.0) {
        d.note = "This speed is higher than any single SSD can read, so the reads may have come from memory. Treat it with caution.";
    }
}

} // namespace kera
