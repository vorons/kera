// Auto-setup step 2, RAM speed from the CPU (spec 4a): every core reads a large buffer at once.
// Reading, not copying, because reading weights is what sets the speed of writing words.

#include "setup/kr_machine.h"

#include "io/kr_os_file.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace kera {

namespace {

volatile uint64_t g_sink; // keeps the compiler from removing the reads

uint64_t sum_range(const uint64_t * p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 += p[i + 0]; a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3];
        a4 += p[i + 4]; a5 += p[i + 5]; a6 += p[i + 6]; a7 += p[i + 7];
    }
    for (; i < n; ++i) {
        a0 += p[i];
    }
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

} // namespace

double measure_cpu_read_gbps(size_t bytes, int threads, double seconds) {
    if (threads < 1) {
        threads = (int) std::max(1u, std::thread::hardware_concurrency());
    }
    const size_t n = bytes / sizeof(uint64_t);
    uint64_t * buf = (uint64_t *) os::aligned_alloc(n * sizeof(uint64_t), 4096);
    if (buf == nullptr) {
        return 0.0;
    }
    const size_t per = n / (size_t) threads;

    // each thread touches its own slice first, so the memory is spread like the reads
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t) {
            ts.emplace_back([=] {
                for (size_t i = (size_t) t * per; i < (size_t) (t + 1) * per; ++i) {
                    buf[i] = i;
                }
            });
        }
        for (auto & th : ts) th.join();
    }

    std::atomic<bool> go{false};
    std::atomic<bool> stop{false};
    std::vector<uint64_t> passes((size_t) threads, 0);
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        ts.emplace_back([&, t] {
            while (!go) { std::this_thread::yield(); }
            uint64_t s = 0;
            while (!stop) {
                s += sum_range(buf + (size_t) t * per, per);
                passes[(size_t) t]++;
            }
            g_sink = s;
        });
    }
    const auto start = std::chrono::steady_clock::now();
    go = true;
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto & th : ts) th.join();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    os::aligned_free(buf);

    uint64_t total_passes = 0;
    for (uint64_t p : passes) total_passes += p;
    return elapsed > 0 ? (double) total_passes * per * sizeof(uint64_t) / elapsed / 1e9 : 0.0;
}

} // namespace kera
