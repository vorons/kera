// Auto-setup step 2 (spec 4a): the graphics chip, measured through ggml's own back-ends, so the numbers
// are what Kera's kernels will see:
//   - memory read speed: a row sum over a large tensor;
//   - a 4-bit matrix-times-vector test (Q4_0), the operation that writing words is made of;
//   - the same 4-bit test on the CPU, and CPU + graphics chip reading memory at the same time.

#include "setup/kr_machine.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace kera {

namespace {

constexpr double GB = 1e9;
constexpr size_t MiB = 1024 * 1024;

struct graph_bench {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * gf = nullptr;
    size_t bytes_per_run = 0; // bytes the graph reads from memory each run
    bool ok = false;

    ~graph_bench() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

ggml_context * new_ctx(int n_tensors) {
    ggml_init_params p = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) n_tensors + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(p);
}

// Seconds for one run of the graph: the best of several, after a warm-up.
double best_run_seconds(ggml_backend_t backend, ggml_cgraph * gf, int min_iters, double min_seconds) {
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        return 0.0;
    }
    ggml_backend_synchronize(backend);
    double best = std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (int i = 0; i < min_iters || total < min_seconds; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, dt);
        total += dt;
        if (i > 200) {
            break;
        }
    }
    return best;
}

// A row sum over `bytes` of F32: reads every byte once.
void build_read_test(graph_bench & b, size_t bytes) {
    const int64_t ne0 = 4096;
    const int64_t rows = std::max<int64_t>(1, (int64_t) (bytes / (ne0 * sizeof(float))));
    b.ctx = new_ctx(4);
    ggml_tensor * a = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, ne0, rows);
    ggml_tensor * r = ggml_sum_rows(b.ctx, a);
    if (!ggml_backend_supports_op(b.backend, r)) {
        return;
    }
    b.gf = ggml_new_graph(b.ctx);
    ggml_build_forward_expand(b.gf, r);
    b.buf = ggml_backend_alloc_ctx_tensors(b.ctx, b.backend);
    if (b.buf == nullptr) {
        return;
    }
    ggml_backend_buffer_clear(b.buf, 0);
    b.bytes_per_run = ggml_nbytes(a);
    b.ok = true;
}

// n_mats 4-bit matrices [k x n] times one vector: reads the weights once per run.
void build_q4_test(graph_bench & b, int n_mats, int64_t k, int64_t n) {
    b.ctx = new_ctx(2 * n_mats + 4);
    ggml_tensor * x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, k, 1);
    std::vector<ggml_tensor *> ws;
    std::vector<ggml_tensor *> ys;
    for (int i = 0; i < n_mats; ++i) {
        ggml_tensor * w = ggml_new_tensor_2d(b.ctx, GGML_TYPE_Q4_0, k, n);
        ggml_tensor * y = ggml_mul_mat(b.ctx, w, x);
        if (!ggml_backend_supports_op(b.backend, y)) {
            return;
        }
        ws.push_back(w);
        ys.push_back(y);
    }
    b.gf = ggml_new_graph(b.ctx);
    for (ggml_tensor * y : ys) {
        ggml_build_forward_expand(b.gf, y);
    }
    b.buf = ggml_backend_alloc_ctx_tensors(b.ctx, b.backend);
    if (b.buf == nullptr) {
        return;
    }

    // real 4-bit data (random weights, quantized), so the kernels take their normal path
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    const int64_t block_rows = 256;
    std::vector<float> rows_f((size_t) (block_rows * k));
    for (float & v : rows_f) v = uni(rng);
    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, k);
    std::vector<uint8_t> rows_q(row_size * (size_t) block_rows);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, rows_f.data(), rows_q.data(), 0, block_rows, k, nullptr);
    for (ggml_tensor * w : ws) {
        for (int64_t r0 = 0; r0 < n; r0 += block_rows) {
            const int64_t nr = std::min(block_rows, n - r0);
            ggml_backend_tensor_set(w, rows_q.data(), (size_t) r0 * row_size, (size_t) nr * row_size);
        }
    }
    std::vector<float> xv((size_t) k);
    for (float & v : xv) v = uni(rng);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));

    size_t bytes = 0;
    for (ggml_tensor * w : ws) bytes += ggml_nbytes(w);
    b.bytes_per_run = bytes;
    b.ok = true;
}

double gbps(const graph_bench & b, int min_iters, double min_seconds) {
    if (!b.ok) {
        return 0.0;
    }
    const double s = best_run_seconds(b.backend, b.gf, min_iters, min_seconds);
    return s > 0 ? b.bytes_per_run / s / GB : 0.0;
}

void set_cpu_threads(ggml_backend_t backend, ggml_backend_dev_t dev, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) {
        fn(backend, n_threads);
    }
}

} // namespace

// llama.cpp's CUDA back-end reports NVIDIA's chips that share the computer's memory (the DGX Spark's GB10, Jetson
// Thor and Orin) as separate cards: its "integrated" flag is off upstream (corrupted output, llama.cpp #15034). They
// are known by name, or, on ARM Linux (where all of them run), by a CUDA chip that may use about all of the
// computer's memory. total_gb is decimal (as the chip reports it), ram_gb binary (GiB). Not by size elsewhere: an
// ordinary card can have about as much memory as the PC (an RTX 5090 in a 32 GB PC), and it doesn't share it.
bool nvidia_shares_memory(const std::string & backend, const std::string & name, double total_gb, double ram_gb) {
    if (backend != "CUDA") {
        return false;
    }
    for (const char * n : { "GB10", "Thor", "Orin", "Spark" }) {
        if (name.find(n) != std::string::npos) {
            return true;
        }
    }
#if defined(__linux__) && defined(__aarch64__)
    // an ordinary card in an ARM computer (a GeForce in an ARM workstation) can match its memory too
    for (const char * n : { "GeForce", "RTX", "Quadro", "Tesla" }) {
        if (name.find(n) != std::string::npos) {
            return false;
        }
    }
    return ram_gb > 0 && total_gb * 1e9 / 1073741824.0 > 0.9 * ram_gb;
#else
    (void) total_gb;
    (void) ram_gb;
    return false;
#endif
}

void measure_gpus(machine_profile & p, const setup_options & opt) {
    ggml_backend_load_all();

    const size_t read_bytes = opt.quick ? 64 * MiB : 1024 * MiB;
    const int q4_mats = opt.quick ? 1 : 8;        // 8 x 37.7 MB: bigger than any graphics chip's cache
    const int64_t q4_k = 4096;
    const int64_t q4_n = opt.quick ? 4096 : 16384;
    const double min_s = opt.quick ? 0.05 : 0.5;

    ggml_backend_dev_t cpu_dev = nullptr;
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_dev = cpu_dev ? cpu_dev : dev;
            continue;
        }
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU && props.type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        gpu_profile g;
        g.name = props.description ? props.description : (props.name ? props.name : "graphics chip");
        g.backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        g.total_gb = props.memory_total / GB;
        g.free_gb = props.memory_free / GB;
        g.max_alloc_gb = ggml_backend_buft_get_max_size(ggml_backend_dev_buffer_type(dev)) / GB;
        g.unified = props.type == GGML_BACKEND_DEVICE_TYPE_IGPU || g.backend == "Metal" ||
                    nvidia_shares_memory(g.backend, g.name, g.total_gb, p.ram_gb);
        p.gpus.push_back(g);
        gpu_dev = gpu_dev ? gpu_dev : dev;
    }

    const int cpu_threads = p.cpu_cores > 0 ? p.cpu_cores : (int) std::max(1u, std::thread::hardware_concurrency());

    if (cpu_dev && opt.measure_gpu) {
        if (opt.progress) opt.progress(0.0, "Testing the CPU on a 4-bit model layer");
        graph_bench b;
        b.backend = ggml_backend_dev_init(cpu_dev, nullptr);
        if (b.backend) {
            set_cpu_threads(b.backend, cpu_dev, cpu_threads);
            build_q4_test(b, q4_mats, q4_k, q4_n);
            p.cpu_q4_gbps = gbps(b, 3, min_s);
        }
        if (b.buf) { ggml_backend_buffer_free(b.buf); b.buf = nullptr; }
        if (b.backend) ggml_backend_free(b.backend);
    }

    if (!gpu_dev || !opt.measure_gpu) {
        p.ram_gbps.both = p.ram_gbps.cpu;
        return;
    }

    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    if (gpu == nullptr) {
        p.notes.push_back("The graphics chip was found but couldn't be started, so its speed wasn't measured.");
        p.ram_gbps.both = p.ram_gbps.cpu;
        return;
    }

    if (opt.progress) opt.progress(0.0, "Testing the graphics chip's memory speed");
    {
        graph_bench b;
        b.backend = gpu;
        const size_t free_bytes = (size_t) (p.gpus.front().free_gb * GB);
        build_read_test(b, free_bytes > 0 ? std::min(read_bytes, free_bytes / 4) : read_bytes);
        p.ram_gbps.gpu = gbps(b, 5, min_s);

        // CPU and graphics chip reading at the same time (they share the memory on unified machines)
        if (b.ok && p.ram_gbps.gpu > 0) {
            if (opt.progress) opt.progress(0.0, "Testing the CPU and graphics chip together");
            std::atomic<bool> stop{false};
            double gpu_bytes = 0.0;
            double gpu_secs = 0.0;
            std::thread t([&] {
                while (!stop) {
                    const auto t0 = std::chrono::steady_clock::now();
                    ggml_backend_graph_compute(gpu, b.gf);
                    ggml_backend_synchronize(gpu);
                    gpu_secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    gpu_bytes += (double) b.bytes_per_run;
                }
            });
            const double cpu_together = measure_cpu_read_gbps(opt.quick ? 64 * MiB : 512 * MiB, cpu_threads, opt.quick ? 0.2 : 1.5);
            stop = true;
            t.join();
            p.ram_gbps.both = cpu_together + (gpu_secs > 0 ? gpu_bytes / gpu_secs / GB : 0.0);
        }
        b.backend = nullptr; // freed below
    }

    if (opt.progress) opt.progress(0.0, "Testing the graphics chip on a 4-bit model layer");
    {
        graph_bench b;
        b.backend = gpu;
        build_q4_test(b, q4_mats, q4_k, q4_n);
        p.gpu_q4_gbps = gbps(b, 5, min_s);
        b.backend = nullptr;
    }
    ggml_backend_free(gpu);
}

} // namespace kera
