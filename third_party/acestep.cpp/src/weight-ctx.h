#pragma once
// weight-ctx.h: format-independent weight loading context for ggml backends
//
// Manages a ggml_context for weight tensors + their backend buffer.
// Used by gguf-weights.h for all model loaders.
//
// Usage:
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = <loader>_load_tensor(&wctx, source, "name");
//   wctx_alloc(&wctx, backend);

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void *         src;
        size_t               nbytes;
        size_t               offset;  // byte offset into dst tensor (0 for regular loads)
    };

    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data, kept alive until wctx_alloc.
    // unique_ptr keeps the data address stable even when the outer vector grows,
    // so src pointers stored in pending stay valid across staging.push_back().
    std::vector<std::unique_ptr<float[]>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    wctx->ctx    = ggml_init(params);
    wctx->buffer = NULL;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

// [Kera] Copy n bytes with several threads. The weights come from a mapped file: read on one thread, page by page,
// they reach a graphics chip at about 0.5 GB a second on a DGX Spark; through ordinary memory first, read by
// several threads, at several GB a second. The bytes are the same, so the model is.
static void wctx_copy_threads(void * dst, const void * src, size_t n) {
    const size_t chunk = (size_t) 8 << 20;
    if (n <= chunk) {
        memcpy(dst, src, n);
        return;
    }
    const size_t   n_chk = (n + chunk - 1) / chunk;
    const unsigned n_thr = (unsigned) std::min<size_t>({ n_chk, (size_t) 8, (size_t) std::thread::hardware_concurrency() });
    std::atomic<size_t> next{ 0 };
    auto                work = [&]() {
        for (size_t i = next.fetch_add(1); i < n_chk; i = next.fetch_add(1)) {
            const size_t off = i * chunk;
            memcpy((char *) dst + off, (const char *) src + off, std::min(chunk, n - off));
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < n_thr; t++) {
        pool.emplace_back(work);
    }
    work();
    for (auto & th : pool) {
        th.join();
    }
}

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    const auto t0 = std::chrono::steady_clock::now();
    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        fprintf(stderr, "[WeightCtx] FATAL: failed to allocate backend buffer\n");
        return false;
    }
    // Mark as weight buffer so ggml_backend_sched assigns ops to the correct
    // backend based on weight location (avoids fallback through expansion).
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto t1    = std::chrono::steady_clock::now();
    size_t     total = 0;
    // [Kera] to a graphics chip: through one buffer of ordinary memory, as big as the largest tensor (see above).
    // On the CPU the backend's own copy is already a plain one, so it is left as it was
    const bool via_memory = !ggml_backend_buffer_is_host(wctx->buffer);
    size_t     max_bytes  = 0;
    for (auto & pc : wctx->pending) {
        max_bytes = std::max(max_bytes, pc.nbytes);
    }
    void * bounce = via_memory && max_bytes > 0 ? malloc(max_bytes) : NULL;
    for (auto & pc : wctx->pending) {
        if (bounce) {
            wctx_copy_threads(bounce, pc.src, pc.nbytes);
            ggml_backend_tensor_set(pc.tensor, bounce, pc.offset, pc.nbytes);
        } else {
            ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        }
        total += pc.nbytes;
    }
    free(bounce);
    // [Kera] the two parts' times, to see where a load's seconds go
    const auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend (alloc %.0f ms, copy %.0f ms)\n",
            wctx->pending.size(), (float) total / (1024 * 1024),
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count());
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
    }
    wctx->buffer = NULL;
    wctx->ctx    = NULL;
}
