// [Kera] hook 3: streamed weights (include/llama-stream.h)

#include "llama-stream-impl.h"

#include <mutex>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

static std::mutex        g_stream_mutex;
static llama_stream_hook g_stream_hook = {};

void llama_set_stream_hook(const llama_stream_hook * hook) {
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    g_stream_hook = hook ? *hook : llama_stream_hook{};
}

llama_stream_hook llama_stream_hook_get() {
    std::lock_guard<std::mutex> lock(g_stream_mutex);
    return g_stream_hook;
}

// the scheduler whose eval callback is running on this thread (set while the stream hook's callback runs)
static thread_local ggml_backend_sched_t g_eval_sched = nullptr;

void llama_stream_tensor_get(const struct ggml_tensor * t, void * data, size_t offset, size_t size) {
    ggml_backend_t backend = nullptr;
    if (g_eval_sched != nullptr) {
        backend = ggml_backend_sched_get_tensor_backend(g_eval_sched, const_cast<ggml_tensor *>(t));
    }
    const ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (backend == nullptr || buf == nullptr || ggml_backend_buffer_is_host(buf)) {
        ggml_backend_tensor_get(t, data, offset, size);
        return;
    }
    ggml_backend_tensor_get_async(backend, t, data, offset, size);
    ggml_backend_synchronize(backend);
}

bool llama_stream_eval_chain::call(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * c = static_cast<llama_stream_eval_chain *>(user_data);
    if (ask) {
        // the scheduler asks node by node, then calls back once for the node that was wanted
        c->asked_first  = c->first  && c->first (t, true, c->first_ud);
        c->asked_second = c->second && c->second(t, true, c->second_ud);
        return c->asked_first || c->asked_second;
    }
    bool ok = true;
    if (c->asked_first) {
        g_eval_sched = c->sched;
        ok = c->first(t, false, c->first_ud) && ok;
        g_eval_sched = nullptr;
    }
    if (c->asked_second) {
        ok = c->second(t, false, c->second_ud) && ok;
    }
    return ok;
}

bool llama_stream_readable(const void * p, size_t n) {
#if defined(_WIN32)
    const char * a = (const char *) p;
    const char * end = a + n;
    while (a < end) {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(a, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT ||
            (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
            return false;
        }
        a = (const char *) mbi.BaseAddress + mbi.RegionSize;
    }
#else
    (void) p;
    (void) n;
#endif
    return true;
}
