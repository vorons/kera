// [Kera] Streamed weights (hook 3, docs/UPSTREAM_PATCHES.md in the Kera repo).
//
// Lets a program keep some weights (a MoE model's routed experts) out of memory at load time and
// bring them in itself while the model runs:
//   - while a model's tensors are created, the hook says how to load each weight: as usual, mapped but
//     not read (the OS file cache serves it), or placed in the hook's own buffer type and not read;
//   - every context of such a model calls the hook's eval callback before its own, so the hook sees
//     each layer's chosen experts before any operation reads them.
// The loader never reads a streamed weight, and nothing else in llama.cpp changes.
#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    enum llama_stream_mode {
        LLAMA_STREAM_NONE = 0, // load as usual
        LLAMA_STREAM_MAP  = 1, // map the file without reading it at load (like a lazy tensor)
        LLAMA_STREAM_BUFT = 2, // allocate it in the buffer type the hook returns, and don't read it
    };

    struct llama_stream_hook {
        // a model is about to be loaded from path; return false to load it as usual
        bool (*begin)(void * user_data, const char * path, bool no_alloc);

        // how to load this weight; for LLAMA_STREAM_BUFT, set *buft. file is the path of the file that
        // holds its data (one of a split model's parts) and offset where the data starts in it
        enum llama_stream_mode (*mode)(void * user_data, const char * name, const char * file, uint64_t offset,
                                       size_t nbytes, ggml_backend_buffer_type_t * buft);

        // a LLAMA_STREAM_MAP weight has been mapped: its data now points into the mapping (may be NULL)
        void (*mapped)(void * user_data, struct ggml_tensor * tensor);

        // the model has loaded (model is NULL if loading failed); not called if begin returned false. Return false
        // if the weights placed in the hook's buffer type can't be brought in: the load then fails, and end is
        // called again with NULL once the model is freed
        bool (*end)(void * user_data, const struct llama_model * model);

        // called before the context's own eval callback, for every context of a model the hook streams
        ggml_backend_sched_eval_callback cb_eval;

        void * user_data;
    };

    // Set the hook used by the following model loads (copied; NULL removes it). Models already loaded
    // keep the hook they were loaded with, which must stay valid until they are freed.
    LLAMA_API void llama_set_stream_hook(const struct llama_stream_hook * hook);

    // For the hook's cb_eval: copy bytes of a tensor the model just computed to host memory, reading them on
    // the queue of the backend that computed it (ggml_backend_tensor_get_async, then a wait on that backend)
    // instead of the separate queue ggml_backend_tensor_get uses. Same as ggml_backend_tensor_get elsewhere.
    LLAMA_API void llama_stream_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

#ifdef __cplusplus
}
#endif
