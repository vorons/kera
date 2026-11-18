// [Kera] hook 3: internals of the stream hook (include/llama-stream.h)
#pragma once

#include "llama-stream.h"

// the hook set by llama_set_stream_hook (all fields NULL if none)
llama_stream_hook llama_stream_hook_get();

// a context's eval callback when its model streams weights: the stream hook's first, then the
// context's own, each called back only for the tensors it asked for
struct llama_stream_eval_chain {
    ggml_backend_sched_eval_callback first     = nullptr;
    void *                           first_ud  = nullptr;
    ggml_backend_sched_eval_callback second    = nullptr;
    void *                           second_ud = nullptr;

    // the context's scheduler, so that llama_stream_tensor_get can find the backend that computed a tensor
    ggml_backend_sched_t sched = nullptr;

    bool asked_first  = false;
    bool asked_second = false;

    static bool call(struct ggml_tensor * t, bool ask, void * user_data);
};

// whether all of [p, p + n) can be read: a streamed weight's experts that weren't brought in can't be on Windows,
// where their pages aren't committed (KERA_STREAM_VERIFY in sched_copy_experts)
bool llama_stream_readable(const void * p, size_t n);
