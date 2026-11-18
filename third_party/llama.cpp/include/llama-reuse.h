// [Kera] Sharing what one sequence has read with another, without reading it again (hook 7, docs/UPSTREAM_PATCHES.md
// in the Kera repo). Exact: the shared cells and state are the ones the model computed for the same positions.
// (Hook 5, copying a passage to a new position, was lossy and was removed with Kera's lossy modes: D-034.)
#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    // [Kera] hook 7: gives seq_id_dst the first n positions seq_id_src has read, without reading them again, in a
    // hybrid memory (attention plus a recurrent state): the attention cells for [0, n) are shared (both sequences hold
    // them, as llama_memory_seq_cp does) and the recurrent state is set from `state`, a partial state
    // (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) saved when seq_id_src had read exactly n tokens. seq_id_dst's own memory is
    // cleared first. With n == 0 it only says whether this memory can (0) and changes nothing. Returns 0, or -1 if the
    // memory isn't a plain hybrid one or the state doesn't load (seq_id_dst is then left empty).
    LLAMA_API int32_t llama_memory_seq_share_hybrid(
            struct llama_context * ctx,
                    llama_seq_id   seq_id_src,
                    llama_seq_id   seq_id_dst,
                       llama_pos   n,
                   const uint8_t * state,
                          size_t   n_state);

#ifdef __cplusplus
}
#endif
