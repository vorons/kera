// [Kera] hook 7: see include/llama-reuse.h (hook 5, copying passages to new positions, was removed: D-034)
#include "llama-reuse.h"

#include "llama-kv-cache.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid.h"

// [Kera] hook 7: see include/llama-reuse.h
int32_t llama_memory_seq_share_hybrid(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   n,
        const uint8_t * state,
               size_t   n_state) {
    llama_memory_t mem = llama_get_memory(ctx);
    // the sparse-attention memory (hybrid_idx) pools keys per sequence and copies only whole sequences
    auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem);
    if (hyb == nullptr || dynamic_cast<llama_memory_hybrid_idx *>(mem) != nullptr || seq_id_src == seq_id_dst || n < 0) {
        return -1;
    }
    if (n == 0) {
        return 0;
    }
    if (state == nullptr || n_state == 0) {
        return -1;
    }
    hyb->seq_rm(seq_id_dst, -1, -1);
    hyb->get_mem_attn()->seq_cp(seq_id_src, seq_id_dst, 0, n);
    if (llama_state_seq_set_data_ext(ctx, state, n_state, seq_id_dst, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        hyb->seq_rm(seq_id_dst, -1, -1);
        return -1;
    }
    return 0;
}
