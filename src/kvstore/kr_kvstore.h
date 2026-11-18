/* KV store (spec P3, section 4b): saves the model's memory of a chat (KV) and reloads it instead of
 * re-reading the text. Its only job is faster chat replies.
 *
 * A saved chunk is valid only with exactly the same text before it and exactly the same cache
 * identity. Reusing KV under another model file, adapter, weight mode or KV precision gives wrong
 * answers silently, so any difference in kr_cache_id must be a miss (section 6 item 5).
 * Frozen in M0; built in M2 (Phase 5b).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kr_kvstore kr_kvstore;

/* Everything that changes the KV for the same tokens (section 4b item 5). */
typedef struct {
    uint64_t model_hash;       /* hash of the model file */
    uint64_t adapter_hash;     /* LoRA adapter, 0 = none */
    uint32_t weight_mode;      /* KR_WEIGHTS_NATIVE, the only mode: see kr_weight_mode */
    uint32_t kv_precision;     /* bits per KV value, 16 by default */
    uint64_t tokenizer_hash;   /* tokenizer and chat-template version */
    uint64_t rope_hash;        /* position (RoPE) settings */
} kr_cache_id;

enum kr_weight_mode {
    KR_WEIGHTS_NATIVE   = 0, /* the model file's own format: the only one. Shadow Bits' modes were never built (M7
                                skipped, N-004), and Kera has no modes that trade quality for speed (D-034) */
};

/* Where loaded KV goes, and where saved KV comes from: opaque handles owned by the engine. */
typedef struct kr_kv_dst kr_kv_dst;
typedef struct kr_kv_src kr_kv_src;

/* How many of the first n tokens already have saved KV under this identity (0 = none). */
int  kr_kv_lookup(kr_kvstore * k, const kr_cache_id * id, const int32_t * tok, int n);

/* Load the saved KV for the first n tokens into dst. Returns the number of tokens loaded. */
int  kr_kv_load(kr_kvstore * k, const kr_cache_id * id, const int32_t * tok, int n, kr_kv_dst * dst);

/* Save the KV for these tokens in the background, so writing never pauses. */
void kr_kv_save_async(kr_kvstore * k, const kr_cache_id * id, const int32_t * tok, int n, const kr_kv_src * src);

/* Keep (pin = 1) or release (pin = 0) these tokens' KV so it is never evicted ("Remember this"). */
void kr_kv_pin(kr_kvstore * k, const kr_cache_id * id, const int32_t * tok, int n, int pin);

/* 1 if the two identities are exactly the same. */
int  kr_cache_id_equal(const kr_cache_id * a, const kr_cache_id * b);

#ifdef __cplusplus
}
#endif
