// KV cache identity (section 4b item 5): any difference is a miss, never a reuse.

#include "kvstore/kr_kvstore.h"

extern "C" int kr_cache_id_equal(const kr_cache_id * a, const kr_cache_id * b) {
    return a->model_hash     == b->model_hash &&
           a->adapter_hash   == b->adapter_hash &&
           a->weight_mode    == b->weight_mode &&
           a->kv_precision   == b->kv_precision &&
           a->tokenizer_hash == b->tokenizer_hash &&
           a->rope_hash      == b->rope_hash;
}
