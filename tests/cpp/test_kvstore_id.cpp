// Section 6 item 5 (wrong-identity test, at the interface level): any change to the cache identity must
// be a miss. The full test against a running model comes with the KV store in M2.

#include "kr_test.h"

#include "kvstore/kr_kvstore.h"

KR_TEST(identity_must_match_exactly) {
    const kr_cache_id base = { 0x1111, 0, KR_WEIGHTS_NATIVE, 16, 0x2222, 0x3333 };
    kr_cache_id same = base;
    KR_CHECK(kr_cache_id_equal(&base, &same));

    kr_cache_id other_model = base;   other_model.model_hash ^= 1;
    kr_cache_id other_adapter = base; other_adapter.adapter_hash = 0x4444;
    kr_cache_id other_mode = base;    other_mode.weight_mode = KR_WEIGHTS_NATIVE + 1; // any other value
    kr_cache_id other_kv = base;      other_kv.kv_precision = 8;
    kr_cache_id other_tok = base;     other_tok.tokenizer_hash ^= 1;
    kr_cache_id other_rope = base;    other_rope.rope_hash ^= 1;
    for (const kr_cache_id * id : { &other_model, &other_adapter, &other_mode, &other_kv, &other_tok, &other_rope }) {
        KR_CHECK(!kr_cache_id_equal(&base, id));
    }
}

KR_TEST_MAIN()
