// The speed planner: Appendix C's numbers exactly, real tensor sizes from GGUF headers, and the
// planner's cases (fits, streamed MoE, streamed dense).

#include "kr_test.h"

#include "planner/kr_plan.h"

#include "gguf.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// The example output printed by reference/planner.py (spec Appendix C).
KR_TEST(appendix_c_numbers) {
    const double mac[3] = { 10, 100, 2.0 }, mini[3] = { 50, 210, 5.0 }, spark[3] = { 110, 231, 11.0 };
    KR_CHECK_NEAR(kera::ceiling_tok_s(32, 32, 32, 4.5, mac[0], mac[1], mac[2]), 0.24, 0.005);
    KR_CHECK_NEAR(kera::ceiling_tok_s(70, 70, 70, 4.5, mini[0], mini[1], mini[2]), 5.33, 0.005);
    KR_CHECK_NEAR(kera::ceiling_tok_s(120, 7.5, 3, 4.5, mac[0], mac[1], mac[2], 0.5), 1.48, 0.005);
    KR_CHECK_NEAR(kera::ceiling_tok_s(35, 4.5, 2, 4.5, mac[0], mac[1], mac[2], 0.75), 4.97, 0.005);
    KR_CHECK_NEAR(kera::ceiling_tok_s(744, 40, 21, 4.5, spark[0], spark[1], spark[2], 0.82), 3.67, 0.005);
    KR_CHECK_NEAR(kera::ceiling_tok_s(744, 40, 21, 4.5, spark[0], spark[1], spark[2], 0.82, true), 5.28, 0.005);
}

static std::string temp_file(const char * name) {
#if defined(_WIN32)
    const char * dir = std::getenv("TEMP");
    return std::string(dir ? dir : ".") + "\\" + name;
#else
    return std::string("/tmp/") + name;
#endif
}

// A fake MoE file: 2 layers, each with attention (always) and experts (4 experts, 2 used), plus
// embeddings and an output layer. Sizes are chosen so the arithmetic is easy to check.
static std::string write_fake_moe() {
    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "fakemoe");
    gguf_set_val_u32(g, "fakemoe.block_count", 2);
    gguf_set_val_u32(g, "fakemoe.expert_count", 4);
    gguf_set_val_u32(g, "fakemoe.expert_used_count", 2);
    auto add = [&](const char * name, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(t, name);
        gguf_add_tensor(g, t);
    };
    add("token_embd.weight", 1000);          // 4,000 bytes
    add("output.weight", 500);               // 2,000 bytes
    add("blk.0.attn_q.weight", 250);         // 1,000 bytes
    add("blk.0.ffn_up_exps.weight", 2000);   // 8,000 bytes
    add("blk.1.attn_q.weight", 250);         // 1,000 bytes
    add("blk.1.ffn_up_exps.weight", 2000);   // 8,000 bytes
    const std::string path = temp_file("kera_test_plan.gguf");
    gguf_write_to_file(g, path.c_str(), true); // header only is enough for the planner
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

KR_TEST(model_sizes_from_gguf) {
    const std::string path = write_fake_moe();
    kera::model_sizes m;
    std::string err;
    KR_CHECK(kera::read_model_sizes(path, m, &err));
    KR_CHECK(m.arch == "fakemoe");
    KR_CHECK(m.n_expert == 4 && m.n_expert_used == 2 && m.n_moe_layers == 2);
    KR_CHECK_NEAR(m.embed_gb, 4000e-9, 1e-15);
    KR_CHECK_NEAR(m.always_gb, 4000e-9, 1e-15);           // output + both attention tensors
    KR_CHECK_NEAR(m.experts_gb, 16000e-9, 1e-15);
    KR_CHECK_NEAR(m.active_experts_gb, 8000e-9, 1e-15);   // half the experts per word
    KR_CHECK_NEAR(m.per_word_gb(), 12000e-9, 1e-15);
    std::remove(path.c_str());
    KR_CHECK(!kera::read_model_sizes(temp_file("kera_no_such_model.gguf"), m, &err));
}

KR_TEST(plan_cases) {
    kera::model_sizes moe;
    moe.always_gb = 3;  // read every word
    moe.embed_gb = 1;
    moe.experts_gb = 60;
    moe.active_experts_gb = 2;
    moe.n_expert = 64;
    moe.n_expert_used = 4;
    kera::plan_input in;
    in.usable_ram_gb = 100;
    in.ram_gb_s = 200;
    in.ssd_gb_s = 5;
    auto r = kera::plan(moe, in);
    KR_CHECK(r.fits);
    KR_CHECK_NEAR(r.ceiling_tok_s, 200.0 / 5.0, 1e-9);     // 5 GB per word from memory
    KR_CHECK_NEAR(r.ssd_gb_per_word, 0.0, 1e-12);

    in.usable_ram_gb = 34;                                   // 4 GB always-used + 30 GB of the 60 GB of experts
    r = kera::plan(moe, in);
    KR_CHECK(!r.fits);
    KR_CHECK_NEAR(r.expert_cache_gb, 30, 1e-9);
    KR_CHECK_NEAR(r.hit_rate, 0.833, 1e-12);                 // half the experts cached: 83% hits (RT2's LRU curve)
    KR_CHECK_NEAR(r.ssd_gb_per_word, 2.0 * (1 - 0.833), 1e-12);
    KR_CHECK_NEAR(r.ceiling_tok_s, 1.0 / (5.0 / 200 + 2.0 * (1 - 0.833) / 5), 1e-9);
    KR_CHECK_NEAR(r.ceiling_overlap_tok_s, 1.0 / (2.0 * (1 - 0.833) / 5 + 0.15 * 5.0 / 200), 1e-9);
    in.hit_rate = 0.9;                                       // a measured hit rate overrides the estimate
    r = kera::plan(moe, in);
    KR_CHECK_NEAR(r.ssd_gb_per_word, 0.2, 1e-12);

    kera::model_sizes dense;
    dense.always_gb = 39;
    dense.embed_gb = 1;
    kera::plan_input in2;
    in2.usable_ram_gb = 30;
    in2.ram_gb_s = 100;
    in2.ssd_gb_s = 2;
    r = kera::plan(dense, in2);
    KR_CHECK(!r.fits);
    KR_CHECK_NEAR(r.ssd_gb_per_word, 10, 1e-9);
    KR_CHECK_NEAR(r.ceiling_tok_s, 1.0 / (39.0 / 100 + 10.0 / 2), 1e-9);
    KR_CHECK(!kera::explain_plan(dense, in2, r).empty());
}

KR_TEST(lru_curve_from_the_routing_traces) {
    KR_CHECK_NEAR(kera::lru_hit_rate(0.0), 0.0, 1e-12);
    KR_CHECK_NEAR(kera::lru_hit_rate(0.25), 0.626, 1e-12);
    KR_CHECK_NEAR(kera::lru_hit_rate(0.375), (0.626 + 0.833) / 2, 1e-12);
    KR_CHECK_NEAR(kera::lru_hit_rate(1.0), 1.0, 1e-12);
    KR_CHECK_NEAR(kera::lru_hit_rate(2.0), 1.0, 1e-12);
}

KR_TEST_MAIN()
