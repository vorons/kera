// The model cards' recommended sampling (src/tune/kr_model_defaults.h), checked against the cards' own
// numbers (docs/DECISIONS.md D-013).

#include "kr_test.h"

#include "tune/kr_model_defaults.h"

using namespace kera;

KR_TEST(families_from_the_gguf_header) {
    KR_CHECK(family_of("nemotron_h_moe", "NVIDIA Nemotron 3 Nano 30B A3B") == model_family::nemotron3);
    KR_CHECK(family_of("nemotron_h_moe", "Nemotron-3-Nano-30B-A3B") == model_family::nemotron3);
    KR_CHECK(family_of("qwen3moe", "Qwen3 30B A3B") == model_family::qwen3);
    KR_CHECK(family_of("qwen3", "Qwen3-8B") == model_family::qwen3);
    KR_CHECK(family_of("qwen3moe", "Qwen3 Coder 30B A3B Instruct") == model_family::qwen3_coder);
    KR_CHECK(family_of("llama", "Llama 3.1 8B") == model_family::unknown);
    // only the card's own models (D-033): others of the same design keep the server's defaults
    KR_CHECK(family_of("nemotron_h", "") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h", "NVIDIA Nemotron Nano 9B v2") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "") == model_family::unknown);
    KR_CHECK(family_of("qwen3", "DeepSeek R1 0528 Qwen3 8B") == model_family::unknown);
    KR_CHECK(family_of("qwen3", "") == model_family::unknown);
    // Nemotron 3 Nano's card is Nano 30B-A3B's alone: the real files' names of the rest of the family, whose cards
    // differ (Super, Ultra and 3.5: 1.0/0.95 everywhere; Nano Omni: 0.6/0.95 thinking), keep their own values
    KR_CHECK(family_of("nemotron_h_moe", "Nemotron Nano 3 30B A3B") == model_family::nemotron3); // LM Studio's file
    KR_CHECK(family_of("nemotron_h_moe", "Nemotron 3 Nano Omni 30B 3B Reasoning") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "Nvidia-Nemotron-3-Nano-Omni-30B-A3B-Reasoning") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "NVIDIA Nemotron 3 Super 120B A12B") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "NVIDIA-Nemotron-3-Ultra-550B") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "NVIDIA Nemotron 3.5 Lightning") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "nemotron_3_5_nano") == model_family::unknown);
    KR_CHECK(family_of("nemotron_h_moe", "Nemotron Nano 3.5 30B") == model_family::unknown);
}

KR_TEST(the_card_only_where_the_file_agrees) {
    sampling_values none;
    KR_CHECK(card_agrees_with_file(model_family::qwen3, "Qwen3 8B", none)); // no defaults of its own
    sampling_values f;
    f.temp = 0.6f, f.top_p = 0.95f, f.top_k = 20;                          // Qwen3-8B's generation_config.json
    KR_CHECK(card_agrees_with_file(model_family::qwen3, "Qwen3 8B", f));
    f.temp = 0.7f, f.top_p = 0.8f;                                         // the 2507 Instruct release's
    KR_CHECK(card_agrees_with_file(model_family::qwen3, "Qwen3 30B A3B Instruct 2507", f));
    KR_CHECK(card_agrees_with_file(model_family::qwen3_coder, "Qwen3 Coder 30B A3B Instruct", f));
    f.temp = 0.3f;                                                         // a fine-tune's own
    KR_CHECK(!card_agrees_with_file(model_family::qwen3, "Qwen3 8B Medical", f));
    sampling_values n;
    n.temp = 1.0f, n.top_p = 1.0f;                                         // Nemotron 3 Nano's generation_config.json
    KR_CHECK(card_agrees_with_file(model_family::nemotron3, "NVIDIA Nemotron 3 Nano 30B A3B", n));
    n.top_p = 0.9f;
    KR_CHECK(!card_agrees_with_file(model_family::nemotron3, "NVIDIA Nemotron 3 Nano 30B A3B", n));
    n.temp = 0.6f, n.top_p = 0.95f;                                        // Nano's tool-call values, not a plain setting
    KR_CHECK(!card_agrees_with_file(model_family::nemotron3, "NVIDIA Nemotron 3 Nano 30B A3B", n));
    n.temp = 1.0f, n.top_p = 1.0f, n.top_k = 20, n.min_p = 0.05f;          // values the card doesn't name: not compared
    KR_CHECK(card_agrees_with_file(model_family::nemotron3, "NVIDIA Nemotron 3 Nano 30B A3B", n));
}

KR_TEST(nemotron_card) {
    sampling_values v;
    KR_CHECK(recommended_sampling(model_family::nemotron3, "", true, false, v));
    KR_CHECK(v.temp == 1.0f && v.top_p == 1.0f);   // reasoning on
    KR_CHECK(recommended_sampling(model_family::nemotron3, "", false, false, v));
    KR_CHECK(v.temp == 0.0f);                      // reasoning off: greedy
    KR_CHECK(recommended_sampling(model_family::nemotron3, "", true, true, v));
    KR_CHECK(v.temp == 0.6f && v.top_p == 0.95f);  // tool calling
    // the card names no top_k or min_p: off, as NVIDIA's own engines leave them (not llama.cpp's 40 and 0.05)
    KR_CHECK(v.top_k == 0 && v.min_p == 0.0f && v.top_k_from_engine && v.min_p_from_engine);
}

KR_TEST(qwen3_cards) {
    sampling_values v;
    KR_CHECK(recommended_sampling(model_family::qwen3, "Qwen3 30B A3B", true, false, v));
    KR_CHECK(v.temp == 0.6f && v.top_p == 0.95f && v.top_k == 20 && v.min_p == 0.0f);
    KR_CHECK(recommended_sampling(model_family::qwen3, "Qwen3 30B A3B", false, true, v));
    KR_CHECK(v.temp == 0.7f && v.top_p == 0.8f && v.top_k == 20);
    KR_CHECK(recommended_sampling(model_family::qwen3, "Qwen3 30B A3B Thinking 2507", false, false, v));
    KR_CHECK(v.temp == 0.6f); // always thinks, whatever its template says
    KR_CHECK(recommended_sampling(model_family::qwen3_coder, "", false, true, v));
    KR_CHECK(v.temp == 0.7f && v.top_p == 0.8f && v.top_k == 20 && v.repeat == 1.05f); // its card's repetition penalty
    KR_CHECK(v.min_p == 0.0f && v.min_p_from_engine && !v.top_k_from_engine); // min_p off, as Qwen's engines
    KR_CHECK(recommended_sampling(model_family::qwen3, "Qwen3 30B A3B", false, false, v) && v.repeat < 0);
    KR_CHECK(!recommended_sampling(model_family::unknown, "", true, true, v));
    // Qwen's GGUF cards: presence penalty 1.5 for quantized files, thinking or not; not 2507, not full precision
    KR_CHECK(recommended_presence_penalty(model_family::qwen3, "Qwen3 0.6B Instruct", true) == 1.5f);
    KR_CHECK(recommended_presence_penalty(model_family::qwen3, "Qwen3 30B A3B Instruct 2507", true) < 0);
    KR_CHECK(recommended_presence_penalty(model_family::qwen3, "Qwen3 8B", false) < 0);
    KR_CHECK(recommended_presence_penalty(model_family::qwen3_coder, "Qwen3 Coder 30B A3B Instruct", true) < 0);
    KR_CHECK(recommended_presence_penalty(model_family::nemotron3, "NVIDIA Nemotron 3 Nano 30B A3B", true) < 0);
}

// the makers' own generation_config.json values, for the models listed from Hugging Face (tools/cards, D-036)
KR_TEST(listed_models_by_name_and_shape) {
    sampling_values v;
    std::string repo;
    // Qwen2.5-1.5B-Instruct: 28 layers, 1536 wide; its generation_config.json: 0.7, 0.8, 20, repetition penalty 1.1
    KR_CHECK(listed_sampling({ "Qwen2.5 1.5B Instruct" }, 28, 1536, v, repo));
    KR_CHECK(repo == "Qwen/Qwen2.5-1.5B-Instruct");
    KR_CHECK_NEAR(v.temp, 0.7, 1e-6);
    KR_CHECK_NEAR(v.top_p, 0.8, 1e-6);
    KR_CHECK(v.top_k == 20);
    KR_CHECK_NEAR(v.repeat, 1.1, 1e-6);
    KR_CHECK(v.min_p < 0); // the makers name none: left to the engine
    // the name's other spellings, and the file's own name when the header's says nothing
    KR_CHECK(listed_sampling({ "", "qwen2.5-1.5b-instruct" }, 28, 1536, v, repo));
    KR_CHECK(listed_sampling({ "Qwen2.5-1.5B-Instruct" }, 28, 1536, v, repo));
    // the name alone isn't enough: another shape is another model
    KR_CHECK(!listed_sampling({ "Qwen2.5 1.5B Instruct" }, 28, 2048, v, repo));
    KR_CHECK(!listed_sampling({ "Qwen2.5 1.5B Instruct" }, 24, 1536, v, repo));
    KR_CHECK(!listed_sampling({ "Qwen2.5 1.5B Instruct" }, 0, 0, v, repo));
    // nor the shape alone: a fine-tune under its own name keeps the engine's defaults
    KR_CHECK(!listed_sampling({ "My Qwen2.5 1.5B Instruct Finetune" }, 28, 1536, v, repo));
    KR_CHECK(!listed_sampling({ "" }, 28, 1536, v, repo));
}

KR_TEST_MAIN()
