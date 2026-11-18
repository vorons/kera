// A model file's precision from its name, and the same model at a higher precision beside it (src/tune/kr_precision.h)

#include "kr_test.h"

#include "tune/kr_precision.h"

using namespace kera;

KR_TEST(the_precision_and_the_models_stem) {
    std::string stem;
    KR_CHECK(precision_of("flux-3-image-klein-BF16.gguf", &stem) == 1);
    KR_CHECK(stem == "flux-3-image-klein");
    KR_CHECK(precision_of("FLUX-3-image-klein-Q8_0.gguf", &stem) == 2);
    KR_CHECK(stem == "flux-3-image-klein");
    KR_CHECK(precision_of("flux-3-image-klein-pup-UD-Q4_K_XL.gguf", &stem) == 5);
    KR_CHECK(stem == "flux-3-image-klein-pup");
    KR_CHECK(precision_of("Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf", &stem) == 5);
    KR_CHECK(stem == "qwen3-tts-12hz-1.7b-base");
    KR_CHECK(precision_of("moss-soundeffect-v2-bf16.gguf", &stem) == 1);
    KR_CHECK(precision_of("qwen_image_fp8_e4m3fn.safetensors", &stem) == 2);
    KR_CHECK(stem == "qwen_image");
    KR_CHECK(precision_of("model-f32.gguf") == 0);
    KR_CHECK(precision_of("Model-IQ4_XS.gguf") == 5);
    KR_CHECK(precision_of("Model-Q6_K.gguf") == 3);
    KR_CHECK(precision_of("Model-Q5_K_M-00001-of-00002.gguf", &stem) == 4);
    KR_CHECK(stem == "model-00001-of-00002");
    // no tag: nothing to tell by (and "f16" inside another word isn't one)
    KR_CHECK(precision_of("sd_turbo.safetensors", &stem) == -1);
    KR_CHECK(stem == "sd_turbo");
    KR_CHECK(precision_of("leaf16.gguf") == -1);
}

KR_TEST(the_same_model_at_a_higher_precision) {
    const uint64_t GB = 1000ull * 1000 * 1000;
    const std::vector<sibling_file> folder = { { "flux-3-image-klein-Q4_K_M.gguf", 5 * GB },
                                               { "flux-3-image-klein-Q8_0.gguf", 9 * GB },
                                               { "flux-3-image-klein-BF16.gguf", 17 * GB },
                                               { "flux-3-image-klein-base-BF16.gguf", 17 * GB },
                                               { "flux-3-image-klein-BF16.safetensors", 17 * GB },
                                               { "other-model-BF16.gguf", GB } };
    KR_CHECK(higher_precision_sibling("flux-3-image-klein-Q4_K_M.gguf", folder, 0) == "flux-3-image-klein-BF16.gguf");
    KR_CHECK(higher_precision_sibling("flux-3-image-klein-Q8_0.gguf", folder, 0) == "flux-3-image-klein-BF16.gguf");
    // only what fits: the 8-bit file for the 4-bit one, nothing for the 8-bit one
    KR_CHECK(higher_precision_sibling("flux-3-image-klein-Q4_K_M.gguf", folder, 10 * GB) == "flux-3-image-klein-Q8_0.gguf");
    KR_CHECK(higher_precision_sibling("flux-3-image-klein-Q8_0.gguf", folder, 10 * GB).empty());
    // the model's own weights already, another model, a name without a tag
    KR_CHECK(higher_precision_sibling("flux-3-image-klein-BF16.gguf", folder, 0).empty());
    KR_CHECK(higher_precision_sibling("flux-3-image-klein-base-Q8_0.gguf", folder, 0) == "flux-3-image-klein-base-BF16.gguf");
    KR_CHECK(higher_precision_sibling("third-model-Q4_K_M.gguf", folder, 0).empty());
    KR_CHECK(higher_precision_sibling("sd_turbo.gguf", folder, 0).empty());
}

KR_TEST_MAIN()
