// Which music models a song is made with when the request names none (src/tune/kr_music_defaults.h): ACE-Step's own
// guide's rows ("Which Model Should I Choose?"), docs/DECISIONS.md D-036.

#include "kr_test.h"

#include "tune/kr_music_defaults.h"

using namespace kera;

namespace {

const uint64_t GB = 1000ull * 1000 * 1000;

std::vector<music_file> all_lms() {
    return { { "acestep-5Hz-lm-0.6B-Q8_0.gguf", GB * 7 / 10 }, { "acestep-5Hz-lm-1.7B-Q8_0.gguf", 2 * GB },
             { "acestep-5Hz-lm-4B-Q8_0.gguf", 4 * GB } };
}

std::vector<music_file> all_dits() {
    return { { "acestep-v15-base-Q8_0.gguf", 3 * GB },     { "acestep-v15-sft-Q8_0.gguf", 3 * GB },
             { "acestep-v15-turbo-Q8_0.gguf", 3 * GB },    { "acestep-v15-xl-base-Q8_0.gguf", 5 * GB },
             { "acestep-v15-xl-sft-Q8_0.gguf", 5 * GB },   { "acestep-v15-xl-turbo-Q8_0.gguf", 5 * GB } };
}

} // namespace

KR_TEST(the_guides_rows_by_graphics_memory) {
    music_choice c = choose_music_models(all_lms(), all_dits(), 8, 16);
    KR_CHECK(c.lm == "acestep-5Hz-lm-0.6B-Q8_0.gguf");
    KR_CHECK(c.dit == "acestep-v15-turbo-Q8_0.gguf");
    c = choose_music_models(all_lms(), all_dits(), 12, 32);
    KR_CHECK(c.lm == "acestep-5Hz-lm-1.7B-Q8_0.gguf");
    KR_CHECK(c.dit == "acestep-v15-turbo-Q8_0.gguf");
    c = choose_music_models(all_lms(), all_dits(), 16, 32);
    KR_CHECK(c.lm == "acestep-5Hz-lm-1.7B-Q8_0.gguf");
    KR_CHECK(c.dit == "acestep-v15-sft-Q8_0.gguf");
    c = choose_music_models(all_lms(), all_dits(), 22, 32);
    KR_CHECK(c.lm == "acestep-5Hz-lm-1.7B-Q8_0.gguf");
    KR_CHECK(c.dit == "acestep-v15-xl-turbo-Q8_0.gguf");
    c = choose_music_models(all_lms(), all_dits(), 24, 32);
    KR_CHECK(c.lm == "acestep-5Hz-lm-4B-Q8_0.gguf");
    KR_CHECK(c.dit == "acestep-v15-xl-sft-Q8_0.gguf");
    KR_CHECK(!c.why.empty());
}

KR_TEST(on_the_cpu_the_planner_by_memory_and_the_turbo_model) {
    const music_choice c = choose_music_models(all_lms(), all_dits(), 0, 64);
    KR_CHECK(c.lm == "acestep-5Hz-lm-4B-Q8_0.gguf");
    KR_CHECK(c.dit == "acestep-v15-turbo-Q8_0.gguf");
    const music_choice small = choose_music_models(all_lms(), all_dits(), 0, 8);
    KR_CHECK(small.lm == "acestep-5Hz-lm-0.6B-Q8_0.gguf");
}

KR_TEST(what_the_folder_has) {
    // the recommended size isn't there: the nearest smaller one, then the nearest larger
    std::vector<music_file> lms = { { "acestep-5Hz-lm-0.6B-Q8_0.gguf", GB }, { "acestep-5Hz-lm-1.7B-Q8_0.gguf", 2 * GB } };
    KR_CHECK(choose_music_models(lms, {}, 48, 64).lm == "acestep-5Hz-lm-1.7B-Q8_0.gguf");
    lms = { { "acestep-5Hz-lm-1.7B-Q8_0.gguf", 2 * GB }, { "acestep-5Hz-lm-4B-Q8_0.gguf", 4 * GB } };
    KR_CHECK(choose_music_models(lms, {}, 8, 16).lm == "acestep-5Hz-lm-1.7B-Q8_0.gguf");
    // the base models only when nothing else is there; SFT before base, as the guide's table rates them
    std::vector<music_file> dits = { { "acestep-v15-base-Q8_0.gguf", 3 * GB }, { "acestep-v15-sft-Q8_0.gguf", 3 * GB } };
    KR_CHECK(choose_music_models({}, dits, 8, 16).dit == "acestep-v15-sft-Q8_0.gguf");
    dits = { { "acestep-v15-base-Q8_0.gguf", 3 * GB } };
    const music_choice one = choose_music_models({}, dits, 8, 16);
    KR_CHECK(one.dit == "acestep-v15-base-Q8_0.gguf");
    KR_CHECK(one.why.empty()); // nothing to choose, nothing to say
    // "sft" and "xl" are whole parts of the name, as in ACE-Step's own interface
    dits = { { "acestep-v15-turbo-shift3-Q8_0.gguf", 3 * GB }, { "acestep-v15-base-Q8_0.gguf", 3 * GB } };
    KR_CHECK(choose_music_models({}, dits, 8, 16).dit == "acestep-v15-turbo-shift3-Q8_0.gguf");
}

KR_TEST(the_highest_precision_that_fits) {
    std::vector<music_file> dits = { { "acestep-v15-turbo-Q4_K_M.gguf", GB * 3 / 2 }, { "acestep-v15-turbo-Q8_0.gguf", 3 * GB },
                                     { "acestep-v15-turbo-BF16.gguf", 5 * GB } };
    KR_CHECK(choose_music_models({}, dits, 0, 64).dit == "acestep-v15-turbo-BF16.gguf");
    KR_CHECK(choose_music_models({}, dits, 0, 8).dit == "acestep-v15-turbo-Q8_0.gguf");  // BF16 is over half of 8 GB
    KR_CHECK(choose_music_models({}, dits, 0, 2).dit == "acestep-v15-turbo-Q4_K_M.gguf"); // none fits: the smallest
    // at the same precision, the plain release before its variants; a merge under a name of its own comes last
    dits = { { "acestep-v15-turbo-shift1-Q8_0.gguf", 3 * GB }, { "acestep-v15-turbo-continuous-Q8_0.gguf", 3 * GB },
             { "acestep-v15-turbo-Q8_0.gguf", 3 * GB }, { "acestep-v15-sftturbo50-BF16.gguf", 5 * GB } };
    KR_CHECK(choose_music_models({}, dits, 0, 64).dit == "acestep-v15-turbo-Q8_0.gguf");
    // a file under another name is taken only when ACE-Step's own aren't there
    dits = { { "my-finetune.gguf", GB }, { "acestep-v15-turbo-Q8_0.gguf", 3 * GB } };
    KR_CHECK(choose_music_models({}, dits, 0, 64).dit == "acestep-v15-turbo-Q8_0.gguf");
}

KR_TEST_MAIN()
