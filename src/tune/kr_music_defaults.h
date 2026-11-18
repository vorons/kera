// Which of the music folder's models a song is made with when the request names none (D-036).
//
// ACE-Step's own guide ("Which Model Should I Choose?", README.md, and docs/en/GPU_COMPATIBILITY.md) names a sound
// model and a planner (its 5 Hz language model) for each memory size. The music engine otherwise takes the first of
// each by file name: the base model before SFT and turbo, the 0.6B planner before 1.7B and 4B, 4-bit before 8-bit.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kera {

struct music_file {
    std::string name;        // the file's name, e.g. "acestep-v15-xl-sft-Q8_0.gguf"
    uint64_t    bytes = 0;
};

struct music_choice {
    std::string lm;          // "" when the folder has none
    std::string dit;
    std::string why;         // the start of a plain sentence for the log (the caller adds which); "" with nothing to choose
};

// gpu_gb: the memory the graphics chip the engine runs on may use (0 when it runs on the CPU); ram_gb: the computer's.
// Only files named as ACE-Step's own releases are ranked ("acestep-5Hz-lm-<size>", "acestep-v15[-xl]-<variant>");
// anything else (a fine-tune under another name) is taken only when nothing ranked is there.
music_choice choose_music_models(const std::vector<music_file> & lms, const std::vector<music_file> & dits,
                                 double gpu_gb, double ram_gb);

} // namespace kera
