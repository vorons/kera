// What Kera learned about each guesser, per model, on this machine (spec 11.1 item 1: the controller picks
// the guesser per model, machine and task). A guesser that runs a model (an MTP head, a draft model)
// costs time every word while it's loaded, even when the switch has it guess nothing, so when its best
// choice was "don't guess", the next `kera serve` leaves it out. Verdicts expire, so it gets tried again.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <mutex>
#include <string>

namespace kera {

struct guess_verdict {
    bool pays = true;
    int64_t time = 0;        // when it was decided (seconds since 1970)
    long long steps = 0;     // the evidence: steps it decided
    double words_per_s = 0;  // its measured speed in those steps
};

class guess_memory {
public:
    explicit guess_memory(std::string path); // a JSON file, e.g. guessers.json in the Kera folder

    // The verdict for a model's guesser, if there is one younger than max_age_days.
    bool get(const std::string & model, const std::string & guesser, guess_verdict & out, int max_age_days = 14) const;

    // Record a verdict and save the file.
    void set(const std::string & model, const std::string & guesser, const guess_verdict & v);

private:
    std::string path_;
    nlohmann::json j_;
    mutable std::mutex mu_;
};

// Names a model file for the memory: its file name and size (cheap, and changes when the file does)
std::string guess_memory_key(const std::string & model_path);

} // namespace kera
