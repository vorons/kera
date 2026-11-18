// The guesser switch (spec 11.1 item 1, M4): with several guessers set up, picks at every step which one
// guesses and how far, by measured speed. llama.cpp asks its guessers in a fixed order (n-gram guessers
// first, because they cost nothing to run, then draft models and heads). Each guesser gets its own
// Appendix B KTuner, which measures the real words per second of its choices, 0 included; 0 means "let
// the next guesser (or none) go instead". So each guesser learns, in the steps where it applies, whether
// it beats what would happen without it.
//
// A step's measurement belongs to the first guesser whose choice decided it:
//   - an n-gram guesser that found a guess (it guesses before it is asked, see hook 2), whatever it
//     chose, even 0 (its 0 then measures what the next guessers do in the steps where it had a guess);
//   - otherwise the first model guesser that chose to guess;
//   - otherwise the last model guesser asked (its 0 meant no guess at all).
//
// One switch per chat. Not thread-safe; the caller locks.
#pragma once

#include <map>
#include <vector>

struct kr_ktuner;

namespace kera {

class guess_switch {
public:
    struct options {
        // the guess lengths each kind of guesser can be given, index 0 being "don't guess"
        std::vector<int> model_ladder = { 0, 1, 2, 3, 4, 5, 6, 7, 8 }; // Appendix B's k_max of 8
        std::vector<int> free_ladder = { 0, 4, 8, 16, 32, 64 };          // n-gram guesses cost nothing to make
        // where each guesser starts (an index into its ladder): low, because the opening test tries the
        // neighbours of the start, and a start too high costs a lot whenever guessing doesn't pay
        int model_start = 2;  // k = 2
        int free_start = 2;   // k = 8
        int trial_tokens = 64;
        int period = 512;
        int max_period = 8192;
        double skip_zero_gain = 1.2;
        int zero_every = 8;
        // While every guesser's choice is "don't guess" and none is testing, guessing is tried only every
        // this many steps: just trying costs a little each step (llama.cpp's server hands each guess to
        // its task queue), and these steps still let a guesser notice when guessing would pay again.
        int probe_every = 16;
    };

    guess_switch();
    explicit guess_switch(const options & o);
    ~guess_switch();
    guess_switch(const guess_switch &) = delete;
    guess_switch & operator=(const guess_switch &) = delete;

    // A generation step starts at `now` (seconds, any steady clock); the previous step ends there.
    // Returns whether to try guessing in this step (see options::probe_every).
    bool step(double now);

    // Hook 2's guesser_len: a model guesser asked before it runs (drafted false: how far may it guess,
    // 0 = skip it), or an n-gram guesser asked after it guessed `n_max` tokens (drafted true: how many to
    // keep, 0 = drop them).
    int ask(int type, bool drafted, int n_max);

    // The step's check accepted this many guesses.
    void accepted(int n_accepted);

    // The reply ended: the time until the next step() isn't a step.
    void stop();

    struct guesser_stats {
        int type = 0;
        bool free = false;
        long long steps = 0;     // steps whose measurement it owned
        long long tokens = 0;    // words those steps wrote
        double seconds = 0;
        int k = 0;               // the guess length it uses now
        bool testing = false;    // in the middle of a test
    };
    std::vector<guesser_stats> stats() const;

private:
    struct guesser {
        kr_ktuner * tuner = nullptr;
        bool free = false;
        guesser_stats st;
    };
    guesser & get(int type, bool free);
    int ladder_k(const guesser & g, int index) const;

    options o_;
    std::map<int, guesser> guessers_;
    bool running_ = false;
    double start_ = 0;
    int tokens_ = 0;
    int owner_ = -1;          // type that owns this step's measurement (-1: none yet)
    bool owner_final_ = false;
    long long idle_steps_ = 0; // steps since guessing was last tried while no guesser wanted to
    bool tried_ = false;       // this step tries guessing
    int unasked_ = 0;          // tried steps in a row in which no guesser had a guess
};

} // namespace kera
