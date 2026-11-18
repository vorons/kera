// The guesser switch (kr_guess_switch.h).

#include "spec/kr_guess_switch.h"

#include "spec/kr_controller.h"

#include <algorithm>

namespace kera {

guess_switch::guess_switch() : guess_switch(options()) {}

guess_switch::guess_switch(const options & o) : o_(o) {}

guess_switch::~guess_switch() {
    for (auto & [type, g] : guessers_) {
        kr_ktuner_destroy(g.tuner);
    }
}

guess_switch::guesser & guess_switch::get(int type, bool free) {
    auto it = guessers_.find(type);
    if (it != guessers_.end()) {
        return it->second;
    }
    guesser & g = guessers_[type];
    g.free = free;
    g.st.type = type;
    g.st.free = free;
    const auto & ladder = free ? o_.free_ladder : o_.model_ladder;
    const int top = std::max(0, (int) ladder.size() - 1);
    const int start = std::min(top, free ? o_.free_start : o_.model_start);
    g.tuner = kr_ktuner_create(start, top, o_.trial_tokens, o_.period, o_.max_period, o_.skip_zero_gain, o_.zero_every);
    kr_ktuner_test_now(g.tuner); // the start is only a guess: measure at once
    return g;
}

int guess_switch::ladder_k(const guesser & g, int index) const {
    const auto & ladder = g.free ? o_.free_ladder : o_.model_ladder;
    if (ladder.empty()) {
        return 0;
    }
    return ladder[(size_t) std::clamp(index, 0, (int) ladder.size() - 1)];
}

bool guess_switch::step(double now) {
    if (running_ && tried_) {
        unasked_ = owner_ >= 0 ? 0 : unasked_ + 1; // tried, but no guesser had anything to say
    }
    if (running_ && owner_ >= 0) {
        guesser & g = guessers_.at(owner_);
        const double dt = std::max(1e-9, now - start_);
        kr_ktuner_record(g.tuner, tokens_, dt);
        g.st.steps += 1;
        g.st.tokens += tokens_;
        g.st.seconds += dt;
    }
    running_ = true;
    start_ = now;
    tokens_ = 1; // every step writes at least one word; accepted guesses add to it
    owner_ = -1;
    owner_final_ = false;

    // until a guesser shows up (an n-gram guesser only does when it finds a guess), keep trying for a while
    bool wanted = guessers_.empty() && unasked_ < o_.probe_every;
    for (const auto & [type, g] : guessers_) {
        wanted = wanted || kr_ktuner_testing(g.tuner) || ladder_k(g, kr_ktuner_current_k(g.tuner)) > 0;
    }
    tried_ = wanted || ++idle_steps_ >= o_.probe_every;
    if (tried_) {
        idle_steps_ = 0;
    }
    return tried_;
}

int guess_switch::ask(int type, bool drafted, int n_max) {
    guesser & g = get(type, drafted);
    const int k = std::min(std::max(0, n_max), ladder_k(g, kr_ktuner_next_k(g.tuner)));
    if (!owner_final_) {
        // an n-gram guesser with a guess decides the step whatever it chose; a model guesser decides it if
        // it guesses, and otherwise only until a later guesser does
        owner_ = type;
        owner_final_ = drafted || k > 0;
    }
    return k;
}

void guess_switch::accepted(int n_accepted) {
    tokens_ += std::max(0, n_accepted);
}

void guess_switch::stop() {
    running_ = false;
    tried_ = false;
}

std::vector<guess_switch::guesser_stats> guess_switch::stats() const {
    std::vector<guesser_stats> out;
    for (const auto & [type, g] : guessers_) {
        guesser_stats s = g.st;
        s.k = ladder_k(g, kr_ktuner_current_k(g.tuner));
        s.testing = kr_ktuner_testing(g.tuner) != 0;
        out.push_back(s);
    }
    return out;
}

} // namespace kera
