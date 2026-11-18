// The guesser switch (src/spec/kr_guess_switch.h) on simulated chats: an n-gram guesser that is often
// right and cheap to check, and a model guesser (an MTP head or a draft model) that costs time to run.
// The switch must keep the guessers that pay, at about the best guess length, drop one that slows
// writing down, and never end up meaningfully slower than not guessing.

#include "kr_test.h"

#include "spec/kr_guess_switch.h"

#include <cstdint>
#include <cstdio>

namespace {

const int NGRAM = 9; // the type numbers only need to differ
const int MODEL = 3;

// deterministic on every platform (std::uniform_real_distribution isn't)
struct rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    double next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (double) (s >> 11) * (1.0 / 9007199254740992.0);
    }
};

struct guesser_sim {
    double a;        // chance each guessed word is right
    double t_draft;  // seconds to guess one word (0 for n-gram)
};

struct world {
    double t_base = 0.05;    // one step without guessing
    double t_check = 0.002;  // each extra word checked
    double t_try = 0.0;      // what just trying to guess costs (llama.cpp's server hands each guess to its queue)
    double p_match = 0.3;    // how often the n-gram guesser finds a guess
    guesser_sim ngram{ 0.95, 0.0 };
    guesser_sim model{ 0.6, 0.01 };
    bool has_ngram = true;
};

// accepted words of k guesses, each right with chance a, stopping at the first wrong one
int accept(rng & r, double a, int k) {
    int n = 0;
    while (n < k && r.next() < a) {
        ++n;
    }
    return n;
}

struct result {
    double rate = 0;  // words per second over the run
    double tried = 0; // share of steps that tried guessing, over the last quarter of the run
};

result run(kera::guess_switch & sw, const world & w, int steps, uint64_t seed) {
    rng r;
    r.s ^= seed;
    double now = 0;
    long long words = 0;
    int tried_late = 0;
    for (int i = 0; i < steps; ++i) {
        const bool try_now = sw.step(now);
        double t = w.t_base;
        int got = 1;
        bool guessed = !try_now; // a step that doesn't try guessing just writes one word
        if (try_now) {
            t += w.t_try;
            tried_late += i >= steps * 3 / 4;
        }
        if (try_now && w.has_ngram && r.next() < w.p_match) {
            const int k = sw.ask(NGRAM, true, 64);
            if (k > 0) {
                const int n = accept(r, w.ngram.a, k);
                sw.accepted(n);
                got += n;
                t += w.t_check * k;
                guessed = true;
            }
        }
        if (!guessed) {
            const int k = sw.ask(MODEL, false, 16);
            if (k > 0) {
                const int n = accept(r, w.model.a, k);
                sw.accepted(n);
                got += n;
                t += (w.model.t_draft + w.t_check) * k;
            }
        }
        now += t;
        words += got;
    }
    sw.step(now);
    return result{ words / now, tried_late / (steps / 4.0) };
}

int k_of(const kera::guess_switch & sw, int type) {
    for (const auto & s : sw.stats()) {
        if (s.type == type) {
            return s.k;
        }
    }
    return -1;
}

} // namespace

KR_TEST(keeps_both_guessers_when_they_pay) {
    world w;
    kera::guess_switch sw;
    const result r = run(sw, w, 40000, 1);
    const int k_ngram = k_of(sw, NGRAM);
    const int k_model = k_of(sw, MODEL);
    std::printf("  n-gram k %d, model k %d, %.1f words/s (no guessing: %.1f)\n", k_ngram, k_model, r.rate, 1 / w.t_base);
    // a = 0.95 with cheap checks: 16 or 32 words are best (141 and 143 words/s alone)
    KR_CHECK(k_ngram == 16 || k_ngram == 32);
    // a = 0.6 with 12 ms per guessed word: k = 2 is best (26.5 words/s against 20 without)
    KR_CHECK(k_model >= 1 && k_model <= 3);
    KR_CHECK(r.rate > 1.4 / w.t_base);
}

KR_TEST(drops_a_guesser_that_slows_writing) {
    world w;
    w.has_ngram = false;
    w.model = { 0.2, 0.03 }; // rarely right and slow to run: any k loses
    kera::guess_switch sw;
    const result r = run(sw, w, 40000, 2);
    std::printf("  model k %d, %.1f words/s (no guessing: %.1f)\n", k_of(sw, MODEL), r.rate, 1 / w.t_base);
    KR_CHECK(k_of(sw, MODEL) == 0);
    // never meaningfully slower than not guessing (Gate 6 (b): within 3%), trials included
    KR_CHECK(r.rate >= 0.97 / w.t_base);
}

KR_TEST(settles_into_probing_when_nothing_pays) {
    world w;
    w.t_try = 0.3 * w.t_base;  // trying costs a lot, as on a tiny model (Kera measured 33%)
    w.ngram = { 0.05, 0.0 };
    w.t_check = 0.01;
    w.model = { 0.1, 0.02 };
    kera::guess_switch sw;
    const result r = run(sw, w, 40000, 4);
    std::printf("  n-gram k %d, model k %d, tried %.1f%% of late steps, %.1f words/s (no guessing: %.1f)\n",
                k_of(sw, NGRAM), k_of(sw, MODEL), 100 * r.tried, r.rate, 1 / w.t_base);
    KR_CHECK(k_of(sw, NGRAM) == 0);
    KR_CHECK(k_of(sw, MODEL) == 0);
    KR_CHECK(r.tried < 0.15);              // mostly probing (1 step in 16), tests included
    KR_CHECK(r.rate >= 0.93 / w.t_base);   // even with trying this expensive
}

KR_TEST(nothing_pays_and_trying_costs_what_it_does_on_real_models) {
    world w;
    w.t_try = 0.03 * w.t_base;  // about 0.3 ms on a 10 ms step
    w.ngram = { 0.05, 0.0 };
    w.t_check = 0.01;
    w.model = { 0.1, 0.02 };
    kera::guess_switch sw;
    const result r = run(sw, w, 40000, 5);
    std::printf("  %.1f words/s (no guessing: %.1f)\n", r.rate, 1 / w.t_base);
    KR_CHECK(r.rate >= 0.97 / w.t_base);   // Gate 6 (b): within 3% of not guessing
}

KR_TEST(an_ngram_guesser_that_never_matches_stops_being_tried) {
    world w;
    w.p_match = 0.0;        // the chat never repeats itself, so the n-gram guesser never has a guess
    w.has_ngram = true;
    w.t_try = 0.2 * w.t_base;
    kera::guess_switch sw;
    // no model guesser either: only the n-gram one is set up
    double now = 0;
    int tried = 0;
    for (int i = 0; i < 1000; ++i) {
        tried += sw.step(now) ? 1 : 0;   // nothing is ever asked
        now += w.t_base;
    }
    std::printf("  tried %d of 1000 steps\n", tried);
    KR_CHECK(tried <= 16 + 1000 / 16 + 1);
}

KR_TEST(an_ngram_guess_that_loses_is_dropped_for_the_model) {
    world w;
    w.ngram = { 0.05, 0.0 };  // n-gram guesses almost always wrong
    w.t_check = 0.01;         // and checking costs (a CPU, or a streamed model)
    w.model = { 0.7, 0.005 };
    kera::guess_switch sw;
    run(sw, w, 40000, 3);
    std::printf("  n-gram k %d, model k %d\n", k_of(sw, NGRAM), k_of(sw, MODEL));
    KR_CHECK(k_of(sw, NGRAM) == 0);
    KR_CHECK(k_of(sw, MODEL) >= 1);
}

KR_TEST(a_reply_gap_is_not_a_step) {
    kera::guess_switch sw;
    sw.step(0.0);
    sw.ask(MODEL, false, 8);
    sw.accepted(2);
    sw.step(0.1);  // the first step: 3 words in 0.1 s
    sw.stop();     // the reply ends
    sw.step(100);  // the next reply starts much later: the gap isn't counted
    sw.step(100.1);
    long long steps = 0;
    double seconds = 0;
    for (const auto & s : sw.stats()) {
        steps += s.steps;
        seconds += s.seconds;
    }
    KR_CHECK(steps == 1);
    KR_CHECK(seconds < 1.0);
}

int main() {
    return kr_test::run_all();
}
