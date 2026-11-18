// The C++ guess-ahead controller must make exactly the decisions of the reference code (spec Appendix B).
// Vectors: tests/data/controller_vectors.json, written by tests/python/gen_controller_vectors.py.

#include "kr_test.h"

#include "spec/kr_controller.h"

#include <nlohmann/json.hpp>

#include <fstream>

static nlohmann::json load_vectors() {
    std::ifstream f(KERA_TEST_DATA "/controller_vectors.json");
    return nlohmann::json::parse(f);
}

struct bytes_args {
    double base, growth, ram_base, ram_growth;
};

static double new_bytes(void * u, int k) {
    const bytes_args * a = (const bytes_args *) u;
    return a->base * (1 + a->growth * k);
}

static double new_ram_bytes(void * u, int k) {
    const bytes_args * a = (const bytes_args *) u;
    return a->ram_base * a->ram_growth * k;
}

KR_TEST(choose_k_matches_reference) {
    const auto v = load_vectors();
    KR_CHECK(v["choose_k"].size() >= 100);
    for (const auto & c : v["choose_k"]) {
        bytes_args args{ c["base"], c["growth"], c["ram_base"], c["ram_growth"] };
        const bool moe = c["moe_ram"];
        double rate = 0.0;
        const int k = kr_choose_k_ex(c["a"], c["t_draft"], c["t_check"], new_bytes, &args, c["ssd"], c["k_max"],
                                     moe ? new_ram_bytes : nullptr, &args, moe ? (double) c["ram"] : 0.0, &rate);
        KR_CHECK(k == c["k"].get<int>());
        KR_CHECK_NEAR(rate, c["rate"].get<double>(), 1e-9 * c["rate"].get<double>());
    }
}

KR_TEST(acceptance_matches_reference) {
    const auto v = load_vectors();
    kr_acceptance t;
    kr_acceptance_init(&t, 0.6, 0.1);
    for (const auto & s : v["acceptance"]) {
        const double a = kr_acceptance_update(&t, s["guessed"], s["accepted"]);
        KR_CHECK_NEAR(a, s["a"].get<double>(), 1e-12);
    }
}

KR_TEST(ktuner_matches_reference_step_by_step) {
    const auto v = load_vectors()["ktuner"];
    kr_ktuner * t = kr_ktuner_create(v["k_start"], v["k_max"], v["trial_tokens"], v["period"], v["max_period"],
                                     v["skip_zero_gain"], v["zero_every"]);
    const auto & ks = v["ks"];
    const auto & tokens = v["tokens"];
    const auto & seconds = v["seconds"];
    int mismatches = 0;
    for (size_t i = 0; i < ks.size(); ++i) {
        const int k = kr_ktuner_next_k(t);
        if (k != ks[i].get<int>()) {
            mismatches++;
        }
        kr_ktuner_record(t, tokens[i], seconds[i]);
    }
    KR_CHECK(mismatches == 0);
    KR_CHECK(kr_ktuner_current_k(t) == v["final_k"].get<int>());
    kr_ktuner_destroy(t);
}

KR_TEST(tokens_per_check_edges) {
    KR_CHECK_NEAR(kr_tokens_per_check(1.0, 4), 5.0, 1e-12);
    KR_CHECK_NEAR(kr_tokens_per_check(0.0, 4), 1.0, 1e-12);
    KR_CHECK_NEAR(kr_tokens_per_check(0.5, 1), 1.5, 1e-12);
}

KR_TEST_MAIN()
