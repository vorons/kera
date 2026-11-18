/* Guess-ahead controller (spec Appendix B, P3): picks how many tokens to guess, 0 = off.
 *
 * kr_choose_k is only the starting guess (the formula in Appendix B). After that, kr_ktuner measures
 * real speed with the test-and-set method from Cascade (arXiv 2506.20675) and has the final say.
 * Ported from reference/controller.py; tests/cpp/test_controller.cpp checks it against that code.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes a check must read for k guesses: from the SSD (new_bytes) or, for MoE models, extra bytes
 * from RAM (new_ram_bytes, NULL for dense models). */
typedef double (*kr_bytes_fn)(void * user, int k);

/* Expected tokens per check when each guess is right with chance a. */
double kr_tokens_per_check(double a, int k);

/* Appendix B choose_k: the k (0..k_max) with the most tokens per second. ssd_gbps and ram_gbps are
 * in GB/s. Returns k and writes the predicted tokens per second to *rate if rate is not NULL. */
int kr_choose_k_ex(double a, double t_draft, double t_check,
                   kr_bytes_fn new_bytes, void * new_bytes_user, double ssd_gbps,
                   int k_max,
                   kr_bytes_fn new_ram_bytes, void * new_ram_bytes_user, double ram_gbps,
                   double * rate);

/* The P3 shape: the acceptance rate comes from the controller's tracker. */
typedef struct kr_ctrl kr_ctrl;
int kr_choose_k(const kr_ctrl * c, double t_draft, double t_check, kr_bytes_fn new_bytes, double ssd_gbps);

/* Running estimate of the per-token acceptance: accepted / (accepted + rejections), both as
 * exponential moving averages (Appendix B AcceptanceTracker). */
typedef struct {
    double accepted;
    double tested;
    double w;
} kr_acceptance;

void   kr_acceptance_init(kr_acceptance * t, double start, double weight); /* defaults: 0.6, 0.1 */
double kr_acceptance_update(kr_acceptance * t, int guessed, int accepted);
double kr_acceptance_rate(const kr_acceptance * t);

/* Measure, don't only predict (Appendix B KTuner). */
typedef struct kr_ktuner kr_ktuner;

kr_ktuner * kr_ktuner_create(int k_start, int k_max, int trial_tokens, int period, int max_period,
                             double skip_zero_gain, int zero_every);
kr_ktuner * kr_ktuner_create_default(int k_start); /* k_max 8, trial 64, period 512, max 8192, 1.2, 8 */
void        kr_ktuner_destroy(kr_ktuner * t);
int         kr_ktuner_next_k(kr_ktuner * t);                         /* the k to use for the next step */
void        kr_ktuner_record(kr_ktuner * t, int tokens, double seconds); /* what that step produced */
int         kr_ktuner_current_k(const kr_ktuner * t);
/* Start a test at the next step instead of after `period` tokens: for a tuner whose starting k is only a
 * guess (no measurements yet), so a bad start costs one test, not a whole period. */
void        kr_ktuner_test_now(kr_ktuner * t);
/* 1 while a test is trying candidates, 0 otherwise. */
int         kr_ktuner_testing(const kr_ktuner * t);

/* The controller: an acceptance tracker plus a tuner. */
struct kr_ctrl {
    kr_acceptance acceptance;
    int           k_max;
    void        * new_bytes_user;
};

#ifdef __cplusplus
}
#endif
