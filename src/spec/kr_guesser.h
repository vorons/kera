/* Guessers (spec P3, section 11.1): every way of guessing the next words sits behind one interface,
 * so the guess-ahead controller (kr_controller.h) can pick between them by measured speed:
 * MTP heads, DFlash, EAGLE-3, a small draft model, the SB4 base, prompt copy, SuffixDecoding,
 * lookahead and Token Recycling. Frozen in M0; guessers are built in M4.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The engine state a guesser may read: the accepted tokens so far. */
typedef struct {
    const int32_t * tokens;
    int             n_tokens;
    void          * engine;   /* opaque; for guessers that run a model */
} kr_ctx;

/* For MoE models: the experts each guessed token is expected to use, per layer, so the store can
 * start reading them early (section 10.5). A guesser that can't tell leaves n_layers at 0. */
typedef struct {
    int       n_layers;
    int       n_used;      /* experts used per token per layer */
    int32_t * experts;     /* [k][n_layers][n_used], filled by propose() */
} kr_routes;

typedef struct {
    const char * name;
    /* Propose up to k tokens after ctx. Returns how many were written to out_tok (0 = no guess).
     * out_routes may be NULL. */
    int  (*propose)(void * self, const kr_ctx * ctx, int k, int32_t * out_tok, kr_routes * out_routes);
    /* After the check: how many of the proposed tokens the full model accepted. */
    void (*accepted)(void * self, int n_accepted);
} kr_guesser;

#ifdef __cplusplus
}
#endif
