#pragma once

// Kera hook (listed in Kera's docs/UPSTREAM_PATCHES.md): an optional controller that picks how many
// tokens to guess at each step of speculative decoding, 0 meaning "don't guess this step", from the
// acceptance and the speed it measures. Without one set, the server behaves exactly as upstream.

#include "llama.h"

struct common_spec_controller {
    virtual ~common_spec_controller() = default;

    // Called at every generation step of a sequence that could guess: how many tokens to guess now
    // (0..n_max). The time between two calls for the same sequence is how long a step took.
    virtual int32_t draft_len(llama_seq_id seq, int32_t n_max) = 0;

    // Called for each guesser, in llama.cpp's order (n-gram guessers first, then draft models), while a
    // step's guess is made. `type` is the guesser's common_speculative_type.
    //   - A guesser that runs a model is asked before it runs (drafted == false): how many tokens it may
    //     guess (0 = skip it this step, so the next guesser may guess instead).
    //   - An n-gram guesser costs nothing to run, so it guesses first and is asked after (drafted ==
    //     true, n_max = how many it guessed): how many of those to keep (0 = drop them all, so the next
    //     guesser may guess instead).
    // The default keeps upstream's behaviour.
    virtual int32_t guesser_len(llama_seq_id seq, int32_t type, bool drafted, int32_t n_max) {
        (void) seq; (void) type; (void) drafted;
        return n_max;
    }

    // The step's check: n_drafted tokens were guessed and n_accepted of them were right.
    virtual void accepted(llama_seq_id seq, int32_t n_drafted, int32_t n_accepted) = 0;

    // The sequence stopped generating (its next call starts a new measurement).
    virtual void reset(llama_seq_id seq) = 0;
};

// Set before the server starts; nullptr (the default) keeps upstream's fixed draft length.
void                     common_set_spec_controller(common_spec_controller * ctrl);
common_spec_controller * common_get_spec_controller();
