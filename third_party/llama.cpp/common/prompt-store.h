#pragma once

// Kera hook (listed in Kera's docs/UPSTREAM_PATCHES.md): an optional second tier for the server's RAM
// prompt cache, such as a store on the SSD, so chats survive restarts and old chats don't need
// re-reading. Without a store set, the server behaves exactly as upstream. The setting lives in this
// shared library so the program and the server library see the same one.

#include "common.h"

#include <list>
#include <vector>

struct common_saved_prompt {
    std::vector<llama_token>             tokens;      // text-only prompts; prompts with images aren't saved
    std::list<common_prompt_checkpoint>  checkpoints; // what hybrid and sliding-window models need to go back
    std::vector<uint8_t>                 state_tgt;   // llama_state_seq_get_data_ext(..., LLAMA_STATE_SEQ_FLAGS_NONE)
    std::vector<uint8_t>                 state_dft;   // the draft model's, if any
};

struct common_prompt_store {
    virtual ~common_prompt_store() = default;

    // A prompt's state is worth keeping (it went into the RAM cache, or a reply just finished). Called
    // on the server thread with a copy the store owns, so it can write it in the background.
    virtual void save(common_saved_prompt && prompt) = 0;

    // Nothing in RAM shares more than min_lcp tokens with tokens_new. Fill `out` with the saved prompt
    // that shares the longest start with tokens_new, if that start is longer than min_lcp tokens.
    virtual bool load(const std::vector<llama_token> & tokens_new, size_t min_lcp, common_saved_prompt & out) = 0;

    // Once the model and its context are loaded: what, besides the model file, decides the numbers a saved state
    // holds (the settings as parsed, whether they came from the command line, LLAMA_ARG_* variables or a preset),
    // and the files those settings name (adapters, control vectors, a draft model), whose contents count too. A store
    // must keep states apart that differ in any of it.
    virtual void set_identity(const std::string & settings, const std::vector<std::string> & files) {
        (void) settings;
        (void) files;
    }
};

// Set before the server starts; nullptr (the default) turns the second tier off.
void                  common_set_prompt_store(common_prompt_store * store);
common_prompt_store * common_get_prompt_store();

// [Kera] hook 8: settings a program fills in where nothing else set them. Called by the server once it has read every
// source of settings (the command line, LLAMA_ARG_* variables, config.ini, presets), before it loads the model, with
// the settings it will use. nullptr (the default) leaves them exactly as upstream.
using common_params_fill = void (*)(common_params & params);
void               common_set_params_fill(common_params_fill fill);
common_params_fill common_get_params_fill();
