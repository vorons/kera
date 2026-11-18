// The settings a model's own card recommends (spec 4c item 2), for the model families whose cards were
// checked (ground rule 5; the sources are in docs/DECISIONS.md D-013). Families not listed keep the
// server's defaults, which come from the GGUF file's own general.sampling values when it has them.
#pragma once

#include <string>
#include <vector>

namespace kera {

enum class model_family {
    unknown,
    nemotron3,  // NVIDIA Nemotron 3 Nano 30B-A3B: hybrid Mamba-2 + attention + MoE (not Super, Ultra, 3.5 or Omni)
    qwen3,      // Qwen3 with the thinking switch, and the 2507 Instruct / Thinking releases (2507 Instruct can't think:
                // its template has no thinking section, so it always gets the non-thinking values)
    qwen3_coder,
};

const char * family_name(model_family f);

// From the GGUF header's general.architecture and general.name. Only the models whose cards were read match (by
// design and by name): another model built on the same design (a fine-tune, a distill, an older generation) is
// unknown and keeps the server's own defaults, as in stock llama.cpp.
model_family family_of(const std::string & arch, const std::string & name);

struct sampling_values {
    float temp = -1.0f; // < 0: no recommendation
    float top_p = -1.0f;
    int top_k = -1;
    float min_p = -1.0f;
    float repeat = -1.0f; // the repetition penalty, where the card names one
    // top_k and min_p filled from the makers' own serving engine where the card names none (0: off, as there; the
    // owner's choice, 9 Nov): used for requests, but not compared with a file's own values
    bool top_k_from_engine = false;
    bool min_p_from_engine = false;
};

// What the card recommends for a chat request: the model thinking first or not, tools offered or not.
// Returns false when it recommends nothing for this case.
bool recommended_sampling(model_family f, const std::string & name, bool thinking, bool tools, sampling_values & out);

// Qwen's own GGUF cards for Qwen3 add, for thinking and non-thinking alike: "We recommend setting presence_penalty to
// 1.5 for quantized models to suppress repetitive outputs". Not for the 2507 releases (their cards leave it optional, and
// Qwen published no GGUFs of them) nor full-precision files. < 0: none.
float recommended_presence_penalty(model_family f, const std::string & name, bool quantized);

// Whether the file's own sampling defaults (general.sampling.*, < 0 where absent) are one of the card's settings, as
// they are in the card's model's own files (its generation_config.json holds one of them). A file that says otherwise
// is another model under the same name (a fine-tune with its own defaults): its own values are kept.
bool card_agrees_with_file(model_family f, const std::string & name, const sampling_values & file);

// A model's own default sampling from its makers' generation_config.json (tools/cards: what transformers and vLLM
// apply when a request sets none, and what llama.cpp's converter writes into a file as general.sampling.*), for the
// models listed in model_cards.inc. Found by the model's name and shape together: one of `names` (general.name, the
// name's parts, the file's name), compared without case and separators, must be the makers' repository's name, and the
// layer count and width must be that model's. Only the values the makers name are set (< 0: left to the engine).
// repo: the makers' repository, for the log.
bool listed_sampling(const std::vector<std::string> & names, int n_layer, int n_embd, sampling_values & out,
                     std::string & repo);

} // namespace kera
