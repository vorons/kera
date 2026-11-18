#pragma once

// Kera hook (listed in Kera's docs/UPSTREAM_PATCHES.md, hook 4): the sampling a model's card recommends
// for a request that doesn't set its own, which can depend on whether the model thinks first and on
// whether tools are offered (Nemotron 3 Nano: one setting for each). Without one set, the server keeps
// its own defaults (the command line's, or the GGUF file's general.sampling values).

#include <string>
#include <utility>
#include <vector>

struct common_sampling_defaults {
    float temp  = -1.0f; // < 0: keep the server's
    float top_p = -1.0f;
    int   top_k = -1;
    float min_p = -1.0f;
    float penalty_repeat  = -1.0f;
    float penalty_present = -1.0f;
    float penalty_freq    = -1.0f;
    std::vector<std::string> stop; // added to a request's stop words where the request gives none
};

struct common_request_defaults {
    virtual ~common_request_defaults() = default;

    // Fill in d for a chat request; return false to keep the server's defaults for it.
    virtual bool sampling(bool thinking, bool tools, common_sampling_defaults & d) = 0;

    // Messages to put before a chat request's own, as (role, content), the way the model's packaging adds them (an
    // Ollama Modelfile's SYSTEM where the request has no system message, and its MESSAGE pairs). None by default.
    virtual std::vector<std::pair<std::string, std::string>> leading_messages(bool request_has_system) {
        (void) request_has_system;
        return {};
    }
};

// Set before the server starts; nullptr (the default) keeps upstream's behaviour.
void                      common_set_request_defaults(common_request_defaults * rd);
common_request_defaults * common_get_request_defaults();
