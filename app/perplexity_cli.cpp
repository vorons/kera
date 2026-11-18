// `kera perplexity`: llama.cpp's perplexity and KL-divergence tool (spec section 6 quality tests), with Kera's
// streaming options, so a model whose experts are streamed can be checked against running it from memory:
//   --stream off|os|kera, --stream-budget-gb N
// Every other option is llama.cpp's.

#include "app/kera_app.h"
#include "stream/kr_stream.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

int llama_perplexity(int argc, char ** argv);

namespace kera {

int run_perplexity_cli(int argc, char ** argv) {
    std::vector<char *> args;
    stream::config sc;
    sc.mode = stream::cache_mode::off;
    std::string model;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--stream" && i + 1 < argc) {
            if (!stream::parse_mode(argv[++i], sc.mode)) {
                std::fprintf(stderr, "--stream takes off, os or kera\n");
                return 2;
            }
            continue;
        }
        if (a == "--stream-budget-gb" && i + 1 < argc) {
            sc.budget_bytes = (uint64_t) (std::atof(argv[++i]) * 1e9);
            continue;
        }
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            model = argv[i + 1];
        }
        args.push_back(argv[i]);
    }
    std::shared_ptr<stream::session> streaming;
    if (sc.mode != stream::cache_mode::off) {
        sc.model_path = model;
        std::string err;
        streaming = stream::start(sc, &err);
        if (!streaming) {
            std::fprintf(stderr, "Kera: can't stream the experts: %s\n", err.c_str());
            return 2;
        }
    }
    const int rc = llama_perplexity((int) args.size(), args.data());
    if (streaming) {
        // totals: perplexity reads the text in batches, so "per word" doesn't apply
        const stream::report r = streaming->snapshot();
        const double asked = (double) std::max<uint64_t>(1, r.cache.requests);
        std::fprintf(stderr, "Kera: streaming over this run, %.0f MB read from the SSD, %.1f%% of the experts used were "
                             "already in memory", (double) r.cache.bytes_read / 1e6, 100.0 * (double) r.cache.hits / asked);
        std::fprintf(stderr, "\n");
        stream::stop();
    }
    return rc;
}

} // namespace kera
