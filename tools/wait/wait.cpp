// `kera wait`: where the wait before a reply goes (spec 4c item 3): loading, reading the message,
// thinking, writing. Sends one chat message to an OpenAI-compatible server (kera serve, or the owner's
// own app's server), reads the streamed reply, and explains the times in plain words.
//
//   kera wait [--url http://127.0.0.1:8080] [--message TEXT | --file FILE] [--thinking on|off] [--repeat N]

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using nlohmann::json;
using clock_type = std::chrono::steady_clock;

namespace {

struct wait_result {
    double first_byte_s = -1;    // the server answered at all (loading shows up here)
    double first_token_s = -1;   // any word, thinking included
    double first_visible_s = -1; // the first word of the answer itself
    double end_s = 0;
    int n_thinking_chunks = 0;
    int n_answer_chunks = 0;
    json timings;
    std::string error;
};

wait_result measure(const std::string & url, const json & body) {
    wait_result r;
    httplib::Client cli(url);
    cli.set_read_timeout(600, 0);
    std::string buf;
    const auto t0 = clock_type::now();
    auto secs = [&] { return std::chrono::duration<double>(clock_type::now() - t0).count(); };
    httplib::Headers headers = { { "Accept", "text/event-stream" } };
    auto res = cli.Post("/v1/chat/completions", headers, body.dump(), "application/json",
        [&](const char * data, size_t len) {
            if (r.first_byte_s < 0) {
                r.first_byte_s = secs();
            }
            buf.append(data, len);
            size_t nl;
            while ((nl = buf.find('\n')) != std::string::npos) {
                std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                if (line.rfind("data: ", 0) != 0 || line == "data: [DONE]") {
                    continue;
                }
                try {
                    const json j = json::parse(line.substr(6));
                    if (j.contains("timings")) {
                        r.timings = j["timings"];
                    }
                    if (!j.contains("choices") || j["choices"].empty()) {
                        continue;
                    }
                    const json & d = j["choices"][0].value("delta", json::object());
                    const bool thinking = d.contains("reasoning_content") && d["reasoning_content"].is_string() &&
                                          !d["reasoning_content"].get<std::string>().empty();
                    const bool answer = d.contains("content") && d["content"].is_string() &&
                                        !d["content"].get<std::string>().empty();
                    if ((thinking || answer) && r.first_token_s < 0) {
                        r.first_token_s = secs();
                    }
                    if (thinking) {
                        r.n_thinking_chunks++;
                    }
                    if (answer) {
                        r.n_answer_chunks++;
                        if (r.first_visible_s < 0) {
                            r.first_visible_s = secs();
                        }
                    }
                } catch (const std::exception &) {
                }
            }
            return true;
        });
    r.end_s = secs();
    if (!res) {
        r.error = "can't reach " + url + " (" + httplib::to_string(res.error()) + ")";
    } else if (res->status != 200) {
        r.error = "the server answered " + std::to_string(res->status);
    }
    return r;
}

void explain(const wait_result & r, bool capped) {
    if (!r.error.empty()) {
        std::printf("%s\n", r.error.c_str());
        return;
    }
    const json & t = r.timings;
    const double prompt_s = t.value("prompt_ms", 0.0) / 1000.0;
    const int prompt_n = t.value("prompt_n", 0);
    const int cache_n = t.value("cache_n", 0);
    const double gen_tps = t.value("predicted_per_second", 0.0);
    const double before_reading = r.first_token_s >= 0 ? std::max(0.0, r.first_token_s - prompt_s) : 0.0;

    std::printf("Where the wait went:\n");
    if (before_reading > 0.5) {
        std::printf("  - %.1f s before Kera started reading (loading the model, or another request first)\n", before_reading);
    }
    if (!t.empty()) {
        std::printf("  - %.2f s reading your message: %d new tokens", prompt_s, prompt_n);
        if (cache_n > 0) {
            std::printf(", and %d tokens of the chat it had already read were reused", cache_n);
        }
        std::printf("\n");
    }
    if (r.first_visible_s >= 0 && r.first_token_s >= 0 && r.n_thinking_chunks > 0) {
        std::printf("  - %.1f s thinking before the answer started (about %d words of thinking)\n",
                    r.first_visible_s - r.first_token_s, r.n_thinking_chunks);
    }
    if (r.first_visible_s >= 0) {
        std::printf("  - the first word of the answer appeared after %.2f s\n", r.first_visible_s);
    } else if (r.n_thinking_chunks > 0) {
        std::printf("  - the reply was all thinking: no answer appeared%s\n", capped ? " within the word limit (--max-tokens)" : "");
    }
    if (gen_tps > 0) {
        std::printf("  - then it wrote %.1f words a second (about %.0f words of text a minute)\n", gen_tps, gen_tps * 45);
    }
    std::printf("Total: %.1f s.\n", r.end_s);
    if (r.n_thinking_chunks > 0 && r.first_visible_s - r.first_token_s > 2 * std::max(prompt_s, 0.1)) {
        // rule one (D-033, D-034): thinking is how the model answers well, so Kera never suggests turning it off
        std::printf("Most of the wait was the model thinking before it answered: that is how this model works, and\n"
                    "Kera leaves it as it is.\n");
    } else if (prompt_s > 1.0 && cache_n == 0) {
        std::printf("Most of the wait was reading the whole chat. If this was a follow-up message, the chat's earlier\n"
                    "part wasn't reused: see whether your app changes earlier messages or turns the prompt cache off.\n");
    }
}

} // namespace

int kera_wait(int argc, char ** argv) {
    std::string url = "http://127.0.0.1:8080", message = "Hi! In one sentence, what is a solid-state drive?", thinking;
    int repeat = 1, max_tokens = -1; // no cap unless asked: a capped reply isn't the real wait
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--url") {
            url = argv[i + 1];
        } else if (a == "--message") {
            message = argv[i + 1];
        } else if (a == "--file") {
            std::ifstream f(argv[i + 1]);
            std::stringstream ss;
            ss << f.rdbuf();
            message = ss.str();
        } else if (a == "--thinking") {
            thinking = argv[i + 1];
        } else if (a == "--repeat") {
            repeat = std::max(1, std::atoi(argv[i + 1]));
        } else if (a == "--max-tokens") {
            max_tokens = std::atoi(argv[i + 1]);
        } else {
            std::fprintf(stderr, "Usage: kera wait [--url URL] [--message TEXT | --file FILE] [--thinking on|off] [--repeat N] [--max-tokens N]\n"
                                 "  --thinking on|off   send what your app sends (the model's own default if left out)\n"
                                 "  --max-tokens N      stop the reply after N words (default: no limit, the whole reply)\n");
            return 2;
        }
    }
    json body = { { "messages", json::array({ { { "role", "user" }, { "content", message } } }) },
                  { "stream", true }, { "timings_per_token", false } };
    if (max_tokens > 0) {
        body["max_tokens"] = max_tokens;
    }
    if (thinking == "on" || thinking == "off") {
        body["chat_template_kwargs"] = { { "enable_thinking", thinking == "on" } };
    }
    for (int i = 0; i < repeat; ++i) {
        if (repeat > 1) {
            std::printf("Try %d of %d%s\n", i + 1, repeat, i > 0 ? " (the same message again: its start should be reused)" : "");
        }
        explain(measure(url, body), max_tokens > 0);
    }
    return 0;
}
