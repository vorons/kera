// The prompt improver (the owner's request for 0.1.2, docs/DECISIONS.md D-030): a chat model rewrites a short
// picture or video description into the fuller one image models do best with, keeping everything the user said.
// Opt-in (--improve-prompt MODEL.gguf): the user's words and the rewrite are both shown.
//
// Kera never changes how a model answers (the owner's first rule, D-033): the chat model is asked through Kera's own
// chat server, exactly as a chat app asks it, so it uses its own chat template, its card's sampling (D-013), thinks as
// it does in a chat, with a new random seed each time, and its whole answer is used, unedited. Kera only makes it
// quicker: a model the app already has loaded is asked where it is, with no loading at all, and the chat server's own
// speed-ups apply (guessing ahead; its instructions kept on the SSD, so the next rewrite doesn't read them again).

#include "app/kera_app.h"
#include "common/kr_paths.h"
#include "app/model_runner.h"

#include <cpp-httplib/httplib.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace kera {

namespace {

// Kera's own instructions (not copied from another project), in two styles: image models that read sentences
// (FLUX, Z-Image, Qwen-Image, SD3, video models) and the older ones that read comma-separated tags (SD 1.x/2.x, SDXL)
const char * k_sentences =
    "You rewrite descriptions for an image generator. Keep every subject, object, colour, count, name, style and "
    "detail the user gave, and contradict none of them. Add what makes a strong picture where the user left it open: "
    "the setting, composition and camera view, lighting, mood, materials and textures, and the style or medium if "
    "none was given. Write one paragraph of two to four plain, concrete sentences, at most 80 words, in the user's "
    "language. "
    "Add no text, letters or signs to the picture unless the user asked for them. Reply with the description only: "
    "no title, no quotes, no explanation.";
const char * k_tags =
    "You rewrite descriptions for a Stable Diffusion image generator, which reads short comma-separated phrases. "
    "Keep every subject, object, colour, count, name, style and detail the user gave, and contradict none of them. "
    "Add phrases for what makes a strong picture where the user left it open: setting, composition, camera view, "
    "lighting, mood, style or medium, and quality words such as highly detailed. Most important first, at most 40 "
    "words, in English. Add no text or signs unless asked. Reply with the phrases only, on one line, no sentences "
    "and no explanation. For example, for \"a cat on a sofa\" reply: a fluffy grey cat curled up on a velvet sofa, "
    "cozy living room, warm afternoon light from a window, soft shadows, shallow depth of field, photo, highly "
    "detailed";

// the server's answer to one chat request: its "content" (the answer, without the thinking the server keeps apart, as
// for any chat app)
bool ask_server(int port, const std::string & system, const std::string & user, std::string & out, std::string & err) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(3600, 0); // a thinking model on a slow computer may take a while
    // nothing about how it answers is set, the seed included: the server uses the model's own settings and a new
    // random seed, as for any chat app, so the same words can give a different rewrite each time (D-033)
    const nlohmann::json body = {
        { "messages", nlohmann::json::array({ { { "role", "system" }, { "content", system } },
                                              { { "role", "user" }, { "content", user } } }) },
        { "stream", false },
    };
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    if (!r) {
        err = "the chat model's server didn't answer";
        return false;
    }
    const nlohmann::json j = nlohmann::json::parse(r->body, nullptr, false);
    if (r->status != 200 || !j.is_object() || !j.contains("choices") || !j["choices"].is_array() || j["choices"].empty()) {
        err = "the chat model's server said: " + r->body.substr(0, 300);
        return false;
    }
    const nlohmann::json & c = j["choices"][0];
    if (c.value("finish_reason", std::string()) == "length") {
        err = "the chat model kept writing until its context was full, without finishing its answer, so there is no "
              "complete rewrite to use (Kera doesn't cut a model short): try another chat model";
        return false;
    }
    const nlohmann::json msg = c.value("message", nlohmann::json::object());
    out = msg.contains("content") && msg["content"].is_string() ? msg["content"].get<std::string>() : std::string();
    // only the blank space around it removed
    const size_t first = out.find_first_not_of(" \t\r\n");
    const size_t last = out.find_last_not_of(" \t\r\n");
    out = first == std::string::npos ? std::string() : out.substr(first, last - first + 1);
    if (out.empty()) {
        err = "the chat model gave no answer";
        return false;
    }
    return true;
}

// The port of the app's model, if the app has this very model loaded and ready: then it is asked there
int app_port_for(const std::string & model_path) {
    httplib::Client cli("127.0.0.1", 8341);
    cli.set_connection_timeout(0, 300000);
    cli.set_read_timeout(2, 0);
    auto r = cli.Get("/api/run");
    if (!r || r->status != 200) {
        return 0;
    }
    const nlohmann::json j = nlohmann::json::parse(r->body, nullptr, false);
    if (!j.is_object() || j.value("state", std::string()) != "ready") {
        return 0;
    }
    std::error_code ec;
    if (!fs::equivalent(fs::u8path(j.value("model", std::string())), fs::u8path(model_path), ec) || ec) {
        return 0;
    }
    const std::string url = j.value("url", std::string()); // http://127.0.0.1:PORT/v1
    const size_t colon = url.rfind(':');
    return colon == std::string::npos ? 0 : std::atoi(url.c_str() + colon + 1);
}

bool ask_chat_model(const std::string & model_path, const std::string & system, const std::string & user,
                    std::string & out, std::string & err) {
    if (const int port = app_port_for(model_path); port > 0) {
        std::fprintf(stderr, "Kera: asking the chat model the app already has loaded\n");
        return ask_server(port, system, user, out, err);
    }
    // otherwise Kera's chat server for it, for this one question (its own log; not remembered as the app's model)
    model_runner runner("prompt-improver.log", false);
    if (!runner.start(model_path, &err)) {
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        const model_runner::status st = runner.get();
        if (st.st == model_runner::state::ready) {
            return ask_server(st.port, system, user, out, err);
        }
        if (st.st == model_runner::state::failed || st.st == model_runner::state::stopped) {
            err = "the chat model didn't load: " + st.error;
            return false;
        }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::minutes(30)) {
            err = "the chat model took more than 30 minutes to load";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

} // namespace

bool improve_prompt(const std::string & chat_model, bool tag_style, const std::string & prompt, std::string & out,
                    std::string & err) {
    std::error_code ec;
    if (!fs::is_regular_file(fs::u8path(chat_model), ec)) {
        err = "no chat model at " + chat_model;
        return false;
    }
    // 0.1.2 kept its rewrites here and gave the same one back for the same words; a chat doesn't (D-033)
    fs::remove(fs::u8path(home_path("prompt-improver.json")), ec);
    std::fprintf(stderr, "Kera: improving the description with %s\n", fs::u8path(chat_model).filename().u8string().c_str());
    return ask_chat_model(chat_model, tag_style ? k_tags : k_sentences, prompt, out, err);
}

} // namespace kera
