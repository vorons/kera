// Which back-end serves each message (src/planner/engine_route.h), with speeds shaped like the Minisforum's on
// Nemotron 3 Nano (16 Oct): ROCm reads a 127K-token document in about 138 s and Vulkan in about 296 s, while Vulkan
// writes about a fifth faster.

#include "kr_test.h"

#include "planner/engine_route.h"

#include <nlohmann/json.hpp>

#include <string>

using namespace kera;

namespace {

backend_speed vulkan() {
    backend_speed s;
    s.backend = "vulkan";
    s.read_s = 0.0005;
    s.read_s_per_depth = 2.88e-8;
    s.write_s = 0.0152;
    s.write_s_per_depth = 6.4e-8;
    return s;
}

backend_speed rocm() {
    backend_speed s;
    s.backend = "hip";
    s.read_s = 0.0004;
    s.read_s_per_depth = 1.09e-8;
    s.write_s = 0.0185;
    s.write_s_per_depth = 5.6e-8;
    return s;
}

std::string chat(const std::string & first, const std::string & reply = "", const std::string & next = "") {
    nlohmann::json m = nlohmann::json::array();
    m.push_back({ { "role", "system" }, { "content", "You are helpful." } });
    m.push_back({ { "role", "user" }, { "content", first } });
    if (!reply.empty()) {
        m.push_back({ { "role", "assistant" }, { "content", reply } });
        m.push_back({ { "role", "user" }, { "content", next } });
    }
    return nlohmann::json{ { "messages", m }, { "stream", true } }.dump();
}

std::string document(size_t bytes) {
    std::string s;
    for (size_t i = 0; s.size() < bytes; ++i) {
        s += "Line " + std::to_string(i) + " of a long report about the year's results. ";
    }
    return s;
}

engine_router both_ready() {
    engine_router r({ vulkan(), rocm() });
    r.set_ready(0, true);
    r.set_ready(1, true);
    return r;
}

} // namespace

KR_TEST(a_chat_message_goes_to_the_faster_writer) {
    engine_router r = both_ready();
    const auto c = r.pick(route_pieces(chat("What's a good name for a cat?"), nullptr), 0, 0);
    KR_CHECK(c.engine == 0);
    KR_CHECK(c.seconds[0] > 0 && c.seconds[0] < c.seconds[1]);
}

KR_TEST(a_long_document_goes_to_the_faster_reader) {
    engine_router r = both_ready();
    const auto c = r.pick(route_pieces(chat(document(450000) + "\nSummarise this."), nullptr), 0, 0);
    KR_CHECK(c.engine == 1);
    KR_CHECK(c.read > 100000);
    KR_CHECK(c.seconds[1] < c.seconds[0] / 1.5);
}

KR_TEST(a_follow_up_stays_where_its_document_was_read) {
    engine_router r = both_ready();
    const std::string doc = document(450000) + "\nSummarise this.";
    const auto first = route_pieces(chat(doc), nullptr);
    const int64_t id = r.started(1, first, 140, 0);
    r.finished(id, 0, 0, 0);
    // the follow-up holds the document and ROCm's own reply: only the new question is read
    const auto next = route_pieces(chat(doc, "The year went well.", "And the costs?"), nullptr);
    const auto c = r.pick(next, 0, 200);
    KR_CHECK(c.engine == 1);
    KR_CHECK(c.read - c.held[1] < 50);
    KR_CHECK(c.held[0] == 0);
    // ...and a new short chat still goes to Vulkan
    KR_CHECK(r.pick(route_pieces(chat("Hello there"), nullptr), 0, 200).engine == 0);
}

KR_TEST(a_busy_engine_counts_against_itself) {
    engine_router r = both_ready();
    const auto doc = route_pieces(chat(document(450000)), nullptr);
    r.started(1, doc, 140, 0);
    // a second, different document while ROCm reads the first: still ROCm, as reading it twice as fast pays more
    // than half the other's remaining time
    const auto other = route_pieces(chat("Another: " + document(450000)), nullptr);
    KR_CHECK(r.pick(other, 0, 10).engine == 1);
    // a medium prompt while ROCm is busy goes to the idle Vulkan
    const auto medium = route_pieces(chat("Read: " + document(20000)), nullptr);
    KR_CHECK(r.pick(medium, 0, 10).engine == 0);
}

KR_TEST(only_ready_engines_are_picked) {
    engine_router r({ vulkan(), rocm() });
    const auto doc = route_pieces(chat(document(450000)), nullptr);
    KR_CHECK(r.pick(doc, 0, 0).engine == -1);
    r.set_ready(0, true);
    KR_CHECK(r.pick(doc, 0, 0).engine == 0); // ROCm still loading: Vulkan rather than waiting
    r.set_ready(1, true);
    KR_CHECK(r.pick(doc, 0, 0).engine == 1);
}

KR_TEST(a_growing_prompt_matches_its_start) {
    // a raw completion prompt that grows: the stretches it had before are held
    engine_router r = both_ready();
    const std::string a = document(30000);
    const auto first = route_pieces(nlohmann::json{ { "prompt", a } }.dump(), nullptr);
    r.started(1, first, 1, 0);
    const auto c = r.pick(route_pieces(nlohmann::json{ { "prompt", a + " More words." } }.dump(), nullptr), 0, 10);
    KR_CHECK(c.held[1] > c.read * 0.9);
}

KR_TEST(the_pieces_of_each_api) {
    double w = 0;
    KR_CHECK(route_pieces(nlohmann::json{ { "system", "be brief" },
                                          { "messages", { { { "role", "user" }, { "content", { { { "type", "text" }, { "text", "hi" } } } } } } },
                                          { "max_tokens", 64 } }.dump(), &w).size() == 2);
    KR_CHECK(w == 64);
    KR_CHECK(route_pieces(nlohmann::json{ { "prompt", "hello" }, { "n_predict", 8 } }.dump(), &w).size() == 1);
    KR_CHECK(w == 8);
    KR_CHECK(route_pieces("not json", &w).empty());
    KR_CHECK(route_pieces(nlohmann::json{ { "model", "x" } }.dump(), &w).empty());
    // the same start gives the same chain; a different start a different one
    const auto a = route_pieces(chat("one"), nullptr);
    const auto b = route_pieces(chat("one", "two", "three"), nullptr);
    const auto c = route_pieces(chat("uno"), nullptr);
    KR_CHECK(a[0].chain == b[0].chain && a[1].chain == b[1].chain);
    KR_CHECK(a[1].chain != c[1].chain);
    KR_CHECK(b[2].reply && !b[3].reply);
}

KR_TEST(replies_teach_the_token_size_and_reply_length) {
    engine_router r = both_ready();
    const int64_t id = r.started(0, route_pieces(chat("hi"), nullptr), 1, 0);
    r.finished(id, 4000, 1000, 1300);
    KR_CHECK(r.bytes_per_token() > 3.6 && r.bytes_per_token() < 4.0);
    // a short message, mostly the template's tokens, teaches nothing about the size
    const double before = r.bytes_per_token();
    r.finished(r.started(0, route_pieces(chat("hi"), nullptr), 1, 0), 30, 32, 8);
    KR_CHECK(r.bytes_per_token() == before);
    KR_CHECK(r.reply_tokens() > 300);
}

// memory for one engine: the other is loaded in its place when that pays
KR_TEST(one_at_a_time_a_long_document_loads_the_faster_reader) {
    engine_route_options o;
    o.one_at_a_time = true;
    engine_router r({ vulkan(), rocm() }, o);
    r.set_ready(0, true);
    r.set_load_s(1, 20);
    // a chat stays on the loaded Vulkan; a medium prompt too (ROCm would read it sooner, but not by its loading time)
    auto c = r.pick(route_pieces(chat("Hello there"), nullptr), 0, 0);
    KR_CHECK(c.engine == 0 && !c.load);
    c = r.pick(route_pieces(chat("Read: " + document(40000)), nullptr), 0, 0);
    KR_CHECK(c.engine == 0 && !c.load);
    // a 127K-token document: ROCm, loading it first (138 s and 20 s of loading against 296 s)
    const std::string doc = document(450000) + "\nSummarise this.";
    c = r.pick(route_pieces(chat(doc), nullptr), 0, 0);
    KR_CHECK(c.engine == 1 && c.load);
    KR_CHECK(c.seconds[1] > 20 && c.seconds[1] < c.seconds[0]);
    // ...but not while Vulkan is busy for longer than loading would save
    const int64_t busy = r.started(0, route_pieces(chat("Another: " + document(450000)), nullptr), 1000, 0);
    KR_CHECK(r.pick(route_pieces(chat(doc), nullptr), 0, 0).engine == 0);
    r.finished(busy, 0, 0, 0);
    // loaded: the document is read there, and its follow-up stays
    r.set_ready(0, false);
    r.set_ready(1, true);
    r.finished(r.started(1, route_pieces(chat(doc), nullptr), 140, 0), 0, 0, 0);
    c = r.pick(route_pieces(chat(doc, "The year went well.", "And the costs?"), nullptr), 0, 200);
    KR_CHECK(c.engine == 1 && !c.load);
}

KR_TEST(one_at_a_time_a_run_of_chats_brings_the_faster_writer_back) {
    engine_route_options o;
    o.one_at_a_time = true;
    engine_router r({ vulkan(), rocm() }, o);
    r.set_ready(1, true);
    r.set_load_s(0, 20);
    // each chat would be about 0.9 s sooner on Vulkan: after about 20 s of that, Vulkan is loaded for the next one
    int n = 0;
    for (; n < 100; ++n) {
        const auto c = r.pick(route_pieces(chat("Question " + std::to_string(n)), nullptr), 0, 1000.0 * n);
        if (c.load) {
            KR_CHECK(c.engine == 0);
            break;
        }
        KR_CHECK(c.engine == 1);
        r.chosen(c);
        r.finished(r.started(1, route_pieces(chat("Question " + std::to_string(n)), nullptr), c.seconds[1], 1000.0 * n), 0, 0, 0);
    }
    KR_CHECK(n >= 15 && n <= 30);
    // not while ROCm is answering someone
    engine_router busy({ vulkan(), rocm() }, o);
    busy.set_ready(1, true);
    busy.set_load_s(0, 20);
    for (int i = 0; i < 40; ++i) {
        const auto c = busy.pick(route_pieces(chat("Question " + std::to_string(i)), nullptr), 0, 0);
        busy.chosen(c);
    }
    busy.started(1, route_pieces(chat("A long answer"), nullptr), 60, 0);
    KR_CHECK(!busy.pick(route_pieces(chat("Hi"), nullptr), 0, 1).load);
    // a changed engine starts the count afresh
    busy.set_ready(1, false);
    busy.set_ready(0, true);
    KR_CHECK(!busy.pick(route_pieces(chat("Hi"), nullptr), 0, 1).load);
}

KR_TEST(one_at_a_time_an_engine_that_failed_is_never_chosen) {
    engine_route_options o;
    o.one_at_a_time = true;
    engine_router r({ vulkan(), rocm() }, o);
    r.set_ready(0, true);
    r.set_failed(1);
    KR_CHECK(r.pick(route_pieces(chat(document(450000)), nullptr), 0, 0).engine == 0);
}

KR_TEST_MAIN()
