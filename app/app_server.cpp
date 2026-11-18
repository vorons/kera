// `kera app`: the downloadable app (spec Phase 0, section 4). Serves a local web page on 127.0.0.1,
// runs auto-setup on first launch, and opens the browser. Nobody has to measure anything by hand.

#include "app/kera_app.h"
#include "app/model_runner.h"

#include "common/kr_paths.h"
#include "planner/kr_models.h"
#include "planner/projector.h"
#include "planner/kr_plan.h"
#include "setup/kr_machine.h"

#include "kera_version.h"
#include "kera_web_assets.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <csignal>
#include <cstdlib>
#endif

namespace kera {

namespace {

struct app_state {
    std::mutex mutex;
    bool has_profile = false;
    machine_profile profile;
    std::vector<std::string> lines;
    bool running = false;
    double progress = 0.0;
    std::string step;
    std::string error;
    std::thread worker;
    bool quick = false;
    std::atomic<bool> quitting{false};
    std::thread quitter; // joined after the server stops, so it never outlives it
    model_runner runner; // the model kept loaded for chat apps (spec 4c item 1)
};

void start_setup(app_state & st) {
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        if (st.running) {
            return;
        }
        st.running = true;
        st.progress = 0.0;
        st.step = "Getting ready";
        st.error.clear();
    }
    if (st.worker.joinable()) {
        st.worker.join();
    }
    st.worker = std::thread([&st] {
        setup_options o;
        o.quick = st.quick;
        o.progress = [&st](double f, const std::string & step) {
            std::lock_guard<std::mutex> lock(st.mutex);
            st.progress = f;
            st.step = step;
        };
        try {
            machine_profile p = run_setup(o);
            const bool saved = save_profile(p, profile_path());
            std::lock_guard<std::mutex> lock(st.mutex);
            st.profile = p;
            st.lines = explain(p);
            st.has_profile = true;
            if (!saved) {
                st.error = "The results couldn't be saved in " + home_dir() + ", so Kera will measure again next time.";
            }
        } catch (const std::exception & e) {
            std::lock_guard<std::mutex> lock(st.mutex);
            st.error = std::string("Measuring this computer failed: ") + e.what();
        }
        std::lock_guard<std::mutex> lock(st.mutex);
        st.running = false;
        st.progress = 1.0;
    });
}

const char * mime_for(const std::string & path) {
    auto ends = [&](const char * ext) {
        const size_t n = std::strlen(ext);
        return path.size() >= n && path.compare(path.size() - n, n, ext) == 0;
    };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".css"))  return "text/css; charset=utf-8";
    if (ends(".js"))   return "text/javascript; charset=utf-8";
    if (ends(".svg"))  return "image/svg+xml";
    if (ends(".png"))  return "image/png";
    if (ends(".ico"))  return "image/x-icon";
    if (ends(".json")) return "application/json";
    return "application/octet-stream";
}

void open_browser(const std::string & url) {
#if defined(_WIN32)
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    const std::string cmd = "open '" + url + "' >/dev/null 2>&1 &";
    if (std::system(cmd.c_str()) != 0) {}
#else
    if (std::getenv("DISPLAY") || std::getenv("WAYLAND_DISPLAY")) {
        const std::string cmd = "xdg-open '" + url + "' >/dev/null 2>&1 &";
        if (std::system(cmd.c_str()) != 0) {}
    }
#endif
}

httplib::Server * g_server = nullptr;

// The models the page lists: GGUF files in the model folders, and Ollama's models by their Ollama names (used where
// they are, D-027); each with whether it can see images
struct listed_model {
    std::string path;
    std::string ollama; // Ollama's name, or ""
    bool vision = false;
    bool image = false; // makes images (the image engine serves it)
};

std::vector<listed_model> listed_models() {
    std::vector<listed_model> out;
    for (const auto & path : find_models(model_folders())) {
        const bool image = is_image_model(path);
        out.push_back({ path, "", !image && !find_projector(path).empty(), image });
    }
    for (const auto & m : ollama_models()) {
        out.push_back({ m.model, m.name, !m.projector.empty() && is_projector(m.projector) });
    }
    return out;
}

#if !defined(_WIN32)
void on_signal(int) {
    if (g_server) g_server->stop();
}
#else
BOOL WINAPI on_console_event(DWORD) {
    if (g_server) g_server->stop();
    return TRUE;
}
#endif

} // namespace

int run_app(int argc, char ** argv) {
    int port = 8341;
    bool browser = true;
    bool quick = false;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
            if (port < 1 || port > 65535) {
                std::fprintf(stderr, "kera app: --port needs a number from 1 to 65535\n");
                return 1;
            }
        } else if (a == "--no-browser") {
            browser = false;
        } else if (a == "--quick") {
            quick = true;
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: kera app [--port N] [--no-browser] [--quick]\n"
                        "Starts Kera's app: a page in your browser that measures this computer and runs models.\n");
            return 0;
        }
    }

    // Kera already running (its window closed, or the icon clicked twice): open its page rather than start a
    // second Kera, which would load a second copy of the last model beside the first and share its log, its
    // settings and its saved chats
    {
        httplib::Client probe("127.0.0.1", port);
        probe.set_connection_timeout(0, 300000);
        probe.set_read_timeout(2, 0);
        if (auto r = probe.Get("/api/status"); r && r->status == 200) {
            const auto j = nlohmann::json::parse(r->body, nullptr, false);
            // the same person's Kera (the same Kera folder): on a shared computer another user's Kera may hold the
            // port, and its page has that user's chats and models
            if (j.is_object() && j.contains("version") && j.value("home", std::string()) == home_dir()) {
                const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/";
                std::printf("Kera is already running at %s: opening it.\n", url.c_str());
                if (browser) {
                    open_browser(url);
                }
                return 0;
            }
        }
    }

    app_state st;
    st.quick = quick;
    {
        machine_profile p;
        if (load_profile(p, profile_path())) {
            st.profile = p;
            st.lines = explain(p);
            st.has_profile = true;
        }
    }
    const bool had_profile = st.has_profile; // read before the first measurement's thread may change it
    if (!had_profile) {
        start_setup(st); // first launch: measure straight away
    }

    httplib::Server svr;
    g_server = &svr;

    std::string host_ok_1;
    std::string host_ok_2;
    svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        // Only this computer's own page may talk to Kera: refuse other Host names (DNS rebinding).
        const std::string host = req.get_header_value("Host");
        if (host != host_ok_1 && host != host_ok_2) {
            res.status = 403;
            res.set_content("Forbidden", "text/plain");
            return httplib::Server::HandlerResponse::Handled;
        }
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_header("Cache-Control", "no-store");
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/api/status", [&](const httplib::Request &, httplib::Response & res) {
        nlohmann::json j;
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            j["version"] = KERA_VERSION;
            j["home"] = home_dir(); // so a second start can tell its own Kera from another user's
            j["commit"] = KERA_GIT_COMMIT;
            j["profile"] = st.has_profile ? st.profile.to_json() : nlohmann::json(nullptr);
            j["explain"] = st.lines;
            j["setup"] = { { "running", st.running }, { "progress", st.progress }, { "step", st.step }, { "error", st.error } };
        }
        res.set_content(j.dump(), "application/json");
    });

    // the models on this computer, with the planner's prediction for each (spec 4a step 5, Phase 1)
    svr.Get("/api/models", [&](const httplib::Request &, httplib::Response & res) {
        plan_input in;
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            have = st.has_profile;
            if (have) {
                in = plan_input_from_profile(st.profile);
            }
        }
        nlohmann::json list = nlohmann::json::array();
        for (const auto & lm : listed_models()) {
            const std::string & path = lm.path;
            model_sizes m;
            std::string err;
            nlohmann::json j = { { "path", path }, { "vision", lm.vision }, { "image", lm.image } };
            if (!lm.ollama.empty()) {
                j["ollama"] = lm.ollama;
            }
            if (lm.image) {
                j["file"] = path.substr(path.find_last_of("/\\") + 1);
                j["lines"] = { "An image model: Run serves it to apps with OpenAI's images API, and kera image makes "
                               "pictures with it in a terminal." };
            } else if (!read_model_sizes(path, m, &err)) {
                j["file"] = lm.ollama.empty() ? path.substr(path.find_last_of("/\\") + 1) : lm.ollama;
                j["error"] = err;
            } else {
                j["file"] = lm.ollama.empty() ? m.file : lm.ollama;
                j["size_gb"] = m.resident_gb() + m.experts_gb;
                j["moe"] = m.moe();
                if (have && in.ram_gb_s > 0) {
                    const plan_result r = plan(m, in);
                    j["fits"] = r.fits;
                    j["ceiling_tok_s"] = r.ceiling_tok_s;
                    j["ceiling_overlap_tok_s"] = r.ceiling_overlap_tok_s;
                    j["lines"] = explain_plan(m, in, r);
                }
            }
            list.push_back(j);
        }
        res.set_content(nlohmann::json{ { "models", list }, { "folders", model_folders() } }.dump(), "application/json");
    });

    svr.Post("/api/models/folder", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.get_header_value("X-Kera") != "1") {
            res.status = 403;
            return;
        }
        std::string folder;
        try {
            folder = nlohmann::json::parse(req.body).value("folder", "");
        } catch (const std::exception &) {
        }
        const bool ok = !folder.empty() && add_model_folder(folder);
        res.status = ok ? 200 : 400;
        res.set_content(nlohmann::json{ { "added", ok } }.dump(), "application/json");
    });

    // the model kept loaded for chat apps: what it is doing, and where to connect
    svr.Get("/api/run", [&](const httplib::Request &, httplib::Response & res) {
        const model_runner::status s = st.runner.get();
        std::string file = s.model.substr(s.model.find_last_of("/\\") + 1);
        for (const auto & m : ollama_models()) {
            if (m.model == s.model) {
                file = m.name; // Ollama's name, not its blob's
            }
        }
        nlohmann::json j = {
            { "state", state_name(s.st) }, { "model", s.model }, { "file", file }, { "seconds", s.seconds },
            { "error", s.error },
        };
        if (s.port > 0) {
            j["url"] = "http://127.0.0.1:" + std::to_string(s.port) + "/v1";
        }
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/api/run", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.get_header_value("X-Kera") != "1") {
            res.status = 403;
            return;
        }
        std::string path;
        try {
            path = nlohmann::json::parse(req.body).value("path", "");
        } catch (const std::exception &) {
        }
        // only a model the page listed, so nothing else can be started through this
        const auto models = listed_models();
        if (std::none_of(models.begin(), models.end(), [&](const listed_model & m) { return m.path == path; })) {
            res.status = 400;
            res.set_content("{\"error\":\"not one of the models Kera found\"}", "application/json");
            return;
        }
        std::string err;
        const bool ok = st.runner.start(path, &err);
        res.status = ok ? 200 : 500;
        res.set_content(nlohmann::json{ { "started", ok }, { "error", err } }.dump(), "application/json");
    });

    svr.Post("/api/run/stop", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.get_header_value("X-Kera") != "1") {
            res.status = 403;
            return;
        }
        st.runner.stop();
        res.set_content("{\"stopped\":true}", "application/json");
    });

    svr.Post("/api/setup", [&](const httplib::Request & req, httplib::Response & res) {
        // a custom header that browsers never send across sites, so another page can't start a check
        if (req.get_header_value("X-Kera") != "1") {
            res.status = 403;
            return;
        }
        start_setup(st);
        res.set_content("{\"started\":true}", "application/json");
    });

    svr.Post("/api/quit", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.get_header_value("X-Kera") != "1") {
            res.status = 403;
            return;
        }
        res.set_content("{\"stopping\":true}", "application/json");
        if (!st.quitting.exchange(true)) {
            st.quitter = std::thread([&svr] {
                std::this_thread::sleep_for(std::chrono::milliseconds(200)); // let the reply go out first
                svr.stop();
            });
        }
    });

    svr.Get(R"(/(.*))", [&](const httplib::Request & req, httplib::Response & res) {
        std::string path = req.matches[1];
        if (path.empty()) {
            path = "index.html";
        }
        for (size_t i = 0; i < kera_web_assets_count; ++i) {
            if (path == kera_web_assets[i].path) {
                res.set_content((const char *) kera_web_assets[i].data, kera_web_assets[i].size, mime_for(path));
                return;
            }
        }
        res.status = 404;
        res.set_content("Not found", "text/plain");
    });

    // one Kera on a port: httplib's default options let a second copy listen on the same one, and requests would go
    // to either at random
    svr.set_socket_options([](socket_t sock) {
#if defined(_WIN32)
        BOOL yes = TRUE;
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &yes, sizeof(yes));
#else
        int yes = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    });
    int bound = -1;
    for (int p = port; p < port + 20; ++p) {
        if (svr.bind_to_port("127.0.0.1", p)) {
            bound = p;
            break;
        }
    }
    if (bound < 0) {
        std::fprintf(stderr, "Kera couldn't start its page: ports %d to %d are all in use.\n", port, port + 19);
        if (st.worker.joinable()) {
            st.worker.join(); // the first measurement, if it is running: a thread left running would end the program
        }
        return 1;
    }
    // the model used last time loads again by itself (spec 4c item 1): once the page has its port, and only on
    // Kera's own port: a copy that had to move to another port (something else holds Kera's) starts no model
    if (had_profile && bound == port) {
        st.runner.start_last();
    } else if (had_profile) {
        std::fprintf(stderr, "Kera: port %d is taken by another program, so the page is on %d and the last model "
                             "isn't started by itself (start it from the page).\n", port, bound);
    }
    host_ok_1 = "127.0.0.1:" + std::to_string(bound);
    host_ok_2 = "localhost:" + std::to_string(bound);
    const std::string url = "http://" + host_ok_1 + "/";

#if defined(_WIN32)
    SetConsoleCtrlHandler(on_console_event, TRUE);
    SetConsoleTitleW(L"Kera");
#else
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#endif

    std::printf("Kera is running at %s\n", url.c_str());
    std::printf("Close this window (or press Ctrl+C) to stop Kera.\n");
    std::fflush(stdout);
    if (browser) {
        open_browser(url);
    }

    svr.listen_after_bind();

    g_server = nullptr;
    if (st.quitter.joinable()) {
        st.quitter.join();
    }
    if (st.worker.joinable()) {
        st.worker.join();
    }
    return 0;
}

} // namespace kera
