// Keeps one model loaded for chat apps (spec 4c item 1): runs `kera serve` for it in the background, says
// when it's ready and where chat apps should connect, and remembers it, so the next time the app starts
// the same model loads again by itself. It never unloads a model for being idle.
#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace kera {

class model_runner {
public:
    enum class state { stopped, loading, ready, failed };

    // log_name: its log in the Kera folder; remember: whether the next app start loads this model again (the app's
    // model does; a model started for one job, like the prompt improver's, doesn't)
    explicit model_runner(std::string log_name = "serve.log", bool remember = true)
        : log_name_(std::move(log_name)), remember_(remember) {}
    ~model_runner();

    // Start serving a model (stopping the one before). Returns false if it can't be started.
    bool start(const std::string & model_path, std::string * error = nullptr);
    void stop();

    // Load the model used last time, if there was one and the file is still there.
    void start_last();

    struct status {
        state st = state::stopped;
        std::string model;   // path
        int port = 0;
        double seconds = 0;  // since it started loading, or how long loading took
        std::string error;   // why it failed, with the end of its log
    };
    status get();

private:
    void watch(unsigned long long pid, int port);
    void stop_locked(); // with ops_ held
    static std::string last_model_file();

    std::mutex ops_;    // one start or stop at a time (the page's requests come on several threads)
    std::mutex mutex_;
    state state_ = state::stopped;
    std::string model_;
    int port_ = 0;
    std::string error_;
    double started_ = 0;
    double ready_after_ = 0;
    unsigned long long pid_ = 0; // process id (Windows: also the handle, in handle_)
    void * handle_ = nullptr;
    std::thread watcher_;
    bool stopping_ = false;
    std::string log_name_;
    bool remember_ = true;
};

const char * state_name(model_runner::state s);

} // namespace kera
