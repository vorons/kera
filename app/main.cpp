// The `kera` program (spec Phase 13 command list). With no command it starts the app, which is what
// double-clicking it does. Model commands reuse llama.cpp's own tool libraries, the way its unified
// `llama` binary does (third_party/llama.cpp/app/llama.cpp).

#include "app/kera_app.h"
#include "common/kr_paths.h"
#include "setup/kr_machine.h"
#include "tune/kr_settings.h"

#include "kera_version.h"
#include "kera_web_assets.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

// llama.cpp's tools, linked as libraries
int llama_server(int argc, char ** argv);
int llama_cli(int argc, char ** argv);
int llama_completion(int argc, char ** argv);
int llama_bench(int argc, char ** argv);
int llama_perplexity(int argc, char ** argv);
int llama_quantize(int argc, char ** argv);

// Kera's tools
int kera_bench(int argc, char ** argv);
int kera_exactness(int argc, char ** argv);
int kera_trace(int argc, char ** argv);
int kera_wait(int argc, char ** argv);

namespace {

const char * g_prog = "kera";

int cmd_help(int argc, char ** argv);

int cmd_version(int, char **) {
    std::printf("Kera %s (commit %s), built on llama.cpp %s\n", KERA_VERSION, KERA_GIT_COMMIT, KERA_LLAMA_CPP_TAG);
    return 0;
}

// Hidden: the SSD read test on an existing file, for cross-checking auto-setup by hand (spec Phase 0 step 6).
int cmd_disktest(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: kera disktest FILE [READ_MIB] [INFLIGHT] [SECONDS]\n");
        return 1;
    }
    const int mib = argc > 2 ? std::atoi(argv[2]) : 1;
    const int inflight = argc > 3 ? std::atoi(argv[3]) : 1;
    const double seconds = argc > 4 ? std::atof(argv[4]) : 4.0;
    int err = 0;
    const double gbps = kera::read_test_file(argv[1], mib, inflight, seconds, &err);
    if (err != 0) {
        std::fprintf(stderr, "Reading failed: error %d\n", err);
        return 1;
    }
    std::printf("%.3f\n", gbps);
    return 0;
}

int cmd_licenses(int, char **) {
    for (size_t i = 0; i < kera_web_assets_count; ++i) {
        const char * p = kera_web_assets[i].path;
        if (std::strcmp(p, "LICENSE") == 0 || std::strcmp(p, "NOTICE") == 0 || std::strcmp(p, "THIRD_PARTY_LICENSES.txt") == 0) {
            std::fwrite(kera_web_assets[i].data, 1, kera_web_assets[i].size, stdout);
            std::printf("\n");
        }
    }
    return 0;
}

struct command {
    const char * name;
    const char * desc;
    bool hidden;
    int (*func)(int, char **);
};

const command g_cmds[] = {
    { "app",        "Start Kera's app in your browser (the default)",            false, kera::run_app },
    { "setup",      "Measure this computer and explain what it can run",          false, kera::run_setup_cli },
    { "plan",       "Predict a model's top speed on this computer",               false, kera::run_plan_cli },
    { "serve",      "Run a model behind OpenAI- and Anthropic-compatible APIs",   false, kera::run_serve },
    { "cli",        "Chat with a model in this window",                           false, kera::run_cli },
    { "image",      "Make a picture from text with an image model",                false, kera::run_image_cli },
    { "video",      "Make a short video from text with a video model",              false, kera::run_video_cli },
    { "music",      "Make a song from a description (and lyrics)",                  false, kera::run_music_cli },
    { "speak",      "Turn text into speech with a speech model",                    false, kera::run_speak_cli },
    { "3d",         "Make a 3D model (.glb) from a picture or a description",       false, kera::run_3d_cli },
    { "sfx",        "Make a sound effect from a description",                       false, kera::run_sfx_cli },
    { "bench",      "Measure a model's speed with the bench prompts (section 7)",  false, kera_bench },
    { "wait",       "Show where the wait before a reply goes (any chat server)",  false, kera_wait },
    { "finetune",   "Teach a model your own examples (a LoRA adapter; needs Python)", false, kera::run_finetune_cli },
    { "completion", "Text completion (llama.cpp)",                                true,  llama_completion },
    { "llama-bench","Stock llama.cpp benchmark, for baselines",                   true,  llama_bench },
    { "exactness",  "Check a model against the reference code's output (tests)",   true,  kera_exactness },
    { "trace",      "Record which experts an MoE model uses (risk test 2)",        true,  kera_trace },
    { "disktest",   "SSD read test on a file, for cross-checks",                   true,  cmd_disktest },
    { "perplexity", "Perplexity and KL divergence, for quality tests (section 6)", true,  kera::run_perplexity_cli },
    { "calibrate",  "Measure a back-end's speed on a model (for serve's choice)",  true,  kera::run_calibrate_cli },
    { "quantize",   "Quantize a model (llama.cpp)",                               true,  llama_quantize },
    { "version",    "Show the version",                                           false, cmd_version },
    { "licenses",   "Show Kera's licence and third-party notices",                false, cmd_licenses },
    { "help",       "Show the commands",                                          false, cmd_help },
};

int cmd_help(int argc, char ** argv) {
    const bool all = argc >= 2 && std::string(argv[1]) == "all";
    std::printf("Usage: %s [command] [options]\n\nCommands:\n", g_prog);
    for (const auto & c : g_cmds) {
        if (all || !c.hidden) {
            std::printf("  %-12s %s\n", c.name, c.desc);
        }
    }
    std::printf("\nWith no command, Kera starts its app. Run '%s help all' for every command,\n"
                "and '%s <command> --help' for a command's options.\n", g_prog, g_prog);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
#if defined(_WIN32)
    // Windows hands main() its words in the computer's old code page, which garbles a file name with an accented or
    // non-Latin letter, say in the user's name (Kera and llama.cpp take file names as UTF-8). Read them again from
    // the wide command line.
    static std::vector<std::string> words;
    static std::vector<char *> word_ptrs;
    int n_words = 0;
    if (LPWSTR * w = CommandLineToArgvW(GetCommandLineW(), &n_words)) {
        if (n_words == argc) {
            for (int i = 0; i < n_words; ++i) {
                const int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, nullptr, 0, nullptr, nullptr);
                std::string s(len > 1 ? len - 1 : 0, '\0');
                if (len > 1) {
                    WideCharToMultiByte(CP_UTF8, 0, w[i], -1, s.data(), len, nullptr, nullptr);
                }
                words.push_back(std::move(s));
            }
            for (auto & s : words) {
                word_ptrs.push_back(s.data());
            }
            word_ptrs.push_back(nullptr);
            argv = word_ptrs.data();
        }
        LocalFree(w);
    }
#endif
    // "kera", whatever folder it was started from (and without Windows' ".exe")
    static std::string prog = argv[0];
    prog = prog.substr(prog.find_last_of("/\\") == std::string::npos ? 0 : prog.find_last_of("/\\") + 1);
    if (prog.size() > 4 && prog.substr(prog.size() - 4) == ".exe") {
        prog.resize(prog.size() - 4);
    }
    g_prog = prog.c_str();

    if (argc < 2 || (argv[1][0] == '-' && std::string(argv[1]) != "--help" && std::string(argv[1]) != "--version")) {
        // no command (or only app options): start the app
        return kera::run_app(argc - 1, argv + 1);
    }
    std::string arg = argv[1];
    if (arg == "--help" || arg == "-h") {
        arg = "help";
    } else if (arg == "--version") {
        arg = "version";
    }

    // the graphics chip's built shaders kept in Kera's folder (llama.cpp's Vulkan back-end, N-011, D-021), unless
    // the setting is off or KERA_SHADER_CACHE=0
    kr_settings_register_builtin();
    const char * sc = std::getenv("KERA_SHADER_CACHE");
    if (sc != nullptr) {
        kr_setting_set("gpu.shader_cache", std::atof(sc));
    }
    if (kr_setting_get("gpu.shader_cache") > 0.5 && std::getenv("GGML_VK_PIPELINE_CACHE") == nullptr) {
        std::string base = kera::home_path("vulkan-pipelines");
#if defined(_WIN32)
        // llama.cpp opens this path as a narrow string, which Windows reads in the old code page: a folder with an
        // accented or non-Latin letter in it (the user's name) would never open. Then the folder's short name (all
        // ASCII) is used, or, without one, no kept shaders.
        if (std::any_of(base.begin(), base.end(), [](char c) { return (unsigned char) c >= 0x80; })) {
            const std::string home = kera::home_dir();
            std::wstring whome(MultiByteToWideChar(CP_UTF8, 0, home.c_str(), -1, nullptr, 0), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, home.c_str(), -1, whome.data(), (int) whome.size());
            wchar_t shortp[MAX_PATH] = {};
            const DWORD n = GetShortPathNameW(whome.c_str(), shortp, MAX_PATH);
            std::string ascii;
            for (DWORD i = 0; i < n && n < MAX_PATH; ++i) {
                ascii += shortp[i] < 0x80 ? (char) shortp[i] : '\0';
            }
            const bool ok = n > 0 && n < MAX_PATH && ascii.find('\0') == std::string::npos;
            base = ok ? ascii + (ascii.empty() || ascii.back() == '\\' ? "" : "\\") + "vulkan-pipelines" : "";
        }
        if (!base.empty()) {
            _putenv_s("GGML_VK_PIPELINE_CACHE", base.c_str());
        }
#else
        setenv("GGML_VK_PIPELINE_CACHE", base.c_str(), 1);
#endif
    }

    for (const auto & c : g_cmds) {
        if (arg == c.name) {
            // llama.cpp's server re-launches the binary for each model in router mode and puts this
            // command back after the binary path (tools/server/server-models.cpp)
#if defined(_WIN32)
            _putenv_s("LLAMA_APP_CMD", c.name);
#else
            setenv("LLAMA_APP_CMD", c.name, 1);
#endif
            return c.func(argc - 1, argv + 1);
        }
    }
    std::fprintf(stderr, "Unknown command '%s'. Run '%s help' to see the commands.\n", arg.c_str(), g_prog);
    return 1;
}
