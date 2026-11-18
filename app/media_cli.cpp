// Music and speech (the owner's request for 0.1.1, docs/DECISIONS.md D-028).
//
// Music: acestep.cpp (third_party/), ACE-Step 1.5 in C++ on ggml, built as programs of its own in the music/ folder
// next to kera, with its own copy of ggml. A song is two steps: a language model writes the lyrics (unless given) and
// the song's audio codes from the description, then a diffusion model and a VAE turn the codes into 48 kHz stereo.
// `kera music` runs both; `kera serve --music DIR` runs its server, with its own page for writing songs.
//
// Speech: llama.cpp's own speech tool (tools/tts, with libmtmd), linked into kera: Qwen3-TTS (ten languages, a voice
// copied from a short sample) and Pocket TTS. `kera speak` finds the model's audio decoder (its mmproj) by itself.

#include "app/kera_app.h"
#include "common/kr_paths.h"
#include "planner/projector.h"
#include "setup/kr_machine.h"
#include "tune/kr_music_defaults.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int llama_tts(int argc, char ** argv); // llama.cpp's tools/tts (llama-tts-impl)

namespace fs = std::filesystem;

namespace kera {

namespace {

std::string music_models_default() {
    return home_path("models/music");
}

bool music_engine_missing(const char * name) {
    std::error_code ec;
    if (fs::is_regular_file(fs::u8path(engine_path("music", name)), ec)) {
        return false;
    }
    std::fprintf(stderr, "Kera: the music engine isn't here (%s): this download was built without it\n",
                 engine_path("music", name).c_str());
    return true;
}

// the music engine counts half the logical CPUs as its cores, which is half the cores where a core runs one thread
// (Apple Silicon, many laptops and virtual machines): it is told the real count (1.8x faster on 4 such cores)
void music_engine_threads() {
    if (std::getenv("ACESTEP_THREADS") != nullptr) {
        return;
    }
    machine_profile p;
    detect_system(p);
    if (p.cpu_cores <= 0) {
        return;
    }
    const std::string n = std::to_string(p.cpu_cores);
#ifdef _WIN32
    _putenv_s("ACESTEP_THREADS", n.c_str());
#else
    setenv("ACESTEP_THREADS", n.c_str(), 0);
#endif
}

void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// Of the folder's planners and sound models, the ones ACE-Step's own guide recommends for this computer's memory
// are the engine's choice for a request that names none (D-036): it otherwise takes the first of each by file name
// (the base model, the 0.6B planner, the lowest precision). A request that names a model still gets that one.
void music_model_defaults(const std::string & models, bool lm_named = false, bool dit_named = false) {
    std::vector<music_file> lms, dits;
    std::error_code ec;
    for (fs::directory_iterator it(fs::u8path(models), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().u8string();
        std::string n = name;
        for (auto & c : n) {
            c = (char) std::tolower((unsigned char) c);
        }
        if (n.size() < 5 || n.compare(n.size() - 5, 5, ".gguf") != 0) {
            continue;
        }
        std::error_code size_ec;
        const uint64_t bytes = (uint64_t) fs::file_size(it->path(), size_ec);
        if (n.rfind("acestep-5hz-lm", 0) == 0) {
            lms.push_back({ name, size_ec ? 0 : bytes });
        } else if (n.rfind("acestep-v15", 0) == 0) {
            dits.push_back({ name, size_ec ? 0 : bytes });
        }
    }
    if (lms.size() < 2 && dits.size() < 2) {
        return; // nothing to choose
    }
    const auto by_name = [](const music_file & a, const music_file & b) { return a.name < b.name; };
    std::sort(lms.begin(), lms.end(), by_name);
    std::sort(dits.begin(), dits.end(), by_name);
    machine_profile p;
    detect_system(p);
    const music_choice c = choose_music_models(lms, dits, gpu_memory_gb(), p.ram_gb);
    if (!c.lm.empty() && std::getenv("ACESTEP_DEFAULT_LM") == nullptr) {
        set_env("ACESTEP_DEFAULT_LM", c.lm);
    }
    if (!c.dit.empty() && std::getenv("ACESTEP_DEFAULT_DIT") == nullptr) {
        set_env("ACESTEP_DEFAULT_DIT", c.dit);
    }
    // said only for what the request leaves to Kera (a named model is the one used)
    const bool say_lm = !lm_named && lms.size() > 1 && !c.lm.empty();
    const bool say_dit = !dit_named && dits.size() > 1 && !c.dit.empty();
    if (!c.why.empty() && (say_lm || say_dit)) {
        const std::string which = (say_dit ? c.dit : "") + (say_dit && say_lm ? " and " : "") + (say_lm ? c.lm : "");
        std::fprintf(stderr, "Kera: %s: %s (--lm and --dit, or the page's own lists, choose others)\n", c.why.c_str(),
                     which.c_str());
    }
}

bool valid_utf8(const std::string & s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (n == 0 || i + n > s.size()) {
            return false;
        }
        for (int k = 1; k < n; ++k) {
            if ((((unsigned char) s[i + k]) >> 6) != 0x2) {
                return false;
            }
        }
        i += n;
    }
    return true;
}

// UTF-16 text (Windows Notepad's "Unicode", with its byte-order mark) as UTF-8
std::string utf16_to_utf8(const std::string & b, bool big_endian) {
    std::string out;
    auto unit = [&](size_t i) {
        return big_endian ? (uint32_t) ((unsigned char) b[i] << 8 | (unsigned char) b[i + 1])
                          : (uint32_t) ((unsigned char) b[i + 1] << 8 | (unsigned char) b[i]);
    };
    for (size_t i = 0; i + 1 < b.size(); i += 2) {
        uint32_t cp = unit(i);
        if (cp >= 0xD800 && cp < 0xDC00 && i + 3 < b.size()) {
            const uint32_t lo = unit(i + 2);
            if (lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                i += 2;
            }
        }
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | cp >> 6), out += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char) (0xE0 | cp >> 12), out += (char) (0x80 | (cp >> 6 & 0x3F)), out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | cp >> 18), out += (char) (0x80 | (cp >> 12 & 0x3F));
            out += (char) (0x80 | (cp >> 6 & 0x3F)), out += (char) (0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// a value that may be "@file": the file's text (false if it can't be read). The words reach the model as the user
// wrote them (rule one, D-033): a byte-order mark and Windows line ends (CR LF) are a file's format, not words, and
// a file in Windows' own code page or UTF-16 is turned into UTF-8 rather than its letters into "?" marks (N-018)
bool text_or_file(const std::string & v, std::string & out) {
    out = v;
    if (v.size() > 1 && v[0] == '@') {
        std::ifstream f(fs::u8path(v.substr(1)), std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "kera music: can't read %s\n", v.substr(1).c_str());
            return false;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        out = ss.str();
        if (out.size() >= 2 && (unsigned char) out[0] == 0xFF && (unsigned char) out[1] == 0xFE) {
            out = utf16_to_utf8(out.substr(2), false);
        } else if (out.size() >= 2 && (unsigned char) out[0] == 0xFE && (unsigned char) out[1] == 0xFF) {
            out = utf16_to_utf8(out.substr(2), true);
        } else if (out.size() >= 3 && out.compare(0, 3, "\xEF\xBB\xBF") == 0) {
            out.erase(0, 3);
        }
        if (!valid_utf8(out)) {
#ifdef _WIN32
            // Notepad's "ANSI": the computer's own code page
            const int nw = MultiByteToWideChar(CP_ACP, 0, out.data(), (int) out.size(), nullptr, 0);
            std::wstring w((size_t) std::max(nw, 0), L'\0');
            MultiByteToWideChar(CP_ACP, 0, out.data(), (int) out.size(), w.data(), nw);
            const int n8 = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
            std::string u((size_t) std::max(n8, 0), '\0');
            WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), u.data(), n8, nullptr, nullptr);
            out = u;
#endif
            if (!valid_utf8(out)) {
                std::fprintf(stderr, "kera music: %s isn't UTF-8 text; save it as UTF-8 so its words reach the model "
                                     "as written\n", v.substr(1).c_str());
                return false;
            }
        }
        std::string lf; // CR LF and a lone CR end a line, as LF
        lf.reserve(out.size());
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i] == '\r') {
                lf += '\n';
                if (i + 1 < out.size() && out[i + 1] == '\n') {
                    ++i;
                }
            } else {
                lf += out[i];
            }
        }
        out = lf;
    }
    return true;
}

void music_help() {
    std::printf(
        "kera music: a song from a description (ACE-Step 1.5, on this computer). For example:\n\n"
        "  kera music -p \"upbeat pop rock with driving guitars\" --duration 60 -o song.wav\n"
        "  kera music -p \"calm lo-fi piano\" --lyrics \"[Instrumental]\" --duration 30 -o piano.wav\n\n"
        "Options:\n"
        "  -p, --prompt TEXT     what the song is like (style, instruments, mood)\n"
        "  --lyrics TEXT|@FILE   the words; \"[Instrumental]\" for none; left out, the model writes them\n"
        "  --duration SECONDS    how long (left out, the model chooses, as the engine does)\n"
        "  --language CODE       the vocals' language (en, fr, zh, ja, ...)\n"
        "  --bpm N               the tempo (the model chooses if left out)\n"
        "  --seed N              the same seed and settings make the same song (lyrics, key and sound)\n"

        "  --steps N             diffusion steps (the model's own: 8 for turbo, 50 for base and SFT)\n"
        "  --guidance N          how closely the sound follows the description (ACE-Step's own: 7 for base\n"
        "                        and SFT; turbo models have it built in)\n"
        "  --shift N             the steps' timing (ACE-Step's own: 3)\n"
        "  -o, --output FILE     .wav or .mp3 (default song.wav: lossless, the engine's own samples; .mp3 is\n"
        "                        smaller and loses some of the sound)\n"
        "  --models DIR          the folder with the model files (default %s)\n"
        "  --lm FILE, --dit FILE which language and diffusion model, if the folder has several (left out,\n"
        "                        the ones ACE-Step's guide recommends for this computer's memory)\n\n"
        "The folder needs one of each, from huggingface.co/Serveurperso/ACE-Step-1.5-GGUF: a language model\n"
        "(acestep-5Hz-lm-*.gguf), the text encoder (Qwen3-Embedding-0.6B-*.gguf), a diffusion model\n"
        "(acestep-v15-turbo-*.gguf) and the VAE (vae-*.gguf). The -BF16 files are the models' own weights and\n"
        "the others smaller copies; with several in the folder, Kera takes the highest precision that fits this\n"
        "computer's memory. To make songs in a page instead:\n"
        "  kera serve --music DIR\n",
        music_models_default().c_str());
}

} // namespace

int run_music_cli(int argc, char ** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        music_help();
        return argc < 2 ? 1 : 0;
    }
    if (music_engine_missing("ace-lm") || music_engine_missing("ace-synth")) {
        return 2;
    }
    music_engine_threads();
    nlohmann::json req = nlohmann::json::object();
    std::string models = music_models_default();
    std::string out = "song.wav";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool v = i + 1 < argc;
        if ((a == "-p" || a == "--prompt") && v) {
            req["caption"] = argv[++i];
        } else if (a == "--lyrics" && v) {
            std::string lyrics;
            if (!text_or_file(argv[++i], lyrics)) {
                return 1;
            }
            req["lyrics"] = lyrics;
        } else if (a == "--duration" && v) {
            req["duration"] = std::atof(argv[++i]);
        } else if (a == "--language" && v) {
            req["vocal_language"] = argv[++i];
        } else if (a == "--bpm" && v) {
            req["bpm"] = std::atoi(argv[++i]);
        } else if (a == "--seed" && v) {
            // both steps: the engine otherwise picks the language model's seed at random each time, so the same
            // --seed gave a different song (different lyrics, key and codes) every run
            const long long seed = std::atoll(argv[++i]);
            req["seed"] = seed;
            req["lm_seed"] = seed;

        } else if (a == "--steps" && v) {
            req["inference_steps"] = std::atoi(argv[++i]);
        } else if ((a == "--guidance" || a == "--shift") && v) {
            // a number, or nothing is set: a word read as 0 would turn the guidance off without saying so
            char * end = nullptr;
            const double value = std::strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || !(value >= 0.0) || value > 100.0) {
                std::fprintf(stderr, "kera music: %s takes a number, not %s\n", a.c_str(), argv[i]);
                return 2;
            }
            req[a == "--guidance" ? "guidance_scale" : "shift"] = value;
        } else if ((a == "-o" || a == "--output") && v) {
            out = argv[++i];
        } else if (a == "--models" && v) {
            models = argv[++i];
        } else if (a == "--lm" && v) {
            req["lm_model"] = fs::u8path(argv[++i]).filename().u8string();
        } else if (a == "--dit" && v) {
            req["synth_model"] = fs::u8path(argv[++i]).filename().u8string();
        } else {
            std::fprintf(stderr, "kera music: unknown option %s (kera music --help lists them)\n", a.c_str());
            return 1;
        }
    }
    if (!req.contains("caption")) {
        std::fprintf(stderr, "kera music: say what the song is like with -p \"...\"\n");
        return 1;
    }
    std::error_code ec;
    if (!fs::is_directory(fs::u8path(models), ec)) {
        std::fprintf(stderr, "kera music: no model folder at %s (kera music --help says what goes in it)\n", models.c_str());
        return 2;
    }
    std::string ext = fs::u8path(out).extension().u8string();
    for (auto & c : ext) {
        c = (char) std::tolower((unsigned char) c);
    }
    if (ext != ".wav" && ext != ".mp3") {
        std::fprintf(stderr, "kera music: the song is written as .wav (lossless) or .mp3, not %s\n", out.c_str());
        return 1;
    }
    req["output_format"] = ext == ".wav" ? "wav16" : "mp3";
    music_model_defaults(models, req.contains("lm_model"), req.contains("synth_model"));

    // the two steps, in a folder of their own: the first writes request0.json next to the request, the second
    // request00.<format> next to that
    fs::path tmp = fs::temp_directory_path(ec);
    if (ec || !fs::is_directory(tmp, ec)) {
        tmp = fs::u8path(home_path("tmp")); // TMP or TMPDIR names a folder that isn't there
    }
    // songs from an earlier run that was stopped part-way (Ctrl+C on Windows ends the engine and kera at once)
    const auto day_ago = fs::file_time_type::clock::now() - std::chrono::hours(24);
    std::vector<fs::path> old; // removed after the scan: removing entries while iterating is unspecified
    for (fs::directory_iterator it(tmp, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code old_ec;
        if (it->path().filename().u8string().rfind("kera-music-", 0) == 0 &&
            fs::last_write_time(it->path(), old_ec) < day_ago && !old_ec) {
            old.push_back(it->path());
        }
    }
    for (const auto & o : old) {
        fs::remove_all(o, ec);
    }
    ec.clear();
    const fs::path work = tmp / ("kera-music-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(work, ec);
    const fs::path request = work / "request.json";
    {
        // text that isn't UTF-8 (a lyrics file saved in an older Windows encoding) is kept, its odd letters replaced
        std::ofstream f(request);
        f << req.dump(1, ' ', false, nlohmann::json::error_handler_t::replace);
        if (!f.good()) {
            std::fprintf(stderr, "kera music: couldn't write its working files in %s\n", work.u8string().c_str());
            fs::remove_all(work, ec);
            return 1;
        }
    }
    std::fprintf(stderr, "Kera: writing the song (step 1 of 2: lyrics and the song's codes)\n");
    int rc = run_program({ engine_path("music", "ace-lm"), "--models", models, "--request", request.u8string() });
    if (rc == 0) {
        // what step 1 made of the description (the model rewrites it in more detail, as it was made to): shown,
        // so a song can be asked for again in those words
        try {
            std::ifstream f(work / "request0.json");
            const nlohmann::json made = nlohmann::json::parse(f);
            if (made.contains("caption") && made["caption"].is_string()) {
                std::fprintf(stderr, "Kera: the song, as the music engine described it: %s\n",
                             made["caption"].get<std::string>().c_str());
            }
        } catch (const std::exception &) {
        }
        std::fprintf(stderr, "Kera: step 2 of 2: turning the codes into sound\n");
        rc = run_program({ engine_path("music", "ace-synth"), "--models", models, "--request",
                           (work / "request0.json").u8string() });
    }
    const fs::path made = work / (std::string("request00.") + (ext == ".wav" ? "wav" : "mp3"));
    if (rc == 0 && fs::is_regular_file(made, ec)) {
        // the song is only removed with the work folder once it is where it was asked for
        const fs::path dest = fs::u8path(out);
        std::error_code copy_ec;
        if (dest.has_parent_path()) {
            fs::create_directories(dest.parent_path(), copy_ec);
        }
        fs::copy_file(made, dest, fs::copy_options::overwrite_existing, copy_ec);
        if (copy_ec) {
            std::fprintf(stderr, "Kera: couldn't write %s (%s): the song is kept at %s\n", out.c_str(),
                         copy_ec.message().c_str(), made.u8string().c_str());
            return 1;
        }
        std::fprintf(stderr, "Kera: wrote %s\n", out.c_str());
    } else if (rc == 0) {
        rc = 1;
        std::fprintf(stderr, "Kera: the music engine finished without a song (its messages are above)\n");
    }
    fs::remove_all(work, ec); // what is left there is only the engine's own working files
    return rc;
}

int run_music_server(int argc, char ** argv) {
    if (music_engine_missing("ace-server")) {
        return 2;
    }
    music_engine_threads();
    // kera serve's address options and the folder, in the engine's words
    std::vector<std::string> a = { engine_path("music", "ace-server") };
    std::string models = music_models_default();
    bool port_set = false;
    int keep = -1; // --keep-loaded / --no-keep-loaded, or -1: Kera decides
    for (int i = 1; i < argc; ++i) {
        const std::string w = argv[i];
        const bool v = i + 1 < argc;
        if (w == "--music" && v) {
            models = argv[++i];
        } else if ((w == "--port" || w == "--host") && v) {
            port_set = port_set || w == "--port";
            a.insert(a.end(), { w, argv[++i] });
        } else if (w == "--keep-loaded" || w == "--no-fa") {
            keep = w == "--keep-loaded" ? 1 : keep;
            a.push_back(w);
        } else if (w == "--no-keep-loaded") {
            keep = 0;
        } else if ((w == "--adapters" || w == "--device" || w == "--max-seq" || w == "--max-batch") && v) {
            a.insert(a.end(), { w, argv[++i] });
        }
    }
    a.insert(a.end(), { "--models", models });
    music_model_defaults(models);
    if (!port_set) {
        a.insert(a.end(), { "--port", "8080" });
    }
    // The engine loads its four models for each song and frees them after (about 5 s a song on the CPU, more where
    // they go to a graphics chip). With 32 GB or more, they stay loaded between songs (the smallest set holds 3-6
    // GB): the same songs, sooner. --no-keep-loaded frees them after each song
    if (keep < 0) {
        // only with one language model and one diffusion model in the folder: with several, the engine gives a
        // request that names none the one used last instead of its own first choice (rule one, D-033)
        int n_lm = 0, n_dit = 0;
        std::error_code ec;
        for (fs::directory_iterator it(fs::u8path(models), ec), end; !ec && it != end; it.increment(ec)) {
            std::string n = it->path().filename().u8string();
            for (auto & c : n) {
                c = (char) std::tolower((unsigned char) c);
            }
            n_lm += n.rfind("acestep-5hz-lm", 0) == 0 ? 1 : 0;
            n_dit += n.rfind("acestep-v15", 0) == 0 ? 1 : 0;
        }
        machine_profile p;
        detect_system(p);
        if (p.ram_gb >= 31.0 && n_lm == 1 && n_dit == 1) {
            a.push_back("--keep-loaded");
            std::fprintf(stderr, "Kera: this computer has %.0f GB of memory, so the music models stay loaded between songs "
                                 "(--no-keep-loaded frees them after each)\n", p.ram_gb);
        } else if (p.ram_gb >= 31.0) {
            std::fprintf(stderr, "Kera: the folder has several music models, so they are loaded for each song (with "
                                 "--keep-loaded, a request that names no model gets the one used last)\n");
        }
    }
    std::fprintf(stderr, "Kera: serving music models from %s with the music engine (its page is at the address below)\n",
                 models.c_str());
    std::fflush(stderr);
    return run_program(a);
}

int run_speak_cli(int argc, char ** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        std::printf(
            "kera speak: speech from text (llama.cpp's own speech models). For example, with Qwen3-TTS\n"
            "(huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF: the model and its mmproj file side by side):\n\n"
            "  kera speak -m Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf -p \"Hello, this is Kera.\" -o hello.wav\n"
            "  kera speak -m ... -p \"...\" --tts-lang fr --tts-speaker-file my-voice.mp3   (a voice copied from a sample)\n\n"
            "The model's audio decoder (its mmproj file) is found next to it. The model's own weights are its\n"
            "full-precision file (...-BF16.gguf or ...-F16.gguf): use it when it fits this computer's memory; the\n"
            "8-bit and 4-bit files are smaller copies. llama.cpp's options follow.\n\n");
        std::fflush(stdout);
        if (argc < 2) {
            return 1;
        }
    }
    std::vector<std::string> a;
    for (int i = 0; i < argc; ++i) {
        a.push_back(argv[i]);
    }
    std::string model;
    bool mm = false;
    bool out = false;
    for (int i = 1; i < argc; ++i) {
        const std::string w = argv[i];
        if ((w == "-m" || w == "--model") && i + 1 < argc) {
            model = argv[i + 1];
        }
        mm = mm || w == "-mm" || w == "--mmproj" || w == "-hf" || w == "--hf-repo";
        out = out || w == "-o" || w == "--output" || w == "--output-file";
    }
    note_higher_precision(model, "-m");
    if (!model.empty() && !mm) {
        const std::string p = find_projector(model);
        if (!p.empty()) {
            a.insert(a.end(), { "-mm", p });
            std::fprintf(stderr, "Kera: the model's audio decoder: %s\n", p.c_str());
        }
    }
    if (!out) {
        a.insert(a.end(), { "--output", "speech.wav" });
    }
    std::vector<char *> ptrs;
    for (auto & s : a) {
        ptrs.push_back(s.data());
    }
    ptrs.push_back(nullptr);
    return llama_tts((int) a.size(), ptrs.data());
}

} // namespace kera
