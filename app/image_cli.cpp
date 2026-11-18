// Models that make images (the owner's request for 0.1.1, docs/DECISIONS.md D-027). The image engine is
// stable-diffusion.cpp (third_party/), built as programs of its own in the image/ folder next to kera, with its own copy
// of ggml: `kera image` runs its command-line program, and `kera serve` runs its server for an image model, with
// OpenAI's images API (/v1/images/generations and /v1/images/edits) and the AUTOMATIC1111 one (/sdapi/v1/txt2img).
// It runs SD 1.x/2.x/XL, SD3.5, FLUX, Qwen-Image, Z-Image and more, on Vulkan, Metal or the CPU.

#include "app/kera_app.h"
#include "tune/kr_precision.h"
#include "common/kr_paths.h"
#include "setup/kr_machine.h"
#include "tune/kr_settings.h"

#include <cpp-httplib/httplib.h>

#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cctype>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace fs = std::filesystem;

namespace kera {

namespace {

std::string engine(const char * name) {
    return engine_path("image", name);
}

#if defined(_WIN32)
std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

// one word of a command line, quoted the way programs split them again (CommandLineToArgvW)
std::wstring quoted(const std::wstring & a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return a;
    }
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (const wchar_t c : a) {
        if (c == L'\\') {
            ++slashes;
            continue;
        }
        out.append(c == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        slashes = 0;
        out.push_back(c);
    }
    out.append(2 * slashes, L'\\');
    out.push_back(L'"');
    return out;
}
#else
volatile pid_t g_child = 0;

// Ctrl+C or the app stopping kera passes on to the engine, which ends its own way
void pass_on(int sig) {
    if (g_child > 0) {
        kill(g_child, sig);
    }
}
#endif

} // namespace

std::string engine_path(const char * folder, const char * name) {
#if defined(_WIN32)
    const std::string file = std::string(name) + ".exe";
#else
    const std::string file = name;
#endif
    const fs::path here = fs::u8path(exe_dir()) / folder / file;
    std::error_code ec;
    if (!fs::is_regular_file(here, ec)) {
        // a build with several configurations (Visual Studio, Ninja Multi-Config) puts kera in bin/Release/ and the
        // engines, built in one configuration, in bin/<folder>/
        const fs::path up = fs::u8path(exe_dir()).parent_path() / folder / file;
        if (fs::is_regular_file(up, ec)) {
            return up.u8string();
        }
    }
    return here.u8string();
}

int run_program(const std::vector<std::string> & argv) {
    // The graphics chip's built shaders, kept between runs (D-021's file, set in main) in a file of each engine's own
    // (image, music, 3d, sfx): each engine has its own ggml, and its shaders are built the first time a picture or a
    // sound needs them, which made the first few after a start slow (D-032)
    static const std::string vk_base = [] {
        const char * v = std::getenv("GGML_VK_PIPELINE_CACHE");
        return std::string(v != nullptr ? v : "");
    }();
    if (!vk_base.empty() && !argv.empty()) {
        const std::string folder = fs::u8path(argv[0]).parent_path().filename().u8string();
        if (folder == "image" || folder == "music" || folder == "3d" || folder == "sfx") {
            const std::string v = vk_base + "-" + folder;
#if defined(_WIN32)
            _putenv_s("GGML_VK_PIPELINE_CACHE", v.c_str()); // ASCII: main made it so
#else
            setenv("GGML_VK_PIPELINE_CACHE", v.c_str(), 1);
#endif
        }
    }
#if defined(_WIN32)
    std::wstring cmd;
    for (const auto & a : argv) {
        cmd += (cmd.empty() ? L"" : L" ") + quoted(widen(a));
    }
    // in a job that ends with kera: the app stops kera serve by ending it, which would leave the engine running
    static HANDLE job = [] {
        HANDLE j = CreateJobObjectW(nullptr, nullptr);
        if (j != nullptr) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof(li));
        }
        return j;
    }();
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
        return 127;
    }
    if (job != nullptr) {
        AssignProcessToJobObject(job, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return (int) code;
#else
    std::vector<char *> args;
    for (const auto & a : argv) {
        args.push_back(const_cast<char *>(a.c_str()));
    }
    args.push_back(nullptr);
    pid_t child = 0;
    if (posix_spawn(&child, argv[0].c_str(), nullptr, nullptr, args.data(), environ) != 0) {
        return 127;
    }
    g_child = child;
    auto old_int = std::signal(SIGINT, pass_on);
    auto old_term = std::signal(SIGTERM, pass_on);
    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    g_child = 0;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}

namespace {

int run_here(const std::vector<std::string> & argv) {
    // the engine picks some of its methods (flash attention, direct convolutions) by timing both on this graphics
    // chip: a close race can go either way from one start to the next, and the picture with it. The first result
    // is remembered in the Kera folder and used from then on, so a seed gives the same picture every time
    if (std::getenv("SD_PROBE_CACHE") == nullptr) {
        const std::string choices = home_path("image-engine-choices.txt");
#ifdef _WIN32
        // as UTF-16: kera has no UTF-8 code page, so a narrow value would reach the engine garbled for a user name
        // with a non-English letter, and the choices would never be kept
        _wputenv_s(L"SD_PROBE_CACHE", widen(choices).c_str());
#else
        setenv("SD_PROBE_CACHE", choices.c_str(), 0);
#endif
    }
    return run_program(argv);
}

bool engine_missing(const char * name) {
    std::error_code ec;
    if (fs::is_regular_file(fs::u8path(engine(name)), ec)) {
        return false;
    }
    std::fprintf(stderr, "Kera: the image engine isn't here (%s): this download was built without it\n", engine(name).c_str());
    return true;
}

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

// the size of each fixed-size GGUF value type (types 0-12); 8 = string, 9 = array
constexpr int k_gguf_sizes[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };

// the tensor names in a model file's header: a GGUF's (read straight from the file, as stable-diffusion.cpp's own
// GGUF files may have names longer than llama.cpp's reader allows) or a safetensors file's (its JSON header); empty if
// the file can't be read
std::vector<std::string> tensor_names(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream f(fs::u8path(path), std::ios::binary);
    char magic[4] = {};
    if (!f.read(magic, 4)) {
        return out;
    }
    if (std::memcmp(magic, "GGUF", 4) == 0) {
        uint32_t version = 0;
        uint64_t n_tensors = 0, n_kv = 0;
        if (!f.read((char *) &version, 4) || !f.read((char *) &n_tensors, 8) || !f.read((char *) &n_kv, 8) || version < 2) {
            return out;
        }
        auto read_str = [&](std::string & s) {
            uint64_t n = 0;
            if (!f.read((char *) &n, 8) || n > (1u << 20)) {
                return false;
            }
            s.resize((size_t) n);
            return (bool) f.read(s.data(), (std::streamsize) n);
        };
        auto skip_value = [&](uint32_t type, auto & self) -> bool {
            if (type == 8) {
                std::string s;
                return read_str(s);
            }
            if (type == 9) {
                uint32_t et = 0;
                uint64_t count = 0;
                if (!f.read((char *) &et, 4) || !f.read((char *) &count, 8) || et > 12) {
                    return false;
                }
                if (et != 8 && et != 9) {
                    f.seekg((std::streamoff) (count * (uint64_t) k_gguf_sizes[et]), std::ios::cur);
                    return (bool) f;
                }
                for (uint64_t i = 0; i < count; ++i) {
                    if (!self(et, self)) {
                        return false;
                    }
                }
                return true;
            }
            if (type > 12) {
                return false;
            }
            f.seekg(k_gguf_sizes[type], std::ios::cur);
            return (bool) f;
        };
        for (uint64_t i = 0; i < n_kv && i < 100000; ++i) {
            std::string key;
            uint32_t type = 0;
            if (!read_str(key) || !f.read((char *) &type, 4) || !skip_value(type, skip_value)) {
                return {};
            }
        }
        for (uint64_t i = 0; i < n_tensors && i < 1000000; ++i) {
            std::string name;
            uint32_t n_dims = 0, type = 0;
            uint64_t offset = 0;
            if (!read_str(name) || !f.read((char *) &n_dims, 4) || n_dims > 8) {
                return {};
            }
            f.seekg((std::streamoff) (8 * n_dims), std::ios::cur);
            if (!f.read((char *) &type, 4) || !f.read((char *) &offset, 8)) {
                return {};
            }
            out.push_back(name);
        }
        return out;
    }
    // safetensors: an 8-byte little-endian length, then a JSON object keyed by tensor name
    f.seekg(0);
    uint64_t n = 0;
    if (!f.read((char *) &n, 8) || n == 0 || n > (100u << 20)) {
        return out;
    }
    std::string header((size_t) n, '\0');
    if (!f.read(header.data(), (std::streamsize) n)) {
        return out;
    }
    const nlohmann::json j = nlohmann::json::parse(header, nullptr, false);
    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() != "__metadata__") {
                out.push_back(it.key());
            }
        }
    }
    return out;
}

// The image engine's own recipes with their makers' step counts (tools/recipes/make_table.py)
struct image_recipe {
    const char * family;
    const char * names; // "|name|name|": the makers' file names without separators and precision
    int steps;
    double cfg, guidance, shift; // < 0: the engine's own
    int width, height;           // 0: the engine's own (and for video: the user's)
    const char * sampler;
    const char * scheduler;
    bool with_picture; // made to work from a picture (editing, references): its steps apply with one too
    bool video;
};

const image_recipe image_recipes[] = {
#include "image_recipes.inc"
};

// The recipe for a model file's name: the name without its precision tag ("-Q8_0", "_fp8_scaled", "-BF16"), its
// packaging words and its separators must be one of the recipe's own names
const image_recipe * find_image_recipe(const std::string & file_name) {
    std::string stem;
    precision_of(file_name, &stem);
    for (bool again = true; again;) {
        again = false;
        for (const char * w : { "scaled", "comfy", "comfyui", "diffusion", "model", "unet", "transformer", "dit" }) {
            const size_t len = std::strlen(w);
            if (stem.size() > len + 1 && stem.compare(stem.size() - len, len, w) == 0) {
                const char d = stem[stem.size() - len - 1];
                if (d == '-' || d == '_' || d == '.') {
                    stem.resize(stem.size() - len - 1);
                    again = true;
                }
            }
        }
    }
    std::string key = "|";
    for (const char c : stem) {
        if (c != '-' && c != '_' && c != '.' && c != ' ') {
            key += c;
        }
    }
    key += "|";
    if (key.size() <= 2) {
        return nullptr;
    }
    for (const auto & r : image_recipes) {
        if (std::strstr(r.names, key.c_str()) != nullptr) {
            return &r;
        }
    }
    return nullptr;
}

// The model card's settings where the command line sets none (as D-013 does for language models): turbo and schnell
// models are distilled to a few steps without classifier-free guidance, and the engine's defaults (20 steps, guidance
// 7) make them slow and over-cooked; the others get their makers' own step counts, not the engine's 20 (the owner's
// choice, D-035). Recognised by the file's name, and for FLUX by its weights; anything else keeps the engine's
// defaults. A few-step model's steps are for drawing from a description: a picture to start from keeps the user's and
// the engine's steps, since with 1 to 4 steps the engine would ignore it. Where a card's own use is with a picture
// (editing, inpainting, control, video), its steps apply with one too.
void model_card_defaults(std::vector<std::string> & a, const std::string & model, bool server) {
    const std::string n = lower(fs::u8path(model).filename().u8string());
    auto has = [&](const char * w) { return n.find(w) != std::string::npos; };
    auto set = [&](std::initializer_list<const char *> flags) {
        for (const auto & w : a) {
            for (const char * f : flags) {
                if (w == f) {
                    return true;
                }
            }
        }
        return false;
    };
    const bool from_picture = set({ "-i", "--init-img", "--end-img", "-r", "--ref-image", "--mask" });
    int steps = 0;
    bool steps_with_picture = false; // the card's steps apply with a starting picture too (D-035)
    double cfg = -1;
    double guidance = -1;    // FLUX's distilled guidance (--guidance), where the card's isn't the engine's 3.5
    int size = 0;            // the card's width and height
    const char * sampler = nullptr;
    const char * scheduler = nullptr;
    // video: the second (high-noise) model's CFG, the flow shift and expert boundary where the maker's differ from
    // the engine's (shift 5, boundary 0.875), and the negative prompt the maker's own pipeline uses when given none
    double high_cfg = -1, shift = -1, boundary = -1;
    const char * negative = nullptr;
    int fps = 0; // video: the frames a second the model makes, where it isn't the engine's 16 (Wan's)
    std::string key; // the name without separators: "wan2.1_t2v_1.3B" and "Wan2.1-T2V-1.3B" are both "wan21t2v13b"
    for (const char c : n) {
        if (c != '-' && c != '_' && c != '.' && c != ' ') {
            key += c;
        }
    }
    auto has_key = [&](const char * w) { return key.find(w) != std::string::npos; };
    auto value_of = [&](std::initializer_list<const char *> flags) -> std::string {
        for (size_t i = 0; i + 1 < a.size(); ++i) {
            for (const char * f : flags) {
                if (a[i] == f) {
                    return a[i + 1];
                }
            }
        }
        return "";
    };
    std::string family;
    if (has_key("wan21") || has_key("wan22")) {
        // Wan's own pipelines (Wan-Video/Wan2.1 generate.py, Wan2.2 wan/configs), as the engine's own recipes give
        // them; steps stay the engine's. Fine-tunes and speed-ups of Wan (step-distilled, Fun, SkyReels...) have
        // settings of their own: the engine's defaults for those
        for (const char * w : { "lightx2v", "lightning", "distill", "causvid", "turbo", "step", "lora", "fun",
                                "skyreels", "phantom", "animate", "fast", "accvid", "selfforcing" }) {
            if (has_key(w)) {
                return;
            }
        }
        static const char * const wan_negative =
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
            "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
        negative = wan_negative;
        const std::string w = value_of({ "-W", "--width" }), h = value_of({ "-H", "--height" });
        const bool p480 = (w == "832" && h == "480") || (w == "480" && h == "832");
        if (has_key("wan22") && has_key("s2v")) {
            // wan/configs/wan_s2v_14B.py: sample_steps 40, sample_shift 3, sample_guide_scale 4.5
            steps = 40, cfg = 4.5, shift = 3, family = "Wan 2.2 S2V-14B";
        } else if (has_key("wan22") && has_key("ti2v")) {
            steps = 50, cfg = 5, family = "Wan 2.2 TI2V-5B";
        } else if (has_key("wan22") && has_key("i2v")) {
            steps = 40, cfg = 3.5, high_cfg = 3.5, boundary = 0.9, family = "Wan 2.2 I2V-A14B";
        } else if (has_key("wan22") && has_key("t2v")) {
            steps = 40, cfg = 3, high_cfg = 4, shift = 12, family = "Wan 2.2 T2V-A14B";
        } else if (has_key("wan21") && (has_key("flf2v") || has_key("vace"))) {
            steps = 50, cfg = 5, shift = 16, family = has_key("vace") ? "Wan 2.1 VACE" : "Wan 2.1 FLF2V";
        } else if (has_key("wan21") && has_key("i2v")) {
            steps = 40, cfg = 5, shift = p480 ? 3 : -1, family = "Wan 2.1 I2V"; // 3 at 480p, as its generate.py does
        } else if (has_key("wan21") && has_key("t2v")) {
            // its README: "If you are using the T2V-1.3B model, we recommend setting the parameter --sample_guide_scale
            // 6", and its --sample_shift "within the range of 8 to 12"; the 14B: generate.py's guidance 5 and shift 5
            if (has_key("13b")) {
                steps = 50, cfg = 6, shift = 8, family = "Wan 2.1 T2V-1.3B";
            } else {
                steps = 50, cfg = 5, family = "Wan 2.1 T2V";
            }
        } else {
            return;
        }
        steps_with_picture = true; // a video's first picture is conditioning, not a denoising strength
    } else if (has_key("ltx23") || has_key("ltx25")) {
        steps_with_picture = true;
        // LTX 2.3 and 2.5, as the engine's recipes give them: the distilled model is made for 8 steps without CFG.
        // LTX makes 24 frames a second (the recipes' fps; the engine writes 16 unless told, so a clip played slow).
        // Lightricks' own pipeline (ltx_pipelines/utils/constants.py) runs the dev models with 30 steps, a default
        // negative prompt and a guider of four parts; the guider is in the engine (--extra-sample-args, see the
        // help), but none of the three is Kera's default for LTX 2.5 until it has been checked on a long clip there
        // (D-042): what is the default was checked, on the owner's own clip.
        fps = 24;
        if (has_key("distill")) {
            if (has_key("ltx23")) {
                steps = 8, cfg = 1, family = "LTX 2.3 distilled";
            } else {
                family = "LTX 2.5 distilled"; // no recipe for its steps yet: only the frame rate
            }
        } else if (has_key("ltx23")) {
            steps = 30, cfg = 3, family = "LTX 2.3";
        } else {
            steps = 20, cfg = 3, family = "LTX 2.5";
        }
    } else if (has("sdxl_lightning") || has("sdxl-lightning")) {
        if (has("x0")) {
            // the 1-step file predicts the picture itself ("sample"/x0), which this engine can't run as made
            std::fprintf(stderr, "Kera: %s is SDXL-Lightning's 1-step x0 file, which this image engine can't run the way "
                                 "it was made; its 2-step file can\n", fs::u8path(model).filename().u8string().c_str());
            return;
        }
        // its 1-, 2-, 4- and 8-step files, named by their step count ("..._8step_unet.safetensors"); Euler with
        // "trailing" timesteps, as its card says (sgm_uniform is that schedule in this engine)
        steps = 4, cfg = 1.0, size = 1024, sampler = "euler", scheduler = "sgm_uniform", family = "SDXL-Lightning";
        for (int k : { 1, 2, 4, 8 }) {
            const std::string w = std::to_string(k) + "step";
            if (has(w.c_str())) {
                steps = k;
            }
        }
    } else if (has("sd_turbo") || has("sd-turbo") || has("sdxl_turbo") || has("sdxl-turbo")) {
        steps = 1, cfg = 1.0, family = "SD-Turbo / SDXL-Turbo"; // 512x512, the engine's own size
    } else if (has("schnell")) {
        steps = 4, cfg = 1.0, size = 1024, family = "FLUX.1-schnell";
    } else if ((has("z_image") || has("z-image") || has("zimage")) && has("turbo")) {
        // the card's 8 model calls are 9 steps in sd.cpp's count (its own z-image-turbo recipe uses 9)
        steps = 9, cfg = 1.0, size = 1024, family = "Z-Image-Turbo";
    } else if (has("flux") && has("dev")) {
        // FLUX dev models are guidance-distilled (they carry guidance_in and take --guidance, not CFG); a de-distilled
        // one has no guidance_in and keeps classifier-free guidance. Steps: the engine's own.
        bool distilled = false;
        for (const auto & t : tensor_names(model)) {
            if (t.size() >= 27 && (t.compare(t.size() - 27, 27, "guidance_in.in_layer.weight") == 0 ||
                                   t.find("guidance_embedder") != std::string::npos)) {
                distilled = true;
                break;
            }
        }
        if (!distilled) {
            if (has("dedistill") || has("de-distill") || has("de_distill")) {
                // its maker's inference.py: 28 steps, true CFG 3.5, negative prompt "bad photo"
                steps = 28, cfg = 3.5, negative = "bad photo", family = "FLUX.1-dev de-distilled";
            } else {
                return; // its own design, unknown here: the engine's defaults
            }
        } else {
            cfg = 1.0, size = 1024;
            if (has("kontext")) {
                steps = 28, steps_with_picture = true, guidance = 2.5, family = "FLUX.1-Kontext-dev";
            } else if (has("fill")) {
                steps = 50, steps_with_picture = true, guidance = 30, family = "FLUX.1-Fill-dev";
            } else if (has("canny")) {
                steps = 50, steps_with_picture = true, guidance = 30, family = "FLUX.1-Canny-dev";
            } else if (has("depth")) {
                steps = 30, steps_with_picture = true, guidance = 10, family = "FLUX.1-Depth-dev";
            } else if (has("krea")) {
                steps = 28, guidance = 4.5, family = "FLUX.1-Krea-dev";
            } else if (has("flux.2") || has("flux2") || has("flux-2") || has("flux_2")) {
                steps = 50, steps_with_picture = true, guidance = 4, family = "FLUX.2-dev";
            } else {
                steps = 50, family = "FLUX.1-dev"; // guidance 3.5: the engine's own
            }
        }
    }
    int width = size, height = size;
    if (family.empty()) {
        // the image engine's own recipe for the model, with its makers' step count (D-036): by the makers' file name
        // exactly, whatever its precision, so a speed-up variant or a fine-tune under another name gets nothing
        if (const image_recipe * r = find_image_recipe(n)) {
            family = r->family;
            steps = r->steps, steps_with_picture = r->with_picture;
            cfg = r->cfg, guidance = r->guidance, shift = r->shift;
            width = r->width, height = r->height;
            sampler = r->sampler, scheduler = r->scheduler;
        }
    }
    if (family.empty()) {
        return;
    }
    char buf[32];
    std::string added;
    auto note = [&](const std::string & s) { added += (added.empty() ? "" : ", ") + s; };
    if ((!from_picture || steps_with_picture) && steps > 0 && !set({ "--steps" })) {
        a.insert(a.end(), { "--steps", std::to_string(steps) });
        note(std::to_string(steps) + " step(s)");
    }
    if (cfg >= 0 && !set({ "--cfg-scale" })) {
        std::snprintf(buf, sizeof(buf), "%.1f", cfg);
        a.insert(a.end(), { "--cfg-scale", buf });
        note(std::string("CFG ") + buf);
    }
    if (guidance >= 0 && !set({ "--guidance" })) {
        std::snprintf(buf, sizeof(buf), "%g", guidance);
        a.insert(a.end(), { "--guidance", buf });
        note(std::string("guidance ") + buf);
    }
    if (sampler != nullptr && !set({ "--sampling-method" })) {
        a.insert(a.end(), { "--sampling-method", sampler });
        note(std::string("sampler ") + sampler);
    }
    if (scheduler != nullptr && !set({ "--scheduler" })) {
        a.insert(a.end(), { "--scheduler", scheduler });
        note(std::string("schedule ") + scheduler);
    }
    if (!from_picture && width > 0 && height > 0 && !set({ "-W", "--width", "-H", "--height" })) {
        a.insert(a.end(), { "-W", std::to_string(width), "-H", std::to_string(height) });
        note(std::to_string(width) + "x" + std::to_string(height));
    }
    if (high_cfg >= 0 && set({ "--high-noise-diffusion-model" }) && !set({ "--high-noise-cfg-scale" })) {
        std::snprintf(buf, sizeof(buf), "%g", high_cfg);
        a.insert(a.end(), { "--high-noise-cfg-scale", buf });
        note(std::string("high-noise CFG ") + buf);
    }
    if (shift >= 0 && !set({ "--flow-shift" })) {
        std::snprintf(buf, sizeof(buf), "%g", shift);
        a.insert(a.end(), { "--flow-shift", buf });
        note(std::string("flow shift ") + buf);
    }
    if (boundary >= 0 && set({ "--high-noise-diffusion-model" }) && !set({ "--moe-boundary" })) {
        std::snprintf(buf, sizeof(buf), "%g", boundary);
        a.insert(a.end(), { "--moe-boundary", buf });
        note(std::string("expert boundary ") + buf);
    }
    if (fps > 0 && !set({ "--fps" })) {
        a.insert(a.end(), { "--fps", std::to_string(fps) });
        note(std::to_string(fps) + " frames a second");
    }
    if (negative != nullptr && !set({ "-n", "--negative-prompt" })) {
        a.insert(a.end(), { "-n", negative });
        note("the maker's default negative prompt (-n \"" + std::string(negative) + "\"; -n gives your own)");
    }
    if (!added.empty()) {
        std::fprintf(stderr, "Kera: using %s's recommended settings: %s, where the command line sets none\n",
                     family.c_str(), added.c_str());
    }
    if (server && steps > 0 && !steps_with_picture && steps <= 9) {
        std::fprintf(stderr, "Kera: these are for drawing from a description; a picture-to-picture request "
                             "(/sdapi/v1/img2img) should send its own steps: the picture counts only when steps - "
                             "floor(steps x strength) is 2 or more (e.g. 4 steps at strength 0.5)\n");
    }
}

// the model a command line names: -m, else the diffusion model
std::string model_of(int argc, char ** argv) {
    std::string diffusion;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) {
            return argv[i + 1];
        }
        if (std::strcmp(argv[i], "--diffusion-model") == 0) {
            diffusion = argv[i + 1];
        }
    }
    return diffusion;
}

// how the command line names that model
const char * model_flag(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) {
            return "-m";
        }
    }
    return "--diffusion-model";
}

bool is_diffusion_arch(std::string arch) {
    static const char * const diffusion[] = { "sd1", "sd2", "sdxl", "sd3", "flux", "flux2", "aura", "ltxv", "hyvid",
                                              "wan", "hidream", "cosmos", "lumina2", "qwen_image", "chroma", "z_image" };
    arch = lower(arch);
    return std::any_of(std::begin(diffusion), std::end(diffusion), [&](const char * d) { return arch == d; });
}

// general.architecture read straight from a GGUF file's header (its key-value part, before the tensors): "" if it
// names none, "?" if it isn't a GGUF or the header can't be read
std::string gguf_header_arch(const std::string & path) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    char magic[4] = {};
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0 || !f.read((char *) &version, 4) ||
        !f.read((char *) &n_tensors, 8) || !f.read((char *) &n_kv, 8) || version < 2) {
        return "?";
    }
    auto read_str = [&](std::string & out) {
        uint64_t n = 0;
        if (!f.read((char *) &n, 8) || n > (1u << 20)) {
            return false;
        }
        out.resize((size_t) n);
        return (bool) f.read(out.data(), (std::streamsize) n);
    };
    auto skip_value = [&](uint32_t type, auto & self) -> bool {
        if (type == 8) {
            std::string s;
            return read_str(s);
        }
        if (type == 9) {
            uint32_t et = 0;
            uint64_t count = 0;
            if (!f.read((char *) &et, 4) || !f.read((char *) &count, 8) || et > 12) {
                return false;
            }
            if (et != 8 && et != 9) {
                f.seekg((std::streamoff) (count * (uint64_t) k_gguf_sizes[et]), std::ios::cur);
                return (bool) f;
            }
            for (uint64_t i = 0; i < count; ++i) {
                if (!self(et, self)) {
                    return false;
                }
            }
            return true;
        }
        if (type > 12) {
            return false;
        }
        f.seekg(k_gguf_sizes[type], std::ios::cur);
        return (bool) f;
    };
    for (uint64_t i = 0; i < n_kv && i < 100000; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_str(key) || !f.read((char *) &type, 4)) {
            return "?";
        }
        if (key == "general.architecture" && type == 8) {
            std::string arch;
            return read_str(arch) ? arch : "?";
        }
        if (!skip_value(type, skip_value)) {
            return "?";
        }
    }
    return "";
}

} // namespace

bool is_image_model(const std::string & path) {
    const std::string ext = lower(fs::u8path(path).extension().u8string());
    if (ext == ".safetensors" || ext == ".ckpt" || ext == ".sft" || ext == ".pt" || ext == ".pth") {
        return true; // llama.cpp reads only GGUF: these are image models (or their parts)
    }
    if (ext != ".gguf") {
        return false;
    }
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        // a GGUF llama.cpp can't read: stable-diffusion.cpp's own allow longer tensor names than llama.cpp's 64
        // letters, and name no design. A language model that is damaged (a download cut short) still names its
        // design in its header: it stays llama.cpp's, which says what is wrong.
        const std::string arch = gguf_header_arch(path);
        return arch != "?" && (arch.empty() || is_diffusion_arch(arch));
    }
    // language models (and their image readers) always name their design; stable-diffusion.cpp's own GGUF files
    // don't, and the converted ones name a diffusion design
    const int64_t id = gguf_find_key(ctx, "general.architecture");
    std::string arch = id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING ? gguf_get_val_str(ctx, id) : "";
    gguf_free(ctx);
    return arch.empty() || is_diffusion_arch(arch);
}

namespace {

// --improve-prompt MODEL.gguf (D-030): the description (-p) rewritten by a chat model before the engine sees it.
// Taken out of the engine's options; false if it was asked for and couldn't be done
bool apply_prompt_improver(std::vector<std::string> & a) {
    std::string chat;
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] == "--improve-prompt") {
            if (i + 1 >= a.size()) {
                std::fprintf(stderr, "Kera: --improve-prompt needs a chat model's file (a .gguf)\n");
                return false;
            }
            chat = a[i + 1];
            a.erase(a.begin() + (std::ptrdiff_t) i, a.begin() + (std::ptrdiff_t) i + 2);
            break;
        }
    }
    if (chat.empty()) {
        return true;
    }
    size_t p = 0;
    for (size_t i = 1; i + 1 < a.size(); ++i) {
        if (a[i] == "-p" || a[i] == "--prompt") {
            p = i + 1;
        }
    }
    if (p == 0) {
        std::fprintf(stderr, "Kera: --improve-prompt: there is no description (-p) to improve\n");
        return false;
    }
    // the older models read comma-separated phrases; the others (FLUX, Z-Image, Qwen-Image, SD3, video) sentences
    std::string model;
    for (size_t i = 1; i + 1 < a.size(); ++i) {
        if (a[i] == "-m" || a[i] == "--model") {
            model = a[i + 1];
        }
    }
    bool tags = false;
    if (!model.empty()) {
        const std::string arch = lower(gguf_header_arch(model));
        const std::string n = lower(fs::u8path(model).filename().u8string());
        tags = arch == "sd1" || arch == "sd2" || arch == "sdxl" ||
               ((arch.empty() || arch == "?") && (n.find("sdxl") != std::string::npos || n.find("sd_turbo") != std::string::npos ||
                                 n.find("sd-turbo") != std::string::npos || n.find("v1-5") != std::string::npos ||
                                 n.find("sd15") != std::string::npos || n.find("sd_1") != std::string::npos));
    }
    std::string better;
    std::string err;
    if (!improve_prompt(chat, tags, a[p], better, err)) {
        std::fprintf(stderr, "Kera: --improve-prompt: %s\n", err.c_str());
        return false;
    }
    std::fprintf(stderr, "Kera: your description: %s\nKera: improved: %s\n", a[p].c_str(), better.c_str());
    a[p] = better;
    return true;
}

} // namespace

int run_image_cli(int argc, char ** argv) {
    if (engine_missing("sd-cli")) {
        return 2;
    }
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        std::printf(
            "kera image: make a picture from text with an image model (stable-diffusion.cpp, on this computer's graphics\n"
            "chip). For example, with SD-Turbo (one step, a few seconds):\n\n"
            "  kera image -m sd_turbo.gguf -p \"a red apple on a wooden table\" --steps 1 --cfg-scale 1 -o apple.png\n\n"
            "FLUX and others come as separate parts: --diffusion-model, --vae, --clip_l, --t5xxl (or --llm). To serve\n"
            "one to apps (OpenAI's images API): kera serve -m MODEL.\n\n"
            "  --improve-prompt CHAT.gguf  a chat model rewrites the description in more detail first, keeping what it\n"
            "                              says; both are shown. It answers as it does in a chat (thinking too)\n\n"
            "The engine's own options follow.\n\n");
        std::fflush(stdout);
        return run_here({ engine("sd-cli"), "--help" });
    }
    std::vector<std::string> a = { engine("sd-cli") };
    for (int i = 1; i < argc; ++i) {
        a.push_back(argv[i]);
    }
    if (!apply_prompt_improver(a)) {
        return 1;
    }
    model_card_defaults(a, model_of(argc, argv), false);
    note_higher_precision(model_of(argc, argv), model_flag(argc, argv));
    return run_here(a);
}

int run_video_cli(int argc, char ** argv) {
    if (engine_missing("sd-cli")) {
        return 2;
    }
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        std::printf(
            "kera video: make a short video from text (or from a picture) with a video model, through the image engine\n"
            "(stable-diffusion.cpp): Wan 2.1 and 2.2, LTX-Video and others. For example, with Wan 2.1's 1.3B model:\n\n"
            "  kera video --diffusion-model wan2.1_t2v_1.3B.gguf --vae wan_2.1_vae.safetensors --t5xxl umt5-xxl.gguf \\\n"
            "             -p \"a cat walking on a beach at sunset\" --video-frames 33 -W 832 -H 480 -o cat.avi\n\n"
            "Where the command line sets none, Kera adds the model maker's own settings and says which: for Wan, its steps\n"
            "(50 for Wan 2.1), CFG, flow shift and default negative prompt; for LTX, its steps, CFG and 24 frames a\n"
            "second. LTX's makers also guide it with three more terms (slower: four passes a step, not two):\n"
            "  --extra-sample-args ltx_stg=1,ltx_stg_blocks=28,ltx_modality=3,ltx_rescale=0.7,ltx_audio_cfg=7\n"
            "The output is an .avi file (MJPEG), which every video player opens. The engine's own options follow.\n\n");
        std::fflush(stdout);
        return run_here({ engine("sd-cli"), "--help" });
    }
    std::vector<std::string> a = { engine("sd-cli"), "-M", "vid_gen" };
    bool out_set = false;
    for (int i = 1; i < argc; ++i) {
        out_set = out_set || std::strcmp(argv[i], "-o") == 0 || std::strcmp(argv[i], "--output") == 0;
        a.push_back(argv[i]);
    }
    if (!out_set) {
        a.insert(a.end(), { "-o", "video.avi" }); // the engine's default is a picture's name
    }
    if (!apply_prompt_improver(a)) {
        return 1;
    }
    model_card_defaults(a, model_of(argc, argv), false);
    note_higher_precision(model_of(argc, argv), model_flag(argc, argv));
    return run_here(a);
}

int run_image_server(int argc, char ** argv) {
    if (engine_missing("sd-server")) {
        return 2;
    }
    // kera serve's address options in the engine's words; Kera's own options for language models are left out
    std::vector<std::string> a = { engine("sd-server") };
    bool port_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string w = argv[i];
        const bool has_value = i + 1 < argc;
        if (w == "--port" && has_value) {
            a.insert(a.end(), { "--listen-port", argv[++i] });
            port_set = true;
        } else if (w == "--host" && has_value) {
            a.insert(a.end(), { "--listen-ip", argv[++i] });
        } else if ((w == "--backend" || w == "--stream" || w == "--stream-budget-gb") && has_value) {
            ++i;
        } else if (w == "--multi-user" || w == "--no-kv-store" || w == "--no-auto-guess" || w == "--no-guess-controller" ||
                   w == "--no-model-defaults" || w == "--metrics" || w == "--jinja" ||
                   w == "--no-mmproj" || w == "-kvu" || w == "--kv-unified" || w == "--no-warmup" ||
                   w == "--cont-batching" || w == "--mlock" || w == "--no-mmap" || w == "--no-webui" ||
                   w == "--no-jinja" || w == "--slots" || w == "--no-slots" || w == "--props") {
            std::fprintf(stderr, "Kera: %s is a language model's option: not used for an image model\n", w.c_str());
            continue; // the image engine would stop at them (its own -n and --flash-attn, the negative prompt and
                      // its attention, are passed on; -b, a chat model's batch size, is its number of pictures)
        } else if ((w == "-c" ||
                    w == "--ctx-size" || w == "-np" || w == "--parallel" || w == "-ngl" || w == "--n-gpu-layers" ||
                    w == "-ctk" || w == "-ctv" || w == "--cache-type-k" || w == "--cache-type-v" || w == "-b" ||
                    w == "-ub" || w == "--batch-size" || w == "--ubatch-size" || w == "-mm" || w == "--mmproj" ||
                    w == "--cache-ram" || w == "-cram" || w == "--spec-type" || w == "-fa" || w == "-a" ||
                    w == "--alias" || w == "--api-key" || w == "--temp" || w == "--n-predict" || w == "-lm" || w == "--load-mode" || w == "--chat-template" ||
                    w == "--reasoning-budget" || w == "--reasoning-format" || w == "-md" || w == "--model-draft" ||
                    w == "--lora" || w == "--cors-origins" || w == "--ctx-checkpoints" || w == "-to" ||
                    w == "--timeout" || w == "-tb" || w == "--threads-batch") && has_value) {
            std::fprintf(stderr, "Kera: %s is a language model's option: not used for an image model\n", w.c_str());
            ++i;
        } else {
            a.push_back(w);
        }
    }
    if (!port_set) {
        a.insert(a.end(), { "--listen-port", "8080" }); // the port chat apps expect, as for language models
    }
    // load the whole model when the server starts, as for language models, rather than on the first picture: the
    // first request then takes as long as the next ones. The same weights, read the same way
    a.push_back("--eager-load");
    // ...and draw one small throwaway picture once it answers (D-032): on a graphics chip, the first pictures after a
    // start are slow while it builds its shaders and sets up its memory, and the chip's clock rises. Only with a
    // graphics chip (on the processor a picture takes seconds to minutes, with nothing to warm), and never changing a
    // picture: each request has its own seed and the same weights
    std::string warm_port = "8080";
    for (size_t i = 1; i + 1 < a.size(); ++i) {
        if (a[i] == "--listen-port") {
            warm_port = a[i + 1];
        }
    }
    bool warm = kr_setting_get("image.warmup") > 0.5 && std::getenv("KERA_NO_WARMUP") == nullptr;
    if (warm) {
        machine_profile prof;
        warm = load_profile(prof, profile_path()) && !prof.gpus.empty();
    }
    std::atomic<bool> server_done{ false };
    std::thread warmer;
    if (warm) {
        const int port = std::atoi(warm_port.c_str());
        warmer = std::thread([port, &server_done] {
            httplib::Client cli("127.0.0.1", port);
            cli.set_connection_timeout(1, 0);
            cli.set_read_timeout(600, 0);
            const auto t0 = std::chrono::steady_clock::now();
            while (!server_done && std::chrono::steady_clock::now() - t0 < std::chrono::minutes(15)) {
                if (auto r = cli.Get("/v1/models"); r && r->status == 200) {
                    const auto w0 = std::chrono::steady_clock::now();
                    const std::string body =
                        R"({"prompt":"a small warm-up picture<sd_cpp_extra_args>{\"seed\": 1, \"sample_params\": {\"sample_steps\": 1}}</sd_cpp_extra_args>",)"
                        R"("n":1,"size":"512x512"})";
                    auto g = cli.Post("/v1/images/generations", body, "application/json");
                    if (g && g->status == 200) {
                        std::fprintf(stderr, "Kera: the graphics chip is warmed up (a throwaway picture, %.1f s): pictures are "
                                             "at full speed from the first\n",
                                     std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
                    }
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }
    // the engine's defaults for every request; a request can still set its own (sd_cpp_extra_args)
    model_card_defaults(a, model_of(argc, argv), true);
    note_higher_precision(model_of(argc, argv), model_flag(argc, argv));
    std::fprintf(stderr, "Kera: an image model: serving it with the image engine (OpenAI's /v1/images/generations)\n");
    std::fflush(stderr);
    const int rc = run_here(a);
    server_done = true;
    if (warmer.joinable()) {
        warmer.join();
    }
    return rc;
}

} // namespace kera
