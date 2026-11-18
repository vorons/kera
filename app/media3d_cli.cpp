// 3D models and sound effects (version 0.1.2, docs/DECISIONS.md D-031).
//
// 3D: trellis.cpp (third_party/), Microsoft's TRELLIS.2 in C++ on ggml, built as programs of its own in the 3d/ folder
// next to kera, with its own copy of ggml. A picture becomes a textured 3D model (.glb): the background is removed,
// DINOv3 reads the picture, three flow models make the shape and its colours, and the mesh is simplified, unwrapped
// and painted. From text, the image engine draws the picture first (`kera 3d -p ...`).
//
// Sound effects: audio.cpp (third_party/), built the same way into the sfx/ folder. MOSS-SoundEffect v2 (Apache-2.0)
// by default; Stable Audio 3 Small SFX also runs, under its own licence.
//
// Kera ships no weights for either: the user downloads them (`kera 3d --help` and `kera sfx --help` say which).

#include "app/kera_app.h"
#include "common/kr_hash.h"
#include "common/kr_paths.h"
#include "kera_version.h"
#include "setup/kr_machine.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace kera {

namespace {

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

// the physical cores: both engines otherwise count logical CPUs (or a fixed 4), which is twice the cores where a core
// runs two threads and too few on a big CPU
std::string physical_cores() {
    machine_profile p;
    detect_system(p);
    return std::to_string(p.cpu_cores > 0 ? p.cpu_cores : 4);
}

// "-<8 hex>" naming this machine's graphics chips, by back-end and name ("" with none): an engine's `--backend best`
// picks one of them, and numbers worked out on one chip can differ in rounding from another's
std::string graphics_chips_key() {
    machine_profile p;
    setup_options o;
    o.measure_disks = false;
    o.measure_gpu = false; // list them only
    measure_gpus(p, o);
    std::string chips;
    for (const auto & g : p.gpus) {
        chips += g.backend + ":" + g.name + ";";
    }
    return chips.empty() ? "" : "-" + hash_bytes(chips.data(), chips.size()).substr(0, 8);
}

bool engine_missing(const char * folder, const char * name, const char * what) {
    std::error_code ec;
    if (fs::is_regular_file(fs::u8path(engine_path(folder, name)), ec)) {
        return false;
    }
    std::fprintf(stderr, "Kera: the %s engine isn't here (%s): this download was built without it\n", what,
                 engine_path(folder, name).c_str());
    return true;
}

bool is_help(int argc, char ** argv) {
    return argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0;
}

// an environment variable for the engines kera starts (on Windows set as UTF-16: kera itself has no UTF-8 code page,
// so a narrow value with a non-English letter in a path would reach the engine garbled)
void set_env(const char * name, const std::string & value) {
#if defined(_WIN32)
    std::wstring w(MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, w.data(), (int) w.size());
    std::wstring wn(name, name + std::strlen(name));
    SetEnvironmentVariableW(wn.c_str(), w.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

fs::path work_dir(const char * what) {
    std::error_code ec;
    fs::path tmp = fs::temp_directory_path(ec);
    if (ec || !fs::is_directory(tmp, ec)) {
        tmp = fs::u8path(home_path("tmp")); // TMP or TMPDIR names a folder that isn't there
    }
    // folders left by a run that was stopped part-way (Ctrl+C on Windows ends the engine and kera at once)
    const std::string prefix = std::string("kera-") + what + "-";
    const auto day_ago = fs::file_time_type::clock::now() - std::chrono::hours(24);
    std::vector<fs::path> old;
    for (fs::directory_iterator it(tmp, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code t_ec;
        if (it->path().filename().u8string().rfind(prefix, 0) == 0 && fs::last_write_time(it->path(), t_ec) < day_ago &&
            !t_ec) {
            old.push_back(it->path());
        }
    }
    for (const auto & o : old) {
        fs::remove_all(o, ec);
    }
    const fs::path w = tmp / (prefix + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ec.clear();
    fs::create_directories(w, ec);
    return w;
}

// the engine's file, made in the work folder, to where it was asked for; false (and why) if it can't be written
bool deliver(const fs::path & made, const std::string & out, const char * what) {
    const fs::path dest = fs::u8path(out);
    std::error_code ec;
    if (dest.has_parent_path()) {
        fs::create_directories(dest.parent_path(), ec);
    }
    fs::copy_file(made, dest, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::fprintf(stderr, "Kera: couldn't write %s (%s)\n", out.c_str(), ec.message().c_str());
        return false;
    }
    std::fprintf(stderr, "Kera: wrote the %s to %s\n", what, out.c_str());
    return true;
}

// ----------------------------------------------------------------------------------------------------------------
// kera 3d
// ----------------------------------------------------------------------------------------------------------------

std::string models_3d_default() {
    return home_path("models/3d");
}

void help_3d() {
    std::printf(
        "kera 3d: a textured 3D model (.glb) from a picture, or from text (TRELLIS.2, on this computer). For example:\n\n"
        "  kera 3d -i chair.png -o chair.glb\n"
        "  kera 3d -p \"a wooden chair on a plain white background\" --image-model sd_turbo.safetensors -o chair.glb\n\n"
        "Options:\n"
        "  -i, --image FILE        the picture (PNG or JPEG): one object, ideally on a plain or transparent background\n"
        "  -p, --prompt TEXT       or a description: the image engine draws it first, exactly as written, then it\n"
        "                          becomes 3D. The 3D step reads one whole object on a plain background best, so say\n"
        "                          so if you want it. Needs an image model: --image-model FILE (any model `kera image`\n"
        "                          runs; turbo models are quickest), or the image engine's own options after `--`\n"
        "  -o, --output FILE       the .glb to write (default model.glb); .glb opens in Windows' 3D Viewer, Blender,\n"
        "                          macOS Preview and most 3D tools\n"
        "  --resolution 512|1024   the shape's resolution (default 1024, the 3D engine's own)\n"
        "  --seed N                the same seed, picture and settings make the same model (default: a new random\n"
        "                          seed each time, as the 3D engine does; with -p the image engine's own default)\n"
        "  --no-texture            the shape only, without colours (quicker)\n"
        "  --models DIR            the folder with the model files (default %s)\n"
        "  --cpu                   run on the processor even if there is a graphics chip (very slow)\n\n"
        "The folder needs TRELLIS.2's weights as GGUF files, from huggingface.co/ilintar/trellis2-gguf (its q4/\n"
        "folder is the smallest, about 5 GB; q8/ is 8-bit; the files at the top are the model's own weights:\n"
        "use those when they fit this computer's graphics memory). Put these in the folder, under these names:\n"
        "  dinov3.gguf, ss_flow.gguf, ss_dec.gguf, shape_flow_512.gguf, shape_dec.gguf, tex_flow_512.gguf, tex_dec.gguf\n"
        "  shape_flow_1024.gguf and tex_flow_1024.gguf      for --resolution 1024\n"
        "  birefnet.gguf                                    removes a picture's background (a picture already cut\n"
        "                                                   out, with transparency, keeps its own), as TRELLIS.2 does\n"
        "Licences: TRELLIS.2 (Microsoft) and BiRefNet are MIT. dinov3.gguf is Meta's DINOv3 picture reader, under\n"
        "Meta's DINOv3 License: by downloading it you accept that licence (its terms are at\n"
        "huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). Kera ships none of these files.\n"
        "A graphics chip with 8 GB or more is recommended; on the processor alone a model takes a very long time.\n",
        models_3d_default().c_str());
}

// the files a run needs, by its options, that the folder doesn't have
std::vector<std::string> missing_3d(const std::string & models, int resolution, bool texture) {
    // birefnet.gguf too: without it the 3D engine cuts a picture's background out by its brightness, which TRELLIS.2's
    // own pipeline never does (rule one, D-033)
    std::vector<std::string> need = { "dinov3.gguf", "ss_flow.gguf", "ss_dec.gguf", "shape_flow_512.gguf",
                                      "shape_dec.gguf", "birefnet.gguf" };
    if (resolution >= 1024) {
        need.push_back("shape_flow_1024.gguf");
    }
    if (texture) {
        need.insert(need.end(), { "tex_flow_512.gguf", "tex_dec.gguf" });
        if (resolution >= 1024) {
            need.push_back("tex_flow_1024.gguf");
        }
    }
    std::vector<std::string> missing;
    std::error_code ec;
    for (const auto & f : need) {
        if (!fs::is_regular_file(fs::u8path(models) / f, ec)) {
            missing.push_back(f);
        }
    }
    return missing;
}

// the picture for `kera 3d -p`: the image engine, with the model card's sampling (kera image's own path), at the
// seed given (none given: the image engine's own default), from the user's description exactly as written (rule
// one, D-033: Kera adds no words of its own)
int draw_picture(const std::string & prompt, const std::string & image_model, const std::vector<std::string> & extra,
                 const std::string & seed, const fs::path & out) {
    const std::string & full = prompt;
    std::fprintf(stderr, "Kera: drawing the picture first (the image engine): \"%s\"\n", full.c_str());
    std::vector<std::string> a = { "image" };
    if (!image_model.empty()) {
        a.insert(a.end(), { "-m", image_model });
    }
    a.insert(a.end(), extra.begin(), extra.end());
    a.insert(a.end(), { "-p", full, "-o", out.u8string() });
    if (!seed.empty()) {
        a.insert(a.end(), { "--seed", seed });
    }
    std::vector<char *> ptrs;
    for (auto & s : a) {
        ptrs.push_back(s.data());
    }
    ptrs.push_back(nullptr);
    return run_image_cli((int) a.size(), ptrs.data());
}

} // namespace

int run_3d_cli(int argc, char ** argv) {
    if (is_help(argc, argv)) {
        help_3d();
        return argc < 2 ? 1 : 0;
    }
    std::string image;
    std::string prompt;
    std::string out = "model.glb";
    std::string models = models_3d_default();
    std::string seed; // none: each engine's own default (the 3D engine's is a new random seed each time, D-033)
    std::string image_model;
    std::vector<std::string> image_extra; // the image engine's own options, after --
    int resolution = 1024;
    bool texture = true;
    bool cpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool v = i + 1 < argc;
        if (a == "--") {
            image_extra.assign(argv + i + 1, argv + argc);
            break;
        } else if ((a == "-i" || a == "--image") && v) {
            image = argv[++i];
        } else if ((a == "-p" || a == "--prompt") && v) {
            prompt = argv[++i];
        } else if ((a == "-o" || a == "--output") && v) {
            out = argv[++i];
        } else if (a == "--models" && v) {
            models = argv[++i];
        } else if ((a == "--resolution" || a == "--res") && v) {
            resolution = std::atoi(argv[++i]);
            if (resolution != 512 && resolution != 1024) {
                std::fprintf(stderr, "kera 3d: --resolution is 512 or 1024\n");
                return 1;
            }
        } else if ((a == "--seed" || a == "-s") && v) {
            seed = argv[++i];
            char * end = nullptr;
            const unsigned long long n = std::strtoull(seed.c_str(), &end, 10);
            // the 3D engine reads the seed as a signed 32-bit number on Windows: above that, every seed was the same
            if (seed.empty() || *end != '\0' || seed[0] == '-' || n > 0x7fffffffull || n == 0) {
                // 0 asks the 3D engine for a new random seed each run: leave --seed out for that
                std::fprintf(stderr, "kera 3d: --seed is a whole number from 1 to 2147483647 (without it, each run "
                                     "gets a new random seed)\n");
                return 1;
            }
        } else if (a == "--no-texture") {
            texture = false;
        } else if (a == "--cpu") {
            cpu = true;
        } else if (a == "--image-model" && v) {
            image_model = argv[++i];
        } else {
            std::fprintf(stderr, "kera 3d: unknown option %s (kera 3d --help lists them)\n", a.c_str());
            return 1;
        }
    }
    if (image.empty() == prompt.empty()) {
        std::fprintf(stderr, "kera 3d: give a picture (-i FILE) or a description (-p \"...\"), one of the two\n");
        return 1;
    }
    if (!prompt.empty() && image_model.empty() && image_extra.empty()) {
        std::fprintf(stderr,
                     "kera 3d: making a 3D model from text draws a picture first, which needs an image model:\n"
                     "  kera 3d -p \"%s\" --image-model MODEL -o %s\n"
                     "MODEL is any model `kera image` runs, for example SD-Turbo (sd_turbo.safetensors from\n"
                     "huggingface.co/stabilityai/sd-turbo, about 5 GB) or a GGUF of it; models in parts (FLUX, Z-Image) take\n"
                     "the image engine's options after `--` (kera image --help lists them).\n",
                     prompt.c_str(), out.c_str());
        return 2;
    }
    if (engine_missing("3d", "trellis-cli", "3D") || (!prompt.empty() && engine_missing("image", "sd-cli", "image"))) {
        return 2;
    }
    std::error_code ec;
#if defined(_WIN32)
    // the Vulkan build of the 3D engine runs two of its own steps through Vulkan's loader, which graphics drivers
    // install; the engine loads it only then, and would stop there without it
    if (fs::exists(fs::u8path(exe_dir()) / "3d" / "ggml-vulkan.dll", ec)) {
        HMODULE vk = LoadLibraryW(L"vulkan-1.dll");
        if (vk == nullptr) {
            std::fprintf(stderr, "kera 3d: this PC has no Vulkan (vulkan-1.dll), which comes with the graphics driver: "
                                 "install or update the graphics chip's driver\n");
            return 2;
        }
        FreeLibrary(vk);
    }
#endif
    if (!fs::is_directory(fs::u8path(models), ec)) {
        std::fprintf(stderr, "kera 3d: no model folder at %s (kera 3d --help says what goes in it)\n", models.c_str());
        return 2;
    }
    const auto missing = missing_3d(models, resolution, texture);
    if (!missing.empty()) {
        std::string list;
        for (const auto & f : missing) {
            list += (list.empty() ? "" : ", ") + f;
        }
        std::fprintf(stderr, "kera 3d: the model folder %s doesn't have %s (kera 3d --help says where they come from)\n",
                     models.c_str(), list.c_str());
        return 2;
    }
    if (!image.empty() && !fs::is_regular_file(fs::u8path(image), ec)) {
        std::fprintf(stderr, "kera 3d: can't find the picture %s\n", image.c_str());
        return 1;
    }

    const fs::path work = work_dir("3d");
    if (!prompt.empty()) {
        const fs::path picture = work / "picture.png";
        const int rc = draw_picture(prompt, image_model, image_extra, seed, picture);
        if (rc != 0 || !fs::is_regular_file(picture, ec)) {
            std::fprintf(stderr, "Kera: the image engine didn't make the picture (its messages are above)\n");
            fs::remove_all(work, ec);
            return rc != 0 ? rc : 1;
        }
        // kept next to the model, so the picture it came from can be seen (and used again with -i)
        fs::path keep = fs::u8path(out);
        keep.replace_extension(".png");
        if (keep.has_parent_path()) {
            fs::create_directories(keep.parent_path(), ec);
        }
        // never over a picture already there (perhaps the one an earlier kera 3d -i used): chair-picture.png, ...
        for (int k = 1; fs::exists(keep, ec) && k < 1000; ++k) {
            fs::path alt = fs::u8path(out);
            alt.replace_filename(fs::u8path(out).stem().u8string() + (k == 1 ? std::string("-picture")
                                                                               : "-picture-" + std::to_string(k)) +
                                 ".png");
            keep = alt;
        }
        fs::copy_file(picture, keep, fs::copy_options::none, ec);
        image = ec ? picture.u8string() : keep.u8string();
        if (!ec) {
            std::fprintf(stderr, "Kera: the picture is %s; now the 3D model\n", keep.u8string().c_str());
        }
    }
    const fs::path made = work / "model.glb";
    std::vector<std::string> a = { engine_path("3d", "trellis-cli"), "--image", image, "--output", made.u8string(),
                                   "--models", models, "--res", std::to_string(resolution),
                                   "--threads", physical_cores() };
    if (!seed.empty()) {
        a.insert(a.end(), { "--seed", seed });
    }
    if (!prompt.empty()) {
        // a drawn picture is never cut out already: BiRefNet removes its background (and says so if it can't run)
        a.insert(a.end(), { "--bg-removal", "birefnet" });
    }
    if (!texture) {
        a.push_back("--no-texture");
    }
    if (cpu) {
        a.insert(a.end(), { "--gpu", "-1" });
    }
    std::fprintf(stderr, "Kera: making the 3D model at resolution %d%s%s (a few minutes on a graphics chip)\n",
                 resolution, seed.empty() ? "" : ", seed ", seed.c_str());
    std::fflush(stderr);
    int rc = run_program(a);
    if (rc == 0 && !fs::is_regular_file(made, ec)) {
        rc = 1;
        std::fprintf(stderr, "Kera: the 3D engine finished without a model (its messages are above)\n");
    } else if (rc == 0 && !deliver(made, out, "3D model")) {
        std::fprintf(stderr, "Kera: the model is kept at %s\n", made.u8string().c_str());
        return 1; // the work folder stays, with the model in it
    }
    fs::remove_all(work, ec);
    return rc;
}

// ----------------------------------------------------------------------------------------------------------------
// kera sfx
// ----------------------------------------------------------------------------------------------------------------

namespace {

std::string models_sfx_default() {
    return home_path("models/sfx");
}

void help_sfx() {
    std::printf(
        "kera sfx: a sound effect from a description (MOSS-SoundEffect, on this computer). For example:\n\n"
        "  kera sfx -p \"a wooden door creaking open slowly\" --duration 4 -o door.wav\n"
        "  kera sfx -p \"steady rain on a tin roof, distant thunder\" --duration 20 --seed 7 -o rain.wav\n\n"
        "Options:\n"
        "  -p, --prompt TEXT     what it sounds like (English or Chinese)\n"
        "  --duration SECONDS    how long (default: the model's own: 10 s for MOSS-SoundEffect, which makes 0.1 to\n"
        "                        30 s; Stable Audio: the longest its file allows)\n"
        "  --seed N              the same seed and settings make the same sound (default: the model's own; Stable\n"
        "                        Audio picks a new random seed each time)\n"
        "  --steps N             denoising steps (the model's own: 100 for MOSS-SoundEffect, 8 for Stable Audio)\n"
        "  -o, --output FILE     the .wav to write (default sound.wav): 48 kHz for MOSS-SoundEffect\n"
        "  --models DIR          the folder with the model file (default %s)\n"
        "  --model FILE          which model file, if not the folder's\n"
        "  --cpu                 run on the processor even if there is a graphics chip\n\n"
        "The folder needs one model file. The default, MOSS-SoundEffect v2 (Apache-2.0: free to use, no sign-up):\n"
        "  moss-soundeffect-v2-q8_0.gguf (4.0 GB), from huggingface.co/audio-cpp/audio.cpp-gguf, in its folder\n"
        "  MOSS-SoundEffect-v2-GGUF (or moss-soundeffect-v2-bf16.gguf, 6.6 GB; with both, the BF16 file is used:\n"
        "  the model's own weights; --model picks another)\n"
        "Many sounds in a row: kera serve --sfx DIR keeps the model loaded (the same sounds, each without the load).\n"
        "Also runs Stable Audio 3 Small SFX (stable-audio-3-small-sfx-q8_0.gguf, 1.7 GB, the same place, folder\n"
        "Stable-Audio-3-Small-SFX-GGUF), which is under the Stability AI Community License (free below USD 1M of\n"
        "yearly revenue, with \"Powered by Stability AI\" shown; its text encoder under Google's Gemma terms).\n"
        "Kera ships no model files. A graphics chip makes it much quicker: MOSS-SoundEffect takes several minutes per\n"
        "sound on a processor.\n",
        models_sfx_default().c_str());
}

// the model families audio.cpp knows, from a file's name ("" if neither)
std::string sfx_family(const std::string & path) {
    const std::string n = lower(fs::u8path(path).filename().u8string());
    if (n.find("moss") != std::string::npos && (n.find("sound") != std::string::npos || n.find("sfx") != std::string::npos)) {
        return "moss_soundeffect";
    }
    if (n.find("stable-audio") != std::string::npos || n.find("stable_audio") != std::string::npos) {
        return "stable_audio";
    }
    return "";
}

// the folder's model: MOSS-SoundEffect first (its full-precision file before Q8_0 and Q4), then Stable Audio; "" if none
std::string find_sfx_model(const std::string & dir) {
    std::vector<std::string> found;
    std::error_code ec;
    for (fs::directory_iterator it(fs::u8path(dir), ec), end; !ec && it != end; it.increment(ec)) {
        const fs::path p = it->path();
        if (lower(p.extension().u8string()) == ".gguf" && !sfx_family(p.u8string()).empty()) {
            found.push_back(p.u8string());
        }
    }
    auto rank = [](const std::string & p) {
        const std::string n = lower(fs::u8path(p).filename().u8string());
        const int family = sfx_family(p) == "moss_soundeffect" ? 0 : 1;
        // the model's own weights first (full precision: f32, bf16, f16), then 8-bit, then 4-bit (rule one, D-033)
        const int smaller = n.find("q4") != std::string::npos ? 2 : n.find("q8") != std::string::npos ? 1 : 0;
        return family * 10 + smaller;
    };
    std::sort(found.begin(), found.end(), [&](const std::string & a, const std::string & b) {
        return rank(a) != rank(b) ? rank(a) < rank(b) : a < b;
    });
    return found.empty() ? "" : found.front();
}

} // namespace

// The engine's name for the graphics back-end Kera's own build sees ("cuda", "vulkan", "metal"), "cpu" without one:
// its server takes a named back-end, not "best"
std::string sfx_server_backend() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU || props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            const std::string reg = lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
            for (const char * known : { "cuda", "vulkan", "metal" }) {
                if (reg.find(known) != std::string::npos) {
                    return known;
                }
            }
        }
    }
    return "cpu";
}

// kera serve --sfx DIR: the sound-effect engine's own server with the folder's model kept loaded (D-039). A sound
// made through it is the same samples as kera sfx makes with the same words and seed; only the model's load (about a
// fifth of a 10 s sound's time on a DGX Spark) is paid once instead of for every sound.
int run_sfx_server(int argc, char ** argv) {
    if (engine_missing("sfx", "audiocpp_server", "sound-effect")) {
        return 2;
    }
    std::string models = models_sfx_default(), model, port = "8080", host = "127.0.0.1";
    bool cpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string w = argv[i];
        const bool v = i + 1 < argc;
        if (w == "--sfx" && v) {
            models = argv[++i];
        } else if (w == "--port" && v) {
            port = argv[++i];
        } else if (w == "--host" && v) {
            host = argv[++i];
        } else if (w == "--cpu") {
            cpu = true;
        }
    }
    // the port is a number, and nothing else: it also names the settings file below
    if (port.empty() || port.size() > 5 || port.find_first_not_of("0123456789") != std::string::npos ||
        std::atoi(port.c_str()) < 1 || std::atoi(port.c_str()) > 65535) {
        std::fprintf(stderr, "kera serve --sfx: --port takes a number from 1 to 65535, not %s\n", port.c_str());
        return 2;
    }
    std::error_code ec;
    if (fs::is_regular_file(fs::u8path(models), ec)) {
        model = models;
    } else if (fs::is_directory(fs::u8path(models), ec)) {
        model = find_sfx_model(models);
    }
    const std::string family = model.empty() ? "" : sfx_family(model);
    if (family.empty()) {
        std::fprintf(stderr, "kera serve --sfx: no sound-effect model in %s (kera sfx --help says which files it runs)\n",
                     models.c_str());
        return 2;
    }
    nlohmann::json cfg = {
        { "host", host }, { "port", std::atoi(port.c_str()) }, { "backend", cpu ? "cpu" : sfx_server_backend() },
        { "device", 0 }, { "threads", std::atoi(physical_cores().c_str()) }, { "lazy_load", false },
        { "models", nlohmann::json::array({ { { "id", "sfx" }, { "family", family },
                                              { "path", fs::absolute(fs::u8path(model), ec).u8string() },
                                              { "task", "gen" }, { "mode", "offline" } } }) },
    };
    const fs::path dir = fs::u8path(home_path("tmp"));
    fs::create_directories(dir, ec);
    const fs::path file = dir / ("sfx-server-" + port + ".json");
    {
        // a folder or host name that isn't UTF-8 is written with its odd letters replaced, not thrown at the user
        std::ofstream f(file);
        f << cfg.dump(1, ' ', false, nlohmann::json::error_handler_t::replace);
        if (!f.good()) {
            std::fprintf(stderr, "kera serve --sfx: couldn't write %s\n", file.u8string().c_str());
            return 1;
        }
    }
    std::fprintf(stderr,
                 "Kera: serving %s with the sound-effect engine, kept loaded. A sound: POST http://%s:%s/v1/tasks/run with\n"
                 "  {\"model\": \"sfx\", \"request\": {\"text\": \"rain on a tin roof\", \"seed\": 5, \"options\": "
                 "{\"duration_sec\": \"10\"}}}\n"
                 "The reply's \"audio\" is the WAV in Base64: the same samples as kera sfx with those words and that seed.\n",
                 fs::u8path(model).filename().u8string().c_str(), host.c_str(), port.c_str());
    std::fflush(stderr);
    return run_program({ engine_path("sfx", "audiocpp_server"), "--config", file.u8string() });
}

int run_sfx_cli(int argc, char ** argv) {
    if (is_help(argc, argv)) {
        help_sfx();
        return argc < 2 ? 1 : 0;
    }
    std::string prompt;
    std::string out = "sound.wav";
    std::string models = models_sfx_default();
    std::string model;
    std::string seed; // none: the model's own default (D-033)
    std::string steps;
    std::string duration; // the user's own number, passed on as written; none: the model's own length (D-033)
    bool cpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool v = i + 1 < argc;
        if ((a == "-p" || a == "--prompt") && v) {
            prompt = argv[++i];
        } else if (a == "--duration" && v) {
            duration = argv[++i];
            char * end = nullptr;
            const double d = std::strtod(duration.c_str(), &end);
            if (duration.empty() || *end != '\0' || !(d > 0) || !std::isfinite(d)) {
                std::fprintf(stderr, "kera sfx: --duration is a number of seconds\n");
                return 1;
            }
        } else if ((a == "--seed" || a == "-s") && v) {
            seed = argv[++i];
            char * end = nullptr;
            const unsigned long long n = std::strtoull(seed.c_str(), &end, 10);
            if (seed.empty() || *end != '\0' || seed[0] == '-' || n > 0xffffffffull) {
                std::fprintf(stderr, "kera sfx: --seed is a whole number from 0 to 4294967295\n");
                return 1;
            }
        } else if (a == "--steps" && v) {
            steps = argv[++i];
            if (std::atoi(steps.c_str()) < 1) {
                std::fprintf(stderr, "kera sfx: --steps is 1 or more\n");
                return 1;
            }
        } else if ((a == "-o" || a == "--output") && v) {
            out = argv[++i];
        } else if (a == "--models" && v) {
            models = argv[++i];
        } else if ((a == "--model" || a == "-m") && v) {
            model = argv[++i];
        } else if (a == "--cpu") {
            cpu = true;
        } else {
            std::fprintf(stderr, "kera sfx: unknown option %s (kera sfx --help lists them)\n", a.c_str());
            return 1;
        }
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "kera sfx: say what it sounds like with -p \"...\"\n");
        return 1;
    }
    if (engine_missing("sfx", "audiocpp_cli", "sound-effect")) {
        return 2;
    }
    std::error_code ec;
    if (model.empty()) {
        if (!fs::is_directory(fs::u8path(models), ec)) {
            std::fprintf(stderr, "kera sfx: no model folder at %s (kera sfx --help says what goes in it)\n", models.c_str());
            return 2;
        }
        model = find_sfx_model(models);
        if (model.empty()) {
            std::fprintf(stderr,
                         "kera sfx: no sound-effect model in %s: it needs moss-soundeffect-v2-q8_0.gguf (kera sfx --help "
                         "says where it comes from)\n", models.c_str());
            return 2;
        }
    } else if (!fs::is_regular_file(fs::u8path(model), ec)) {
        std::fprintf(stderr, "kera sfx: can't find the model %s\n", model.c_str());
        return 2;
    }
    const std::string family = sfx_family(model);
    // MOSS-SoundEffect makes 0.1 to 30 s, after rounding the length to tenths as the engine does (it checks too; this
    // says so before its 2 GB load)
    if (family == "moss_soundeffect" && !duration.empty()) {
        std::ostringstream tenths;
        tenths << std::fixed << std::setprecision(1) << std::strtod(duration.c_str(), nullptr);
        const double d = std::strtod(tenths.str().c_str(), nullptr);
        if (d < 0.1 || d > 30) {
            std::fprintf(stderr, "kera sfx: MOSS-SoundEffect makes 0.1 to 30 seconds\n");
            return 1;
        }
    }

    // the model's text encodings, kept per model file in the Kera folder: the same description again (another seed)
    // then skips its 2 GB text encoder (the engine reads AUDIOCPP_TEXT_CACHE; the same sound, byte for byte). The
    // folder is named after the file's contents, this build and the back-end, so an encoding is only ever reused by
    // the same model on the same engine (rule one, D-033); folders of other builds are removed.
    if (std::getenv("AUDIOCPP_TEXT_CACHE") == nullptr) {
        int h_err = 0;
        const std::string mh = model_hash(model, &h_err); // "xxh128:<hex>", remembered by path, size and time
        // the engine itself, by content (a version label set when the build was configured can stay the same across
        // rebuilds of a changed engine)
        const std::string eh = model_hash(engine_path("sfx", "audiocpp_cli"), &h_err);
        const size_t e_colon = eh.find(':');
        const std::string build = "-" + (e_colon == std::string::npos ? std::string(KERA_VERSION "-" KERA_GIT_COMMIT)
                                                                     : eh.substr(e_colon + 1, 16));
        const fs::path root = fs::u8path(home_path("sfx-text-cache"));
        for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string n = it->path().filename().u8string();
            if (n.find(build) == std::string::npos) {
                std::error_code rm_ec;
                fs::remove_all(it->path(), rm_ec);
            }
        }
        ec.clear();
        const size_t colon = mh.find(':');
        if (colon != std::string::npos) {
            // the processor's, or the graphics chips': a text read on another chip can differ in rounding
            const fs::path dir = root / (mh.substr(colon + 1) + build + (cpu ? "-cpu" : graphics_chips_key()));
            // the oldest beyond 256 prompts (4 MB each) go
            std::vector<std::pair<fs::file_time_type, fs::path>> files;
            for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
                std::error_code t_ec;
                files.emplace_back(fs::last_write_time(it->path(), t_ec), it->path());
            }
            ec.clear();
            if (files.size() > 256) {
                std::sort(files.begin(), files.end());
                for (size_t i = 0; i + 256 < files.size(); ++i) {
                    fs::remove(files[i].second, ec);
                }
            }
            set_env("AUDIOCPP_TEXT_CACHE", dir.u8string());
        }
    }
    const fs::path work = work_dir("sfx");
    const fs::path made = work / "sound.wav";
    std::vector<std::string> a = { engine_path("sfx", "audiocpp_cli"), "--task", "gen", "--model", model,
                                   "--backend", cpu ? "cpu" : "best", "--threads", physical_cores(),
                                   "--text", prompt, "--out", made.u8string() };
    if (!seed.empty()) {
        a.insert(a.end(), { "--seed", seed });
    }
    if (!family.empty()) {
        a.insert(a.end(), { "--family", family });
    }
    // the length is the same option for both families (duration_sec), passed only when given; the steps are named
    // differently
    if (!duration.empty()) {
        a.insert(a.end(), { "--duration-seconds", duration });
    }
    if (!steps.empty()) {
        if (family == "stable_audio") {
            a.insert(a.end(), { "--num-inference-steps", steps });
        } else {
            a.insert(a.end(), { "--request-option", "num_steps=" + steps });
        }
    }
    std::fprintf(stderr, "Kera: making %s of sound with %s%s%s\n",
                 duration.empty() ? "the model's own length" : (duration + " s").c_str(),
                 fs::u8path(model).filename().u8string().c_str(), seed.empty() ? "" : ", seed ", seed.c_str());
    std::fflush(stderr);
    int rc = run_program(a);
    if (rc == 0 && !fs::is_regular_file(made, ec)) {
        rc = 1;
        std::fprintf(stderr, "Kera: the sound-effect engine finished without a sound (its messages are above)\n");
    } else if (rc == 0 && !deliver(made, out, "sound")) {
        std::fprintf(stderr, "Kera: the sound is kept at %s\n", made.u8string().c_str());
        return 1;
    }
    fs::remove_all(work, ec);
    return rc;
}

} // namespace kera
