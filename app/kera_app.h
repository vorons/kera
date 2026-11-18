// Kera's own commands, dispatched by app/main.cpp.
#pragma once

#include "planner/backend_pick.h"

#include <string>
#include <vector>

namespace kera {

int run_app(int argc, char ** argv);    // kera app: the local web page (the default with no command)
int run_setup_cli(int argc, char ** argv); // kera setup: measure this computer and explain it
int run_plan_cli(int argc, char ** argv);  // kera plan: the speed ceiling for a model on this computer
int run_finetune_cli(int argc, char ** argv); // kera finetune: a LoRA from your own examples (Phase 10)
int run_perplexity_cli(int argc, char ** argv); // kera perplexity: llama.cpp's, with Kera's streaming options
// --backend auto|vulkan|hip|cpu: the llama.cpp device to use ("" = llama.cpp's own choice); loads the optional
// ROCm add-on for hip. Returns false (and why) if it can't.
bool choose_backend(const std::string & which, std::string & device, std::string * error);
bool hip_addon_installed(); // the optional ROCm add-on's files are in Kera's folder
int run_calibrate_cli(int argc, char ** argv); // kera calibrate: a back-end's speed on a model, for serve's choice
// kera serve's choice between Vulkan and the ROCm add-on (--backend auto or the setting gpu.auto_backend,
// docs/DECISIONS.md D-022): the faster for this model and how it has been used, each measured once per model.
// argv is kera serve's command line without Kera's own options; "" leaves the choice to llama.cpp.
std::string auto_backend(const std::string & model, int argc, char ** argv);
// Each back-end's measured speed on the model (measuring it once if needed); empty without two to choose between
std::vector<backend_speed> backend_speeds(const std::string & model, int argc, char ** argv);
// kera serve --backend auto when memory holds the model twice (docs/DECISIONS.md D-024): the model on both back-ends,
// each request sent to the one that finishes it sooner. argv is kera serve's whole command line, llama_args the part
// llama.cpp reads. Returns kera serve's exit code, or -1 if it can't (too little memory, one back-end): then the
// caller picks one back-end for the model.
int serve_each_message(const std::string & model, int argc, char ** argv, int n_llama, char ** llama_args);
// Notes each reply's shape (llama.cpp hook 6) for the next start's choice, until stop_noting_replies(), which also
// says how the replies' times compared with the back-end's measurement
void note_replies(const std::string & model, const std::string & backend, int argc, char ** argv);
void stop_noting_replies();
int run_serve(int argc, char ** argv);     // kera serve: llama.cpp's server plus saved chat states on the SSD
int run_cli(int argc, char ** argv);       // kera cli: llama.cpp's chat with the model card's sampling
// Models that make images (D-027): the image engine's programs in image/ next to kera
int run_image_cli(int argc, char ** argv);    // kera image: a picture from text
int run_image_server(int argc, char ** argv); // kera serve for an image model (argv: kera serve's command line)
bool is_image_model(const std::string & path); // a diffusion model, not a language model
int run_video_cli(int argc, char ** argv);    // kera video: a short video from text, through the image engine
// Music and speech (D-028): the music engine's programs in music/ next to kera; speech is llama.cpp's own
int run_music_cli(int argc, char ** argv);    // kera music: a song from a description (and lyrics)
int run_music_server(int argc, char ** argv); // kera serve --music DIR: the music engine's server and page
int run_speak_cli(int argc, char ** argv);    // kera speak: speech from text (Qwen3-TTS, Pocket TTS)
// 3D models and sound effects (D-031): the 3D engine's programs in 3d/ and the sound-effect engine's in sfx/
int run_3d_cli(int argc, char ** argv);       // kera 3d: a .glb from a picture, or from text through kera image
int run_sfx_cli(int argc, char ** argv);      // kera sfx: a sound effect from a description
int run_sfx_server(int argc, char ** argv);   // kera serve --sfx DIR: the sound-effect model kept loaded (D-039)
// An engine's program in a folder next to kera (image/, music/, 3d/, sfx/), and running a program in this window
std::string engine_path(const char * folder, const char * name);
// The prompt improver (D-030, opt-in): a chat model (a GGUF file) rewrites a picture's description in more detail,
// keeping what it says; tag_style for the models that read comma-separated phrases (SD 1.x/2.x, SDXL). The model
// answers as it does in a chat (D-033), a new random seed included: nothing about how it answers is changed. False,
// with why, if it couldn't.
bool improve_prompt(const std::string & chat_model, bool tag_style, const std::string & prompt, std::string & out,
                    std::string & err);
int run_program(const std::vector<std::string> & argv); // waits; its exit code, 127 if it didn't start
// The memory of the best graphics chip Kera's back-ends see, in GB; 0 without one
double gpu_memory_gb();
// Says so when the same model at a higher precision is next to the named file and fits this computer's memory (D-036):
// the model's own weights are its full-precision file. option: how the file is named on the command line ("-m").
void note_higher_precision(const std::string & model, const char * option);

} // namespace kera
