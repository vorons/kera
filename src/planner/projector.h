// Models that see images (the owner's request for 0.1.1, docs/DECISIONS.md D-027). llama.cpp runs a vision model
// as two files: the language model and its image reader (a "multimodal projector", mmproj, a GGUF whose
// architecture is "clip"), and it sees images only when told where the reader is. Kera finds it by itself: next to
// the model (the way Hugging Face's GGUF repos ship them), or in Ollama's store (its manifests name a model's
// projector layer), so `kera serve -m model.gguf` and the app's Run button just work. Ollama's models are listed by
// their Ollama names, so the app shows the models already on the computer where they are.
#pragma once

#include <string>
#include <vector>

namespace kera {

// The image reader for a model file, or "" if none is found. Next to the model: the files with "mmproj" in their
// name that are image readers; with several, the one whose name has the most in common with the model's. In
// Ollama's store (a blob in .../models/blobs): the projector its manifest lists with the model.
// A folder's only reader beside its only model is taken only on evidence that it is this model's: its file name
// names no model ("mmproj-F16.gguf"), or its own name (general.name) is the model's or one of the model's base models.
// Otherwise the model stays text-only, as in stock llama.cpp, and *left_out (if given) names the reader passed over.
std::string find_projector(const std::string & model, std::string * left_out = nullptr);

// Whether a file is an image reader (a GGUF whose architecture is "clip")
bool is_projector(const std::string & file);

struct ollama_model {
    std::string name;       // as Ollama shows it, e.g. "qwen2.5vl:7b" or "hf.co/user/repo:Q4_K_M"
    std::string model;      // the model's file (a blob)
    std::string projector;  // its image reader, or ""
    // the rest of its Modelfile, as Ollama gives it to the model (blobs; "" where the manifest has none): its chat
    // template (Ollama's own Go template), its SYSTEM text, its PARAMETERs (JSON), its MESSAGE pairs (JSON), and its
    // ADAPTERs (LoRA files)
    std::string template_file;
    std::string system_file;
    std::string params_file;
    std::string messages_file;
    std::vector<std::string> adapters;
};

// The models in Ollama's store: $OLLAMA_MODELS, else ~/.ollama/models (folder overrides both, for tests)
std::vector<ollama_model> ollama_models(const std::string & folder = "");

// The manifest entry for a model file in Ollama's store (a blob in .../models/blobs); false for any other file
bool ollama_model_of(const std::string & model, ollama_model & out);

// Whether a GGUF file carries its own chat template (tokenizer.chat_template)
bool has_chat_template(const std::string & gguf);

} // namespace kera
