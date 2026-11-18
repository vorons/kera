// Finding a vision model's image reader (src/planner/projector.h): next to the model, and in Ollama's store.

#include "kr_test.h"

#include "planner/projector.h"

#include "gguf.h"

#include <filesystem>
#include <initializer_list>
#include <utility>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace kera;

namespace {

fs::path fresh_dir(const char * name) {
    const fs::path d = fs::temp_directory_path() / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

// a GGUF file with only its architecture: enough for telling an image reader from a model
void gguf_with_arch(const fs::path & file, const char * arch) {
    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", arch);
    gguf_write_to_file(ctx, file.u8string().c_str(), false);
    gguf_free(ctx);
}

// a GGUF file with its architecture and string keys (general.type, general.name, ...), and a base model's name
void gguf_with(const fs::path & file, const char * arch, std::initializer_list<std::pair<const char *, const char *>> kv,
               const char * base_model = nullptr) {
    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture", arch);
    for (const auto & [k, v] : kv) {
        gguf_set_val_str(ctx, k, v);
    }
    if (base_model != nullptr) {
        gguf_set_val_u32(ctx, "general.base_model.count", 1);
        gguf_set_val_str(ctx, "general.base_model.0.name", base_model);
    }
    gguf_write_to_file(ctx, file.u8string().c_str(), false);
    gguf_free(ctx);
}

} // namespace

KR_TEST(a_lone_reader_only_on_evidence_it_is_the_models) {
    // a text model beside another model's reader (the deep check: it described a blue square as something else)
    const fs::path a = fresh_dir("kera-test-projector-lone-a");
    gguf_with(a / "qwen2.5-1.5b-instruct-q4_k_m.gguf", "qwen2", { { "general.name", "qwen2.5-1.5b-instruct" } });
    gguf_with(a / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", "clip",
              { { "general.type", "mmproj" }, { "general.name", "SmolVLM 256M Instruct" } });
    std::string left;
    KR_CHECK(find_projector((a / "qwen2.5-1.5b-instruct-q4_k_m.gguf").u8string(), &left).empty());
    KR_CHECK(fs::u8path(left).filename() == "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf");
    // another size's reader, named for it (no metadata): never taken this way
    const fs::path b = fresh_dir("kera-test-projector-lone-b");
    gguf_with_arch(b / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "qwen2vl");
    gguf_with_arch(b / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", "clip");
    KR_CHECK(find_projector((b / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf").u8string()).empty());
    // a plainly named reader whose own name is the model's: taken
    const fs::path c = fresh_dir("kera-test-projector-lone-c");
    gguf_with(c / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "qwen3vl", { { "general.name", "Qwen3-VL-8B-Instruct" } });
    gguf_with(c / "mmproj-F16.gguf", "clip", { { "general.type", "mmproj" }, { "general.name", "Qwen3 VL 8B Instruct" } });
    KR_CHECK(fs::u8path(find_projector((c / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf").u8string())).filename() ==
             "mmproj-F16.gguf");
    // ... but not when its own name is another model's
    const fs::path d = fresh_dir("kera-test-projector-lone-d");
    gguf_with(d / "SmolVLM-256M-Instruct-Q8_0.gguf", "llama", { { "general.name", "SmolVLM 256M Instruct" } });
    gguf_with(d / "mmproj-F16.gguf", "clip", { { "general.type", "mmproj" }, { "general.name", "Granite Docling 258M" } });
    KR_CHECK(find_projector((d / "SmolVLM-256M-Instruct-Q8_0.gguf").u8string()).empty());
    // a fine-tune keeps its base model's reader (unsloth's Devstral and Mistral Small 3.1's)
    const fs::path e = fresh_dir("kera-test-projector-lone-e");
    gguf_with(e / "Devstral-Small-2507-Q4_K_M.gguf", "llama", { { "general.name", "Devstral Small 2507" } },
              "Mistral Small 3.1 24B Instruct 2503");
    gguf_with(e / "mmproj-F16.gguf", "clip",
              { { "general.type", "mmproj" }, { "general.name", "Mistral-Small-3.1-24B-Instruct-2503" } });
    KR_CHECK(fs::u8path(find_projector((e / "Devstral-Small-2507-Q4_K_M.gguf").u8string())).filename() ==
             "mmproj-F16.gguf");
}

KR_TEST(among_one_models_readers_the_nearest_precision) {
    // llama.cpp's own rule: the reader whose number format is nearest the model file's, then the more precise
    const fs::path d = fresh_dir("kera-test-projector-bits");
    gguf_with_arch(d / "Qwen3-TTS-12Hz-1.7B-Base-BF16.gguf", "qwen3tts");
    gguf_with_arch(d / "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", "qwen3tts");
    gguf_with_arch(d / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", "clip");
    gguf_with_arch(d / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-BF16.gguf", "clip");
    KR_CHECK(fs::u8path(find_projector((d / "Qwen3-TTS-12Hz-1.7B-Base-BF16.gguf").u8string())).filename() ==
             "mmproj-Qwen3-TTS-12Hz-1.7B-Base-BF16.gguf");
    KR_CHECK(fs::u8path(find_projector((d / "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf").u8string())).filename() ==
             "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf");
    // ggml-org's vision names: an f16 model gets the f16 reader; Q8_0 and Q4_K_M models the Q8_0 one
    const fs::path v = fresh_dir("kera-test-projector-bits-vision");
    for (const char * m : { "SmolVLM-256M-Instruct-f16.gguf", "SmolVLM-256M-Instruct-Q8_0.gguf",
                            "SmolVLM-256M-Instruct-Q4_K_M.gguf" }) {
        gguf_with_arch(v / m, "llama");
    }
    gguf_with_arch(v / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", "clip");
    gguf_with_arch(v / "mmproj-SmolVLM-256M-Instruct-f16.gguf", "clip");
    KR_CHECK(fs::u8path(find_projector((v / "SmolVLM-256M-Instruct-f16.gguf").u8string())).filename() ==
             "mmproj-SmolVLM-256M-Instruct-f16.gguf");
    KR_CHECK(fs::u8path(find_projector((v / "SmolVLM-256M-Instruct-Q8_0.gguf").u8string())).filename() ==
             "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf");
    KR_CHECK(fs::u8path(find_projector((v / "SmolVLM-256M-Instruct-Q4_K_M.gguf").u8string())).filename() ==
             "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf");
}

KR_TEST(next_to_the_model_the_closest_image_reader) {
    const fs::path d = fresh_dir("kera-test-projector");
    gguf_with_arch(d / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "qwen2vl");
    gguf_with_arch(d / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", "clip");
    gguf_with_arch(d / "mmproj-gemma-3-4b-f16.gguf", "clip");
    gguf_with_arch(d / "mmproj-not-really.gguf", "llama"); // the name says so, the file doesn't
    const std::string p = find_projector((d / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").u8string());
    KR_CHECK(fs::u8path(p).filename() == "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf");
    KR_CHECK(is_projector(p));
    KR_CHECK(!is_projector((d / "mmproj-not-really.gguf").u8string()));
}

KR_TEST(another_models_reader_is_never_attached) {
    // a folder of several models: a text model gets no reader, even a similar model's
    const fs::path d = fresh_dir("kera-test-projector-others");
    gguf_with_arch(d / "Llama-3-8B-Instruct-Q4_K_M.gguf", "llama");
    gguf_with_arch(d / "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "qwen2");
    gguf_with_arch(d / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "qwen2vl");
    gguf_with_arch(d / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", "clip");
    KR_CHECK(find_projector((d / "Llama-3-8B-Instruct-Q4_K_M.gguf").u8string()).empty());
    KR_CHECK(find_projector((d / "Qwen2.5-7B-Instruct-Q4_K_M.gguf").u8string()).empty());
    KR_CHECK(!find_projector((d / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").u8string()).empty());
    // a download's own folder: one model, one reader with a plain name
    const fs::path e = fresh_dir("kera-test-projector-plain");
    gguf_with_arch(e / "gemma-3-4b-it-Q4_K_M.gguf", "gemma3");
    gguf_with_arch(e / "mmproj-F16.gguf", "clip");
    KR_CHECK(fs::u8path(find_projector((e / "gemma-3-4b-it-Q4_K_M.gguf").u8string())).filename() == "mmproj-F16.gguf");
    // another size of the same model is another model: its reader doesn't fit
    const fs::path f = fresh_dir("kera-test-projector-sizes");
    gguf_with_arch(f / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "qwen2vl");
    gguf_with_arch(f / "Qwen3-TTS-12Hz-0.6B-Base-Q8_0.gguf", "qwen3tts");
    gguf_with_arch(f / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", "clip");
    gguf_with_arch(f / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", "clip");
    KR_CHECK(find_projector((f / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf").u8string()).empty());
    KR_CHECK(find_projector((f / "Qwen3-TTS-12Hz-0.6B-Base-Q8_0.gguf").u8string()).empty());
    // a reader whose name gives no size isn't taken for one size of the model among several
    const fs::path h = fresh_dir("kera-test-projector-nosize");
    gguf_with_arch(h / "gemma-3-12b-it-Q4_K_M.gguf", "gemma3");
    gguf_with_arch(h / "gemma-3-4b-it-Q4_K_M.gguf", "gemma3");
    gguf_with_arch(h / "mmproj-gemma3-f16.gguf", "clip");
    KR_CHECK(find_projector((h / "gemma-3-12b-it-Q4_K_M.gguf").u8string()).empty());
    // a short name in a shared folder, a reader named without "-it", and a split model
    const fs::path g = fresh_dir("kera-test-projector-short");
    gguf_with_arch(g / "gemma3.gguf", "gemma3");
    gguf_with_arch(g / "mmproj-gemma3.gguf", "clip");
    gguf_with_arch(g / "gemma-3-12b-it-Q4_K_M.gguf", "gemma3");
    gguf_with_arch(g / "mmproj-gemma-3-12b-f16.gguf", "clip");
    gguf_with_arch(g / "Llama-4-Scout-Q4_K_M-00001-of-00002.gguf", "llama4");
    gguf_with_arch(g / "mmproj-Llama-4-Scout-BF16.gguf", "clip");
    KR_CHECK(fs::u8path(find_projector((g / "gemma3.gguf").u8string())).filename() == "mmproj-gemma3.gguf");
    KR_CHECK(fs::u8path(find_projector((g / "gemma-3-12b-it-Q4_K_M.gguf").u8string())).filename() ==
             "mmproj-gemma-3-12b-f16.gguf");
    KR_CHECK(fs::u8path(find_projector((g / "Llama-4-Scout-Q4_K_M-00001-of-00002.gguf").u8string())).filename() ==
             "mmproj-Llama-4-Scout-BF16.gguf");
}

KR_TEST(a_model_without_one_gets_none) {
    const fs::path d = fresh_dir("kera-test-projector-none");
    gguf_with_arch(d / "llama-3-8b.gguf", "llama");
    gguf_with_arch(d / "mmproj-not-really.gguf", "llama");
    KR_CHECK(find_projector((d / "llama-3-8b.gguf").u8string()).empty());
}

KR_TEST(ollamas_models_by_name_with_their_image_readers) {
    const fs::path d = fresh_dir("kera-test-ollama");
    fs::create_directories(d / "blobs");
    gguf_with_arch(d / "blobs" / "sha256-aaa", "qwen2vl");
    gguf_with_arch(d / "blobs" / "sha256-bbb", "clip");
    gguf_with_arch(d / "blobs" / "sha256-ccc", "llama");
    const fs::path lib = d / "manifests" / "registry.ollama.ai" / "library";
    fs::create_directories(lib / "qwen2.5vl");
    std::ofstream(lib / "qwen2.5vl" / "7b") << R"({"layers":[
        {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa"},
        {"mediaType":"application/vnd.ollama.image.projector","digest":"sha256:bbb"},
        {"mediaType":"application/vnd.ollama.image.template","digest":"sha256:ddd"}]})";
    fs::create_directories(d / "manifests" / "hf.co" / "someone" / "Llama-3-GGUF");
    std::ofstream(d / "manifests" / "hf.co" / "someone" / "Llama-3-GGUF" / "Q4_K_M") << R"({"layers":[
        {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:ccc"}]})";
    fs::create_directories(lib / "gone");
    std::ofstream(lib / "gone" / "latest") << R"({"layers":[
        {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:missing"}]})";

    const auto models = ollama_models(d.u8string());
    KR_CHECK(models.size() == 2); // the one whose file is gone isn't listed
    KR_CHECK(models[0].name == "hf.co/someone/Llama-3-GGUF:Q4_K_M" && models[0].projector.empty());
    KR_CHECK(models[1].name == "qwen2.5vl:7b");
    KR_CHECK(fs::u8path(models[1].model).filename() == "sha256-aaa");
    KR_CHECK(fs::u8path(models[1].projector).filename() == "sha256-bbb");
    // and from the blob alone
    KR_CHECK(fs::u8path(find_projector((d / "blobs" / "sha256-aaa").u8string())).filename() == "sha256-bbb");
    KR_CHECK(find_projector((d / "blobs" / "sha256-ccc").u8string()).empty());
}

KR_TEST_MAIN()
