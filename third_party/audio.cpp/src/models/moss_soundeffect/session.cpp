#include "engine/models/moss_soundeffect/session.h"

#include "engine/models/moss_soundeffect/assets.h"
#include "engine/models/moss_soundeffect/dit.h"
#include "engine/models/moss_soundeffect/vae.h"
#include "engine/models/moss_soundeffect/text_encoder.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/noise.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace engine::models::moss_soundeffect {
namespace {
constexpr const char * kFamily = "moss_soundeffect";

// Kera: the text encoder's output, kept on disk. AUDIOCPP_TEXT_CACHE names a folder for one model file (Kera makes
// one per file); each prompt's encoding is a file named by its hash. A prompt seen before (the same sound with
// another seed) then needs no text encoder at all: its 2 GB of weights are never loaded. The cached floats are the
// encoder's own output, byte for byte, so the sound is the same.
std::filesystem::path text_cache_file(const std::string & prompt) {
    const char * dir = std::getenv("AUDIOCPP_TEXT_CACHE");
    if (dir == nullptr || *dir == '\0') return {};
    uint64_t h = 1469598103934665603ULL; // FNV-1a
    for (unsigned char c : prompt) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    std::ostringstream name;
    name << std::hex << std::setw(16) << std::setfill('0') << h << "-" << std::dec << prompt.size() << ".f32";
    return std::filesystem::u8path(dir) / name.str();
}

constexpr uint32_t kTextCacheMagic = 0x4B544331; // "KTC1"

bool read_text_cache(const std::filesystem::path & file, const std::string & prompt, std::vector<float> & out) {
    if (file.empty()) return false;
    std::ifstream f(file, std::ios::binary);
    uint32_t magic = 0;
    uint64_t n_prompt = 0, n = 0;
    if (!f.read(reinterpret_cast<char *>(&magic), 4) || magic != kTextCacheMagic ||
        !f.read(reinterpret_cast<char *>(&n_prompt), 8) || n_prompt != prompt.size()) return false;
    std::string stored(n_prompt, '\0');
    if (!f.read(stored.data(), static_cast<std::streamsize>(n_prompt)) || stored != prompt) return false; // a hash collision
    if (!f.read(reinterpret_cast<char *>(&n), 8) || n == 0 || n > (1ULL << 28)) return false;
    out.resize(static_cast<size_t>(n));
    return static_cast<bool>(f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(n * sizeof(float))));
}

void write_text_cache(const std::filesystem::path & file, const std::string & prompt, const std::vector<float> & v) {
    if (file.empty() || v.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    // written under a name of its own, then renamed: a reader never sees half a file
    const auto tmp = std::filesystem::path(file.u8string() + "." +
                                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        const uint64_t n_prompt = prompt.size(), n = v.size();
        f.write(reinterpret_cast<const char *>(&kTextCacheMagic), 4);
        f.write(reinterpret_cast<const char *>(&n_prompt), 8);
        f.write(prompt.data(), static_cast<std::streamsize>(n_prompt));
        f.write(reinterpret_cast<const char *>(&n), 8);
        f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
        if (!f.good()) {
            f.close();
            std::filesystem::remove(tmp, ec);
            return;
        }
    }
    std::filesystem::rename(tmp, file, ec);
    if (ec) std::filesystem::remove(tmp, ec);
}

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
            std::shared_ptr<const Assets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "MOSS-SoundEffect v2");
        core::set_backend_round_bf16_fusion(execution_context().backend(), true);
        if (task.task != runtime::VoiceTaskKind::AudioGeneration || task.mode != runtime::RunMode::Offline)
            throw std::runtime_error("MOSS-SoundEffect v2 supports offline sound-effect generation");
        attention_ = core::parse_attention_preference(
            runtime::find_option(options.options, {"moss_soundeffect.attention"}).value_or("auto"),
            "moss_soundeffect.attention");
        storage_ = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"moss_soundeffect.weight_type"}).value_or("native"));
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest &) override {
        if (!dit_) {
            // Kera: the text encoder is made in run(), only when a prompt isn't in the text cache
            auto & execution = execution_context();
            auto dit = std::make_unique<DitRuntime>(execution, assets_, storage_, attention_);
            auto vae = std::make_unique<VaeRuntime>(execution, assets_);
            dit_ = std::move(dit);
            vae_ = std::move(vae);
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("MOSS-SoundEffect v2 generation");
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "MOSS-SoundEffect v2");
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
        if (!request.text_input || request.text_input->text.empty())
            throw runtime::InvalidRequestError("MOSS-SoundEffect v2 requires a text prompt");
        const auto start = std::chrono::steady_clock::now();
        if (!request.option_arrays.empty())
            throw runtime::InvalidRequestError("MOSS-SoundEffect v2 does not accept list-valued request options");
        const auto duration_text = runtime::find_option(request.options, {"duration_sec"}).value_or("10");
        // Use shared option validation, retaining the original double for decimal rounding.
        runtime::parse_positive_finite_float_option(request.options, {"duration_sec"});
        const double requested_duration = std::stod(duration_text);
        // Format the double directly: rounding an intermediate float changes decimal ties.
        std::ostringstream duration_suffix;
        duration_suffix << std::fixed << std::setprecision(1) << requested_duration;
        const double duration = std::stod(duration_suffix.str());
        if (duration < 0.1 || duration > assets_->max_duration_sec)
            throw runtime::InvalidRequestError("MOSS-SoundEffect v2 duration_sec must be between 0.1 and 30");
        const int steps = runtime::parse_int_option(request.options, {"num_steps"}).value_or(100);
        const float guidance = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(4.0F);
        const float shift = runtime::parse_positive_finite_float_option(request.options, {"sigma_shift"}).value_or(5.0F);
        const auto seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(0);
        if (steps < 1) throw runtime::InvalidRequestError("MOSS-SoundEffect v2 num_steps must be positive");
        auto prompt = request.text_input->text;
        const auto append = runtime::find_option(request.options, {"append_duration_suffix"});
        if (!append || runtime::parse_bool_option(*append, "append_duration_suffix")) {
            prompt += " duration: " + duration_suffix.str() + "s";
        }
        const auto text_start = std::chrono::steady_clock::now();
        const auto encode_cached = [&](const std::string & text) {
            std::vector<float> v;
            // Kera: only with the file's own weights, which is what Kera's folder for the file stands for
            const auto file = storage_ == assets::TensorStorageType::Native ? text_cache_file(text) : std::filesystem::path{};
            if (read_text_cache(file, text, v)) return v;
            if (!text_) text_ = std::make_unique<TextEncoderRuntime>(execution_context(), assets_, storage_);
            v = text_->encode(text);
            write_text_cache(file, text, v);
            return v;
        };
        const auto positive = encode_cached(prompt);
        const auto negative = encode_cached(runtime::find_option(request.options, {"negative_prompt"}).value_or(""));
        // Kera: its 2 GB freed before the sound model runs (it is made again if a later prompt needs it)
        text_.reset();
        debug::timing_log_scalar("moss_soundeffect.text.total_ms", debug::elapsed_ms(text_start));
        dit_->prepare_context(positive, negative);
        const auto sample_start = std::chrono::steady_clock::now();
        const int64_t frames = int64_t(assets_->sample_rate) * assets_->max_duration_sec / assets_->vae.hop_length;
        auto noise = sampling::generate_normal_noise(static_cast<size_t>(assets_->dit.latent_dim * frames), seed);
        core::round_f32_to_bf16_in_place(noise);
        const auto latent = dit_->sample(noise, steps, guidance, shift);
        debug::timing_log_scalar("moss_soundeffect.dit.sample_ms", debug::elapsed_ms(sample_start));
        auto audio = vae_->decode(latent, static_cast<size_t>(duration * assets_->sample_rate));
        debug::timing_log_scalar("audio.duration_sec", duration);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        runtime::TaskResult result;
        result.audio_output = std::move(audio);
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    core::AttentionPreference attention_ = core::AttentionPreference::Auto;
    assets::TensorStorageType storage_ = assets::TensorStorageType::Native;
    std::unique_ptr<TextEncoderRuntime> text_;
    std::unique_ptr<DitRuntime> dit_;
    std::unique_ptr<VaeRuntime> vae_;
};
}

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_soundeffect_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = kFamily;
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}
