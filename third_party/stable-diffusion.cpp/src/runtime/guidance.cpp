#include "runtime/guidance.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include "core/util.h"

namespace sd::guidance {

    static bool has_tensor(const sd::Tensor<float>* tensor) {
        return tensor != nullptr && !tensor->empty();
    }

    bool is_adaptive_projected_guidance_enabled(const AdaptiveProjectedGuidanceParams& params) {
        return params.eta != 1.0f || params.momentum != 0.0f || params.norm_threshold > 0.0f;
    }

    AdaptiveProjectedGuidanceParams parse_adaptive_projected_guidance_args(const char* extra_sample_args) {
        AdaptiveProjectedGuidanceParams params;
        for (const auto& [key, value] : parse_key_value_args(extra_sample_args, "extra sample arg")) {
            float parsed = 0.0f;
            if (key == "apg_eta") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.eta = parsed;
            } else if (key == "apg_momentum") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.momentum = parsed;
            } else if (key == "apg_norm_threshold") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.norm_threshold = parsed;
            } else if (key == "apg_norm_threshold_smoothing") {
                if (!parse_strict_float(value, parsed)) {
                    LOG_WARN("ignoring invalid APG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.norm_threshold_smoothing = parsed;
            }
        }
        return params;
    }

    bool parse_skip_layer_guidance_uncond_arg(const char* extra_sample_args) {
        bool uncond = false;
        for (const auto& [key, value] : parse_key_value_args(extra_sample_args, "extra sample arg")) {
            if (key == "slg_uncond") {
                if (!parse_strict_bool(value, uncond)) {
                    LOG_WARN("ignoring invalid SLG extra sample arg '%s=%s'", key.c_str(), value.c_str());
                }
            }
        }
        return uncond;
    }

    LTXGuiderParams parse_ltx_guider_args(const char* extra_sample_args) {
        LTXGuiderParams params;
        for (const auto& [key, value] : parse_key_value_args(extra_sample_args, "extra sample arg")) {
            float parsed = 0.0f;
            if (key == "ltx_stg_blocks") {
                std::vector<int> blocks;
                bool ok = !value.empty();
                size_t begin = 0;
                while (ok && begin <= value.size()) {
                    size_t end       = value.find('+', begin);
                    std::string item = value.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
                    ok               = !item.empty() && item.find_first_not_of("0123456789") == std::string::npos && item.size() < 5;
                    if (ok) {
                        blocks.push_back(std::atoi(item.c_str()));
                    }
                    if (end == std::string::npos) {
                        break;
                    }
                    begin = end + 1;
                }
                if (!ok) {
                    LOG_WARN("ignoring invalid LTX guider extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                params.stg_blocks = std::move(blocks);
            } else if (key == "ltx_stg" || key == "ltx_modality" || key == "ltx_rescale" || key == "ltx_audio_cfg") {
                if (!parse_strict_float(value, parsed) || !std::isfinite(parsed)) {
                    LOG_WARN("ignoring invalid LTX guider extra sample arg '%s=%s'", key.c_str(), value.c_str());
                    continue;
                }
                if (key == "ltx_stg") {
                    params.stg_scale = parsed;
                } else if (key == "ltx_modality") {
                    params.modality_scale = parsed;
                } else if (key == "ltx_rescale") {
                    params.rescale = parsed;
                } else {
                    params.audio_cfg = parsed;
                }
            }
        }
        if (params.stg_scale != 0.0f && params.stg_blocks.empty()) {
            LOG_WARN("ltx_stg needs ltx_stg_blocks; STG is off");
            params.stg_scale = 0.0f;
        }
        return params;
    }

    void ltx_guider_combine(float* out,
                            const float* x,
                            const float* cond,
                            const float* uncond,
                            const float* ptb,
                            const float* mod,
                            const float* mask,
                            size_t n,
                            float c_out,
                            float c_skip,
                            float cfg_scale,
                            const LTXGuiderParams& params) {
        // pred = cond + (cfg - 1)(cond - uncond) + stg (cond - ptb) + (modality - 1)(cond - mod), on the denoised
        // predictions, as MultiModalGuider.calculate does
        double sum_c = 0.0, sum_c2 = 0.0, sum_p = 0.0, sum_p2 = 0.0;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            const float base = x[i] * c_skip;
            const float dc   = cond[i] * c_out + base;
            float pred       = dc;
            if (uncond != nullptr) {
                pred += (cfg_scale - 1.0f) * (dc - (uncond[i] * c_out + base));
            }
            if (ptb != nullptr) {
                pred += params.stg_scale * (dc - (ptb[i] * c_out + base));
            }
            if (mod != nullptr) {
                pred += (params.modality_scale - 1.0f) * (dc - (mod[i] * c_out + base));
            }
            out[i] = pred;
            if (mask == nullptr || mask[i] != 0.0f) {
                sum_c += dc;
                sum_c2 += static_cast<double>(dc) * dc;
                sum_p += pred;
                sum_p2 += static_cast<double>(pred) * pred;
                count++;
            }
        }
        if (params.rescale == 0.0f || count < 2) {
            return;
        }
        // factor = std(cond) / std(pred), the unbiased spread over the counted values; pred is scaled by
        // rescale * factor + (1 - rescale)
        const double var_c = (sum_c2 - sum_c * sum_c / count) / (count - 1);
        const double var_p = (sum_p2 - sum_p * sum_p / count) / (count - 1);
        if (!(var_c > 0.0) || !(var_p > 0.0)) {
            return;
        }
        const float factor = params.rescale * static_cast<float>(std::sqrt(var_c / var_p)) + (1.0f - params.rescale);
        for (size_t i = 0; i < n; i++) {
            out[i] *= factor;
        }
    }

    std::vector<float> parse_guidance_schedule_from_spec(std::string spec) {
        std::vector<float> schedule;

        while (!spec.empty()) {
            auto sep     = spec.find('+');
            auto segment = spec.substr(0, sep);

            auto x = segment.find('x');
            if (x == std::string::npos) {
                LOG_ERROR("Invalid guidance schedule segment: '%s' (expected <guidance>x<count>)", segment.c_str());
                return {};
            }

            float guidance;
            int count;

            auto guidance_str = segment.substr(0, x);
            auto count_str    = segment.substr(x + 1);

            try {
                size_t idx = 0;
                guidance   = std::stof(guidance_str, &idx);
                if (idx != guidance_str.size()) {
                    LOG_ERROR("Invalid guidance value in guidance schedule: '%s'", guidance_str.c_str());
                    return {};
                }
            } catch (const std::exception&) {
                LOG_ERROR("Invalid guidance value in guidance schedule: '%s'", guidance_str.c_str());
                return {};
            }

            try {
                size_t idx = 0;
                count      = std::stoi(count_str, &idx);
                if (idx != count_str.size()) {
                    LOG_ERROR("Invalid count in guidance schedule: '%s'", count_str.c_str());
                    return {};
                }
            } catch (const std::exception&) {
                LOG_ERROR("Invalid count in guidance schedule: '%s'", count_str.c_str());
                return {};
            }

            if (count <= 0) {
                LOG_ERROR("Guidance schedule count must be positive");
                return {};
            }

            schedule.insert(schedule.end(), count, guidance);

            if (sep == std::string::npos) {
                break;
            }

            spec = spec.substr(sep + 1);
        }

        return schedule;
    }

    std::vector<float> parse_guidance_schedule(const char* extra_sample_args) {
        std::vector<float> guidance_schedule;
        std::string guidance_schedule_str = "";
        for (const auto& [key, value] : parse_key_value_args(extra_sample_args, "extra sample arg")) {
            float parsed = 0.0f;
            if (key == "guidance_schedule") {
                guidance_schedule_str = value;
            }
        }

        if (!guidance_schedule_str.empty()) {
            guidance_schedule = parse_guidance_schedule_from_spec(guidance_schedule_str);
        }
        return guidance_schedule;
    }

    ClassifierFreeGuidance::ClassifierFreeGuidance(float guidance_scale,
                                                   float image_guidance_scale)
        : guidance_scale_(guidance_scale),
          image_guidance_scale_(image_guidance_scale) {
    }

    GuiderOutput ClassifierFreeGuidance::forward(const GuidanceInput& input,
                                                 GuiderOutput previous,
                                                 std::optional<float> scale_override) const {
        (void)previous;
        float guidance_scale = scale_override.value_or(guidance_scale_);

        GuiderOutput output;
        if (!has_tensor(input.pred_cond)) {
            return output;
        }

        const sd::Tensor<float>& pred_cond = *input.pred_cond;
        output.pred                        = pred_cond;
        if (has_tensor(input.pred_uncond)) {
            const sd::Tensor<float>& pred_uncond = *input.pred_uncond;
            if (has_tensor(input.pred_img_uncond)) {
                const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
                output.pred                              = pred_img_uncond +
                                                           image_guidance_scale_ * (pred_uncond - pred_img_uncond) +
                                                           guidance_scale * (pred_cond - pred_uncond);

            } else {
                output.pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond);
            }
        } else if (has_tensor(input.pred_img_uncond)) {
            const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
            output.pred                              = pred_img_uncond + guidance_scale * (pred_cond - pred_img_uncond);
        }

        return output;
    }

    AdaptiveProjectedGuidance::AdaptiveProjectedGuidance(float guidance_scale,
                                                         float image_guidance_scale,
                                                         AdaptiveProjectedGuidanceParams params)
        : guidance_scale_(guidance_scale),
          image_guidance_scale_(image_guidance_scale),
          params_(params) {
    }

    static sd::Tensor<float> calculate_guidance_delta(const sd::Tensor<float>& pred_cond,
                                                      const sd::Tensor<float>* pred_uncond,
                                                      const sd::Tensor<float>* pred_img_uncond,
                                                      float guidance_scale,
                                                      float image_guidance_scale) {
        if (pred_img_uncond != nullptr) {
            if (pred_uncond != nullptr && guidance_scale == 1.0f) {
                return *pred_uncond - *pred_img_uncond;
            }
            if (pred_uncond != nullptr) {
                return pred_cond +
                       (*pred_uncond * (image_guidance_scale - guidance_scale) +
                        *pred_img_uncond * (1.0f - image_guidance_scale)) /
                           (guidance_scale - 1.0f);
            }
            return pred_cond - *pred_img_uncond;
        }
        return pred_cond - *pred_uncond;
    }

    GuiderOutput AdaptiveProjectedGuidance::forward(const GuidanceInput& input,
                                                    GuiderOutput previous,
                                                    std::optional<float> scale_override) const {
        (void)previous;
        float guidance_scale = scale_override.value_or(guidance_scale_);

        GuiderOutput output;
        if (!has_tensor(input.pred_cond)) {
            return output;
        }

        const sd::Tensor<float>& pred_cond = *input.pred_cond;
        output.pred                        = pred_cond;
        if (has_tensor(input.pred_uncond)) {
            const sd::Tensor<float>& pred_uncond = *input.pred_uncond;
            if (has_tensor(input.pred_img_uncond)) {
                const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
                output.pred                              = pred_img_uncond +
                                                           image_guidance_scale_ * (pred_uncond - pred_img_uncond) +
                                                           guidance_scale * (pred_cond - pred_uncond);
            } else {
                output.pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond);
            }
        } else if (has_tensor(input.pred_img_uncond)) {
            const sd::Tensor<float>& pred_img_uncond = *input.pred_img_uncond;
            output.pred                              = pred_img_uncond + guidance_scale * (pred_cond - pred_img_uncond);
        }
        if (!has_tensor(input.pred_uncond) && !has_tensor(input.pred_img_uncond)) {
            return output;
        }

        const sd::Tensor<float>* pred_uncond     = input.pred_uncond;
        const sd::Tensor<float>* pred_img_uncond = input.pred_img_uncond;

        sd::Tensor<float> deltas = calculate_guidance_delta(pred_cond,
                                                            pred_uncond,
                                                            pred_img_uncond,
                                                            guidance_scale,
                                                            image_guidance_scale_);
        if (params_.momentum != 0.0f) {
            if (momentum_buffer_.shape() != deltas.shape()) {
                momentum_buffer_ = sd::Tensor<float>::zeros_like(deltas);
            }
            deltas += params_.momentum * momentum_buffer_;
            momentum_buffer_ = deltas;
        }

        float diff_norm        = 0.0f;
        const int standard_res = 2 * 1024 / 8;  // Use SDXL as the standard resolution (1024x1024, 8x8 patches, 4=2x2 channels)
        if (params_.norm_threshold > 0.0f) {
            diff_norm = std::sqrt((deltas * deltas).sum()) * standard_res / std::sqrt(static_cast<float>(deltas.numel()));
        }

        float apg_scale_factor = 1.0f;
        if (params_.norm_threshold > 0.0f) {
            if (diff_norm > 0.0f) {
                if (params_.norm_threshold_smoothing <= 0.0f) {
                    apg_scale_factor = std::min(1.0f, params_.norm_threshold / diff_norm);
                } else {
                    float x          = params_.norm_threshold / diff_norm;
                    apg_scale_factor = x / std::pow(1.0f + std::pow(x, 1.0f / params_.norm_threshold_smoothing),
                                                    params_.norm_threshold_smoothing);
                }
            }
        }

        deltas *= apg_scale_factor;
        if (params_.eta != 1.0f) {
            float cond_norm_sq = (pred_cond * pred_cond).sum();
            if (cond_norm_sq != 0.0f) {
                float projection_scale = (pred_cond * deltas).sum() / cond_norm_sq;
                deltas += (params_.eta - 1.0f) * (projection_scale * pred_cond);
            }
        }

        output.pred = pred_cond;
        if (pred_uncond != nullptr) {
            if (guidance_scale_ != 1.0f) {
                output.pred = pred_cond + (guidance_scale_ - 1.0f) * deltas;
            } else if (pred_img_uncond != nullptr) {
                output.pred = pred_cond + (image_guidance_scale_ - 1.0f) * deltas;
            }
        } else if (pred_img_uncond != nullptr) {
            output.pred = *pred_img_uncond + guidance_scale_ * deltas;
        }

        return output;
    }

    SkipLayerGuidance::SkipLayerGuidance(std::vector<int> layers,
                                         float scale,
                                         float start,
                                         float stop)
        : layers_(std::move(layers)),
          scale_(scale),
          start_(start),
          stop_(stop) {
    }

    bool SkipLayerGuidance::is_enabled_for_step(const GuidanceInput& input) const {
        if (layers_.empty() || input.schedule_size == 0) {
            return false;
        }

        int start_step = static_cast<int>(start_ * static_cast<float>(input.schedule_size));
        int stop_step  = static_cast<int>(stop_ * static_cast<float>(input.schedule_size));
        return input.step > start_step && input.step < stop_step;
    }

    const std::vector<int>& SkipLayerGuidance::layers() const {
        return layers_;
    }

    GuiderOutput SkipLayerGuidance::forward(const GuidanceInput& input,
                                            GuiderOutput output,
                                            std::optional<float> /*scale_override*/) const {
        if (scale_ == 0.0f || !is_enabled_for_step(input) || !input.predict_skip_layer) {
            return output;
        }

        if (output.pred.empty() || !has_tensor(input.pred_cond)) {
            return GuiderOutput();
        }

        output.pred_skip_layer = input.predict_skip_layer();
        if (output.pred_skip_layer.empty()) {
            return GuiderOutput();
        }

        output.pred += (*input.pred_cond - output.pred_skip_layer) * scale_;
        return output;
    }

}  // namespace sd::guidance
