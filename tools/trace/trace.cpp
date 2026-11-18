// `kera trace`: routing traces for MoE models (M1 risk test 2; calibrates the planner in M3 and the
// expert-aware guessing in M6).
//
// Reads conversations (JSON lines: {"id": ..., "messages": [...]} rendered with the model's own chat
// template, or {"id": ..., "text": "..."}), runs them through the model in one pass each, and saves
// which experts every MoE layer chose for every token. Reading a conversation in one pass picks the
// same experts as writing it word by word, so the traces stand for real use. With --hidden-every N it
// also saves, for every Nth token, the residual stream after each MoE layer, so the next MoE layer's
// choice can be predicted from it offline (next-layer lookahead).
//
// Output folder: meta.json, tokens.npy (int32), topk_L<layer>.npy (int32 [tokens, experts used]) and
// hidden_L<layer>.npy (float32 [sampled tokens, n_embd]).

#include "chat.h"
#include "ggml-backend.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

// a minimal .npy writer (NumPy format 1.0)
void write_npy(const std::string & path, const char * descr, const std::vector<size_t> & shape, const void * data, size_t bytes) {
    std::string dims;
    for (size_t i = 0; i < shape.size(); ++i) {
        dims += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
    }
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    const size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    const uint16_t hl = (uint16_t) header.size();
    f.write((const char *) &hl, 2);
    f.write(header.data(), (std::streamsize) header.size());
    f.write((const char *) data, (std::streamsize) bytes);
}

struct trace_state {
    int n_used = 0;
    int64_t n_embd = 0;
    int hidden_every = 0;
    std::map<int, std::vector<int32_t>> topk;      // layer -> [tokens * n_used]
    std::map<int, std::vector<float>> hidden;      // layer -> [sampled tokens * n_embd]
    std::map<int, int64_t> hidden_seen;            // layer -> tokens seen so far
    std::vector<char> buf;
    std::string hidden_name;                       // e.g. "nemotron_h_block_out"
};

// "name-12" -> ("name", 12)
bool split_name(const char * full, std::string & base, int & layer) {
    const char * dash = std::strrchr(full, '-');
    if (dash == nullptr || dash[1] == '\0') {
        return false;
    }
    char * end = nullptr;
    const long v = std::strtol(dash + 1, &end, 10);
    if (*end != '\0') {
        return false;
    }
    base.assign(full, dash);
    layer = (int) v;
    return true;
}

bool trace_cb(struct ggml_tensor * t, bool ask, void * user) {
    auto * st = (trace_state *) user;
    std::string base;
    int layer = 0;
    if (!split_name(t->name, base, layer)) {
        return false;
    }
    const bool want_topk = base == "ffn_moe_topk";
    // the residual stream after MoE layers only (their routing comes earlier in the same graph)
    const bool want_hidden = st->hidden_every > 0 && base == st->hidden_name && st->topk.count(layer) > 0;
    if (!want_topk && !want_hidden) {
        return false;
    }
    if (ask) {
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    st->buf.resize(nbytes);
    ggml_backend_tensor_get(t, st->buf.data(), 0, nbytes);
    if (want_topk && t->type == GGML_TYPE_I32) {
        // [n_used, n_tokens], possibly with a row stride
        const int64_t n_used = t->ne[0], n_tok = t->ne[1];
        st->n_used = (int) n_used;
        auto & v = st->topk[layer];
        for (int64_t j = 0; j < n_tok; ++j) {
            const int32_t * row = (const int32_t *) (st->buf.data() + j * t->nb[1]);
            v.insert(v.end(), row, row + n_used);
        }
    }
    if (want_hidden && t->type == GGML_TYPE_F32) {
        const int64_t n_embd = t->ne[0], n_tok = t->ne[1];
        st->n_embd = n_embd;
        int64_t & seen = st->hidden_seen[layer];
        auto & v = st->hidden[layer];
        for (int64_t j = 0; j < n_tok; ++j, ++seen) {
            if (seen % st->hidden_every == 0) {
                const float * row = (const float *) (st->buf.data() + j * t->nb[1]);
                v.insert(v.end(), row, row + n_embd);
            }
        }
    }
    return true;
}

std::string render(const common_chat_templates * tmpls, const json & conv) {
    if (conv.contains("text")) {
        return conv["text"].get<std::string>();
    }
    common_chat_templates_inputs in;
    in.add_generation_prompt = false;
    for (const auto & m : conv["messages"]) {
        common_chat_msg msg;
        msg.role = m["role"].get<std::string>();
        msg.content = m.value("content", std::string());
        if (m.contains("reasoning_content")) {
            msg.reasoning_content = m["reasoning_content"].get<std::string>();
        }
        in.messages.push_back(msg);
    }
    return common_chat_templates_apply(tmpls, in).prompt.str();
}

} // namespace

int kera_trace(int argc, char ** argv) {
    std::string model_path, input, out_dir, hidden_name = "nemotron_h_block_out";
    int hidden_every = 0, threads = 0, max_tokens = 32768, ubatch = 512;
    bool repack = true;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--model") model_path = argv[i + 1];
        else if (a == "--input") input = argv[i + 1];
        else if (a == "--out") out_dir = argv[i + 1];
        else if (a == "--hidden-every") hidden_every = std::atoi(argv[i + 1]);
        else if (a == "--hidden-name") hidden_name = argv[i + 1];
        else if (a == "--threads") threads = std::atoi(argv[i + 1]);
        else if (a == "--max-tokens") max_tokens = std::atoi(argv[i + 1]);
        else if (a == "--ubatch") ubatch = std::atoi(argv[i + 1]);
        else if (a == "--repack") repack = std::atoi(argv[i + 1]) != 0;
        else {
            std::fprintf(stderr, "Unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty() || input.empty() || out_dir.empty()) {
        std::fprintf(stderr, "Usage: kera trace --model FILE.gguf --input CONVERSATIONS.jsonl --out DIR "
                             "[--hidden-every N] [--hidden-name NAME] [--threads N] [--max-tokens N] [--ubatch N] [--repack 0|1]\n");
        return 2;
    }

    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    // --repack 0: keep the weights memory-mapped from the file. The CPU's faster repacked layout copies
    // them into memory, which a model bigger than the RAM can't afford
    mp.use_extra_bufts = repack;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) {
        std::fprintf(stderr, "Can't load %s\n", model_path.c_str());
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, "");

    trace_state st;
    st.hidden_every = hidden_every;
    st.hidden_name = hidden_name;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) max_tokens;
    // bigger steps read each expert fewer times when the model doesn't fit in memory; the experts
    // chosen don't depend on the step size
    cp.n_batch = (uint32_t) std::max(2048, ubatch);
    cp.n_ubatch = (uint32_t) ubatch;
    if (threads > 0) {
        cp.n_threads = cp.n_threads_batch = threads;
    }
    cp.cb_eval = trace_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "Can't create a context\n");
        return 2;
    }

    std::ifstream in(input);
    std::string line;
    std::vector<int32_t> all_tokens;
    json segments = json::array();
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const json conv = json::parse(line);
        const std::string text = render(tmpls.get(), conv);
        int n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, true, true);
        std::vector<llama_token> toks((size_t) n);
        llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), n, true, true);
        if ((int) toks.size() > max_tokens) {
            toks.resize((size_t) max_tokens);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t i = 0; i < toks.size(); i += cp.n_batch) {
            const int nb = (int) std::min(toks.size() - i, (size_t) cp.n_batch);
            if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, nb)) != 0) {
                std::fprintf(stderr, "decode failed\n");
                return 1;
            }
        }
        llama_synchronize(ctx);
        segments.push_back({ { "id", conv.value("id", std::string()) }, { "start", all_tokens.size() }, { "length", toks.size() } });
        all_tokens.insert(all_tokens.end(), toks.begin(), toks.end());
        std::fprintf(stderr, "%s: %zu tokens (total %zu)\n", conv.value("id", std::string("?")).c_str(), toks.size(), all_tokens.size());
    }

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(out_dir), ec);
    write_npy(out_dir + "/tokens.npy", "<i4", { all_tokens.size() }, all_tokens.data(), all_tokens.size() * 4);
    json layers = json::array();
    for (const auto & [layer, v] : st.topk) {
        const size_t n_tok = v.size() / (size_t) st.n_used;
        if (n_tok != all_tokens.size()) {
            std::fprintf(stderr, "layer %d: %zu tokens traced, expected %zu\n", layer, n_tok, all_tokens.size());
        }
        write_npy(out_dir + "/topk_L" + std::to_string(layer) + ".npy", "<i4", { n_tok, (size_t) st.n_used }, v.data(), v.size() * 4);
        layers.push_back(layer);
    }
    for (const auto & [layer, v] : st.hidden) {
        write_npy(out_dir + "/hidden_L" + std::to_string(layer) + ".npy", "<f4", { v.size() / (size_t) st.n_embd, (size_t) st.n_embd },
                  v.data(), v.size() * 4);
    }
    char arch[128] = "";
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    const json meta = {
        { "model", model_path.substr(model_path.find_last_of("/\\") == std::string::npos ? 0 : model_path.find_last_of("/\\") + 1) },
        { "desc", desc }, { "architecture", arch }, { "n_layer", llama_model_n_layer(model) },
        { "moe_layers", layers }, { "n_expert_used", st.n_used }, { "n_embd", llama_model_n_embd(model) },
        { "hidden_every", hidden_every }, { "hidden_name", hidden_name }, { "segments", segments },
    };
    std::ofstream(out_dir + "/meta.json") << meta.dump(1) << "\n";
    std::printf("Traced %zu tokens over %zu MoE layers into %s\n", all_tokens.size(), layers.size(), out_dir.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
