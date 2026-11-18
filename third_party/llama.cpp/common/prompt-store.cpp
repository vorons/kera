#include "prompt-store.h"

// [Kera] see prompt-store.h
static common_prompt_store * g_prompt_store = nullptr;

void common_set_prompt_store(common_prompt_store * store) {
    g_prompt_store = store;
}

common_prompt_store * common_get_prompt_store() {
    return g_prompt_store;
}

// [Kera] hook 8, see prompt-store.h
static common_params_fill g_params_fill = nullptr;

void common_set_params_fill(common_params_fill fill) {
    g_params_fill = fill;
}

common_params_fill common_get_params_fill() {
    return g_params_fill;
}
