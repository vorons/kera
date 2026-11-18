#include "request-defaults.h"

// [Kera] see request-defaults.h
static common_request_defaults * g_request_defaults = nullptr;

void common_set_request_defaults(common_request_defaults * rd) {
    g_request_defaults = rd;
}

common_request_defaults * common_get_request_defaults() {
    return g_request_defaults;
}
