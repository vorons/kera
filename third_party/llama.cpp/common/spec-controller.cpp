#include "spec-controller.h"

// [Kera] see spec-controller.h
static common_spec_controller * g_spec_controller = nullptr;

void common_set_spec_controller(common_spec_controller * ctrl) {
    g_spec_controller = ctrl;
}

common_spec_controller * common_get_spec_controller() {
    return g_spec_controller;
}
