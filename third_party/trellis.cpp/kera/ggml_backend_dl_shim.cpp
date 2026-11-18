// Added by Kera (docs/UPSTREAM_PATCHES.md, "trellis.cpp"); not part of upstream trellis.cpp.
//
// With GGML_BACKEND_DL, ggml's back-ends (the CPU variants, Vulkan, CUDA, Metal) are libraries loaded when the program
// starts, so nothing can link their functions. trellis.cpp calls four of them by name; this file gives the same calls
// through ggml's registry, and loads the back-ends next to the program before main() runs. Built only with
// GGML_BACKEND_DL (CMakeLists.txt), so trellis.cpp's sources stay as upstream wrote them.

#include "ggml-backend.h"

// declared here rather than through ggml-cpu.h and ggml-cuda.h, whose declarations would be imports on Windows
extern "C" {
ggml_backend_t ggml_backend_cpu_init(void);
bool ggml_backend_is_cpu(ggml_backend_t backend);
void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);
#ifdef TRELLIS_USE_CUDA
ggml_backend_t ggml_backend_cuda_init(int device);
#endif
}

namespace {

struct load_backends {
    load_backends() { ggml_backend_load_all(); }
};
const load_backends g_load_backends;

ggml_backend_reg_t reg_named(const char * name) {
    return ggml_backend_reg_by_name(name);
}

} // namespace

extern "C" {

ggml_backend_t ggml_backend_cpu_init(void) {
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    return dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    ggml_backend_dev_t dev = backend_cpu ? ggml_backend_get_device(backend_cpu) : nullptr;
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : reg_named("CPU");
    void * fn = reg ? ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr;
    if (fn != nullptr) {
        ((ggml_backend_set_n_threads_t) fn)(backend_cpu, n_threads);
    }
}

#ifdef TRELLIS_USE_CUDA
ggml_backend_t ggml_backend_cuda_init(int device) {
    ggml_backend_reg_t reg = reg_named("CUDA");
    if (reg == nullptr || device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, (size_t) device), nullptr);
}
#endif

} // extern "C"
