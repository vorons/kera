Kera's optional ROCm (HIP) back-end for AMD Radeon 8060S graphics (gfx1151: Ryzen AI Max, Strix Halo)
=====================================================================================================

Kera runs on Vulkan by default. This add-on lets you try AMD's ROCm instead: on this chip it can read prompts
faster, while Vulkan often writes faster. Nothing changes unless you ask for it.

To install: unzip all of this file into the Kera folder, so that the folders named "hip" and ".kpack" sit next
to kera.exe. To remove it, delete those two folders.

To use it, add --backend hip to a command, for example:

    kera serve -m your-model.gguf --backend hip

(--backend vulkan uses Vulkan only; without --backend Kera keeps its usual choice.)

Before relying on it, check that it gives the same answers as Vulkan on your PC: ask the same question with the
temperature at 0, once with --backend vulkan and once with --backend hip. The answers should match, or differ only
where two words were almost equally likely. (The exact check, `kera exactness` on Kera's tiny test models against
reference outputs, is run by Kera's developers once they can reach the Minisforum.)

llama.cpp's own tests on this chip found that, without HIP_LAUNCH_BLOCKING=1, batched work could give wrong
results (set it with: set HIP_LAUNCH_BLOCKING=1). Kera keeps Vulkan as the default until ROCm passes these
checks and is faster on the Minisforum.

The files: ggml-hip.dll is llama.cpp's HIP back-end (MIT licence). amdhip64_7.dll, rocm_kpack.dll and
amd_comgr.dll are AMD's ROCm runtime, and hipblas, rocblas, rocsolver and hipblaslt are AMD's maths libraries.
Their kernels for this chip are in the rocblas and hipblaslt folders and in ..\.kpack (where these libraries
look for them). All of them come from AMD's ROCm packages; their licences are in the licenses folder. Nothing
else needs installing.
They sit in this folder, not next to kera.exe, so that the copy in your graphics driver keeps working for
other programs; Kera points Windows at this folder only when you ask for --backend hip.
