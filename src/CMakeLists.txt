# kera-core: Kera's own engine parts (spec section 4). The interfaces in each folder's kr_*.h are the
# P3 shapes frozen in M0.

set(KERA_IO_OS_SOURCES)
if (WIN32)
    set(KERA_IO_OS_SOURCES io/io_windows.cpp)
elseif (APPLE)
    set(KERA_IO_OS_SOURCES io/io_macos.cpp)
else()
    set(KERA_IO_OS_SOURCES io/io_linux.cpp)
endif()

add_library(kera-core STATIC
    common/hash.cpp
    common/paths.cpp
    io/io_pool.cpp
    ${KERA_IO_OS_SOURCES}
    kvstore/cache_id.cpp
    kvstore/prompt_disk.cpp
    planner/backend_pick.cpp
    planner/chat_share.cpp
    planner/engine_route.cpp
    planner/projector.cpp
    planner/models.cpp
    planner/planner.cpp
    setup/detect.cpp
    setup/explain.cpp
    setup/measure_disk.cpp
    setup/measure_gpu.cpp
    setup/measure_ram.cpp
    setup/profile.cpp
    setup/setup.cpp
    spec/controller.cpp
    spec/guess_switch.cpp
    spec/guess_memory.cpp
    stream/expert_cache.cpp
    stream/region.cpp
    stream/session.cpp
    tune/model_defaults.cpp
    tune/music_defaults.cpp
    tune/precision.cpp
    tune/settings.cpp
)

target_include_directories(kera-core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/generated
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor   # nlohmann/json.hpp
)
# the streamed experts' buffer type is defined with ggml's backend interface (stream/session.cpp)
target_include_directories(kera-core PRIVATE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/src)

find_package(Threads REQUIRED)
target_link_libraries(kera-core PUBLIC ggml llama Threads::Threads)
target_link_libraries(kera-core PRIVATE vendor::hash) # xxHash, bundled with llama.cpp
kera_target_warnings(kera-core)

if (WIN32)
    target_compile_definitions(kera-core PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
