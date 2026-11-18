# Unit tests (spec P7: unit tests plus the tiny-model exactness test pass on the CPU).

function(kera_add_test name)
    add_executable(${name} cpp/${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
    target_link_libraries(${name} PRIVATE kera-core)
    target_compile_definitions(${name} PRIVATE KERA_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
    kera_target_warnings(${name})
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

kera_add_test(test_settings)
kera_add_test(test_controller)
kera_add_test(test_io)
kera_add_test(test_setup)
kera_add_test(test_kvstore_id)
kera_add_test(test_store_fake)
kera_add_test(test_hash)
kera_add_test(test_plan)
kera_add_test(test_prompt_disk)
kera_add_test(test_stream_cache)
kera_add_test(test_model_defaults)
kera_add_test(test_music_defaults)
kera_add_test(test_precision)
kera_add_test(test_guess_switch)
kera_add_test(test_sched_cancel)
kera_add_test(test_backend_pick)
kera_add_test(test_chat_share)
kera_add_test(test_engine_route)
kera_add_test(test_projector)

# The fake store defines the kr_store functions itself, so it doesn't link the real ones (there are none yet).

# llama.cpp's own test of chat templates and tool-call parsing, run on the pinned copy (spec 4c item 4):
# the formats of Nemotron 3 (qwen3_coder style), GLM, Kimi K2, DeepSeek, Qwen3 and gpt-oss, with each
# model's own template from third_party/llama.cpp/models/templates. It uses llama.cpp's internal
# functions, which a Windows DLL doesn't export, so (as upstream does) it's skipped there; the parsing
# code is the same on every platform, and Linux and macOS run it.
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    set(KERA_LLAMA_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
    add_executable(llama-test-chat ${KERA_LLAMA_DIR}/tests/test-chat.cpp)
    target_include_directories(llama-test-chat PRIVATE ${KERA_LLAMA_DIR}/tools/server)
    target_link_libraries(llama-test-chat PRIVATE server-context llama-common)
    target_compile_features(llama-test-chat PRIVATE cxx_std_17)
    add_test(NAME llama-test-chat COMMAND llama-test-chat WORKING_DIRECTORY ${KERA_LLAMA_DIR})
endif()

