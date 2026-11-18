# The `kera` program: the downloadable app, its commands, and the web page built into it.

set(KERA_EMBED_PAIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/web/index.html=index.html"
    "${CMAKE_CURRENT_SOURCE_DIR}/web/app.css=app.css"
    "${CMAKE_CURRENT_SOURCE_DIR}/web/app.js=app.js"
    "${CMAKE_SOURCE_DIR}/assets/brand/icons/favicon.svg=favicon.svg"
    "${CMAKE_SOURCE_DIR}/assets/brand/icons/favicon.ico=favicon.ico"
    "${CMAKE_SOURCE_DIR}/assets/brand/icons/apple-touch-icon.png=apple-touch-icon.png"
    "${CMAKE_SOURCE_DIR}/assets/brand/svg/kera-logo-on-dark.svg=kera-logo-on-dark.svg"
    "${CMAKE_SOURCE_DIR}/assets/brand/svg/kera-logo-on-light.svg=kera-logo-on-light.svg"
    "${CMAKE_SOURCE_DIR}/LICENSE=LICENSE"
    "${CMAKE_SOURCE_DIR}/NOTICE=NOTICE"
    "${KERA_THIRD_PARTY_LICENSES}=THIRD_PARTY_LICENSES.txt"
    "${CMAKE_SOURCE_DIR}/bench/prompts/short_chat.json=bench/prompts/short_chat.json"
    "${CMAKE_SOURCE_DIR}/bench/prompts/code.json=bench/prompts/code.json"
    "${CMAKE_SOURCE_DIR}/bench/prompts/long_context.json=bench/prompts/long_context.json"
    "${CMAKE_SOURCE_DIR}/bench/prompts/tool_call.json=bench/prompts/tool_call.json"
)
set(KERA_EMBED_DEPS "")
foreach(pair IN LISTS KERA_EMBED_PAIRS)
    string(FIND "${pair}" "=" eq REVERSE)
    string(SUBSTRING "${pair}" 0 ${eq} src)
    list(APPEND KERA_EMBED_DEPS "${src}")
endforeach()

set(KERA_WEB_CPP ${CMAKE_BINARY_DIR}/generated/kera_web_assets.cpp)
# "|" separates the pairs on the command line: ";" would need escaping that differs between generators
string(REPLACE ";" "|" KERA_EMBED_PAIRS_ARG "${KERA_EMBED_PAIRS}")
add_custom_command(
    OUTPUT ${KERA_WEB_CPP}
    COMMAND ${CMAKE_COMMAND} "-DOUT=${KERA_WEB_CPP}" "-DPAIRS=${KERA_EMBED_PAIRS_ARG}" -P ${CMAKE_SOURCE_DIR}/cmake/kera_embed.cmake
    DEPENDS ${KERA_EMBED_DEPS} ${CMAKE_SOURCE_DIR}/cmake/kera_embed.cmake
    COMMENT "Embedding Kera's web page"
    VERBATIM)

set(KERA_APP_SOURCES
    main.cpp
    app_server.cpp
    model_runner.cpp
    setup_cli.cpp
    plan_cli.cpp
    finetune_cli.cpp
    perplexity_cli.cpp
    backend_choice.cpp
    backend_auto.cpp
    calibrate_cli.cpp
    serve.cpp
    serve_router.cpp
    image_cli.cpp
    media_cli.cpp
    precision_hint.cpp
    prompt_improver.cpp
    media3d_cli.cpp
    ${CMAKE_SOURCE_DIR}/tools/bench/bench.cpp
    ${CMAKE_SOURCE_DIR}/tools/exactness/exactness.cpp
    ${CMAKE_SOURCE_DIR}/tools/trace/trace.cpp
    ${CMAKE_SOURCE_DIR}/tools/wait/wait.cpp
    ${KERA_WEB_CPP}
)
if (WIN32)
    list(APPEND KERA_APP_SOURCES windows/kera.rc)
endif()

add_executable(kera ${KERA_APP_SOURCES})
target_include_directories(kera PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/generated)
# where `kera finetune` finds tools/finetune when run from a build outside the source tree
# where a build finds tools/finetune when it is run from its build tree. A download built on someone's own machine
# gives another name here, so that the machine's folders are not in the program (packaging/linux/build-spark.sh)
set(KERA_SOURCE_DIR_IN_PROGRAM "${CMAKE_SOURCE_DIR}" CACHE STRING "Kera: the source folder's name compiled into the program")
target_compile_definitions(kera PRIVATE KERA_SOURCE_DIR="${KERA_SOURCE_DIR_IN_PROGRAM}")
target_link_libraries(kera PRIVATE
    kera-core
    cpp-httplib
    llama-common
    llama-server-impl
    llama-cli-impl
    llama-completion-impl
    llama-bench-impl
    llama-perplexity-impl
    llama-tts-impl
    llama-quantize-impl
)
kera_target_warnings(kera)

if (WIN32)
    target_link_libraries(kera PRIVATE shell32 psapi)
endif()

# The Windows build fetches LLVM's OpenMP runtime (GGML_OPENMP_FETCH); its copy step lives in
# llama.cpp's folder, which is left out of the default build, so ask for it here
if (TARGET ggml-openmp-runtime)
    add_dependencies(kera ggml-openmp-runtime)
endif()
