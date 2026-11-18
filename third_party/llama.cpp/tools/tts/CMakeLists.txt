# [Kera] llama-tts-impl: the tool's logic, reusable by an app (as llama-cli-impl is)

set(TARGET llama-tts-impl)
add_library(${TARGET} tts.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(${TARGET} PUBLIC llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-tts)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-tts-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
