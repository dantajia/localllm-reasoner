cmake_minimum_required(VERSION 3.14)
project(LocalLLM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(LLAMA_DIR "${PROJECT_SOURCE_DIR}/llama.cpp-master")
set(LLAMA_BUILD_DIR "${LLAMA_DIR}/build")
set(LLAMA_INCLUDE "${LLAMA_DIR}/include")
set(GGML_INCLUDE "${LLAMA_DIR}/ggml/include")
set(COMMON_INCLUDE "${LLAMA_DIR}/common")

include_directories(
    ${LLAMA_INCLUDE}
    ${GGML_INCLUDE}
    ${COMMON_INCLUDE}
)

set(LLAMA_LIB_DIR "${LLAMA_BUILD_DIR}/src/Release")
set(GGML_LIB_DIR "${LLAMA_BUILD_DIR}/ggml/src/Release")
set(COMMON_LIB_DIR "${LLAMA_BUILD_DIR}/common/Release")

link_directories(
    ${LLAMA_LIB_DIR}
    ${GGML_LIB_DIR}
    ${COMMON_LIB_DIR}
)

add_executable(localllm main.cpp)

find_library(LLAMA_LIB llama PATHS ${LLAMA_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_LIB ggml PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB ggml-cpu PATHS ${GGML_LIB_DIR} NO_DEFAULT_PATH)

if(LLAMA_LIB)
    target_link_libraries(localllm PRIVATE ${LLAMA_LIB})
else()
    message(FATAL_ERROR "llama.lib not found in ${LLAMA_LIB_DIR}")
endif()

if(GGML_LIB)
    target_link_libraries(localllm PRIVATE ${GGML_LIB})
endif()

if(GGML_BASE_LIB)
    target_link_libraries(localllm PRIVATE ${GGML_BASE_LIB})
endif()

if(GGML_CPU_LIB)
    target_link_libraries(localllm PRIVATE ${GGML_CPU_LIB})
endif()

if(WIN32)
    target_link_libraries(localllm PRIVATE ws2_32)
endif()

if(MSVC)
    target_compile_options(localllm PRIVATE /O2 /Oi /utf-8)
    target_link_options(localllm PRIVATE /LTCG)
endif()
