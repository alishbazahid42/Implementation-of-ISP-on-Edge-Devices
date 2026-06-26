cmake_minimum_required(VERSION 3.18)
project(syenet_bench CUDA CXX)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# SM 7.5 = Turing (Quadro T1000, RTX 20xx)
# Add more as needed: 72 = Xavier, 87 = Orin, 80 = A100, 86 = RTX 30xx
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90" CACHE STRING "CUDA architectures")

# ── TensorRT ──────────────────────────────────────────────────────────────────
# Set TRT_DIR to your TensorRT install root if not in system paths.
# e.g. cmake -DTRT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.x"
if(DEFINED TRT_DIR)
    set(TRT_INCLUDE "${TRT_DIR}/include")
    set(TRT_LIB_DIR "${TRT_DIR}/lib")
else()
    # Common default locations
    find_path(TRT_INCLUDE NvInfer.h
        PATHS
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.9.0.34/include"
        "C:/TensorRT/include"
        "/usr/include/x86_64-linux-gnu"
        "/usr/local/tensorrt/include"
    )
    find_path(TRT_LIB_DIR nvinfer.lib nvinfer.so
        PATHS
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.9.0.34/lib"
        "C:/TensorRT/lib"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/local/tensorrt/lib"
    )
endif()

if(NOT TRT_INCLUDE)
    message(FATAL_ERROR "TensorRT include not found. "
            "Install TensorRT from https://developer.nvidia.com/tensorrt "
            "and pass -DTRT_DIR=<path>")
endif()

message(STATUS "TensorRT include : ${TRT_INCLUDE}")
message(STATUS "TensorRT lib dir : ${TRT_LIB_DIR}")

# ── STB image (header-only, place stb_image.h in include/) ────────────────────
# Download from: https://github.com/nothings/stb/blob/master/stb_image.h

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cu
    src/trt_engine.cu
    src/benchmark.cu
    src/metrics.cu
    src/nvml_monitor.cu
    src/report_gen.cpp
)

add_executable(syenet_bench ${SOURCES})

target_include_directories(syenet_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TRT_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(syenet_bench PRIVATE
    CUDA::cudart
    CUDA::nvml
    ${TRT_LIB_DIR}/nvinfer.lib
    ${TRT_LIB_DIR}/nvinfer_plugin.lib
    ${TRT_LIB_DIR}/nvonnxparser.lib
)

# Windows: copy TRT DLLs next to exe
if(WIN32)
    add_custom_command(TARGET syenet_bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TRT_LIB_DIR}/nvinfer.dll"
            "${TRT_LIB_DIR}/nvinfer_plugin.dll"
            "${TRT_LIB_DIR}/nvonnxparser.dll"
            $<TARGET_FILE_DIR:syenet_bench>
        COMMENT "Copying TensorRT DLLs"
    )
endif()

# Compiler options
target_compile_options(syenet_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --generate-line-info
        -O3
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        /O2 /W3
    >
)

set_target_properties(syenet_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
