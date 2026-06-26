cmake_minimum_required(VERSION 3.18)
project(sr_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Target architectures:
#   53 = Jetson Nano (Maxwell)        62 = Jetson TX2
#   72 = Xavier                        87 = Jetson Orin Nano/NX (Ampere)
#   75 = Turing RTX 20xx / Quadro RTX  80 = A100        86 = RTX 30xx / A-series
#   89 = RTX 40xx / Ada                (Tesla V100 = 70)
# On Jetson, trim to just the local arch for faster builds:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=87 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "53;62;70;72;75;80;86;87;89")
endif()

add_executable(sr_infer
  src/main.cu
  src/model.cu
  src/kernels.cu
  src/utils.cu
)
target_compile_options(sr_infer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)
