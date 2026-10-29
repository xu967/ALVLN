cmake_minimum_required(VERSION 3.20)

# 关键：必须在 project() 之前指定，否则 CMake 会采用 nvcc 的默认架构 (sm_52)
# 本机 GPU 为 GeForce RTX 2080 Ti = compute capability 7.5 -> sm_75。不要写 120。
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)

project(cuda_smoke LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")

add_executable(cuda_smoke vector_add.cu)
set_target_properties(cuda_smoke PROPERTIES
  CUDA_ARCHITECTURES 75
  CUDA_SEPARABLE_COMPILATION OFF
)
