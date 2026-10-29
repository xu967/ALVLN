cmake_minimum_required(VERSION 3.20)

# 关键：必须在 project() 之前指定，否则 CMake 3.28 会采用 nvcc 默认架构 (sm_52)。
# 本机 GPU 为 RTX 2080 Ti = compute capability 7.5 -> sm_75。不要写 120/sm_120。
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)

project(gpu_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UAVRGB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
set(UAVRGB_CORE_LIB "${UAVRGB_SRC}/build/libuavrgb_core.a")

if(NOT EXISTS "${UAVRGB_CORE_LIB}")
  message(FATAL_ERROR "找不到预构建的 ${UAVRGB_CORE_LIB}，请先构建 src/")
endif()

# 单文件基准；直接链接 src/ 已构建的 parity 版 CUDA 包装（只读，不改 src/）。
add_executable(gpu_bench gpu_bench.cu)
target_include_directories(gpu_bench PRIVATE "${UAVRGB_SRC}/include")
target_link_libraries(gpu_bench PRIVATE "${UAVRGB_CORE_LIB}")
set_target_properties(gpu_bench PROPERTIES
  CUDA_ARCHITECTURES 75
  CUDA_SEPARABLE_COMPILATION OFF
)

message(STATUS "uavrgb_core static lib : ${UAVRGB_CORE_LIB}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
