cmake_minimum_required(VERSION 3.18)
project(tinyvln_numeric_probes LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(tinyvln_gradcheck tests/gradcheck.cpp)
target_include_directories(tinyvln_gradcheck PRIVATE include)
enable_testing()
add_test(NAME tinyvln_gradcheck COMMAND tinyvln_gradcheck)
# Real CUDA training and ONNX exports are planned but intentionally not faked.
