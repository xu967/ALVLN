# CfSitlSystem：gz-sim8 System 插件（纯 C++17，不碰 CUDA，不受仓库 sm_75 影响）。
# 两种用法：
#   1) 被 sim/cf_sitl/CMakeLists.txt 以 add_subdirectory(gz) 包含（CF_SITL_BUILD_GZ=ON）：
#      直接链接父工程的 cf_sitl target（名字可用 CF_SITL_TARGET 覆盖）。
#   2) 独立构建：cmake -S sim/cf_sitl/gz -B <build> [-DCF_SITL_LIB=/path/libcf_sitl.so]
#      [-DCF_SITL_INCLUDE=<dir>]；不给 CF_SITL_LIB 时 find_library(cf_sitl)，
#      可用 CF_SITL_LIB_DIR 提示搜索目录。
#   测试桩：-DCF_SITL_GZ_STUB=ON 用 tests/stub_cf_sitl.cpp 代替真库（只用于链路联调）。
cmake_minimum_required(VERSION 3.22)
project(cf_sitl_gz CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(CF_SITL_GZ_STUB "用 tests/stub_cf_sitl.cpp 桩库代替真库（联调用，不是固件在环）" OFF)
set(CF_SITL_TARGET "cf_sitl" CACHE STRING "父工程里 SITL 库的 target 名")
set(CF_SITL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../include" CACHE PATH "cf_sitl.h 所在目录")
set(CF_SITL_LIB "" CACHE FILEPATH "独立构建时 libcf_sitl.so/.a 的路径")
set(CF_SITL_LIB_DIR "" CACHE PATH "find_library(cf_sitl) 的提示目录")

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

if(CF_SITL_GZ_STUB)
  add_library(cf_sitl_stub SHARED tests/stub_cf_sitl.cpp)
  target_include_directories(cf_sitl_stub PUBLIC "${CF_SITL_INCLUDE}")
  set(_cf_sitl_link cf_sitl_stub)
  message(WARNING "CfSitlSystem 链接的是测试桩 cf_sitl_stub，不是固件在环库")
elseif(TARGET ${CF_SITL_TARGET})
  set(_cf_sitl_link ${CF_SITL_TARGET})
else()
  if(NOT CF_SITL_LIB)
    find_library(CF_SITL_LIB NAMES cf_sitl HINTS "${CF_SITL_LIB_DIR}" "${CF_SITL_LIB_DIR}/lib")
  endif()
  if(NOT CF_SITL_LIB)
    message(FATAL_ERROR "找不到 libcf_sitl：传 -DCF_SITL_LIB=<path> 或 -DCF_SITL_LIB_DIR=<dir>，"
                        "或 -DCF_SITL_GZ_STUB=ON 用桩库联调")
  endif()
  add_library(cf_sitl_import UNKNOWN IMPORTED)
  set_target_properties(cf_sitl_import PROPERTIES IMPORTED_LOCATION "${CF_SITL_LIB}")
  set(_cf_sitl_link cf_sitl_import)
endif()

add_library(CfSitlSystem SHARED CfSitlSystem.cc CfSitlState.cc)
target_include_directories(CfSitlSystem PRIVATE "${CF_SITL_INCLUDE}")
target_link_libraries(CfSitlSystem PRIVATE gz-sim8::gz-sim8 gz-plugin2::register ${_cf_sitl_link})
# 插件由 gz 以 dlopen 加载；把 libcf_sitl 所在目录写进 RUNPATH，运行时不用再设 LD_LIBRARY_PATH。
set_target_properties(CfSitlSystem PROPERTIES BUILD_RPATH_USE_ORIGIN ON INSTALL_RPATH_USE_LINK_PATH ON)
if(CF_SITL_LIB)
  get_filename_component(_cf_sitl_dir "${CF_SITL_LIB}" DIRECTORY)
  set_property(TARGET CfSitlSystem APPEND PROPERTY BUILD_RPATH "${_cf_sitl_dir}")
endif()

# FlattenState 的字段顺序测试（不需要 Gazebo 进程，也不链接 SITL 库）
find_package(gz-msgs10 REQUIRED)
enable_testing()
add_executable(test_flatten_state tests/test_flatten_state.cc CfSitlState.cc)
target_include_directories(test_flatten_state PRIVATE "${CF_SITL_INCLUDE}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_flatten_state PRIVATE gz-msgs10::gz-msgs10)
add_test(NAME cf_sitl_gz_flatten_state COMMAND test_flatten_state)
