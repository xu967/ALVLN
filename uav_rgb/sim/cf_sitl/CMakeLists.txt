# cf_sitl：E1 MINI 固件在环仿真库（独立 GPLv3 目录，见 LICENSE；目录外的代码不得链接）。
# 构建（产物 libcf_sitl.so 直接落在构建目录根）：
#   cmake -S sim/cf_sitl -B /root/rivermind-data/uav_rgb/tmp/build-cf-sitl -G Ninja \
#         -DCF_DRONE_DIR=/root/claude-code-backup-20261028/code/CF-Drone
#   cmake --build /root/rivermind-data/uav_rgb/tmp/build-cf-sitl --parallel 8
# 纯 C/C++，不碰 CUDA；固件源码构建时从 CF_DRONE_DIR 原样 #include，不复制进仓库。
cmake_minimum_required(VERSION 3.22)
project(cf_sitl C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)   # 固件与 plant.h 用 M_PI
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CF_DRONE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../CF-Drone" CACHE PATH "CF-Drone 固件源码目录（只读）")
if(NOT EXISTS "${CF_DRONE_DIR}/CF-Drone.ino")
  message(FATAL_ERROR "CF_DRONE_DIR=${CF_DRONE_DIR} 里没有 CF-Drone.ino；传 -DCF_DRONE_DIR=<CF-Drone 目录>")
endif()
option(CF_SITL_BUILD_GZ "一并构建 gz/ 的 Gazebo 插件（需要 gz-sim8 开发包）" OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# 确定性：关掉 FMA 收缩与 fast-math；x86-64 默认 SSE2，无 x87 中间精度问题
set(CF_SITL_DET_FLAGS -ffp-contract=off -fno-fast-math)

add_library(cf_sitl SHARED src/cf_sitl.cpp fw/fw_sitl.cpp src/json_min.cpp exec/executor.c)
# fw/stub 与 CF_DRONE_DIR 只给 fw/fw_sitl.cpp 用（Arduino/ESP32 打桩头 + 固件 .ino 原文）
target_include_directories(cf_sitl
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/exec" "${CMAKE_CURRENT_SOURCE_DIR}/fw"
          "${CMAKE_CURRENT_SOURCE_DIR}/fw/stub" "${CF_DRONE_DIR}")
target_link_libraries(cf_sitl PRIVATE Eigen3::Eigen m)
# 只导出 cf_sitl_*（src/cf_sitl.cpp 用 pragma 把头文件声明设成 default）；固件全局符号全部本地绑定
target_compile_options(cf_sitl PRIVATE ${CF_SITL_DET_FLAGS} -fvisibility=hidden -Wall)
target_compile_options(cf_sitl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden -fno-gnu-unique -Wno-reorder>)
target_link_options(cf_sitl PRIVATE -Wl,-Bsymbolic -Wl,--no-undefined)
set_target_properties(cf_sitl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# 冒烟：reset → arm_takeoff(2.0) → 独立积分 15 s，确认到 READY
add_executable(cf_sitl_smoke src/smoke_main.cpp)
target_link_libraries(cf_sitl_smoke PRIVATE cf_sitl)
target_compile_options(cf_sitl_smoke PRIVATE -Wall)
target_compile_definitions(cf_sitl_smoke PRIVATE
  CF_SITL_DEFAULT_PARAMS="${CMAKE_CURRENT_SOURCE_DIR}/../dynamics/e1_mini.json")

enable_testing()
# 下面三个目录由 SITL-tools agent 提供，存在才进
foreach(sub proc selftest tests)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${sub}/CMakeLists.txt")
    add_subdirectory(${sub})
  endif()
endforeach()
if(CF_SITL_BUILD_GZ AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gz/CMakeLists.txt")
  add_subdirectory(gz)
endif()
