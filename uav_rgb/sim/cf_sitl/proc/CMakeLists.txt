# cf_sitl_proc：独立进程封装，由顶层 CMake 在本目录存在时 add_subdirectory。
# 依赖父工程的 target cf_sitl 与 ../include；json_min 单独再编一份（库用 -fvisibility=hidden，
# 不导出 Json 的符号）。proc/selftest/tests 三个目录都可能先到，所以 target 用 if(NOT TARGET) 守护。
if(NOT TARGET cf_sitl_tools_json)
  add_library(cf_sitl_tools_json STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../src/json_min.cpp)
  target_include_directories(cf_sitl_tools_json PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_compile_features(cf_sitl_tools_json PUBLIC cxx_std_17)
endif()

add_executable(cf_sitl_proc cf_sitl_proc.cpp)
target_include_directories(cf_sitl_proc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cf_sitl_proc PRIVATE cf_sitl cf_sitl_tools_json)
target_compile_features(cf_sitl_proc PRIVATE cxx_std_17)
target_compile_options(cf_sitl_proc PRIVATE -Wall -Wextra)
# 可执行文件放构建根目录，3DGS 侧按 <build>/cf_sitl_proc 找
set_target_properties(cf_sitl_proc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
