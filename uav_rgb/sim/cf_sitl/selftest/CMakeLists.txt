# cf_sitl_selftest：6 个场景的自检程序（用法见 selftest_main.cpp 文件头），由顶层 CMake add_subdirectory。
if(NOT TARGET cf_sitl_tools_json)
  add_library(cf_sitl_tools_json STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../src/json_min.cpp)
  target_include_directories(cf_sitl_tools_json PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_compile_features(cf_sitl_tools_json PUBLIC cxx_std_17)
endif()

add_executable(cf_sitl_selftest
  selftest_main.cpp scenario_frame.cpp scenario_hover.cpp scenario_actuator.cpp scenario_sticks.cpp
  scenario_determinism.cpp scenario_battery.cpp scenario_longrun.cpp)
target_include_directories(cf_sitl_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cf_sitl_selftest PRIVATE cf_sitl cf_sitl_tools_json)
target_compile_features(cf_sitl_selftest PRIVATE cxx_std_17)
target_compile_options(cf_sitl_selftest PRIVATE -Wall -Wextra)
# 放构建根目录：check.sh 与 README 按 <build>/cf_sitl_selftest 调用
set_target_properties(cf_sitl_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
