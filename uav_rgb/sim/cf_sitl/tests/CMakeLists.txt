# cf_sitl 的 CTest：json 读写、参数文件约定、reset 确定性、proc 协议往返、外部/独立模式一致性。
# 顶层已 enable_testing()。依赖参数文件/真库的测试在条件不满足时退出 77 → CTest 显示 Skipped 而不是 Failed。
if(NOT TARGET cf_sitl_tools_json)
  add_library(cf_sitl_tools_json STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../src/json_min.cpp)
  target_include_directories(cf_sitl_tools_json PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_compile_features(cf_sitl_tools_json PUBLIC cxx_std_17)
endif()
if(NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

set(CF_SITL_PARAMS "${CMAKE_CURRENT_SOURCE_DIR}/../../dynamics/e1_mini.json")

add_executable(test_json_min test_json_min.cpp)
target_link_libraries(test_json_min PRIVATE cf_sitl_tools_json)

add_executable(test_params_file test_params_file.cpp)
target_link_libraries(test_params_file PRIVATE cf_sitl_tools_json)

add_executable(test_reset_determinism test_reset_determinism.cpp)
target_link_libraries(test_reset_determinism PRIVATE cf_sitl)

add_executable(test_proc_protocol test_proc_protocol.cpp)
target_link_libraries(test_proc_protocol PRIVATE cf_sitl_tools_json)

add_executable(test_external_vs_standalone test_external_vs_standalone.cpp)
target_link_libraries(test_external_vs_standalone PRIVATE cf_sitl cf_sitl_tools_json Eigen3::Eigen)

foreach(t test_json_min test_params_file test_reset_determinism test_proc_protocol test_external_vs_standalone)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  target_compile_features(${t} PRIVATE cxx_std_17)
  target_compile_options(${t} PRIVATE -Wall -Wextra)
endforeach()

add_test(NAME cf_sitl_json_min COMMAND test_json_min)
add_test(NAME cf_sitl_params_file COMMAND test_params_file "${CF_SITL_PARAMS}")
add_test(NAME cf_sitl_reset_determinism COMMAND test_reset_determinism "${CF_SITL_PARAMS}")
add_test(NAME cf_sitl_proc_protocol COMMAND test_proc_protocol $<TARGET_FILE:cf_sitl_proc> "${CF_SITL_PARAMS}")
add_test(NAME cf_sitl_external_vs_standalone COMMAND test_external_vs_standalone "${CF_SITL_PARAMS}")
set_tests_properties(cf_sitl_params_file cf_sitl_reset_determinism cf_sitl_proc_protocol cf_sitl_external_vs_standalone
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
set_tests_properties(cf_sitl_json_min PROPERTIES TIMEOUT 30)
