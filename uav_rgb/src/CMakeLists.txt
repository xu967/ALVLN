cmake_minimum_required(VERSION 3.20)
# 关键：必须在 project() 之前强制指定，否则 CMake 3.28 采用 nvcc 默认架构 (52)。
# 本机 Turing sm_75（RTX 2080 Ti）；不要写 120。
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
project(uavrgb_core LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")

add_library(uavrgb_core
  action.cpp
  tensor.cpp
  ops_cpu.cpp
  ops_weighted.cpp
  conv_cpu.cpp
  conv_gradcheck.cpp
  nn_cpu.cpp
  linear_cuda.cu
  conv_cuda.cu
  softmax_cuda.cu
  nn_cuda.cu
  model.cpp
  dataset.cpp
  batch_reader.cpp
  run_owner.cpp
  checkpoint.cpp
  class_weights.cpp
  eval_metrics.cpp
  metrics_json.cpp
  trainer.cpp
  trainer_eval.cpp
  infer.cpp
)
target_include_directories(uavrgb_core PUBLIC include)

# 新设备状态层不反向依赖 trainer；现有 host Tensor 路径继续独立使用。
add_library(uavrgb_device
  device/buffer.cpp
  device/context.cpp
  device/state_validation.cpp
  device/state.cpp
)
target_include_directories(uavrgb_device PUBLIC include)
target_link_libraries(uavrgb_device PUBLIC uavrgb_core CUDA::cudart)

enable_testing()
add_executable(test_action tests/test_action.cpp)
target_link_libraries(test_action PRIVATE uavrgb_core)
add_test(NAME test_action COMMAND test_action)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE uavrgb_core)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_gradcheck tests/test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE uavrgb_core)
add_test(NAME test_gradcheck COMMAND test_gradcheck)

add_executable(test_linear_cuda tests/test_linear_cuda.cpp)
target_link_libraries(test_linear_cuda PRIVATE uavrgb_core)
add_test(NAME test_linear_cuda COMMAND test_linear_cuda)

add_executable(test_conv_cuda tests/test_conv_cuda.cpp)
target_link_libraries(test_conv_cuda PRIVATE uavrgb_core)
add_test(NAME test_conv_cuda COMMAND test_conv_cuda)

add_executable(test_softmax_cuda tests/test_softmax_cuda.cpp)
target_link_libraries(test_softmax_cuda PRIVATE uavrgb_core)
add_test(NAME test_softmax_cuda COMMAND test_softmax_cuda)

add_executable(test_nn_cuda tests/test_nn_cuda.cpp)
target_link_libraries(test_nn_cuda PRIVATE uavrgb_core)
add_test(NAME test_nn_cuda COMMAND test_nn_cuda)

# 训练器 CLI 与端到端自检（合成数据，不依赖真实采集）。
add_executable(uavrgb_train train_main.cpp)
target_link_libraries(uavrgb_train PRIVATE uavrgb_core)

# 常驻推理进程：闭环评测按帧喂 RGB，避免每步重启进程。
add_executable(uavrgb_infer infer_main.cpp)
target_link_libraries(uavrgb_infer PRIVATE uavrgb_core)

add_executable(test_infer_io tests/test_infer_io.cpp)
target_link_libraries(test_infer_io PRIVATE uavrgb_core)
add_test(NAME test_infer_io COMMAND test_infer_io)

add_executable(test_trainer tests/test_trainer.cpp)
target_link_libraries(test_trainer PRIVATE uavrgb_core)
add_test(NAME test_trainer COMMAND test_trainer)

add_executable(test_overfit tests/test_overfit.cpp)
target_link_libraries(test_overfit PRIVATE uavrgb_core)
add_test(NAME test_overfit COMMAND test_overfit)

# T4：--eval-only 与训练期同权重/同数据的 validation 事件必须逐字节对账。
add_executable(test_eval_only tests/test_eval_only.cpp)
target_link_libraries(test_eval_only PRIVATE uavrgb_core)
add_test(NAME test_eval_only COMMAND test_eval_only)

# 类别权重 + 碰撞过滤（数值 / 数据集 / 端到端不平衡实验）。
add_executable(test_weighted_loss tests/test_weighted_loss.cpp)
target_link_libraries(test_weighted_loss PRIVATE uavrgb_core)
add_test(NAME test_weighted_loss COMMAND test_weighted_loss)

add_executable(test_dataset_filter tests/test_dataset_filter.cpp)
target_link_libraries(test_dataset_filter PRIVATE uavrgb_core)
add_test(NAME test_dataset_filter COMMAND test_dataset_filter)

add_executable(test_dataset_split tests/test_dataset_split.cpp)
target_link_libraries(test_dataset_split PRIVATE uavrgb_core)
add_test(NAME test_dataset_split COMMAND test_dataset_split)

add_executable(test_class_weights_train tests/test_class_weights_train.cpp)
target_link_libraries(test_class_weights_train PRIVATE uavrgb_core)
add_test(NAME test_class_weights_train COMMAND test_class_weights_train)

# EVAL_STANDARD 强制报告的宏指标；test_trainer 只校验键名，数值靠这一项。
add_executable(test_eval_metrics tests/test_eval_metrics.cpp)
target_link_libraries(test_eval_metrics PRIVATE uavrgb_core)
add_test(NAME test_eval_metrics COMMAND test_eval_metrics)

# 首批并行训练基础设施；GPU 用例串行运行，CPU reader/锁包含故障注入。
add_executable(test_device_state tests/test_device_state.cpp)
target_link_libraries(test_device_state PRIVATE uavrgb_device)
target_link_options(test_device_state PRIVATE "-Wl,--wrap=cudaEventSynchronize")
add_test(NAME test_device_state COMMAND test_device_state)

add_executable(test_device_transfer_errors tests/test_device_transfer_errors.cpp)
target_link_libraries(test_device_transfer_errors PRIVATE uavrgb_device)
target_link_options(test_device_transfer_errors PRIVATE
  "-Wl,--wrap=cudaStreamSynchronize" "-Wl,--wrap=cudaSetDevice")
add_test(NAME test_device_transfer_errors COMMAND test_device_transfer_errors)

add_executable(test_batch_reader tests/test_batch_reader.cpp)
target_link_libraries(test_batch_reader PRIVATE uavrgb_core)
target_link_options(test_batch_reader PRIVATE "-Wl,--wrap=pread")
add_test(NAME test_batch_reader COMMAND test_batch_reader)

add_executable(test_run_owner tests/test_run_owner.cpp)
target_link_libraries(test_run_owner PRIVATE uavrgb_core)
add_test(NAME test_run_owner COMMAND test_run_owner)
