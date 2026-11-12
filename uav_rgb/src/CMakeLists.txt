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
  ops_soft.cpp
  ops_spatial.cpp
  sha256.cpp
  soft_targets.cpp
  conv_cpu.cpp
  conv_gradcheck.cpp
  nn_cpu.cpp
  linear_cuda.cu
  conv_cuda.cu
  softmax_cuda.cu
  spatial_ce_cuda.cu
  nn_cuda.cu
  model.cpp
  model_dev.cu
  model_dev_film.cu
  model_dev_aux.cu
  augment_dev.cu
  dataset.cpp
  aux_sidecar.cpp
  batch_reader.cpp
  run_owner.cpp
  checkpoint.cpp
  class_weights.cpp
  eval_metrics.cpp
  metrics_json.cpp
  trainer.cpp
  trainer_eval.cpp
  trainer_dev.cpp
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

# 显存分配失败的报错必须带 requested/free/total（2026-11-03 a50 OOM 复盘）。
add_executable(test_cuda_alloc_errors tests/test_cuda_alloc_errors.cpp)
target_link_libraries(test_cuda_alloc_errors PRIVATE uavrgb_core CUDA::cudart)
add_test(NAME test_cuda_alloc_errors COMMAND test_cuda_alloc_errors)

add_executable(test_nn_cuda tests/test_nn_cuda.cpp)
target_link_libraries(test_nn_cuda PRIVATE uavrgb_core)
add_test(NAME test_nn_cuda COMMAND test_nn_cuda)

# 设备常驻训练一步 vs host 算子逐层一步：同 batch/同初始权重，参数差 < 1e-5。
add_executable(test_device_train_step tests/test_device_train_step.cpp)
target_link_libraries(test_device_train_step PRIVATE uavrgb_core)
add_test(NAME test_device_train_step COMMAND test_device_train_step)

# 设备侧 uint8→float（+数据增强）的下标与数值语义。
add_executable(test_augment_dev tests/test_augment_dev.cpp)
target_link_libraries(test_augment_dev PRIVATE uavrgb_core CUDA::cudart)
add_test(NAME test_augment_dev COMMAND test_augment_dev)

# 训练器 CLI 与端到端自检（合成数据，不依赖真实采集）。
add_executable(uavrgb_train train_main.cpp)
target_link_libraries(uavrgb_train PRIVATE uavrgb_core)

# 常驻推理进程：闭环评测按帧喂 RGB，避免每步重启进程。
add_executable(uavrgb_infer infer_main.cpp)
target_link_libraries(uavrgb_infer PRIVATE uavrgb_core)

add_executable(test_infer_io tests/test_infer_io.cpp)
target_link_libraries(test_infer_io PRIVATE uavrgb_core)
add_test(NAME test_infer_io COMMAND test_infer_io)

# 部署推理路径的输入布局（交叉审核 F2）：非方形输入、双帧顺序、prev-action 字节，
# 与自己摆成 NCHW 的 host 参考前向逐值对拍；转置下标、两帧对调、R/B 互换都会红。
add_executable(test_infer_layout tests/test_infer_layout.cpp)
target_link_libraries(test_infer_layout PRIVATE uavrgb_core)
add_test(NAME test_infer_layout COMMAND test_infer_layout)

add_executable(test_trainer tests/test_trainer.cpp)
target_link_libraries(test_trainer PRIVATE uavrgb_core)
add_test(NAME test_trainer COMMAND test_trainer)

add_executable(test_overfit tests/test_overfit.cpp)
target_link_libraries(test_overfit PRIVATE uavrgb_core)
add_test(NAME test_overfit COMMAND test_overfit)

# 翻转增强：标签左右互换、prev-action 同步互换、深度目标按列镜像（交叉审核 F4）。
add_executable(test_flip_augment tests/test_flip_augment.cpp)
target_link_libraries(test_flip_augment PRIVATE uavrgb_core)
add_test(NAME test_flip_augment COMMAND test_flip_augment)

# 双帧训练读 prev_index 那一帧、平面顺序与部署一致（交叉审核 F5）。
add_executable(test_two_frame_prev_index tests/test_two_frame_prev_index.cpp)
target_link_libraries(test_two_frame_prev_index PRIVATE uavrgb_core)
add_test(NAME test_two_frame_prev_index COMMAND test_two_frame_prev_index)

# T4：--eval-only 与训练期同权重/同数据的 validation 事件必须逐字节对账。
add_executable(test_eval_only tests/test_eval_only.cpp)
target_link_libraries(test_eval_only PRIVATE uavrgb_core)
add_test(NAME test_eval_only COMMAND test_eval_only)

# 类别权重 + 碰撞过滤（数值 / 数据集 / 端到端不平衡实验）。
add_executable(test_weighted_loss tests/test_weighted_loss.cpp)
target_link_libraries(test_weighted_loss PRIVATE uavrgb_core)
add_test(NAME test_weighted_loss COMMAND test_weighted_loss)

# 软标签（蒸馏）损失。
add_executable(test_soft_loss tests/test_soft_loss.cpp)
target_link_libraries(test_soft_loss PRIVATE uavrgb_core)
add_test(NAME test_soft_loss COMMAND test_soft_loss)

add_executable(test_soft_targets_train tests/test_soft_targets_train.cpp)
target_link_libraries(test_soft_targets_train PRIVATE uavrgb_core)
add_test(NAME test_soft_targets_train COMMAND test_soft_targets_train)

add_executable(test_soft_targets_identity tests/test_soft_targets_identity.cpp)
target_link_libraries(test_soft_targets_identity PRIVATE uavrgb_core)
add_test(NAME test_soft_targets_identity COMMAND test_soft_targets_identity)

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

# L2 文本分支：token 影响 logits、embedding 解析梯度对拍、侧车展开、checkpoint v4 往返。
add_executable(test_text_branch tests/test_text_branch.cpp)
target_link_libraries(test_text_branch PRIVATE uavrgb_core)
add_test(NAME test_text_branch COMMAND test_text_branch)

add_executable(test_batch_reader tests/test_batch_reader.cpp)
target_link_libraries(test_batch_reader PRIVATE uavrgb_core)
target_link_options(test_batch_reader PRIVATE "-Wl,--wrap=pread")
add_test(NAME test_batch_reader COMMAND test_batch_reader)

# STUDENT-P2：逐格 CE 算子、UAVRAUX1 侧车读写、checkpoint v4/v5/v6 交叉读。
add_executable(test_spatial_ce tests/test_spatial_ce.cpp)
target_link_libraries(test_spatial_ce PRIVATE uavrgb_core)
add_test(NAME test_spatial_ce COMMAND test_spatial_ce)

add_executable(test_aux_sidecar tests/test_aux_sidecar.cpp)
target_link_libraries(test_aux_sidecar PRIVATE uavrgb_core)
add_test(NAME test_aux_sidecar COMMAND test_aux_sidecar)

add_executable(test_checkpoint_versions tests/test_checkpoint_versions.cpp)
target_link_libraries(test_checkpoint_versions PRIVATE uavrgb_core)
add_test(NAME test_checkpoint_versions COMMAND test_checkpoint_versions)

add_executable(test_run_owner tests/test_run_owner.cpp)
target_link_libraries(test_run_owner PRIVATE uavrgb_core)
add_test(NAME test_run_owner COMMAND test_run_owner)

# STUDENT-SELECT：--save-every-epoch 的逐轮 ckpt 文件名 / 内容 / 默认关。
add_executable(test_save_every_epoch tests/test_save_every_epoch.cpp)
target_link_libraries(test_save_every_epoch PRIVATE uavrgb_core)
add_test(NAME test_save_every_epoch COMMAND test_save_every_epoch)

# 训练启动只算一次数据集 SHA256（config / 契约 / 软标签 / 侧车共用），按进程读字节数验证。
add_executable(test_sha_once tests/test_sha_once.cpp)
target_link_libraries(test_sha_once PRIVATE uavrgb_core)
add_test(NAME test_sha_once COMMAND test_sha_once)

# warmup 覆盖全程的告警；resume 时 ckpt_best.bin 比 ckpt.bin 新/旧的处理。
add_executable(test_resume_best_warn tests/test_resume_best_warn.cpp)
target_link_libraries(test_resume_best_warn PRIVATE uavrgb_core)
add_test(NAME test_resume_best_warn COMMAND test_resume_best_warn)

# uavrgb_train 入口的退出码（eval-only 目录归属 / 坏 config 不 abort / --seed 范围 / eval+resume 互斥）。
add_executable(test_train_cli tests/test_train_cli.cpp)
target_link_libraries(test_train_cli PRIVATE uavrgb_core)
add_dependencies(test_train_cli uavrgb_train)
add_test(NAME test_train_cli COMMAND test_train_cli $<TARGET_FILE:uavrgb_train>)

# D3：warmup、余弦退火与 AdamW 解耦衰减的手算数值测试（卡 C1a）。
add_executable(test_lr_adamw tests/test_lr_adamw.cpp)
target_link_libraries(test_lr_adamw PRIVATE uavrgb_core CUDA::cudart)
add_test(NAME test_lr_adamw COMMAND test_lr_adamw)

# H8：增强随机数种子随 --seed 变化且 resume 继承后同步重播（卡 C1a）。
add_executable(test_augment_seed tests/test_augment_seed.cpp)
target_link_libraries(test_augment_seed PRIVATE uavrgb_core)
add_test(NAME test_augment_seed COMMAND test_augment_seed)

# H9/D2：--log-grad-norm 每轮裁剪前梯度范数的中位数、最大值与裁剪步数占比（卡 C1a）。
add_executable(test_log_grad_norm tests/test_log_grad_norm.cpp)
target_link_libraries(test_log_grad_norm PRIVATE uavrgb_core)
add_dependencies(test_log_grad_norm uavrgb_train)
add_test(NAME test_log_grad_norm COMMAND test_log_grad_norm $<TARGET_FILE:uavrgb_train>)

# D6：续训时继承源 run 的 --max-grad-norm 且显式传参可覆盖（卡 C1a）。
add_executable(test_resume_grad_norm tests/test_resume_grad_norm.cpp)
target_link_libraries(test_resume_grad_norm PRIVATE uavrgb_core)
add_dependencies(test_resume_grad_norm uavrgb_train)
add_test(NAME test_resume_grad_norm COMMAND test_resume_grad_norm $<TARGET_FILE:uavrgb_train>)

# C6：--eval-only 从 config.json 或 ckpt 契约读取碰撞过滤口径，缺记录或冲突时拒绝（卡 C1a）。
add_executable(test_eval_only_drop_collision tests/test_eval_only_drop_collision.cpp)
target_link_libraries(test_eval_only_drop_collision PRIVATE uavrgb_core)
add_dependencies(test_eval_only_drop_collision uavrgb_train)
add_test(NAME test_eval_only_drop_collision COMMAND test_eval_only_drop_collision $<TARGET_FILE:uavrgb_train>)

# D9：卷积反传 conv2d_backward / conv2d_backward_cuda 的输入梯度 dx 有限差分检查（卡 C1a）。
add_executable(test_conv_dx_gradcheck tests/test_conv_dx_gradcheck.cpp)
target_link_libraries(test_conv_dx_gradcheck PRIVATE uavrgb_core)
add_test(NAME test_conv_dx_gradcheck COMMAND test_conv_dx_gradcheck)

# 第 7 条：--cuda-blocking-sync / UAVRGB_CUDA_BLOCKING_SYNC=1 降低自旋 CPU 占用且数值逐位不变（卡 C1a）。
add_executable(test_cuda_blocking_sync tests/test_cuda_blocking_sync.cpp)
target_link_libraries(test_cuda_blocking_sync PRIVATE uavrgb_core)
add_dependencies(test_cuda_blocking_sync uavrgb_train)
add_test(NAME test_cuda_blocking_sync COMMAND test_cuda_blocking_sync $<TARGET_FILE:uavrgb_train>)
