# TinyAerialVLN V7.1 本地测试证据报告

> 在当前容器执行，不代表用户 Ubuntu RTX 5090 / SG2002 / ESP32 实测。2026-10-28。

## 已执行验证

- code/tests: **13 PASS**；场景确定性、日志处理、无指标伪造、非法路径、NaN 与部分写入。
- sim/tests: **2 PASS**；静态 SDF 不能冒充物理闭环，未实现功能明确拒绝。
- HTTP GET: `/api/health`, `/api/status`, `/api/runs`, `/api/scenes`, `/api/audit` 返回 200。
- HTTP 保护：非法 run=../etc/passwd 返回 404，未知路由 404，POST 405。
- Chromium：使用 `set_content` 测试页面切换与数据注入，前端无 JS pageerror；未完成浏览器直接对 localhost 访问。
- 当前 `doctor` 检测不到 nvcc、gz、训练可执行文件与正式配置；`sim doctor` 给出预期 GZ_NOT_INSTALLED。

## 未执行的设备与研究实验

- Gazebo 四旋翼电机模型、真实 RGB 机载传感、动力学和撞击检测。
- CF-Drone 软件在环、固件修改、任何真实 UART/失效保护测试。
- C++/CUDA 训练、蒸馏、模型转换、SG2002 实测、真机飞行。

## 代码测试原始输出

```text
test_audit_explicitly_states_limits (test_dashboard_v7_1.DashboardV71Tests.test_audit_explicitly_states_limits) ... ok
test_generated_scenes_are_not_flight_simulator (test_dashboard_v7_1.DashboardV71Tests.test_generated_scenes_are_not_flight_simulator) ... ok
test_html_has_five_pages_without_fabricated_metrics (test_dashboard_v7_1.DashboardV71Tests.test_html_has_five_pages_without_fabricated_metrics) ... ok
test_nans_rejected (test_dashboard_v7_1.DashboardV71Tests.test_nans_rejected) ... ok
test_no_run_no_fabricated_metrics (test_dashboard_v7_1.DashboardV71Tests.test_no_run_no_fabricated_metrics) ... ok
test_reads_only_completed_lines_and_warns (test_dashboard_v7_1.DashboardV71Tests.test_reads_only_completed_lines_and_warns) ... ok
test_runs_reject_symlink_and_escape (test_dashboard_v7_1.DashboardV71Tests.test_runs_reject_symlink_and_escape) ... ok
test_status_no_false_flight_claim (test_dashboard_v7_1.DashboardV71Tests.test_status_no_false_flight_claim) ... ok
test_dashboard_empty_no_fake_scores (test_infrastructure.InfrastructureTest.test_dashboard_empty_no_fake_scores) ... ok
test_refuses_nonexistent_training_binary (test_infrastructure.InfrastructureTest.test_refuses_nonexistent_training_binary) ... ok
test_rejects_invalid_obstacle_count (test_infrastructure.InfrastructureTest.test_rejects_invalid_obstacle_count) ... ok
test_rejects_path_escape (test_infrastructure.InfrastructureTest.test_rejects_path_escape) ... ok
test_scene_deterministic_and_well_formed (test_infrastructure.InfrastructureTest.test_scene_deterministic_and_well_formed) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.929s

OK
```

## 仿真测试原始输出

```text
test_static_scene_rejected_as_flight (test_v7_contract.V7Contract.test_static_scene_rejected_as_flight) ... ok
test_unimplemented_camera_cannot_false_pass (test_v7_contract.V7Contract.test_unimplemented_camera_cannot_false_pass) ... ok

----------------------------------------------------------------------
Ran 2 tests in 1.953s

OK
```
