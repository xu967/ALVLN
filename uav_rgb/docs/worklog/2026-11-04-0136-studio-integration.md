# 接入 UAV Navigation Studio（桌面工作台）

- 时间：2026-11-04 01:36
- 范围：`ui/`、`sim/worlds/uav_rgb.sdf`、`sim/tools/test_spawn.py`、`sim/tools/test_student.py`、
  `tools/check.sh`、`README.md`、`sim/README.md`、`docs/HANDOFF.md`、`docs/TASKS.md`
- 结论：完成（代码接入 + 自检 + 本机 L3 相机验证）；未提交
- 来源：`uav_project_code_with_studio_20261104 (1).zip`

## 做了什么

### 0. 归档完整性

第一份 `uav_project_code_with_studio_20261104.zip` 只有 **2097152 字节（正好 2 MiB）**，
没有中央目录，`unzip` 直接失败；按本地文件头能救出 402/403 条目，断在
`code/ALVLN/frontend/preview_simulation.png`。第二份 `(1).zip` 5,435,563 字节，
957 条目，EOCD 齐全，`unzip -t` 无错误——用它。

两包里的 `code/uav_rgb` 完全相同，且**就是我们仓库的快照**（`src/train_main.cpp` 与
我们逐字节一致，`src/trainer.cpp` 里已有我们 2026-11-03 加的 `vision_only` 守卫）。
所以**不能整目录覆盖**，只能按差异挑。

### 1. 新增文件（6 个）

`ui/studio.py`、`ui/test_studio.py`、`ui/launch_studio.sh`、`ui/install_desktop_entry.sh`、
`ui/README_STUDIO.md`、`docs/worklog/2026-11-04-uav-navigation-studio.md`。

`docs/#U76d1#U63a7#U7cfb#U7edf#U4ea4#U63a5.md` **未复制**：与仓库已有
`docs/监控系统交接.md` 逐字节相同，只是文件名 URL 编码损坏。

### 2. 修改文件（10 个，逐个 diff 确认方向）

| 文件 | 改动 |
| --- | --- |
| `sim/worlds/uav_rgb.sdf` | +54 行：新增 `/uav/chase_camera`（640×360@15Hz）与 `/uav/top_camera`（480×360@10Hz），均无质量/无碰撞 |
| `sim/tools/test_spawn.py` | 3 个 legacy golden sha256 随模板更新 |
| `sim/tools/test_student.py` | 启动超时 0.3→1.5 s，避免冷启动抖动误判 |
| `ui/trainer_native.py` | 默认训练器 `src/build/` → `build/check/`，支持 `UAV_RGB_TRAIN_BIN`（兼容旧 `UAVRGB_TRAIN_BIN`） |
| `ui/test_trainer_demo.py` | 固定 `sleep(0.5)` 改成轮询 metrics ready，修 flake |
| `tools/check.sh` | 加 `ui/test_studio.py` |
| `README.md`、`sim/README.md`、`docs/HANDOFF.md`、`docs/TASKS.md` | 文档补充 |

**没有采用** `sim/tools/eval_closed_loop.py`：快照版是旧的（还是我们已废弃的
`unique_geometry`），覆盖会回退 2026-11-03 的地图覆盖统计修复。

### 3. 整合时修的两处不匹配

1. **字段名对不上**：studio 读 `eval.json` 的 `unique_geometry`，我们已改名
   `unique_maps`（`dedup_key` 含颜色/左右，会把 80 集算成 80 个几何）。
   改 `ui/studio.py` 三处：优先读 `unique_maps`，回退 `unique_geometry` 兼容旧报告。
2. **`lifecycle.json` 从不写 `pid` / `running`**（原工作室代码的 bug）：
   启动只写 `starting`，收尾写终态，而恢复逻辑只在 `status=="running"` 且 pid 存活时
   才认 → **「关闭桌面后子进程继续跑、重开恢复」这条功能是死的**，在跑的训练会被
   判成 `incomplete_unobserved` 并写错误 summary。已在 `subprocess.Popen` 成功后补写
   `{"status": "running", "pid": ...}`。

### 4. 重建 `build/check`

`ui/trainer_native.py` 的默认训练器换成了 `build/check/uavrgb_train`，而那份是
2026-11-03 11:35 编的旧二进制（缺 11-03 的 `--vision-only` 守卫与内联 test 修复）。
已 `cmake --build build/check` 重建，与 `src/build` 只差 20 字节（nvcc 临时文件名）。

## 没做什么

- **没有接入 `ALVLN/frontend` + `server/`**。原因：那是冻结的 TinyAerialVLN V8 管理壳，
  顶层 README 明确写「历史网页原型，不是桌面应用后端」；后端 `config/registry.json`
  为空、不接本项目的 trainer/eval/Gazebo；与 studio.py 不共享状态、日志、run 目录或话题；
  而且它已经完整存在于我们的 `../ALVLN` 克隆里。结论：只作归档参考。
- **没有修 `install_desktop_entry.sh` 的 `$HOME/Desktop` 硬编码**。原因：本机 XDG 桌面是
  `$HOME/桌面`，桌面快捷方式不会生成，但应用菜单项仍可用，不阻塞。
- **没有跑 `ui/studio.py` 的完整 GUI**。原因：本会话无交互桌面，只做了无 GUI 的自检与
  Gazebo 话题级验证。
- **没有测三路相机同时渲染对采集吞吐的影响**。原因：本轮不采集；但这是接入模板后
  的真实代价，见「风险」。
- **没有提交/推送**。

## 待确认

- 模板多两路相机后，采集吞吐会下降多少？chip 渲染像素从 2.3 MPix/s 变成约 7.5 MPix/s
  （30Hz×320×240 + 15Hz×640×360 + 10Hz×480×360）。要不要按外观 v1 的规矩先测一次
  ≤1.25× 的比值门限？如果要长期采集，建议测；如果不采，就先这样。
- 已采集的 480 集校园数据用的是**旧模板**（无 chase/top 相机）。合并新旧数据前，
  `world_sha256` 会对不上，需要按 provenance 分开。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `unzip -t` 完整包 | 通过（957 条目，无错误） |
| `py_compile ui/studio.py ui/trainer_native.py ui/test_studio.py` | 通过 |
| `python3 ui/test_studio.py` | **8/8 通过**（含 `test_current_world_template_has_three_camera_sensors_and_keeps_fpv`） |
| `python3 sim/tools/test_spawn.py` | PASS（20 seeds × 2 modes） |
| `python3 sim/tools/test_student.py` | PASS |
| `python3 ui/test_monitor.py` | 74 项通过，1 项跳过 |
| `python3 ui/test_trainer_demo.py` | 92 项通过 |
| `bash tools/check.sh --python-only` | PASS（含新加的 studio 自检） |
| `ctest --test-dir build/check` | **25/25 通过**（169.9 s） |
| Gazebo 真实出帧（本机 L3，工作室作者没条件做） | **通过**：`/uav/camera`、`/uav/chase_camera`、`/uav/top_camera` 三个话题都存在，chase 收到带 header 的帧，top 测得 480×360 |
| GUI 交互 | 未跑 |

L3 命令（NVIDIA EGL 无头渲染，跑完已清理进程与临时文件）：

```bash
python3 sim/tools/scene_gen.py --seed 0 --out /tmp/w.sdf
setsid env -u DISPLAY XDG_RUNTIME_DIR=/tmp/xdg GZ_PARTITION=studio_cam \
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  gz sim -s -r --headless-rendering /tmp/w.sdf &
gz topic -l | grep camera      # → /uav/camera /uav/chase_camera /uav/top_camera
```

## 风险与遗留

- **模板变更影响所有未来采集**：多两路渲染，吞吐可能下降；且新旧数据的 `world_sha256` 不同。
- `build/check` 与 `src/build` 现在是两份二进制，重建哪一份要分清（studio 用 `build/check`，
  eval 的 `student.py` 也用 `build/check`）。以后跑完实验记得两边都重建，或统一用一份。
- `install_desktop_entry.sh` 的 `$HOME/Desktop` 在本机不生效（桌面目录是 `$HOME/桌面`）。
- 工作室作者未在真实 Gazebo 上验证的部分（三视角帧率、GPU 占用）本轮只验到「话题存在且出帧」，
  没有测长时间稳定性。
