# 三项核实：渲染是否吃 GPU / 校园仿真缺陷 / 校园基线文档真伪

- 时间：2026-11-03 13:31
- 范围：`sim/tools/*`（只读）、`docs/experiments/2026-11-03-campus-baseline.md`、`docs/worklog/2026-11-03-1100-campus-baseline.md`
- 结论：完成（核实），**未修改任何代码或文档**
- 方式：三个 subagent 并行取证 + 本会话对关键结论独立复验

## 做了什么

### 1. 渲染是否用 GPU（结论：没有，一帧都没有）

- `DISPLAY=:20 glxinfo -B` → `OpenGL renderer string: llvmpipe (LLVM 20.1.2, 256 bits)`，
  `Accelerated: no`。`:20` 是 Xorg + `Driver "dummy"`（`/etc/X11/selkies-dummy.conf`）。
- 实跑 1 实例 × 4 步采集（seed 424242），运行期 54 次 `nvidia-smi` 采样全部 `1 MiB / 0%`；
  gz sim 进程 `/proc/<pid>/maps` 只有 `libEGL_mesa` / `libgallium` / `libGLX_mesa`，
  无任何 nvidia 库，fd 无 `/dev/nvidia*`。
- **反证 README 的历史结论**：`sim/README.md:19-27` 称「容器没有 NVIDIA EGL」，
  实测该说法不成立。真正原因是容器环境把 glvnd 搜索路径指到了 conda 目录：
  `__EGL_VENDOR_LIBRARY_DIRS=/opt/conda/share/glvnd/egl_vendor.d`（其中只有 Mesa），
  而 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 明明存在。
- 覆盖该变量后 `eglinfo -B` 报 `EGL vendor string: NVIDIA`；以 surfaceless EGL 起
  `gz sim -s -r` 可出帧，进程加载 `libEGL_nvidia`、打开 `/dev/nvidia0`，
  nvidia-smi 变为 `133 MiB / 6–7%`。
- 顺带纠正：本机 `nproc=56` 但 cgroup `cpu.max=1200000 100000`，**实际只有 12 核**。

### 2. 校园仿真代码审计

已确认的缺陷（本会话复验过的标 ✓）：

| # | 位置 | 问题 | 严重度 |
| --- | --- | --- | --- |
| 1 | `split_manifest.py:173-186,204-212` ✓ | `--verify` 仍用 `sample_pillars` 判跨 split，完全忽略 `map_id`；对合法 campus 集合会大面积假阳性并 exit 1 | 高（隐性：当前无人调用 `--verify`，也无 campus 采集清单） |
| 2 | `scene_gen.py:322-324` ✓ | `task=campus` 直接 `campus.build(template, seed, scene_id)`，`--appearance` / `--spawn-mode` 被静默丢弃，但 manifest 仍记录它们 → provenance 与实际不符 | 高 |
| 3 | `campus.py:17-23` ✓ | `material()` 把纹理**绝对路径**写进 SDF，`world_sha256` 因此绑死本机路径 | 中 |
| 4 | `split_manifest.py:101` ✓ | 非 campus 分支 `meta.get("geometry_gid")` 直接信任 manifest，不复算；被改过的 manifest 能骗过 `preprocess.py` 的防泄漏门 | 中 |
| 5 | `make_counterfactual.py` / `make_decisive.py` / `probe_text_effect.py` | 未排除 campus，仍套用 l1 的 relation→waypoint 约定，产物会被 `campus_identity` 拒绝 | 中 |
| 6 | `campus.py:139` | `clearance.ok=True` 硬编码，未实际计算 | 低 |
| 7 | `eval_closed_loop.py:544` | campus 报告里 `goal_radius` 硬编码 0.5，改半径后报告与明细矛盾 | 低 |

已验证**不是**问题的：split 无泄漏（seed 0..2999 每 seed 单 split、三族固定 train/val/test）、
几何全可达（1500 seeds A* 无失败）、所有实体都有 `<collision>`、同 seed 两次 build 字节一致、
三张纹理用固定种子可复现、campus 的 0.5 未与 `REACH_RADIUS=1.5` 混用、
base/l1 世界 SDF 与 HEAD 逐字节相同（2 task × 2 appearance × 60 seeds）。

### 3. 两份校园文档核对

属实的：6 集 138 步、教师 6/6 无碰撞进 0.5 m、标签重算 0/138、早先 3 集因
`YAW_RIGHT` 占 4.35%<5% 被拒、CTest 25/25、`goal_radius=0.5` 未混用旧口径。

对不上的（本会话复验过 ✓）：

- ✓ `docs/experiments/2026-11-03-campus-baseline.md:52` 写 `--seeds 16:24`，
  但 `eval-smoke-fixed/eval.json` 实为 `requested_seeds=[16]`、`episodes_requested=1`。
  照文档复跑得不到同样结果。
- ✓ `train-smoke/config.json` 的 `seed=42`，违反项目「训练固定 seed=0」硬规则，文档未声明豁免。
- 「6/6 无碰撞进入目标区」是**教师**结果；学生是 0/1（3 步、只出 FORWARD_SLOW、最终 4.07 m）。
- `visible_goal_reach_v1` 名不副实：运行时判定只有「无碰撞 + 曾进入 0.5 m」，无可见性检查。
- 第一次 eval 失败（`completion=0.0`、`scene_gen` `KeyError: min_pair_gap_m`）未记入文档。
- 「审计 PASS」含 4 项 SKIP，文档未说明。
- 离线缺每类召回（`EVAL_STANDARD.md` 要求四项）。
- worklog「验证到哪一步」无表格无命令；文件名 1100 但 mtime 12:16，时间线矛盾。

## 没做什么

- **没有修改任何代码或文档**。原因：`campus.py` / `collect_parallel.py` / `scene_gen.py`
  当前归另一个并发会话（工作区有大量未提交改动），我改动会与其冲突。列出的缺陷只做记录。
- **没有切到 GPU 渲染**。原因：换渲染器等于换像素，新旧数据不可混训，必须先做逐像素
  差异对比与吞吐 pilot；且改动点在他人正在编辑的文件里。
- **没有跑完整采集循环验证 GPU 下的吞吐/稳定性**。原因：只验证了「能出帧」。
- **没有启动 Gazebo 做多实例并发的 GPU 显存/上下文测试**。
- **没有修 `--verify`**。原因：当前无人调用它，且没有 campus 采集清单可测；属于潜伏问题。

## 待确认

- GPU 渲染的收益是否值得这份风险？现有单步 1.59 s 中 1.525 s 是仿真推进（RTF 0.63），
  渲染未必是瓶颈——**提速幅度未测，不能假定**。
- 是否接受「新旧数据集按渲染后端分开，不可混训」这条口径。
- 校园基线是否应等文档口径修正后再作为基线引用。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `DISPLAY=:20 glxinfo -B` | 通过：renderer 为 llvmpipe，Accelerated: no |
| 采集期 `nvidia-smi` 54 次采样 | 通过：全部 1 MiB / 0% |
| gz 进程 `/proc/<pid>/maps` | 通过：无 nvidia 库 |
| `eglinfo -B`（覆盖 vendor filenames） | 通过：EGL vendor = NVIDIA |
| `gz sim -s -r` surfaceless EGL 出帧 | 通过：出图、打开 /dev/nvidia0、nvidia-smi 133 MiB |
| `eval-smoke-fixed/eval.json` 字段 | 通过：确为 `requested_seeds=[16]`，与文档 `16:24` 不符 |
| `train-smoke/config.json` | 通过：确为 `seed=42` |
| campus 代码审计 | 只读静态审计 + 少量脚本验证；未跑 Gazebo 采集 |
| 两份文档核对 | 数字类已对磁盘产物逐项比对 |

## 风险与遗留

- 若继续用 `split_manifest --verify` 验收 campus 数据，会得到「大面积泄漏」的错误结论。
- 纹理绝对路径在仓库迁移或 `--keep-worlds` 产物转移后，会让 world 哈希失效、贴图丢失。
- `task=campus` 静默忽略 `--appearance`/`--spawn-mode` 但 manifest 照记，会污染 provenance。
- 校园基线目前只有一个学生 episode 的证据（SR=0.0），不足以支撑任何「能跑」的结论。
