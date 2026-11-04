# campus v2 + Studio 界面 + 数据审计修复

- 时间：2026-11-04 02:57
- 范围：`sim/tools/campus.py`、`sim/tools/audit_dataset.py`、`sim/tools/collect_parallel.py`、
  `sim/tools/eval_closed_loop.py`、`sim/tools/test_*`、`sim/worlds/uav_rgb.sdf`、
  `sim/worlds/templates/`、`tools/preprocess.py`、`ui/studio.py`、`ui/test_studio.py`
- 结论：代码改动完成并通过自检；**未采集、未训练、未跑闭环**；未提交

## 做了什么

### 1. 顶视相机高度与构图（`sim/worlds/uav_rgb.sdf`）

`studio_top_camera` 从 `3.5 0 18` 改为 `2.5 0 6`。

- 只降高度（18→6，中心仍 3.5）时实测发现**出生点 x=-0.5 落在画面边缘**；
  把中心挪到 2.5 后整条航线（出生 -0.5 → 目标最远 6.5）进画面。
- 实测（NVIDIA EGL 无头，seed 0）：内容包围盒 101×71 → **397×343**，
  占画面 **4.1% → 78.8%**，无人机可见。

### 2. 模板内容寻址归档（`sim/worlds/templates/`）

- `git show dd5c9cb:sim/worlds/uav_rgb.sdf` 的 sha256 **正好等于**
  旧 480 集 manifest 的 `template_sha256 = 7d7f9472…`，验证了"旧数据可字节复现"。
- 归档三份：`7d7f9472`（v1 数据用）、`3267bb7f`（加 chase/top 后）、
  `70ef2c0a`（本次顶视相机修好后）。

### 3. 数据审计两条 bug（`sim/tools/audit_dataset.py`）

| 检查 | 原判定 | 根因 | 修法 |
| --- | --- | --- | --- |
| `time_pairing` FAIL 1/7572 | bug | audit 用 50 ms 管所有 gap，`collect.py:88` 的 `DEPTH_STALE_NS=100 ms`；失败那步 depth 恰好 100 ms | 拆成 pose ≤50 ms / depth ≤100 ms，与 collect 的 `>` 语义严格对齐 |
| `campus_teacher_reach` FAIL 5/320 | bug | 把"到点"与"整集无碰撞"写在同一检查里；6 集其实全部到点（0.092–0.352 m），失败的是碰撞 | 拆成 `campus_teacher_reach` + 新增 `campus_no_collision` |
| `safe_collision` 报"另有已排除集 1 步" | bug | quarantine 记账重复 | 隔离集单独记 `quarantined{episodes, safe_collision_steps}` |
| `safe_collision` FAIL 1/962 | **真实缺陷** | 教师在 island 旁射线全堵→SAFE→惯性蹭碰 | 不改代码，靠 quarantine |

用新逻辑对真实数据离线复算：train 全量 7572 步 `depth>100ms` 0 步、`pose>50ms` 0 步；
6 个失败集 reach 全 PASS、no_collision 全 FAIL。

### 4. campus v2（`sim/tools/campus.py`）

- `VERSION = "campus.v1"` → `"campus.v2"`（v1 的 480 集冻结）
- plaza 的 `building_back`：`x=8, 1×9×5` → `x=14, 1×12×6`
- 布局随机范围放宽：目标 `x [4.8,5.4]→[3.6,6.5]`、`y [1.4,1.65]→[1.55,2.0]`、island `x [2.0,2.5]→[2.0,3.0]`
- tree_road 树排 `±3.1 → ±3.35`（目标 |y| 上限 2.0 后净空只剩 0.7 m，低于 0.9 m 契约）
- 2000 seed 扫描：0 个 A* 不可达、0 个净空不足、0 个目标超出 28° 视野、0 个被遮挡

### 5. campus.v2 的跨文件打通

v2 会让三处下游只认 `campus.v1` 的代码失效，已修：

| 文件 | 问题 | 改法 |
| --- | --- | --- |
| `tools/preprocess.py:106` | 白名单只认 `campus.v1`，**v2 数据直接拒绝打包** | 改用 `campus.VERSION` |
| `sim/tools/collect_parallel.py:643` | 清单硬编码 `geometry_version="campus.v1"` | `campus.VERSION` |
| `sim/tools/eval_closed_loop.py:136` | `task_version == "campus.v1"` 才用校园指令生成器 | 改按前缀 `startswith("campus.")`（v1 数据仍要能评测） |
| `sim/tools/eval_closed_loop.py:568` | 报告 `geometry_version` 写死 | `campus.VERSION` |

### 6. Studio 界面 20 项改造（`ui/studio.py`）

按性价比分批完成，全部经截图核对：

- **图表列** 574 → 约 878 px（`event_text width=80 → 1`）；标题/图例不再重叠；补 x 轴刻度
- **指标两级化**：2 张 hero 卡（Test Macro-F1 / 闭环 SR）+ 5 个 chip；SR 副行显示集数与地图数
- **混淆矩阵热力图**（颜色=行内占比）+ 每类 recall 柱状图；诊断区移到事件日志上方
- **Experiments 接真实数据**：`discover_runs` 扫 `$DATA_ROOT/uav_rgb` 两层，
  `eval_index` 按 `eval.json` 的 `ckpt` 反查 run —— 实测 **40 份 eval.json 中 30 份带 ckpt、
  指向 18 个 run、0 条失效**。表从空表变为 **91 条真实记录**
- **每集表 + 失败筛选**：新增 **"未到达"** 分类。`eval-cl-full` 80 集中 **76 集**属于此类
  （既没撞也没卡），这是旧界面完全看不到的失败模式
- **多评测 SR 规则**：1 份显示数值、多份显示 `多评测(N)`、0 份显示 `—`；
  **不取 latest**（`student-base` 挂 goal_stop/goal_reach/goal_reach_ever 三种口径，混报比空更糟）
- **Gazebo 启动**改用 `collect_parallel.gz_env()` + `gz sim -s -r --headless-rendering`
- **仿真页相机行** 40 px → 216/195 px（Canvas `height=300 → 1`）
- **主题**：输入框 `#0F1A2B`、按钮 `#1E293B`、正文 10pt
- **`DATA_ROOT` 配错不再崩**，弹对话框后正常启动

## 没做什么

- **没有采集 v2 数据**（约 45 分钟），没有训练，没有跑闭环
- **没有把 plaza 亮度拉到 train 水平**，见下节实测
- **没有做同族留出对照实验**（零成本但需要重划 split + 训练）
- **没有提交**：改动跨 12 个文件 + 1 个新目录，需确认归属后再按主题拆分提交
- `sim/tools/material.py` 不存在（外观逻辑在 `campus.py::material()` 与
  `scene_gen.py::appearance_params()`），此前方案中"复用外观 v1"是空操作

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `bash tools/check.sh` | **EXIT=0** |
| `ctest --test-dir build/check` | **25/25 通过**（173.5 s） |
| `test_spawn` | PASS（20 seeds × 2 modes），`GOLDEN_LEGACY` 已随模板更新 |
| `test_campus` | PASS（48 scenes / 3 topology holdouts） |
| `test_audit_dataset` | PASS（新增"到点但有碰撞"回归） |
| `test_collect` | 32 项通过（新增 depth gap 100/101 ms 边界） |
| `test_studio` | 11 项（原 8 + 新 3） |
| `test_monitor` / `test_trainer_demo` | 74 / 92 项 |
| 顶视相机实拍 | 6 m 下内容占画面 78.8%，无人机可见 |
| plaza 首帧亮度（真实渲染） | 见下节 |

### plaza 外观实测（这一轮最重要的负面结论）

真实渲染（NVIDIA EGL 无头，`/uav/camera` 首帧灰度均值）：

| seed | 旧实测 | 静态估算 | 挪墙后实测 | ambient 0.45 后 |
| --- | --- | --- | --- | --- |
| 16 | 73.2 | 102.6 | 83.8 | 86.8 |
| 20 | 88.8 | 114.2 | 90.5 | — |
| 22 | 58.6 | 102.8 | 75.4 | — |
| 41 | 50.2 | 98.8 | 68.0 | — |
| **均值** | **67.7** | 105.1 | **79.4** | — |

- 挪墙有效：**+11.7 灰度级**
- ambient 0.289→0.45 只再涨 **+3.0**，基本无效
- train 族是 **102.25 ± 7.5**，仍差约 15

**结论**：plaza 与 train 的差距是**构图**（画面里是一堵墙），不是光度。
继续调光照无法消除，而消除构图差异等于取消跨族实验。
因此"外观匹配的跨族对照"这个去混淆方案**概念上不成立**。

## 待确认

1. **是否提交**：12 个修改文件 + `sim/worlds/templates/` 新目录，建议按
   「campus v2」「审计修复」「Studio 界面」「模板归档 + 顶视相机」四个主题拆
2. **是否做同族留出对照**：在 building_lane 内按 `geometry_gid` 留出图做 test，
   与 plaza 跨族 test 对比。零采集成本，是唯一能分开"没学会"与"没泛化"的实验
3. **campus v2 是否现在就采集**：改布局后旧 480 集冻结，新数据必须整包重采

## 风险与遗留

- **v1 与 v2 不能混包**：`preprocess.py` 现在只认 `campus.VERSION`，
  v1 数据**无法再重新打包**（已有 `packed-clean/dataset.uavr` 不受影响）
- **旧闭环结果不可字节复现**：模板从 `7d7f9472` 变到 `70ef2c0a`；
  已归档三份模板，复现时用对应副本
- **Studio 旧实例（pid 119970）仍跑旧代码**，需重启才能看到新界面
- **轨迹分析地图的坐标轴在宽扁容器里被压扁**，刻度标签挤成一串
  （截图 `final_sim.png` 可见），功能不受影响，属收尾项
- **plaza 亮度仍未达标**（79.4 vs 目标 94–110），若 v2 数据直接用于跨族实验，
  "0/80" 的归因问题只解决了一半
- 数据审计在**不传 quarantine 时**会因 5–6 个真实碰撞集持续报红，
  这是刻意暴露缺陷，不能悄悄放过
