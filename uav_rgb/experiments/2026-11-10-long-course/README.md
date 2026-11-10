# 2026-11-10 长课程实验（longcourse）

把 K 个 base 柱阵块首尾相接成一条长课程，不训练，用既有权重测每一块的存活率，量出成功率随航程怎么衰减。

- 课程生成：`sim/tools/scene_course.py`（`--task course_base --blocks K`）。第 0 块与同 seed 的 base 柱阵逐项相同；第 k 块是同 split 的另一个 base 几何，沿 x 平移 5.0·k m。成功线 = 6.0 + 5.0·(K−1) m。
- 入口：`run.sh`（预注册写在文件头；教师 1 臂 + 学生 9 臂，每臂 20 集，5 路并行）。`summarize.py` 随时可跑，写 `summary.md`。
- 用法：`mkdir -p $OUT && cp run.sh summarize.py $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &`，`$OUT = /root/rivermind-data/uav_rgb/exp-long-course-20261110`。
- 逐块存活判据：第 k 块通过 = 首次 `pose_gt.x >= block_goal_x[k]`，且到这一步（含）为止没有接触碰撞、`|y| <= 3.0`。`S_k` 是累计存活率，分母是请求集数。

## 电池

SITL 起始 SoC 是 0.5（`sim/dynamics/e1_mini.json` 的 `sim.value.battery_soc_init`），11-09 探针里 READY 后约 106 s 开始掉高、约 123 s 进 FAILSAFE。
教师 K=3 约 85 步、复位后 87 s 内过线（冒烟结果见 `/root/rivermind-data/uav_rgb/tmp/phase12-20261110/reports/longcourse.md`）；
学生比教师慢，K=3 约 95–100 步，离掉压起点很近。summarize.py 把采集中止（FAILSAFE）单列，并给出每集 `sim_s` 与末 10 步最低高度。

起始 SoC 能不能改成 1.0：不用重编库。`battery_soc_init` 每次 reset 从参数 JSON 读（`sim/cf_sitl/src/plant.h` 的 `Params::load` 与 `reset()`），
插件按 `$UAVRGB_REPO_ROOT/sim/dynamics/e1_mini.json` 解析参数路径，`collect.py --cf-sitl-params` 与 `cf_sitl_exec.DEFAULT_PARAMS` 读同一个路径。
把参数文件拷到另一个根目录、改 `battery_soc_init` 并设 `UAVRGB_REPO_ROOT` 即可。但参数值进指纹：`cf_sitl_v1:1bc8fedf:lib31` 会变成 `cf_sitl_v1:06e5f017:lib31`，
结果与今天的 K=1 不是同一动力学档；推力按 `(V_bus/3.7)^1.5` 缩放，满电开路电压 4.20 V 对 0.5 SoC 的 3.82 V，同占空比推力约高 15%，闭环行为变不变没有跑过（未核实）。
本实验不改起始 SoC。

## 没接的部分

训练打包（`tools/preprocess.py`、`audit_dataset.py`）没接课程：`preprocess` 按版本白名单拒收 `uavrgb.course_base.*`，放开版本门后 split 复算因 episode manifest 不带 `course` 而抛错，仍拒收（`test_scene_course.py` 里有用例锁住）。要拿课程数据训练时再接。

## 第二轮（2026-11-10 21:15 起跑，入口 `run_v2.sh`）

第一轮在 20:46 因时序混杂中止，数据留在 `exp-long-course-20261110`（`summary.md` 与 `DONE.v1` 在该目录）。第二轮输出 `/root/rivermind-data/uav_rgb/exp-long-course-v2-20261110`，用法同第一轮，换成 `run_v2.sh`。

- 混杂的来源：`--window-from observation` 下动作窗口从观测帧起算，观测到派发的仿真时延越长，动作生效越短。K=3 世界每步 A* 重规划约 145 ms（K=1 约 35 ms；教师臂是动作本身，学生臂是影子教师标签），
  观测到派发 0.22 s（K=1 0.13 s），每步动作生效 0.77–0.79 s（K=1 参照 0.871 s）。等帧、深度解码、推理、帧龄都与 K 无关；PNG 落盘在派发之后，不计入这段时延。证据与数字见报告。
- 修法：`sim/tools/teacher.py` 的 `astar_path` 把栅格可走性按场景缓存（提交 `fcbf2fe`，路径逐位不变，`sim/tools/test_astar_cache.py` 锁定）。`--no-aux-cameras` 与降低 `--camera-rate` 标定过，都没有把生效时长拉回（去辅助相机后仿真更快，同样的墙钟计算折成更长的仿真时延；降相机频率使帧更旧）。
- 标定：`calibrate_timing.sh`（教师，seed 1 / 113 各 2 集，2 条车道，输出 `$OUT/calib`）。读表：`python summarize.py --timing "标签=目录1,目录2" ...`（`--tol 0.02` 改标记阈值）。
- 臂：teacher（20 集）、guard2-calswa-s0..2（60 集）、calswa-s0..2（60 集），5 条车道；`guard2-s*`（ckpt_best）这一轮不跑。P1–P3 预注册原文与第一轮逐字相同，新增的只有有效性条件（生效时长与转角对 K=1 参照 ±3%）。
- 汇总：`python summarize.py [输出目录] [--title 文字] [--present-only]`。`--present-only` 的分母是目录里有轨迹的集数，没有 `eval.json` 的臂按 `episodes/seed_N` 读，用于被中止的第一轮和跑到一半的进度查看。
  新增三节：第 8 节时序健康（逐臂生效时长、稳态前飞位移、转角，对 K=1 参照 0.871 s / 26.1° 的偏离，超 3% 标 ⚠；`≥40 步的集` 单列，把短集的 A* 开销效应与整体偏差分开）、
  第 9 节护栏臂「数步数」航向（26.1° × 净转向数）与真实航向的差（20 步一段的均值、标准差、90 分位）、第 10 节每块块末 `|y|` 的分布。
