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
