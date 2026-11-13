# 2026-11-13 无人机 VLN 复核验证

- 范围：评测可信性、视频重建、实际数据覆盖；没有训练新的语言策略。
- 基线：开发仓库 `f7323c4`，GitHub `xu967/ALVLN` 的 `46f8ca1`，fetch 后远端一致。
- 环境：RTX 2080 Ti，22528 MiB，UUID `GPU-4910bb04-2d00-ca81-f64a-2031568758ad`；cgroup 12核；有卡模式。
- 时间：2026-11-13，Asia/Shanghai。以下数字是本轮实测，明确标为历史重算的除外。
- 令 `D=/root/rivermind-data/uav_rgb`，`X=$D/tmp/vln-review-20261113-session`，`R=$D/tmp/vln-render-review-20261113`。

## 1. 构建、回归和未通过项

```bash
source env.sh
export CUDA_VISIBLE_DEVICES=GPU-4910bb04-2d00-ca81-f64a-2031568758ad
export UAV_RGB_BUILD_DIR="$DATA_ROOT/uav_rgb/tmp/vln-review-20261113-session/build-check"
export CF_DRONE_DIR="$UAV_RGB_ROOT/../CF-Drone"
bash tools/check.sh
```

真实输出：C++/CUDA 构建完成，CTest **43/44通过**，`test_cuda_blocking_sync` 失败，脚本退出8；没有改该测试或放宽阈值。
它验证阻塞同步占用低于非阻塞的35%；首轮 `off=0.984148` 核、`cli=0.398577`、`env1=0.403897`，因此失败。
单独复跑 `ctest --test-dir "$UAV_RGB_BUILD_DIR" -R '^test_cuda_blocking_sync$' --output-on-failure` 仍失败：
`off=0.988139`、`cli=0.401910`、`env1=0.392682`。四次训练退出码均0，没有checkpoint逐字节比较失败。
这些结果只定位到性能门槛未过，不能宣称已确定根因或全套绿。原始日志 `$X/check.log`、`$X/check-blocking-retry.log`。

全量入口在CTest处停止，后续分别执行，不能把它们写成同一次 `check.sh` 全通过：

| 检查 | 命令/记录 | 真实结果 |
| --- | --- | --- |
| Python 全套 | `bash tools/check.sh --python-only`，`$X/check-python.log` | 退出0，有SKIP；含16个3DGS测试文件 |
| 固件在环 | `cmake -S sim/cf_sitl -B "$X/build-cf-sitl" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCF_DRONE_DIR="$CF_DRONE_DIR"`，构建后 `ctest --test-dir "$X/build-cf-sitl" --output-on-failure --no-tests=error` | 9/9通过；`$X/check-sitl.log` |
| 固件自检 | `"$X/build-cf-sitl/cf_sitl_selftest" --params sim/dynamics/e1_mini.json --out "$X/build-cf-sitl/selftest-out"` | 33通过、0失败、0跳过 |
| VLN 回归 | `$UAV_RGB_PYTHON sim/3dgs/test_vln.py` | 改前针对runner：5 failures、2 errors；改后18/18通过，`$X/test-vln-after.log` |
| 视频回归 | `$D/3dgs-venv/bin/python -m unittest sim.3dgs.test_video` | 11/11通过；`$R/video-tests.log` |
| 真正GPU渲染 | `RUN_3DGS_REAL=1 TORCH_CUDA_ARCH_LIST=7.5 TORCH_EXTENSIONS_DIR=$D/tmp/torch-extensions-sm75 TMPDIR=$D/tmp $D/3dgs-venv/bin/python -m unittest sim.3dgs.test_renderer` | 12/12通过、0跳过，135.124s；`$R/renderer-real-tests.log` |
| 实际视频编码 | `$D/3dgs-venv/bin/python -m sim.3dgs.video` | VP8/H264 编码读取PASS，25帧、body_px86，high档libx264；`$R/video-selftest.log` |

Python套件未覆盖：需要DISPLAY的3项GUI测试、未开启的平均checkpoint再生实验、缺msgpack的NuRec解包用例，以及各文件显式可选用例。
其中默认跳过的真实GPU渲染2项已通过上表单独启用检查补跑；C++与cf_sitl也已独立运行，结果按各自记录报告。

## 2. 真实 GPU + cf_sitl 小规模闭环

```bash
source env.sh
D="$DATA_ROOT/uav_rgb"
X="$D/tmp/vln-review-20261113-session"
"$D/3dgs-venv/bin/python" -m sim.3dgs.run_vln \
  --scene "$D/3dgs-scenes/sage3d_841150/scene.json" \
  --policy teacher --limit 2 --out "$X/v2-teacher"
"$D/3dgs-venv/bin/python" -m sim.3dgs.run_vln \
  --scene "$D/3dgs-scenes/sage3d_841150/scene.json" \
  --policy forward --limit 1 --out "$X/v2-forward"
```

教师输出目录预先创建为空，验证其可用；两次均退出0。复跑需要换新目录。
动力学 `cf_sitl_v1:1bc8fedf:lib31`，指标 `3dgs.vln.v2`，hold=0.5s。

| 策略 | 有效/请求 | SR | OSR | SPL | NE均值(m) | 碰撞率 | 用时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 特权教师 | 2/2 | 1 | 1 | 0.862229 | 0.439949 | 0 | 14.4s |
| 盲飞 | 1/1 | 0 | 0 | 0 | 2.993820 | 1 | 9.2s |

日志 `$X/v2-teacher.log`、`$X/v2-forward.log`，总表在对应目录 `vln_eval.json`。仅为链路冒烟，不是泛化或算法比较。
新教师两集task_id不同；manifest均为`diag-vln-eval`。真实调用 `preprocess.load_episode` 默认拒收，输出：
`split_policy='diag-vln-eval' 是诊断性划分，需显式 --allow-diagnostic-split`。

需求核验：第二句原文要求取衣架并放到衣服旁，本程序只评到目标附近的导航子任务；不能解释成无人机完成操作。

## 3. 旧轨迹复核：计分与录像错误

读取既有 `$D/exp-vln-m1-20261113/teacher` 的249集 actions/oracle，给参考路补动作前起点，离线重算37.75s：

| 指标 | v1原记录 | v2起点重算 |
| --- | --- | --- |
| SPL | 0.6770121550 | 0.6811217032 |
| SR | 0.8433734940 | 相同 |
| OSR | 0.8554216867 | 相同 |
| 碰撞率 | 0.1485943775 | 相同 |

`$X/m1-v2-spl-recompute.json` 保留差异。没有重飞249集、没有修改旧记录，SPL变化不是策略改进。

另对教师249+盲飞249集比较旧HEAD `video.episode_result` 与修复后函数，旧录像重建入口共**9次假成功**（教师2、盲飞7）。
例 `forward/.../tr_89_14`：旧 `SUCCESS reached (min 0.23 m)` → 新 `NO STOP 3.1 m`；主run_vln旧报告本身没有这项录像错误。

真实SITL的10个0.25s窗口：总计2.5s，30条trace；旧每条一帧/10fps会播放3s。修复后重采样25帧/2.5s，默认0.5s不改变帧序列。

## 4. 渲染与语言数据的真实限制

真实探针：
`$D/3dgs-venv/bin/python tools/scene_probe_quality.py --scene $D/3dgs-scenes/sage3d_841150/scene.json --route 0 --out $R/probe-sage841150`。
输出：路径8.53m、17帧、low=0、lapvar min/p10/median=106.9/343.3/548.1、hole_max=0，旧阈值PASS。
实际查看联系表，末段物体仍有明显涂抹；PASS只是粗锐度筛查。构造棋盘RGB配全零depth得lapvar1040400、hole_frac=1、low=False、route_pass=True，说明空洞没有参与合格判定。

本地4429句SAGE指令用现有`instruction.tokenize`统计：71.4718% token是UNK，96.5455%句子超过9 token，中位14 token。
文本接线和词表必须重新设计；显示自然语言不是模型使用语言的证据。

Wanderland本轮GPU双向探针：东向36.88m/62帧、西向36.92m/63帧，两者low_frac=0，hole最大2.06%/4.67%，旧阈值PASS。
v2碰撞模型的几何规划也可达，最小净空0.8m/1.0m；但地形标记`agl_model_unverified=true`，不能直接视为可飞。
主会话查看侧向偏航联系表也确认车辆/树叶涂抹、部分近墙视角糊。后续视角覆盖核验见
[数据专项报告](../reviews/2026-11-13-vln-data-baselines.md) 的实测更新；它不是无人机完整闭环飞行成绩。
补充偏航±90°共48帧：东向4/24低质（16.67%，原10%门槛FAIL），西向2/24（8.33%，PASS）。
因此“前视PASS”不能推广到转身搜索；低纹理也会降低锐度，需要与原始照片/采集朝向共同判断是否缺数据。

## 5. 未验证事项

- 没有新grounder、条件学生训练、中文导航或真机任务完成证据。
- 未重跑全部公开基线，历史分支结果按原任务/口径保留，不能混进当前VLN主表。
- 未改3DGS光栅化、教师/执行器参数；渲染覆盖不足与控制惯性仍需按计划实验。
- 全量入口的CUDA CPU占用性能测试仍失败；不能宣称所有测试通过。
- GUI交互、Gazebo实时闭环以及约120s长航程的真实机体稳定性没有在本轮验收。
- 全量入口在CTest失败后未执行其后面的Python产物→C++ UAVR读回/软标签身份桥接检查；本次未改这些格式，实现卡E时必须补跑。

## 6. 2026-11-14 恢复会话追加：卡 A 与回归

本节基于审查分支 `review/uav-vln-20261113` 的新变更；测试数只覆盖相应命令，不能与上节 11-13 的指标混为一轮训练结果。

- `collect_3dgs.run_episode` 新增独立 `stop_requested`。显式 STOP 将动作改为一个 `SAFE_NEUTRAL` 窗口，碰撞/异常优先；目标内 SAFE 不自动成功，目标外 STOP 计失败。A* 教师只在目标半径内发 STOP。`run_vln` 默认 `3dgs.vln.v3`，旧三连 SAFE 显式选择 `--stop-mode legacy-safe` 并记为 v2。
- `StudentAdapter`/DAgger 校验条件类型。文本网络缺指令或全 PAD、纯视觉网络收到 token、heading-code 权重被当作 prev-action 输入时拒绝；未知训练来源只警告并记录。此项是输入安全合同，不是把 SAGE 文本接入模型。
- `max_scale_m=0` 与 `None` 统一表示关闭过滤；非法值拒绝。`nav_to_world` 统一校验有限的 4×4 刚体变换。cf_sitl 的步长必须是至少一个 1 ms tick 的整数倍；旧视频的碰撞失败优先于 goal 标记。
- 集成复核补出最后预算步碰撞被 `rows_ok == steps` 覆盖的问题；现已让碰撞、截断和动力学异常优先，并新增末步回归。

| 检查 | 结果 |
| --- | --- |
| 改动相关 `test_collect_eval`、`test_vln`、`test_video`、`test_dynamics_sitl`、`test_student_inputs` | 86 tests，52.920 s，全部通过 |
| 真正 GPU renderer + collision 回归（agent报告，`render-tests.log`） | 43/43，无跳过；RTX 2080 Ti / sm75 |
| 本地场景刚体/视图检查（agent报告） | 13/13 场景通过 |
| `tools/check.sh --python-only` | 退出 0，有 3 项 GUI SKIP；末步 completed 回归补入前运行，相关 86 项当前代码已另行通过 |
| cf_sitl CTest / selftest | 9/9；自检 33 通过、0失败、0跳过 |
| SAGE 特权教师显卡冒烟，`sage3d_841150` 两集 | 2/2，SR=1、OSR=1、碰撞率 0；这是 waypoint 已知的 A* 教师管线验证，不是语言策略效果 |
| 完整 `tools/check.sh` | C++/CUDA 构建通过；CTest 43/44。唯一失败 `test_cuda_blocking_sync`：CPU核占用 `off=0.987944`，`cli=0.397291`、`env1=0.382025`，未达到现有 `<0.35×off` 阈值。没有放宽断言；脚本在 CTest 处停止 |

记录目录：`/root/rivermind-data/uav_rgb/tmp/vln-review-20261113/`；完整构建与SITL日志另在 `.../tmp/vln-review-20261113-session/`。这些临时证据未纳入Git，仓库只保留摘要和可重跑命令。

本轮没有训练语言 grounder/学生、扩数据、重跑 249 集、验证真机抖动或改 PID。后续按计划卡 B/C/D 先完成视角覆盖、控制分因和语言反事实闭环；现存可视化呈现指令与终点只表明任务/结果被记录，不表明模型使用了语言。
