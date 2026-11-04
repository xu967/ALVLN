# cf_blackbox 黑盒骨架 + cf_sil 退出默认矩阵（2026-11-05）

- 结论：新增纯 Python 黑盒状态模型和唯一参数文件；`cf_blackbox` 进入
  collect/eval 的 `--dynamics` choices；`cf_sil` 退出 `collect_parallel.py` 和
  `eval_closed_loop.py` 的默认训练/评测矩阵，只在 `collect.py` 保留手动入口。
- 本轮**未运行 Gazebo**，未做真机标定，未 commit/push。
- cf_blackbox 当前参数全部是占位值，`calibrated=false`；没有 ULog 时保持该状态。

## 1. 文件与接口

| 文件 | 改动 |
|---|---|
| `sim/tools/cf_blackbox.py`（新） | 纯 Python 黑盒模型、参数加载/校验、manifest 块 |
| `sim/dynamics/cf_blackbox_v1.json`（新） | 唯一参数文件（占位值） |
| `sim/tools/test_cf_blackbox.py`（新） | 54 项纯函数自检 |
| `sim/tools/collect.py` | 加 `cf_blackbox` choice/参数校验；manifest 加 `dynamics_experimental`；cf_sil 标实验 |
| `sim/tools/collect_parallel.py` | `DYNAMICS_CHOICES` 去掉 cf_sil、加 cf_blackbox；删 cf_sil 透传 |
| `sim/tools/eval_closed_loop.py` | 复用 `cp.DYNAMICS_CHOICES`；删 cf_sil 透传 |
| `sim/tools/test_collect.py`（追加） | 编排矩阵断言 |
| `sim/tools/test_eval_closed_loop_args.py`（追加） | eval 矩阵断言 |
| `docs/experiments/2026-11-05-cf-blackbox.md`（新） | 本文件 |

`CfBlackbox` 接口（时间单位秒）：

```python
m = cf_blackbox.CfBlackbox()                 # 默认从 cf_blackbox_v1.json 读参数
m.dispatch(YAW_LEFT, t=0.0)                  # 排到 t + latency_s 生效
s = m.step(FORWARD_SLOW, dt=0.1)             # step 的 action 只在变化时入队
# s: t/x/y/z/vx/vy/vz/yaw/yaw_deg/yaw_rate/active_action/path_m
#    + pose=(x,y,z) / velocity=(vx,vy,vz)
```

- z 固定为 `z_target_m`（默认 2.0 m），`vz` 恒 0；不做高度动力学。
- `SAFE_NEUTRAL` 不主动刹车，速度只按 `exp(-drag_k*dt)` 被动衰减。
- `FORWARD_SLOW` 在机体系 +X 给固定加速度，终端速度 = `forward_accel_mps2 / drag_k`。
- `YAW_LEFT/RIGHT` 目标角速率 `±yaw_rate_rad_s`；`yaw_tau_s=0` 为瞬时响应，
  否则按一阶滞后解析解积分。
- 延迟队列按生效时刻稳定排序，`step` 在段内按生效时刻切分积分。

`CfBlackboxParams`：`from_json` / `from_dict` / `profile()` / `terminal_speed_mps`
/ `manifest()`。`profile()` 形如 `cf_blackbox_v1:<sha8>`，参数一变档名就变。

## 2. 参数（sim/dynamics/cf_blackbox_v1.json）

| 字段 | 值 | 说明 |
|---|---|---|
| `forward_accel_mps2` | 1.3787 | 占位：对齐 3DGS pilot 的 8° 倾角 `g*tan8°` |
| `drag_k` | 1.0 /s | 占位：线性阻力 |
| `latency_s` | 0.15 | 占位：派发到生效 |
| `yaw_rate_rad_s` | 0.785398 | 占位：45°/s |
| `yaw_tau_s` | 0.15 | 占位：偏航一阶滞后 |
| `z_target_m` | 2.0 | 训练固定高度 |
| `max_tilt_deg` | 30.0 | 校准上限：`forward_accel <= g*tan(max_tilt)` |
| `calibrated` | false | 无 ULog/实测数据，保持占位 |
| `source` | 占位说明 | 必填非空，缺失直接拒绝 |

对应终端速度 = `1.3787 / 1.0 ≈ 1.38 m/s`；这只是占位，不是真机实测值。

参数校验：缺失/非有限/非正、`profile_version` 不以 `cf_blackbox` 开头、
`max_tilt_deg` 不在 (0,90)、加速度超过倾角上限、非法动作、`dt<0`/NaN，
全部抛 `ValueError`，不静默用默认值。

## 3. CLI 变化

- `collect_parallel.py` 和 `eval_closed_loop.py` 的 `--dynamics` choices：
  `x3_velocity` / `cf_stab` / `cf_blackbox`。默认仍是 `x3_velocity`。
- `cf_sil` 从上述两个编排器移除，`--cf-sil-params` 和
  `--no-velocity-control` 透传一并删除。`cf_sil.py`、`test_cf_sil.py` 未删。
- `cf_blackbox` 是纯 Python 模型，**Gazebo collect/eval 不驱动它**：
  选中后在参数校验阶段明确报错，避免把 x3 物理写成 cf_blackbox 数据。
  3DGS collector 应直接 `import cf_blackbox`，用 `CfBlackboxParams.profile()` /
  `manifest()` 写自己的 `dynamics_profile` / `dynamics_params`。
  注意：当前 `sim/3dgs/dynamics.py` 仍带一份独立实现和
  `sim/dynamics/cf_blackbox_3dgs.json`，需要那边 owner 迁移；本文件数值已对齐它。
- `collect.py` 可手动跑 cf_sil 作参考：

```bash
python3 sim/tools/collect.py --dynamics cf_sil \
  --cf-sil-params sim/dynamics/cf_sil_v1.json \
  --policy forward --steps 5 --hold 0.2 --out /tmp/cf_sil_manual
```

  该档的 manifest 会带 `dynamics_experimental: true`。

## 4. manifest / profile

- cf_blackbox 的标准 manifest 块由 `CfBlackboxParams.manifest()` 提供：
  `profile`、`sha256`、`params_file`、`calibrated`、`source`、
  动力学参数、`terminal_speed_mps`、`altitude_assumption="fixed_z"`。
- 3DGS collector 应写：
  `dynamics_profile = params.profile()`；
  `dynamics_params = params.manifest()`。
  与 `tools/preprocess.py` 已有的 `dynamics_profile` 混包门禁直接兼容。
- Gazebo collect 的 manifest 新增 `dynamics_experimental`：cf_sil 为 true，
  x3/cf_stab 为 false；其余字段不变。

## 5. 参数单源现状

- cf_blackbox 本身已单源：代码不内置默认值，`CfBlackbox()` 从 JSON 读。
- 仍未收敛的重复实现/参数（**不在本次文件所有权内，未硬改**）：
  - `sim/3dgs/dynamics.py:18-20` + `sim/dynamics/cf_blackbox_3dgs.json`：
    另一份 `BlackboxDynamics` 和参数文件（`forward_tilt_deg=8`、
    `yaw_rate_deg_s=45`、`z_m=2.0`），目前不 import 本模块。本文件的数值已对齐它，
    但 3DGS collector 仍应改为 `import cf_blackbox` 并只读
    `cf_blackbox_v1.json`，否则仍是两份参数。
  - `rl_teacher/core.py:107-108` `advance()`：`latency=0.15, tilt_deg=2.0,
    drag_k=1.0, yaw_rate=.6`；
  - `rl_teacher/core.py:272`：`physics_profile='cf_stab_approx_v1_dt0.04_latency0.15_k1_tilt2'`；
  - `rl_teacher/train_ppo.py:55`：同一 profile 字符串；
  - `sim/tools/teacher_predictive.py:18` `max_speed=.65`；
  - `sim/tools/teacher_predictive.py:24` `latency=.15`。
- 下一步应由这些文件的 owner 改成从 `cf_blackbox_v1.json` 读取；在那之前，
  改 JSON 参数不会自动同步 3DGS / teacher_predictive / rl_teacher。

## 6. 测试（未运行 Gazebo）

```bash
python3 sim/tools/test_cf_blackbox.py
python3 sim/tools/test_collect.py
python3 sim/tools/test_eval_closed_loop_args.py
python3 sim/tools/test_cf_sil.py          # 保留档，仍应通过
```

原始结果（2026-11-05，系统 `/usr/bin/python3`）：

```
test_cf_blackbox: 54 项通过
test_collect: 58 项通过
test_eval_closed_loop_args: PASS
test_cf_sil: 49 项通过
```

相关回归（未改文件，确认没被影响）：

```
test_collect_parallel: Ran 2 tests ... OK
test_eval_metrics: PASS
test_eval_provenance: Ran 9 tests ... OK
test_eval_summary: PASS
test_audit_dataset: PASS
```

CLI 拒绝路径也已核对（rc=2，未起 Gazebo）：

```
collect_parallel.py: error: argument --dynamics: invalid choice: 'cf_sil'
  (choose from 'x3_velocity', 'cf_stab', 'cf_blackbox')
eval_closed_loop.py: error: argument --dynamics: invalid choice: 'cf_sil'
collect.py: error: cf_blackbox 参数 cf_blackbox_v1:<sha8> 校验通过，但该档是纯 Python
  模型，Gazebo collect 未接入；...
collect_parallel.py: error: cf_blackbox 是纯 Python 模型档，Gazebo 并行采集未接入；...
```

`test_cf_blackbox.py` 覆盖：默认 JSON/profile/sha8、悬停、前飞终端速度与方向、
SAFE 被动滑行（不刹车、继续前移）、左右偏航、偏航一阶滞后、latency、
乱序延迟队列、固定 z、重复同动作不重复叠加延迟、显式 dispatch 与 step(action) 混用、参数/动作/dt 校验。

## 7. 没做什么 / 风险

- 没运行 Gazebo；cf_blackbox 的 Gazebo 闭环没有接，选中会明确失败。
- 没做真机标定/系统辨识；没有 ULog 时 `calibrated=false` 是刻意保留的状态。
- 没改 `teacher_predictive.py` / `rl_teacher/` / `sim/3dgs/` / `ui/` /
  `teacher/train.py` / `src/`（所有权外）；重复默认和 3DGS 独立实现仍在。
- 没删 `cf_sil.py` / `test_cf_sil.py`；cf_sil 仍可用 `collect.py` 手动跑，
  但不再出现在并行采集/闭环评测的默认矩阵里。
- 风险：`collect_parallel.py` / `eval_closed_loop.py` 的 `cf_blackbox` choice
  目前只是占位；在 3DGS collector 接入前，用户选它会得到显式报错，不是静默降级。
- 风险：3DGS 现有 `sim/3dgs/dynamics.py` 的接口是 `reset/state/advance/step`，
  本模块是 `dispatch/step` 且 `step` 返回 dict；迁移时需要 3DGS 侧做薄适配
  （不在本次所有权内）。
- 风险：cf_blackbox 的 `z_target_m=2.0` 与 Gazebo collect 现有 `Z_TARGET=2.1`
  不同；两档数据不要按高度口径混报。
