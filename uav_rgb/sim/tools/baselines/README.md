# 外部基线包装器（sim/tools/baselines/）

把公开的小型避障网络（PULP-Dronet v3、Tiny-PULP-Dronet v3、DroNet）放进我们的闭环评测，评测代码一行不改。
包装器 `baseline_infer.py` 说与 `uavrgb_infer` 同一套行协议，`StudentPolicy` 把它当成一个学生推理程序。

**范围**：这里只放适配代码。第三方源码、权重、下载的控制代码都在数据盘 `/root/rivermind-data/uav_rgb/baselines/`（环境变量 `BASELINE_ROOT` 可改），不进仓库。
**零样本臂只量域差异**（公开权重在我们的画面上能走多远），不能用来比较网络结构的优劣；结构对比要用「同结构在我们数据上重训」的臂（`BASELINE_WEIGHTS` 接重训权重，见下文）。

## 用法

```bash
# 1. 占位 checkpoint：评测只检查「文件存在」并记 sha256，内容不读。仓库的 .gitignore 挡了 *.ckpt，所以不放二进制，现场生成：
printf 'stub ckpt (占位文件，外部基线只检查它存在；不要当权重)\n' > $OUT/stub.ckpt
# 2. 照常评测，把 --infer-bin 指到启动器：
python sim/tools/eval_closed_loop.py --policy student --ckpt $OUT/stub.ckpt \
    --infer-bin sim/tools/baselines/launch_pulp_v3.sh  --task base --split test ... --out $OUT/pulp_v3
```

`--infer-bin` 只认一个可执行文件、不能带环境变量，所以每个基线一个启动器，环境变量写死在启动器里（`exec` 教师 venv 的 python 跑 `baseline_infer.py`）。

| 启动器 | 基线 | 备注 |
| --- | --- | --- |
| `launch_pulp_v3.sh` | PULP-Dronet v3，ResBlock、宽度 1.0、带旁路 | 320,928 参数，41.1 M MAC（`count_params.py` 实测，见报告） |
| `launch_pulp_tiny.sh` | Tiny-PULP-Dronet v3，深度可分离、宽度 0.125 | 2,784 参数，1.13 M MAC |
| `launch_dronet.sh` | DroNet（ResNet-8，Loquercio 2018） | 321,634 个张量元素（含 BN 滑动统计）；Keras 权重转 npz，用 torch 算子重写前向 |
| `launch_dronet_recal.sh` | DroNet 重标定（次要臂） | 碰撞阈值取离线校准帧中位数、转向尺度取 90 °/s，两处偏离原文 |
| `launch_forward.sh` | 盲前进 | 协议自检：应与 `--policy forward` 逐步一致 |

包装器的行为全部由环境变量决定，默认值在 `baseline_infer.py` 文件顶部的配置区（每个参数一行，注释写出处）。`BASELINE` 没有默认值：漏设直接报错退出 2，不会静默跑成别的基线。

## 协议

- 启动 `<bin> --ckpt <file> [--text id,...]`；先在 stdout 写一行 hello JSON `{in_h,in_w,in_c,num_classes:4,text_dim,prev_action_dim,aux_classes}`。
- hello 声明 240×320×3 原生分辨率，所以 `StudentPolicy` 的缩放是同尺寸直通；各基线在 `step()` 里自己做灰度、裁剪、缩放。
- 每步 stdin 读一帧 HWC uint8 RGB，stdout 回一行 `{"action","probs","infer_ms","raw"}`；`raw` 带网络原始输出（steer、p_coll），`StudentPolicy` 忽略多余字段。
- stdin EOF 退出码 0；帧被截断退出码 1；构造失败退出码 1，原因写 stderr。stdout 只许协议行：第三方库的 `print` 被重定向到 stderr。
- 动作：0 SAFE_NEUTRAL，1 FORWARD_SLOW（约 0.25 m），2 YAW_LEFT（偏航角为正，约 25°），3 YAW_RIGHT；每个动作保持 1.0 s。

## 输出到 4 个动作的映射

最近邻量化：把一个决策周期（1.0 s）内网络想做的转角与位移，量化到最近的原语。

```
yaw_cmd = STEER_SIGN * steer * S * HOLD_S          # 本步想转的角度（度）
|yaw_cmd| >= 12.5 度  → 转向优先：yaw_cmd > 0 → YAW_LEFT，否则 YAW_RIGHT
否则 p_coll <= P_CRIT → FORWARD_SLOW，否则 SAFE_NEUTRAL
steer 或 p_coll 不是有限数 → SAFE_NEUTRAL
```

转向优先的原因：离散动作不能边转边进，朝向错了前进只会离障碍更近。没有按原文做时间低通：原控制律里有低通滤波（PULP `config_dronet.h` L49–50，DroNet `deep_navigation.cpp` L114–115），我们 1 Hz 决策且动作保持 1 s，滤波常数针对 10–30 Hz 的原系统，直接搬会让输出迟钝，没做。

| 参数 | 取值 | 出处 |
| --- | --- | --- |
| 转向门槛 | 12.5°（半个 YAW 步长） | 「转 0 步还是 1 步」的最近邻分界；YAW 步长 25° 取公共任务书 §1 的 23–27° 中值 |
| S，PULP 系 | 90 °/s | `pulp-dronet` 的 `config_dronet.h` L45 `YAW_SCALING 90.0f`（同文件另两组配置为 120，L67、L89；取了 Crazyflie 应用默认那一组） |
| S，DroNet | 20 °/s | `rpg_public_dronet` 的 `outdoor.yaml` L31 `SpeedSettingsMaxRotationSpeedCurrent: 20.0`（Bebop 室外配置） |
| P_CRIT | 0.7 | PULP `config_dronet.h` L55、L77、L99 `CRITICAL_PROB_COLL`；DroNet `deep_navigation.cpp` L117 `critical_prob 0.7`，刹停条件见 L58 |
| FORWARD 位移 | 0.25 m | 公共任务书 §1；只有连续速度映射 `map_velocity` 用 |
| HOLD_S | 1.0 s | 公共任务书 §1 |

控制代码从两个仓库的 master 取原文，保存在数据盘（下载日期 2026-11-10，commit 号当时没记，未核实）：
`baselines/pulp-dronet/src_ctrl/{config_dronet.h,app_dronet.c}` 来自 `pulp-platform/pulp-dronet` 的 `tiny-pulp-dronet-v3/drone-applications/crazyflie-dronet-app/{inc,src}/`；
`baselines/dronet/src_ctrl/{outdoor.yaml,deep_navigation.cpp}` 来自 `uzh-rpg/rpg_public_dronet` 的 `drone_control/dronet/{configs,dronet_control/src}/`。

`map_velocity(vx, yaw_rate)` 是给输出连续速度的基线（DiffPhysDrone、NavRL 一类）预留的：`|yaw_rate*hold| >= 12.5°` → YAW，否则 `vx*hold >= 0.125 m` → FORWARD。目前没有基线调用它，只有测试覆盖，接入时需要再核对符号约定。

## 灰度输入的取法（`BASELINE_PREP`）

三个网络的输入都是 200×200 灰度。
`native`（默认，各自原文的裁法）：QVGA 320×240 中心裁 200×200，DroNet 取底部居中；
`resize_crop`：高缩到 200 再中心裁宽，保留完整竖直视场；
`squash`：整幅压成方形，保留完整水平视场，宽高比失真。
我们的相机水平视场 60°，`native` 只保留中间 ±19.8°（fx=277.13 px），这是零样本的固有偏差之一。

## 符号（STEER_SIGN）

约定：`steer > 0` 表示左转（PULP README：YawRate > 0 为逆时针即左），对应我们的 YAW_LEFT（偏航角为正）。
公开权重在我们的合成画面上不一定遵守这个约定，所以各基线的符号用 train 分区的教师帧单独测，写进各自启动器。
测法与全部数值见 `experiments/2026-11-10-baselines-zeroshot/offline_check.py` 与报告 `reports/baselines-run.md`：

分组：画面内最近的柱（3 m 内）落在左半或右半（左 1,101 帧 / 右 1,122 帧）；「裁剪视场内」只取柱落在 native 裁剪视场（水平 ±19°）里的帧（左 297 / 右 352），那才是网络真正看到的。
指标：右组与左组 steer 均值之差，95% 区间按「集」自助法（2,000 次，seed 0）；差为正说明柱在右时网络给出更大的 steer，即朝左避开，与约定一致。

| 基线 | 裁剪视场内：右 − 左（95% 区间） | 全 60° 画面内：右 − 左（95% 区间） | 镜像检验 corr(steer, 翻转输入的 steer) | STEER_SIGN |
| --- | --- | --- | --- | --- |
| pulp_v3 | +0.162（0.129, 0.194） | +0.070（0.052, 0.087） | +0.093 | +1，实测 |
| pulp_tiny | +0.193（0.176, 0.211） | +0.083（0.074, 0.092） | −0.637 | +1，实测 |
| dronet | −0.0002（−0.064, 0.073） | +0.055（0.027, 0.087） | +0.379 | +1，**沿用约定，判不出** |

DroNet 在裁剪视场内两组均值相同，翻转输入后 steer 基本不变号（只有 12% 的帧变号，偶部均值 0.307 占主导），说明它在我们的合成画面上没有可用的转向信号，符号取 +1 只是约定。
数据：`/root/rivermind-data/uav_rgb/e1-pipeline-20261107-0417/collection/train` 的 255 集、10,200 帧（全取，不抽样），原始数值在 `/root/rivermind-data/uav_rgb/tmp/phase12-20261110/baselines-run/offline/offline_check.json`。

## 怎么加一个新基线

1. 在 `baseline_infer.py` 里加一个 `Adapter` 子类：`__init__` 里加载权重（第三方代码用 `importlib` 按路径加载，不装包、不改源码），`step(frames, prev_action)` 返回 `(动作, raw 字典)`；输出是 steer + 碰撞概率就直接调 `map_steer_coll`，输出连续速度就调 `map_velocity`。
2. 把它登记进文件底部的 `REGISTRY`。
3. 配置区里给它的新常数加一行（带出处注释）；新增一个 `launch_<name>.sh`（复制现有启动器，改 `BASELINE=`，写死该基线的符号与阈值）。
4. 在 `sim/tools/test_baseline_infer.py` 里加测试：映射纯函数直接测；真权重的测试在文件缺失时 SKIP 并打印原因。
5. 先跑 `--out` 指向非空目录的 `eval_closed_loop.py`（不起 Gazebo，只走参数解析与 StudentPolicy 探针），再离线对帧核对符号与碰撞概率分布，最后才上闭环。

## 权重与第三方代码（下载命令）

下载走镜像 `https://ghfast.top/<原 URL>`（次选 `https://gh-proxy.com/<原 URL>`），直连 GitHub 大文件不稳定。只取需要的文件，不整仓克隆。

```bash
B=/root/rivermind-data/uav_rgb/baselines
M=$B/pulp-dronet/src/tiny-pulp-dronet-v3/model && mkdir -p $M
RAW=https://ghfast.top/https://raw.githubusercontent.com/pulp-platform/pulp-dronet/master/tiny-pulp-dronet-v3/model
for f in pulp-dronet-v3-resblock-1.0.pth tiny-pulp-dronet-v3-dw-pw-0.125.pth dronet_v3.py; do curl -fL -m 120 -o $M/$f $RAW/$f; done
# DroNet：UZH 原站，约 14 kB/s，要断点续传
mkdir -p $B/dronet && curl -fL -C - -o $B/dronet/dronet_model.zip http://rpg.ifi.uzh.ch/data/dronet_model.zip
cd $B/dronet && unzip -o dronet_model.zip -d model     # 得到 model/dronet/best_weights.h5 与 model_struct.json
```

DroNet 的 Keras 权重要转成 npz（包装器读 `dronet_best_weights.npz`；键 = h5 数据集路径把 `/` 换成 `|`，例如 `batch_normalization_1|batch_normalization_1|beta:0`）。需要 h5py，教师 venv 里没有，用 `$B/venv-h5`（下面的片段实测得到与现有 npz 键集合与数值逐元素相同的文件）：

```bash
$B/venv-h5/bin/python - <<'EOF'
import h5py, numpy as np
B = "/root/rivermind-data/uav_rgb/baselines/dronet"
out = {}
with h5py.File(f"{B}/model/dronet/best_weights.h5", "r") as f:
    f.visititems(lambda n, o: out.__setitem__(n.replace("/", "|"), o[()]) if isinstance(o, h5py.Dataset) else None)
np.savez(f"{B}/dronet_best_weights.npz", **out)
EOF
```

转换后用 `experiments/2026-11-10-baselines-zeroshot/check_h5_vs_npz.py` 对拍 h5 与 npz 逐张量一致。
PULP 的模型代码顶部 `import nemo`（量化库）：包装器用桩模块替代 `PACT_IntegerAdd`（浮点前向里等价于 `a + b`），不改第三方源码也不装 NEMO。

## 同结构重训权重（`BASELINE_WEIGHTS`）

`BASELINE_WEIGHTS=<.pth>` 让 `pulp_v3` / `pulp_tiny` 加载我们自己训练的同结构权重（`state_dict` 或带 `state_dict` 键的字典均可），其余映射不变；空 = 官方公开权重。重训任务卡见报告。

## 测试

`sim/tools/test_baseline_infer.py`（`tools/check.sh --python-only` 会收集）：
盲前进经真实 `StudentPolicy` 走完整协议；映射纯函数（12.5° 转向门槛、碰撞阈值、连续速度、符号翻转、非有限输入）；探针适配器核对 `StudentPolicy` 送进来的帧逐字节正确；
启动器静态检查（环境变量、可执行位、预注册默认值）。教师 venv 或权重文件缺失时，真权重的端到端测试 SKIP 并打印原因，不算失败。
