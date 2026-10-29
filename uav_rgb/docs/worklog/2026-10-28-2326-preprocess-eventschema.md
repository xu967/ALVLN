# 2026-10-28 23:26 数据预处理流水线 + EventSchema 冻结

## 做了什么

### 1. `tools/preprocess.py`（新增）

- 输入：一个或多个 `sim/tools/collect.py` 产出的 episode 目录
  （`episode_manifest.json` + `actions.jsonl` + `rgb_%05d.png`）。
- 对每帧 `rgb_XXXXX.png`：Pillow `BILINEAR` 缩放到 `--size` 指定 W×H，
  RGB uint8，HWC → **CHW（NCHW 数据集布局）**。
- 标签取 `actions.jsonl` 的 `action_expert`（0..3），按 `step` 与
  `rgb_%05d.png` 一一匹配；缺帧、重复 step、标签越界、相机尺寸与 manifest
  不符都会直接失败。
- 输出 `<out>/dataset.uavr`：48 字节小端文件头 + N×C×H×W uint8 图像 +
  N 个 uint8 标签 + 尾部紧凑 JSON meta。临时文件写完 `os.replace` 原子落盘。
- split 来自每个 episode manifest，并按 `scene_id` 强制隔离；同一 scene_id
  出现在不同 split 时退出码 1，不产出文件。
- `--size WxH` 可配置，**默认 160x120**（320×240 的一半，保持 4:3）；
  `128x128` 会压扁画面（对照实验），`224x168` 等也可用。
  meta 的 `resize.aspect` 显式标 `preserve`/`squash`。
- meta 记录：schema、H/W/C、NCHW、标签名、resize 参数、split_policy、
  各 split 样本数、scene→split、episode 的 `sample_start/sample_count`、
  源目录。H/W 以实际参数为准，无 128 写死。

### 2. `tools/DATASET_FORMAT.md`（新增）

- 文件头逐字段偏移表、图像/标签/meta 区布局、读取顺序、split 推导方式。
- 分辨率/纵横比/角分辨率表（hfov=60° 假设）：
  - 128×128：0.4688°/px，压扁（水平/垂直角分辨率比约 1.28），
    10 m 外 0.4 m 立柱约 4.9 px；
  - 160×120（默认）：0.3750°/px，保持 4:3，约 6.1 px；
  - 224×168：0.2679°/px，保持 4:3，约 8.6 px；
  - 320×240 原始：0.1875°/px，约 12.2 px。
- 明确 `observation.hpp` 的 `SimObservation` 是 128×128 HWC 冻结合同，
  本数据集默认 160×120 NCHW，转换由训练器/适配层负责（未改该头文件）。

### 3. `tools/make_synthetic_episode.py`（新增）

- 生成 3 条合成 episode（非真实采集）：scene 42/train 6 步、
  scene 42/train 4 步、scene 43/val 5 步，共 15 样本；标签 = 步内 `step%4`。
- 偶数全局样本为整帧常量色（双线性缩放后应逐像素精确不变）；奇数样本为
  R 随 x、G 随 y 的线性梯度，B 恒 77（校验 NCHW/通道/宽高方向）。

### 4. `tools/read_dataset.cpp`（新增，独立编译）

- 纯 stdlib C++17，从文件头读取 N/H/W/C，不含 128 假设。
- 校验 magic/version/dtype/offset/文件长度/标签范围/meta JSON 边界。
- `--selfcheck` 额外校验合成数据集的 15 个标签、8 个常量帧逐像素
  RGB 精确值、7 个梯度帧的布局/通道/方向、meta 关键字段。

### 5. `src/include/uavrgb/event.hpp`（新增，纯头文件）

- 冻结 `metrics.jsonl` 事件：`EventType`（train_step/validation/status/error）、
  `Split`、`Severity`，`MetricsEvent` 含 step/total_steps/loss/action_accuracy/
  lr/elapsed_sec/eta_sec/split/severity/message/code，附四类工厂函数。
- 冻结 `results.csv` 行 `ResultsRow`：epoch/train_ce/dev_ce/dev_accuracy，
  表头常量 `kResultsCsvHeader`。
- 字段单位/取值范围/有效性在注释中写明；未接入 `src/CMakeLists.txt`，
  未修改 `src/include/uavrgb/observation.hpp`。

### 6. `tools/test_event.cpp`（新增，独立编译）

- 校验枚举尺寸、结构体默认值、工厂函数赋值、to_string 映射、results.csv 表头。

## 验证（真实命令 + 输出）

```bash
$ python3 -m py_compile tools/preprocess.py tools/make_synthetic_episode.py
py_compile OK

$ python3 tools/preprocess.py --help
usage: preprocess.py [-h] --out OUT [--size WxH] episodes [episodes ...]
...
  --size WxH  输出分辨率；160x120 保持 4:3，128x128 会压扁画面（对照实验）
              (default: (160, 120))

$ python3 tools/make_synthetic_episode.py --out /tmp/uavrgb_selfcheck/synth
synthetic episodes: 3  samples: 15  -> /tmp/uavrgb_selfcheck/synth

$ python3 tools/preprocess.py --out /tmp/uavrgb_selfcheck/ds160 \
    /tmp/uavrgb_selfcheck/synth/ep1 /tmp/uavrgb_selfcheck/synth/ep2 \
    /tmp/uavrgb_selfcheck/synth/ep3
episodes: 3  samples: 15  scenes: 2
splits: train=10 val=5 test=0
resize: bilinear 320x240 RGB -> 160x120 NCHW uint8 (aspect=preserve)
wrote /tmp/uavrgb_selfcheck/ds160/dataset.uavr (865125 bytes, meta 1062 bytes)

$ g++ -std=c++17 -O2 -Wall -Wextra tools/read_dataset.cpp -o /tmp/read_dataset
$ /tmp/read_dataset /tmp/uavrgb_selfcheck/ds160/dataset.uavr --selfcheck
header: magic=UAVR version=1 header_size=48 n=15 h=120 w=160 c=3 images=uint8 labels=uint8
labels: 0=5 1=4 2=3 3=3
OK: header/labels/meta structure valid
selfcheck: labels 15/15 OK
selfcheck: constant frames 8/8 OK (逐像素精确, RGB 顺序)
selfcheck: gradient frames 7/7 OK (R->x, G->y, B=77, NCHW)
selfcheck: meta 字段/文件长度 OK
RESULT: PASS (structure + synthetic pixel/label checks)
```

同样的 `--selfcheck` 在 `--size 128x128`（h=128 w=128，aspect=squash，
738403 bytes）和 `--size 224x168`（h=168 w=224，aspect=preserve，
1694565 bytes）也输出 `RESULT: PASS`，证明读端按文件头尺寸工作。

split 隔离负例：

```bash
$ python3 tools/preprocess.py --out /tmp/uavrgb_selfcheck/ds_conflict \
    .../ep1 .../ep3 .../ep3_bad   # ep3_bad 把 scene 43 从 val 改成 train
preprocess: scene_id=43 同时出现在 split=val 与 train（必须按场景隔离）
exit=1                     # 未产出输出目录内容
```

EventSchema：

```bash
$ g++ -std=c++17 -O2 -Wall -Wextra -I src/include tools/test_event.cpp -o /tmp/test_event
$ /tmp/test_event
sizeof: MetricsEvent=112 ResultsRow=32 EventType=1 Split=1 Severity=1
RESULT: PASS (all checks)

$ cmake --build src/build
ninja: no work to do.       # exit 0；event.hpp 未被现有目标包含，不影响构建
```

## 没做什么

- **没接真实采集数据**：所有验证只用 `tools/make_synthetic_episode.py`
  生成的合成 episode；未跑 `sim/tools/collect.py`，未接触 Gazebo。
- **没做数据增强**（翻转/裁剪/颜色抖动等）、**没做 shuffle**、没做类别均衡。
- 没做归一化（数据集保持原始 uint8，均值/方差交给训练器）。
- 只实现双线性缩放，没有 `--nearest` 选项（任务允许二选一）。
- 没有把 split 写成 per-sample 二进制字节；split 只在 meta 的 episode 区间里，
  C++ 训练器需要解析 JSON（或后续再冻结一个 split 索引格式）。
- 没做数据集压缩、CRC/SHA256、增量合并、多数据集拼接。
- 没改 `src/CMakeLists.txt`、`src/*.cpp`、`src/*.cu`、`src/tests/`、`sim/`、
  `ui/`、`experiments/`；`event.hpp` 未加入任何构建目标。
- 没改 `src/include/uavrgb/observation.hpp`（按协调者要求保持冻结合同）。
- 没做真实相机标定；角分辨率表基于 hfov=60°、320×240 的几何假设
  （manifest 里 fx/fy/cx/cy 仍为 null）。
- 没做大规模性能/内存基准（当前实现逐帧流式写，避免整库驻留内存，
  但未在 2000 episode 规模实测）。

## 风险 / 遗留

1. **默认 160×120 与冻结 `SimObservation` 的 128×128 不一致**：
   `observation.hpp` 是学生在线输入合同，本数据集默认 160×120 NCHW。
   训练器/部署适配层必须显式做尺寸与 HWC/CHW 转换；若要求数据集直接对齐
   合同，可改用 `--size 128x128`，但会压扁 4:3 画面（见 DATASET_FORMAT §4）。
2. meta 里的 `source_dir` 是绝对路径，换机器/移动数据后仅作追溯，不能当稳定 ID；
   稳定标识用 `episode_id/scene_id/seed`。
3. `--selfcheck` 绑定合成数据集的固定 15 样本协议；对真实数据集只能做结构校验。
   梯度帧只做单调性/方向校验，未逐像素比对 Pillow 的精确重采样值（常量帧才是
   逐像素精确校验）。
4. split 隔离只保证“同一 scene_id 不跨 split”；不保证 episode 级去重
   （同一 scene 的多条 episode 仍可能高度相似），也未实现 scene 分组抽样。
5. 并发风险：本次 `cmake --build src/build` 为 no-op，只能说明当时无待编译改动；
   `event.hpp` 尚未被任何源文件包含，真正接入训练器时需再做一次完整编译。
