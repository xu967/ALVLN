# UAVR 二进制数据集格式（v1，冻结）

`tools/preprocess.py` 产出 `<out>/dataset.uavr`。文件为小端、无填充、可直接 `mmap`
或按偏移 `read()`；H/W/C 一律以文件头为准，**读端不得假定 128×128**。

## 1. 文件布局

```
+---------------- 48 B 文件头 ----------------+
| N × C × H × W uint8 图像（NCHW，连续）      |
| N × 1  uint8 标签（0..3）                    |
| meta JSON（UTF-8，长度 = meta_len）          |
+----------------------------------------------+
```

### 1.1 文件头（48 字节，little-endian）

| 偏移 | 字节 | 字段 | 含义 |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `"UAVR"` |
| 4 | 2 | version | u16，当前为 `1` |
| 6 | 2 | header_size | u16，固定 `48` |
| 8 | 4 | n_samples | u32，样本数 N |
| 12 | 2 | height | u16，图像高 H |
| 14 | 2 | width | u16，图像宽 W |
| 16 | 1 | channels | u8，固定 `3`（RGB） |
| 17 | 1 | image_dtype | u8，`1` = uint8 |
| 18 | 1 | label_dtype | u8，`1` = uint8 |
| 19 | 1 | reserved | u8，`0` |
| 20 | 8 | images_offset | u64，固定 `48` |
| 28 | 8 | labels_offset | u64 = `48 + N*C*H*W` |
| 36 | 8 | meta_offset | u64 = `labels_offset + N` |
| 44 | 4 | meta_len | u32，meta JSON 字节数 |

文件总长必须等于 `meta_offset + meta_len`，后面不得再有数据。

### 1.2 图像区

- `N × C × H × W` 个 uint8，**NCHW**，通道顺序 R,G,B，行优先，无 padding。
- 样本 `i` 的起始偏移：`images_offset + i * C*H*W`。
- 像素 `(n, ch, y, x)` 的偏移：`images_offset + ((n*C + ch)*H + y)*W + x`。
- 注意：`src/include/uavrgb/observation.hpp` 的 `SimObservation` 是 **HWC**
  且尺寸 **160×120**。本数据集默认 **160×120 NCHW**，
  两者的布局转换由训练器或部署适配层负责；`--size 128x128` 仅用于
  分辨率对照实验，会压扁画面（见 §4），也需要同步修改在线输入适配。
- 预处理缩放：Pillow `BILINEAR`，从 `episode_manifest.json` 的
  `camera.width/height`（collect.py 为 320×240）缩到 `--size` 指定 W×H。

### 1.3 标签区

- N 个 uint8，动作映射与 `src/include/uavrgb/action.hpp` 一致：
  `0=SAFE_NEUTRAL, 1=FORWARD_SLOW, 2=YAW_LEFT, 3=YAW_RIGHT`。
- 标签来源：`actions.jsonl` 的 `action_expert`（教师标签），按 `step` 排序，
  与 `rgb_%05d.png`（`%05d` = step）一一对应；缺帧/重复 step 会使预处理失败。

### 1.4 meta 区（UTF-8 JSON，紧凑分隔符）

关键字段（只允许追加，不改名/改单位）：

```json
{
  "schema": "uavrgb.dataset.v1",
  "created_utc": "...Z",
  "images": {"layout":"NCHW","dtype":"uint8","height":120,"width":160,"channels":3},
  "labels": {"dtype":"uint8","num_classes":4,
             "names":["SAFE_NEUTRAL","FORWARD_SLOW","YAW_LEFT","YAW_RIGHT"],
             "source":"actions.jsonl:action_expert"},
  "resize": {"method":"bilinear","from":[320,240],"to":[160,120],"aspect":"preserve"},
  "split_policy": "scene_id",
  "collision_flags": "000000000000000",
  "collision_dropped": 0,
  "splits": {"train":10,"val":5,"test":0},
  "num_samples": 15,
  "scenes": [{"scene_id":42,"split":"train"},{"scene_id":43,"split":"val"}],
  "episodes": [
    {"episode_id":1,"scene_id":42,"seed":7,"version":"synthetic","split":"train",
     "geometry_gid":"e8c96d776cd0ba5c",
     "num_steps":6,"sample_start":0,"sample_count":6,"source_dir":"/abs/path/ep1"}
  ]
}
```

- `resize.aspect`：`preserve` = 目标纵横比与原图相同（如 160×120、224×168）；
  `squash` = 强行拉伸到目标方形/异形（如 128×128）。
- `collision_flags`：N 个 `'0'/'1'` 字符，与**全局样本顺序**对齐，`'1'` = 该步
  `actions.jsonl` 的 `collision=true`。`LoadOptions.drop_collision=true` 时训练器
  按它剔除；缺失该字段会直接报错，不会静默当成"无碰撞"。`collision_dropped`
  是打包时用 `--drop-collision` 已经丢掉的步数（此时 flags 里不会再出现 `'1'`）。
  为什么重要：专家在撞墙那一步的动作本身就是错的，不剔除等于把错标签喂给学生。
- `split_policy` 恒为 `scene_id`：预处理强制**同一 scene_id 只能属于一个 split**，
  冲突直接报错退出。`geometry_gid` 是 `sim/tools/split_manifest.py` 算出的几何组 ID；
  有该字段时预处理还强制**同一 gid 只能属于一个 split**（同几何不同 scene_id/seed
  也会被拦）。旧数据/合成夹具没有该字段，仍按 scene_id 检查。
  训练器按 `episodes[].sample_start/sample_count/split`
  切出 train/val/test，并独立校验 scene_id 不跨 split、区间不重叠且覆盖全部样本。
  scene_id 遵守 `EpisodeMeta` 的 int64 整数合同，不要按帧随机拆分。
- `source_dir` 仅用于追溯，可能含机器相关绝对路径。

## 2. 读取顺序（C++）

1. 读 48 字节头，校验 `magic=="UAVR"`、`version==1`、`header_size==48`、
   `channels==3`、dtype 均为 1、三个 offset 与文件长度自洽。
2. 用头里的 `H/W/C` 计算样本字节数 `C*H*W`，**不要写死**。
3. 标签区读取后校验 `0..3`。
4. meta 区按 `meta_len` 读 JSON；trainer 可用任意 JSON 库解析，或只取
   `episodes[].sample_start/sample_count/split` 做 split 过滤。

参考实现：`tools/read_dataset.cpp`（独立编译，见文件头注释）。

## 3. 预处理命令

```bash
.venv-tools/bin/python tools/preprocess.py --out /tmp/ds_160 /tmp/ep0 /tmp/ep1
python3 tools/preprocess.py --size 128x128 --out /tmp/ds_128 /tmp/ep0     # 压扁对照
python3 tools/preprocess.py --size 224x168 --out /tmp/ds_224 /tmp/ep0
```

`--size` 默认 **160×120**（320×240 的精确一半，保持 4:3）。同一批输入的所有
episode 相机尺寸必须一致。

## 4. 分辨率 / 纵横比 / 角分辨率

假设：水平 FOV = 60°，源图 320×240（4:3）。由此垂直 FOV =
`2*atan(tan(30°)*240/320) ≈ 46.83°`。水平角分辨率 = `60° / W`；
垂直角分辨率 = `46.83° / H`。目标“10 m 外 0.4 m 宽立柱”的角宽 =
`2*atan(0.2/10) ≈ 2.292°`，水平像素数 = `2.292 / (60/W)`。

| 输出尺寸 | 纵横比处理 | 水平角分辨率 | 垂直角分辨率 | 10 m 外 0.4 m 立柱水平宽度 | 10 m 处每像素水平尺寸 |
|---|---|---:|---:|---:|---:|
| 320×240（原始） | 无缩放 | 0.1875°/px | 0.1951°/px | ≈ 12.2 px | ≈ 3.3 cm |
| 224×168 | preserve（4:3） | 0.2679°/px | 0.2787°/px | ≈ 8.6 px | ≈ 4.7 cm |
| **160×120（默认）** | preserve（4:3） | 0.3750°/px | 0.3902°/px | ≈ 6.1 px | ≈ 6.5 cm |
| 128×128 | **squash（4:3→1:1）** | 0.4688°/px | 0.3658°/px | ≈ 4.9 px | ≈ 8.2 cm |

结论：

- 160×120 与 224×168 保持 4:3，不改变物体形状；像素数 19200 / 37632，
  与 128×128（16384）同量级。
- 128×128 把 320×240 同时映射到 128×128：水平压缩 2.5×、垂直压缩 1.875×，
  物体在画面上被**纵向拉长**（水平/垂直角分辨率之比约 1.28），且 10 m 外
  0.4 m 立柱只剩约 4.9 px 宽（160×120 为 6.1 px，少约 20%）。
  它只适合作为“压扁对照”实验，不应作为主配置。
- 以上按 hfov=60° 的假设计算；真实相机标定（fx/fy/cx/cy）仍为 null 时，
  这些数字是几何近似，采集器 manifest 的 `camera` 字段需在标定后更新。

## 5. 版本策略

- `version` 变更仅在布局/字段语义不兼容时发生；新增 meta 字段不升版本。
- 读端遇到未知 `version` 必须拒绝，不得猜测。
- 单文件不存 per-sample split 字节；split 只能从 meta 的 episode 区间推导。
