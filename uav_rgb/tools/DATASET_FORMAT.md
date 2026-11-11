# UAVR 二进制数据集格式（v1，冻结；§6 为可选侧车 UAVRAUX1）

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
- `aux_sidecar`（新版 preprocess 追加）：`{"file":"dataset.aux","format":"UAVRAUX1",
  "depth":true|false,"depth_grid":[15,20]|null,"band_edges_m":[...],"envelope_radius_m":0.102}`，
  记录同目录侧车的存在与口径（见 §6）。旧数据集没有该字段也没有侧车，读端照常。
- `cond`（`preprocess.py --cond heading-code` 追加，W1-9）：`"heading_code.v1"`。只有这种包才有该键；没有 = 侧车 `prev_action`
  列是上一步动作（原语义）。有它时那一列是航向码（§6.2），评测端的条件一致性闸门（`sim/tools/nav_cond.py`）读它。

### 1.5 episode_manifest.json 的「有效」与「完整」是两个字段

打包只读 `validity`（三项都要 true），「跑没跑完」另看 `completed`，不要混用：

| 字段 | 含义 | Gazebo `sim/tools/collect.py` | 3DGS `sim/3dgs/collect_3dgs.py` |
| --- | --- | --- | --- |
| `validity.physics_verified` | 帧与位姿可信：有帧、无 `error`、无动力学异常 | 还要求跑满/goal-stop（历史口径，评测集 validity 全 false） | 只看帧/异常：碰撞截断（`truncated_on_collision`）、评测即停（`stopped_on_collision`）的集也是 true |
| `completed` | 跑满 `requested_steps` 或 goal-stop（`stopped_on_goal`） | 内部变量 `completed`，决定 validity | 顶层字段（2026-11-09 起）；碰撞截断 / 评测即停 / 异常 / error 都是 false |

- 训练打包（`preprocess.py`）看 `validity` + `stopped_on_collision`（闭环评测产物拒收），**不看 `completed`**：
  `--truncate-on-collision` 采的集 `completed=false` 照样可打包，撞的那一行由 `--drop-collision` 丢。
- 统计「到达 / 跑满」用 `completed` 与 `stopped_on_goal`，不要拿 `physics_verified` 代替。
- 旧 3DGS manifest 没有 `completed`，读端按 `num_steps == requested_steps or stopped_on_goal` 补算。

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
- 逐样本的新字段放**侧车**而不是改 `dataset.uavr`（§6）：`tools/join_soft_targets.py`、
  `tools/baseline_priors.py`、`tools/train_student_with_video.py` 都写死 `version==1`
  并按整文件 SHA256 绑定软标签；升版本会让它们全部失效。

## 6. UAVRAUX1 侧车 `dataset.aux`（STUDENT-P2：上一帧链 / prev-action / 深度分带目标）

与 `dataset.uavr` 同目录、同名换后缀（`<name>.aux`），新版 `tools/preprocess.py` 总是写。
`dataset.uavr` 的头、像素区、标签区与 v1 逐字节相同，旧读端照常读；但 meta JSON 多了
`aux_sidecar` 键，所以**同一批 episode 重打包后整文件 SHA256 与旧包不同**——按该 SHA 绑定的
P1 软标签（UAVSOFT1）要对新包重新 `join_soft_targets`，旧 `.bin` 会被训练器拒绝。
读端（`src/aux_sidecar.cpp`、`tools/read_dataset.cpp`）文件不存在 → 当旧数据集；存在但不自洽 →
直接报错，不半读。

### 6.1 头（128 字节，little-endian）

| 偏移 | 类型 | 字段 | 说明 |
|---:|---|---|---|
| 0 | `char[8]` | magic | `"UAVRAUX1"` |
| 8 | `u64` | N | 样本数，必须等于 `dataset.uavr` 的 N |
| 16 | `u32` | flags | bit0 = 有深度分带段；其余位必须为 0 |
| 20 | `u16` | depth_h | 深度目标网格高 = H/8（无深度时 0） |
| 22 | `u16` | depth_w | 深度目标网格宽 = W/8（无深度时 0） |
| 24 | `u8` | depth_classes | 分带数 K = 8 |
| 25 | `u8` | unknown_code | 255 |
| 26 | `u16` | reserved | 0 |
| 28 | `u8[32]` | uavr_sha256 | 同目录 `dataset.uavr` 整文件 SHA256（绑定，防「新数据集 + 旧侧车」） |
| 60 | `u8[32]` | payload_sha256 | 偏移 128 起全部 payload 的 SHA256 |
| 92 | `f32[7]` | band_edges_m | 分带上界 0.3/0.5/0.8/1.2/1.8/2.5/4.0 m；带 0 = <0.3 m，带 k = [edge(k−1), edge(k))，带 7 = ≥4 m（含无穷远/天空） |
| 120 | `f32` | envelope_radius_m | 打包时用的机体包络半径（E1：0.102 m） |
| 124 | `u32` | reserved | 0 |

### 6.2 payload（紧接文件头）

| 段 | 类型 | 说明 |
|---|---|---|
| prev_index | `u32[N]` | 全局样本下标：同 episode 内 step−1 的样本；首帧、或上一步被打包侧 `--drop-collision` 丢掉时 = 自身下标。读端由此派生 `first_frame[i] = (prev_index[i] == i)` 供工具/分析用；训练器目前直接按 prev_index 取上一帧，不读 first_frame |
| prev_action | `u8[N]` | 上一步**执行**的动作（`actions.jsonl` 的 `action_executed`，旧数据退回 `action_expert`）；首帧 = 0（SAFE_NEUTRAL）。注意：上一步被丢掉时它仍取该步真实执行的动作。`--cond heading-code` 的包（meta `cond="heading_code.v1"`）这一列改写航向码：按集从头累计**已执行**的 YAW_LEFT−YAW_RIGHT 得 n（含被丢掉的步），取该步执行前的 n，\|n\|≤1→0、n≥2→2、n≤−2→3，码 1 不用，首帧 0（`sim/tools/nav_cond.py`）；训练器同样用 `--prev-action` 读它 |
| depth（flags bit0） | `u8[N·depth_h·depth_w]` | 行优先 `[n][r][c]`，0..K−1 为分带（K−1 含「没打到东西」），255 = 未知（损失屏蔽） |

深度目标的生成（`preprocess.aux_payload`）：源分辨率深度图 → 每格（源图按 H/8×W/8 均分）
取最小已知深度 → 按机体包络膨胀（深度 z 的格把 z 扩散到像素半径 `f·r_env/z` 内的邻格，
f 取**该样本所属 episode** 的 manifest `camera.fx`，没有则按 60° HFOV 推算；多相机混包时逐 episode 分别膨胀）→ `searchsorted(band_edges)`。

**深度 0 的语义按来源区分**，判据是 manifest 的 `depth_format`（它同时决定解码方式；不按文件扩展名猜）：

| `depth_format` | 写端 | 0 的含义 | 目标 |
|---|---|---|---|
| `png16_mm` | `collect.py --save-depth`（uint16 毫米；写端把 `inf`——20 m 远裁剪内没打到——和负值编成 0，>65.535 m 截到 65535，见 `sim/tools/test_save_depth.py`） | 没打到东西 = 天空/超远 | **第 8 带（≥4 m）**，是有效监督 |
| `npy_float32_m` | `collect_3dgs.py --save-depth`（float32 米） | alpha 空洞（渲染没覆盖）；负值/NaN/inf 同样 | **255 未知**，损失屏蔽 |

格内聚合：未知像素不参与 min；「没东西」像素参与（整格只有它 → 第 8 带）；整格全未知 → 255。
包络膨胀只由近处有限深度的格向外扩散（未知/没东西的格半径为 0），但两者都会被邻近近处格覆盖。
一个包要么全部样本有深度，要么全无。

**深度来源 `preprocess.py --depth-source`**（W1-8）：缺省 `file` = 读 episode 的深度文件，即上面的全部流程，输出与加这个开关之前逐字节相同。
`analytic` = 不读深度文件，由几何真值解析算：`actions.jsonl` 的 `pose_before{x,y,z}` 与 `yaw_before`（动作前的机体位姿，度，逆时针为正）、
manifest 的 `vehicle`（只认 `e1_mini`：相机在机体中心前 0.045 m、上 0.015 m，朝向 = 机头，水平安装）、同目录 `scene.json` 的 `pillars`
（轴对齐方柱，或带 `radius` 的 forest 竖直圆柱）。每格取 3×3 条子射线（落在源图子块中心），与每根柱子和地面（z = 0）求交，
深度 = 沿光轴的深度（= 水平射线距离 · cosφ），命中点高度超过柱顶的射线不算打到（+inf，第 8 带），格内取最小；之后与文件路径同一套：
按该集 `camera.fx` 做包络膨胀 → `searchsorted(band_edges)`。解析深度没有「未知」，不产生 255。**忽略机体 roll/pitch**（慢速前进倾角 < 10°，
未量化）。waypoint 任务、3DGS episode、非 `e1_mini`、缺 `scene.json` / `pillars` / 位姿的集在写任何输出之前报错。
meta 的 `aux_sidecar.depth_source` 记 `"analytic"`；缺省 `file` 不加这个键，旧包的 meta 不变。侧车头、payload 布局与读端不变。

### 6.3 训练器约定

`--two-frame` / `--prev-action` 需要侧车存在，`--depth-aux` 需要 flags bit0；缺了直接报错，
不静默降级。旧数据集（无侧车）不带这些开关时训练行为与之前逐字节一致。

两处 `--drop-collision` 对上一帧链的作用不同，别混用：

| 丢碰撞步的位置 | 碰撞步在文件里 | 下一步的 prev_index | 下一步的 prev_action |
|---|---|---|---|
| 打包侧 `preprocess.py --drop-collision` | 不在（N 已减少） | 指向自身（当首帧） | 仍是碰撞步真实执行的动作 |
| 训练器侧 `--drop-collision`（`LoadOptions`） | 在，只是不进 train/val/test 下标 | **仍指向碰撞步**，双帧输入照样把它当上一帧读进来 | 同上 |

也就是说训练器侧过滤只影响「哪些样本当目标」，不改侧车链；想让上一帧链也绕开碰撞步，
必须在打包时丢。
