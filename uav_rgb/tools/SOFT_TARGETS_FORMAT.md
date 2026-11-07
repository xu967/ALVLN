# 绑定数据身份的软标签（UAVSOFT1）

`join_soft_targets.py` 将教师 logits 转为概率，按 UAVR 全局样本序输出。
缺失、重复、非有限 logits，或指令侧车的 episode/区间与 UAVR 不符时直接失败。
缺失标签不能填均匀分布：这会改变混合损失的监督目标，并非回退硬标签。

二进制全部小端，头固定 120 字节：

| 偏移 | 类型 | 内容 |
| ---: | --- | --- |
| 0 | 8 字节 | ASCII `UAVSOFT1` |
| 8 | uint64 | 样本数 N |
| 16 | uint32 | 类别数 C（当前 4） |
| 20 | uint32 | flags：0 无文本侧车，1 有侧车；其余拒绝 |
| 24 | 32 字节 | 完整 UAVR 文件 SHA256（包含图像、标签、样本顺序及 split 元数据） |
| 56 | 32 字节 | `instructions.jsonl` 原始文件 SHA256；无侧车时全部为 0 |
| 88 | 32 字节 | 概率 payload SHA256 |
| 120 | N×C×float32 | 概率；每行有限、非负、和为 1 |

训练器在训练开始前核验以上身份、长度和概率范围，防止同长度数据错配。
同目录 `.json` 是人工追溯信息，不参与读端验证；权威身份在二进制头内。
仅增加/删除空白也会改变文本侧车身份，须重新拼接。SHA256 用于意外错配和
文件损坏检测，不是签名，也不保证教师预测准确。

## 无侧车数据（flags=0）与信念教师

纯视觉数据集（3DGS pilot、DAgger 自采集）没有 `instructions.jsonl`：拼接器直接用 UAVR 元数据
里的 `episodes` 区间（`episode_id / sample_start / sample_count / source_dir`），头里 flags=0、
侧车哈希全 0；C++ 读端要求此时数据目录里也不存在侧车。episode 目录先按
`--episodes-dir/<split>/<id>/` 找，找不到用 `source_dir`（3DGS 采集是 `route_N/ep_xxxxx` 布局）。

`sim/tools/teacher_belief.py` 产出的 npz 键同样是 `{episode_id}/{step}/logits`，但值是
`log p`（信念边缘化后的概率），拼接时必须 `--temperature 1.0` 才还原 p；它附带的
`teacher_version / k / alpha_steps / aggregate / seed` 会写进 `<out>.json`
（`teacher_version`、`belief_*` 字段），多份 npz 这些键必须一致。

旧的裸 float32 `.bin` 会被明确拒绝。请用原来的 `--uavr-dir`、
`--episodes-dir`、`--soft` 和 `--temperature` 再运行拼接器，并输出到新的路径，
保留旧实验产物。无需重训教师；但必须持有覆盖所有样本的教师导出文件。

验证：`tools/test_join_soft_targets.py --native-check build/check/test_soft_targets_identity`。
它覆盖 Python→C++ 真正读写链路、缺失/非法标签、同长度换图像、换文本、损坏概率和旧格式拒绝。
CTest 的 `test_soft_targets_identity` 另验证 SHA256 已知向量、分块边界；
`test_soft_targets_train` 验证完整训练仍按全局样本序使用软标签。
