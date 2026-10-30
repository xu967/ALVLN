# L1 指令词表与模板模块

- 时间：2026-10-31 01:15–01:25
- 范围：`sim/tools/instruction.py`、`sim/tools/test_instruction.py`
- 结论：完成词表/模板/目标色解析；尚未接入 scene_gen/collect。

## 做了什么

- 新增 `instruction.py`：
  - 固定词表（颜色、动词、关系、stop 等；`<pad>`/`<unk>`）。
  - `tokenize(text)`：小写字母切词，未知词 -> `<unk>`。
  - `make_instruction(color, relation=None)`：单目标模板
    `fly to the {color} pillar and stop`；关系模板
    `pass the {color} pillar on your {left|right} and stop`。
  - `parse_target_color(text)`：唯一颜色才返回，多色/无颜色返回 None。
- 新增 `test_instruction.py`：词表、模板、大小写、未知词、多色、非法颜色/关系。

## 没做什么

- **没有改 `scene_gen.py`**：T5b 采集正在用它，采集结束后再加彩色柱。
- **没有改 `collect.py`**：采集期间文件锁，instruction 字段等 T5b 结束。
- **没有生成真实语言数据**：L1 pilot 要等场景改造。
- **没有接模型**：文本编码/融合是 L2。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `test_instruction.py` | PASS |

## 风险与遗留

- 词表固定；真实指令超出词表会落到 `<unk>`，L1 只用模板，不触发。
- 目前只支持单目标颜色；多目标/关系组合要扩模板与审计。
