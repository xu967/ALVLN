#!/usr/bin/env python3
"""L1 最小指令词表与模板。

指令只描述「目标颜色 + 可选左右关系」，不含坐标/距离；token 供
`uavrgb_infer --text` 使用。词表固定，未知词统一映射到 <unk>。
"""
import re

VOCAB = [
    "<pad>", "<unk>",
    "fly", "go", "pass", "to", "the",
    "red", "blue", "yellow", "green",
    "pillar", "and", "stop", "on", "your", "left", "right",
    "before", "near", "through", "gap",
]
COLORS = ("red", "blue", "yellow", "green")
RELATIONS = ("left", "right")
_ID = {w: i for i, w in enumerate(VOCAB)}
PAD = _ID["<pad>"]
UNK = _ID["<unk>"]


def tokenize(text):
    """小写化后按字母切词；未知词 -> <unk>。空串返回 []。"""
    return [_ID.get(w, UNK) for w in re.findall(r"[a-z]+", text.lower())]


def make_instruction(color, relation=None):
    """生成模板指令；relation=None 为「飞到目标柱」，left/right 为「经过目标柱」。"""
    if color not in COLORS:
        raise ValueError(f"颜色不在词表: {color!r}")
    if relation is None:
        return f"fly to the {color} pillar and stop"
    if relation not in RELATIONS:
        raise ValueError(f"关系不在词表: {relation!r}")
    return f"pass the {color} pillar on your {relation} and stop"


def parse_target_color(text):
    """从指令里取唯一目标颜色；没有或出现多个颜色返回 None。"""
    colors = [w for w in re.findall(r"[a-z]+", text.lower()) if w in COLORS]
    return colors[0] if len(colors) == 1 else None
