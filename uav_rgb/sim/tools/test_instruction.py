#!/usr/bin/env python3
"""instruction.py 自检：词表、模板、目标色解析、未知词。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instruction as ins


def main():
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)

    toks = ins.tokenize("Fly to the RED pillar and stop")
    want = [ins._ID[w] for w in ("fly", "to", "the", "red", "pillar", "and", "stop")]
    check(toks == want, f"tokenize 结果不符: {toks} != {want}")
    check(all(0 <= t < len(ins.VOCAB) for t in toks), "token 越界")

    text = ins.make_instruction("red")
    check(text == "fly to the red pillar and stop", f"模板不符: {text}")
    check(ins.parse_target_color(text) == "red", "目标色解析失败")
    check(ins.tokenize(text) == want, "模板 tokenize 与手写不一致")

    rel = ins.make_instruction("blue", "left")
    check(rel == "pass the blue pillar on your left and stop", f"关系模板不符: {rel}")
    check(ins.parse_target_color(rel) == "blue", "关系模板目标色解析失败")

    check(ins.tokenize("xyz") == [ins.UNK], "未知词未映射到 <unk>")
    check(ins.tokenize("") == [], "空串应返回空列表")
    check(ins.parse_target_color("fly to the red and blue pillar") is None,
          "多颜色应返回 None")
    try:
        ins.make_instruction("purple")
        fails.append("非法颜色未抛 ValueError")
    except ValueError:
        pass
    try:
        ins.make_instruction("red", "up")
        fails.append("非法关系未抛 ValueError")
    except ValueError:
        pass

    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print("test_instruction: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
