#!/usr/bin/env python3
"""eval_closed_loop 参数自检：--seeds 解析、缺省集数、格式错误。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ec


def main():
    assert ec.parse_seeds("23, 25:28") == [23, 25, 26, 27], ec.parse_seeds("23, 25:28")
    assert ec.parse_seeds("7") == [7]
    for bad in ("23,abc", "-1", "5:5", "9:3", "5,5"):
        try:
            ec.parse_seeds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad!r}")
    # 关键回归：传 20 个 seed 不显式给 --episodes 时，目标集数必须是 20，不是 3。
    assert ec.episode_target(None, list(range(20))) == 20
    assert ec.episode_target(None, []) == 3
    assert ec.episode_target(20, [1, 2, 3]) == 20

    # L1 去重键必须包含彩色柱位置：同几何、同目标色但目标柱不同的任务不能合并；
    # 同几何换色（反事实对）必须是两个不同任务。
    def pillar(x, y, color=None):
        p = {"x": x, "y": y, "z": 1.5, "size": [0.4, 0.4, 3.0]}
        if color:
            p["color"] = color
        return p
    # 同色柱对：left/right 两个任务的彩色柱完全相同，只有关系词不同。
    a = [pillar(1.0, 0.0, "red"), pillar(2.0, 1.0, "red"), pillar(3.0, -1.0)]
    c = [pillar(1.0, 0.6, "red"), pillar(2.0, 1.0, "red"), pillar(3.0, -1.0)]
    left = {"pillars": a, "relation": "left"}
    right = {"pillars": a, "relation": "right"}
    assert ec.dedup_key(left, "l1") != ec.dedup_key(right, "l1"), \
        "同色柱对的 left/right 两个任务被当成重复"
    assert ec.dedup_key(left, "l1") != ec.dedup_key(
        {"pillars": c, "relation": "left"}, "l1"), "同色不同目标柱被当成重复"
    assert ec.dedup_key(left, "l1") == ec.dedup_key(dict(left), "l1")
    assert ec.dedup_key(left, "base") == ec.dedup_key(right, "base")

    # L2 文本消融：keep 原样；blank 空串；shuffle 把 left/right 反过来。
    meta = {"target_color": "red", "relation": "left",
            "instruction": "pass the red pillar on your left and stop"}
    assert ec.override_instruction(meta, "keep") is meta
    assert ec.override_instruction(meta, "blank")["instruction"] == ""
    shuffled = ec.override_instruction(meta, "shuffle")
    assert shuffled["instruction"] == "pass the red pillar on your right and stop", shuffled
    try:
        ec.override_instruction({"target_color": "red"}, "shuffle")
    except ValueError:
        pass
    else:
        raise AssertionError("缺 relation 的 shuffle 应报错，不能换成别的颜色")
    print("test_eval_closed_loop_args: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
