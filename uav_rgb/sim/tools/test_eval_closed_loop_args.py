#!/usr/bin/env python3
"""eval_closed_loop 参数自检：--seeds 解析、缺省集数、协议绑定、动力学档。"""
import json
import os
import sys
import tempfile

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

    # 2026-11-05 决定：cf_sil 退出默认评测矩阵，cf_blackbox 只留名字给 3DGS。
    assert "cf_sil" not in ec.DYNAMICS_PROFILES, ec.DYNAMICS_PROFILES
    assert "cf_blackbox" in ec.DYNAMICS_PROFILES, ec.DYNAMICS_PROFILES
    assert "x3_velocity" in ec.DYNAMICS_PROFILES and "cf_stab" in ec.DYNAMICS_PROFILES
    assert ec.DYNAMICS_PROFILES == ec.cp.DYNAMICS_CHOICES

    # 协议绑定：只比较 protocol 中出现的键；未知键（如 protocol_id）忽略。
    actual = {"window_from": "observation", "hold_s": 0.2, "steps": 80,
              "task": "l1", "appearance_version": "v1", "dynamics": "cf_blackbox",
              "camera_profile": ec.CAMERA_PROFILE}
    assert ec.protocol_mismatches(dict(actual), actual) == []
    assert ec.protocol_mismatches({"steps": 80, "hold_s": 0.2}, actual) == []
    assert ec.protocol_mismatches({"steps": 80.0}, actual) == []  # int/float 数值相等
    assert ec.protocol_mismatches({"steps": 81}, actual) == [("steps", 81, 80)]
    assert ec.protocol_mismatches({"protocol_id": "x", "dynamics": "cf_stab"}, actual) == [
        ("dynamics", "cf_stab", "cf_blackbox")]
    # 默认（非 strict）只记录 mismatch，不在这里抛错；严格模式由 main 判。

    with tempfile.TemporaryDirectory() as directory:
        good = os.path.join(directory, "protocol.json")
        with open(good, "w", encoding="utf-8") as f:
            json.dump(actual, f)
        assert ec.load_protocol(good) == actual
        bad = os.path.join(directory, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{not json")
        try:
            ec.load_protocol(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("坏 JSON 必须报错，不能静默当成无协议")
        non_object = os.path.join(directory, "list.json")
        with open(non_object, "w", encoding="utf-8") as f:
            json.dump([1, 2], f)
        try:
            ec.load_protocol(non_object)
        except ValueError:
            pass
        else:
            raise AssertionError("protocol 顶层不是对象必须报错")

    # 录像画质档：写进 eval.json 的是生效档名（环境变量 UAVRGB_VIDEO_QUALITY 覆盖），未知档名报错
    import view_layout as vl
    saved = os.environ.pop(vl.QUALITY_ENV, None)
    try:
        assert ec.record_views_quality(None) == vl.DEFAULT_QUALITY and ec.record_views_quality("high") == "high"
        os.environ[vl.QUALITY_ENV] = "low"
        assert ec.record_views_quality("high") == "low"
        del os.environ[vl.QUALITY_ENV]
        try:
            ec.record_views_quality("ultra")
        except ValueError:
            pass
        else:
            raise AssertionError("未知画质档必须报错")
    finally:
        if saved is not None:
            os.environ[vl.QUALITY_ENV] = saved

    print("test_eval_closed_loop_args: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
