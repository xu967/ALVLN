#!/usr/bin/env python3
"""eval_closed_loop.py 指标纯函数自检：成功/碰撞/误前向/百分位/空集。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ev


def main():
    assert ev.percentile([], 50) is None
    assert ev.percentile([1, 2, 3, 4], 50) == 2
    assert ev.percentile([1, 2, 3, 4], 95) == 4
    assert ev.percentile([1, 2, 3, 4], 0) == 1

    actions1 = [
        {"collision": False, "action_executed": 1, "infer_ms": 100.0},
        {"collision": False, "action_executed": 3, "infer_ms": 5.0},
        {"collision": False, "action_executed": 1, "infer_ms": 6.0},
    ]
    oracle1 = [{"pose_gt": {"x": 0.0}}, {"pose_gt": {"x": 3.0}},
               {"pose_gt": {"x": 6.2}}]
    e1 = ev.summarize_episode(actions1, oracle1, 6.0)
    assert e1["success"] is True and e1["collision"] is False, e1
    assert e1["infer_ms"] == [100.0, 5.0, 6.0], e1
    assert e1["max_x"] == 6.2, e1

    actions2 = [
        {"collision": False, "action_executed": 0, "infer_ms": 9.0},
        {"collision": True, "action_executed": 1, "infer_ms": 4.0},
    ]
    oracle2 = [{"pose_gt": {"x": 0.0}}, {"pose_gt": {"x": 2.0}}]
    e2 = ev.summarize_episode(actions2, oracle2, 6.0)
    assert e2["success"] is False and e2["collision"] is True, e2
    assert e2["false_forward"] is True, e2

    e1["actions"] = {1: 2, 3: 1}
    e2["actions"] = {0: 1, 1: 1}
    agg = ev.aggregate([e1, e2])
    assert agg["episodes_done"] == 2, agg
    assert agg["success_rate"] == 0.5 and agg["collision_rate"] == 0.5, agg
    assert agg["false_forward_rate"] == 0.5, agg
    assert agg["action_distribution"]["SAFE_NEUTRAL"] == 1, agg
    assert agg["action_distribution"]["FORWARD_SLOW"] == 3, agg
    assert agg["action_distribution"]["YAW_LEFT"] == 0, agg
    assert agg["action_distribution"]["YAW_RIGHT"] == 1, agg
    im = agg["inference_ms"]
    assert im["n_warm"] == 3, im          # 每集首帧（100/9ms）不计入 warm
    assert im["p50"] == 5.0 and im["p95"] == 6.0, im
    assert im["cold_first_ms"] == [100.0, 9.0], im

    empty = ev.aggregate([])
    assert empty["episodes_done"] == 0 and empty["success_rate"] is None, empty
    assert empty["inference_ms"]["p50"] is None, empty

    print("test_eval_metrics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
