#!/usr/bin/env python3
"""eval_closed_loop.py 指标纯函数自检：成功/碰撞/误前向/百分位/空集/新口径汇总。"""
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
    # 旧 success 之外的新字段：无 waypoint 时 reach/final/task 均为 False。
    assert e1["reach_ever"] is False and e1["final_in_radius"] is False, e1
    assert e1["task_success"] is False and e1["stable_termination"] is None, e1
    assert e1["metric_version"] == ev.METRIC_VERSION, e1

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

    # requested / attempted / skipped / failed 的分母必须分开：
    #   2 集完成、请求 4 集、实际尝试 3 个 seed（1 个失败）、跳过 2 个。
    # completion 只除以尝试数（skipped 不进分母）；execution 才除以请求数。
    e1["geometry_gid"] = "g1"
    e2["geometry_gid"] = "g1"
    agg2 = ev.aggregate([e1, e2], requested=4, attempted=3,
                        failed_count=1, skipped_count=2)
    assert agg2["requested_success_rate"] == 0.25, agg2   # 1/4
    assert abs(agg2["completion_rate"] - 2 / 3) < 1e-12, agg2  # 2/3 attempted
    assert agg2["execution_rate"] == 0.5, agg2                 # 2/4 requested
    assert agg2["failed_count"] == 1 and agg2["skipped_count"] == 2, agg2
    assert agg2["unique_gids"] == 1, agg2
    assert agg2["wilson_ci"]["success"] == ev.wilson_ci(1, 2), agg2
    assert agg2["wilson_ci"]["requested"] == ev.wilson_ci(1, 4), agg2

    # 明确回归：1 集完成、1 集 skipped、请求 2 集 => completion 仍是 1.0（只看尝试），
    # execution 0.5（skipped 表现为未执行），requested SR 0.5（skipped 不是成功）。
    agg_skip = ev.aggregate([e1], requested=2, attempted=1,
                            failed_count=0, skipped_count=1)
    assert agg_skip["completion_rate"] == 1.0, agg_skip
    assert agg_skip["execution_rate"] == 0.5, agg_skip
    assert agg_skip["requested_success_rate"] == 0.5, agg_skip

    # 失败后补扫 seed（attempted > requested）时 execution 仍 <=1。
    agg_retry = ev.aggregate([e1], requested=1, attempted=2,
                             failed_count=1, skipped_count=0)
    assert agg_retry["completion_rate"] == 0.5, agg_retry   # 1/2 attempted
    assert agg_retry["execution_rate"] == 1.0, agg_retry    # 1/1 requested

    # 2 集完成、请求 2、跳过 5 => completion 与 execution 都是 1.0。
    agg3 = ev.aggregate([e1, e2], requested=2, attempted=2,
                        failed_count=0, skipped_count=5)
    assert agg3["completion_rate"] == 1.0 and agg3["execution_rate"] == 1.0, agg3
    assert agg3["skipped_count"] == 5, agg3

    empty = ev.aggregate([])
    assert empty["episodes_done"] == 0 and empty["success_rate"] is None, empty
    assert empty["inference_ms"]["p50"] is None, empty
    assert empty["requested_success_rate"] is None, empty
    assert empty["execution_rate"] is None and empty["unique_gids"] == 0, empty
    assert empty["wilson_ci"] == {"success": None, "requested": None}, empty
    assert empty["stable_termination_rate"] is None, empty

    # Wilson 95%：19/47 与 HANDOFF §4.3 的 [0.276, 0.547] 对齐。
    lo, hi = ev.wilson_ci(19, 47)
    assert round(lo, 3) == 0.276 and round(hi, 3) == 0.547, (lo, hi)
    assert ev.wilson_ci(0, 0) is None
    assert ev.wilson_ci(0, 10)[0] == 0.0
    assert ev.wilson_ci(10, 10)[1] == 1.0

    print("test_eval_metrics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
