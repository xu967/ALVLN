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

    # SPL（2026-11-07 新列，只加不改）：p_i = oracle xy 折线，l_i 由调用方给。
    assert ev.path_length_m([(0, 0), (3, 4), (3, 4)]) == 5.0
    assert ev.path_length_m([]) == 0.0 and ev.path_length_m([(1, 1)]) == 0.0
    assert ev.trajectory_xy([{"pose_gt": {"x": 1.0}}, {"pose_gt": {"x": 1.0, "y": 2.0}}, {}]) == [(1.0, 2.0)]
    assert ev.spl_of(True, 8.0, 6.0) == 0.75 and ev.spl_of(False, 8.0, 6.0) == 0.0
    assert ev.spl_of(True, 5.0, 6.0) == 1.0          # 比教师参考路还短：截成 S_i
    assert ev.spl_of(True, 6.0, None) is None and ev.spl_of(True, None, 6.0) is None
    assert ev.spl_of(True, 0.0, 0.0) == 1.0
    oracle_xy = [{"pose_gt": {"x": 0.0, "y": 0.0}}, {"pose_gt": {"x": 3.0, "y": 0.0}},
                 {"pose_gt": {"x": 6.2, "y": 0.0}}]
    e_spl = ev.summarize_episode(actions1, oracle_xy, 6.0, shortest_path_m=5.0)
    assert e_spl["success"] is True and abs(e_spl["path_length_m"] - 6.0) < 1e-12, e_spl   # 插值到 x=6.0 过线点
    assert e_spl["shortest_path_m"] == 5.0 and abs(e_spl["spl"] - 5.0 / 6.0) < 1e-12, e_spl
    e_nospl = ev.summarize_episode(actions1, oracle_xy, 6.0)
    assert e_nospl["spl"] is None and e_nospl["shortest_path_m"] is None, e_nospl
    assert abs(e_nospl["path_length_m"] - 6.0) < 1e-12, e_nospl
    # p_i 截到首次进入目标区（插值到边界）：过线后继续飞的那段不进分母（base 没有 goal-stop）；从未到达则整条。
    traj_over = [(0.0, 0.0), (3.0, 0.0), (6.2, 0.0), (9.0, 0.0), (9.0, 2.0)]
    assert abs(ev.goal_path_length_m(traj_over, goal_x=6.0) - 6.0) < 1e-12
    assert ev.goal_path_length_m(traj_over, goal_x=20.0) == 11.0
    assert ev.goal_path_length_m([], goal_x=6.0) is None
    assert ev.truncate_at_goal(traj_over, goal_x=6.0) == [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)]
    assert ev.truncate_at_goal(traj_over, goal_x=-1.0) == [(0.0, 0.0)]          # 起点已在区内
    assert ev.truncate_at_goal([(1.0, 1.0)], goal_x=0.0) == [(1.0, 1.0)]
    # waypoint 任务：量到进入 radius 圆盘的那一点；radius 为 None 时退回验收线口径。
    wp_traj = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)]
    assert abs(ev.goal_path_length_m(wp_traj, goal_x=99.0, waypoint={"x": 5.0, "y": 0.0}, radius=1.5) - 3.5) < 1e-12
    assert abs(ev.goal_path_length_m(wp_traj, goal_x=3.0, waypoint={"x": 5.0, "y": 0.0}, radius=None) - 3.0) < 1e-12
    # 斜着进圆：(0,0)->(4,4) 与以 (4,2) 为心 r=2 的圆，截点在圆周上（|P-C| = 2）且在起点一侧。
    cut = ev.truncate_at_goal([(0.0, 0.0), (4.0, 4.0)], waypoint={"x": 4.0, "y": 2.0}, radius=2.0)
    assert len(cut) == 2 and abs(((cut[1][0] - 4.0) ** 2 + (cut[1][1] - 2.0) ** 2) ** 0.5 - 2.0) < 1e-9, cut
    assert cut[1][0] < 3.0, cut
    # 线与圆不相交：原样返回。
    assert ev.truncate_at_goal([(0.0, 0.0), (4.0, 4.0)], waypoint={"x": 4.0, "y": 0.0}, radius=2.0) == [(0.0, 0.0), (4.0, 4.0)]
    assert ev.truncate_at_goal([(0.0, 0.0), (1.0, 0.0)], waypoint={"x": 9.0, "y": 0.0}, radius=2.0) == [(0.0, 0.0), (1.0, 0.0)]
    oracle_over = [{"pose_gt": {"x": x, "y": y}} for x, y in traj_over]
    e_over = ev.summarize_episode(actions1, oracle_over, 6.0, shortest_path_m=6.0)
    assert abs(e_over["path_length_m"] - 6.0) < 1e-12 and abs(e_over["spl"] - 1.0) < 1e-12, e_over
    assert e_over["max_x"] == 9.0, e_over        # 旧字段仍看整条轨迹
    e_wp = ev.summarize_episode(actions1, oracle_over, 6.0, waypoint={"x": 9.0, "y": 0.0},
                                stop_radius=0.5, goal_radius=1.5, shortest_path_m=8.0)
    assert e_wp["success"] is True and abs(e_wp["path_length_m"] - 7.5) < 1e-12, e_wp   # 进 1.5 m 圆在 x=7.5
    # l_i 对 waypoint 同样截到圆盘边界：给 radius 比不给短，且差值 <= radius。
    pillars_wp = [{"x": 1.5, "y": 0.0, "size": [0.4, 0.4, 3.0]}]
    l_full = ev.teacher_shortest_path_m(pillars_wp, (-0.5, 0.0), waypoint={"x": 4.0, "y": -2.0}, clearance=0.4)
    l_cut = ev.teacher_shortest_path_m(pillars_wp, (-0.5, 0.0), waypoint={"x": 4.0, "y": -2.0}, clearance=0.4, radius=1.5)
    assert l_cut is not None and 0.0 < l_full - l_cut <= 1.5 + 1e-9, (l_full, l_cut)
    # 视线抽捷径：阶梯路径拉直；碰到障碍的那段不跳。
    stair = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert ev.shortcut_path(stair, lambda p, q: True) == [(0, 0), (3, 3)]
    assert ev.shortcut_path(stair, lambda p, q: False) == stair
    assert ev.shortcut_path([(0, 0), (1, 1)], lambda p, q: True) == [(0, 0), (1, 1)]
    blocked_mid = lambda p, q: not (p == (0, 0) and q in [(2, 2), (3, 3)])   # noqa: E731
    assert ev.shortcut_path(stair, blocked_mid) == [(0, 0), (1, 1), (3, 3)]
    # 旧 oracle 只有 x：没有 xy 轨迹，p_i 为 None，其余字段与之前完全一样。
    assert e1["path_length_m"] is None and e1["spl"] is None, e1
    for k, v in e1.items():
        if k not in ("path_length_m", "shortest_path_m", "spl", "actions", "geometry_gid"):
            assert ev.summarize_episode(actions1, oracle1, 6.0)[k] == v, k
    assert ev.summarize_episode([], [], 6.0)["spl"] is None
    # 汇总：spl 只在有 l_i 的集上平均，spl_n 单列；requested_spl 只有全部完成集都有 spl 时才给。
    e_spl["actions"], e_nospl["actions"] = {}, {}
    agg_spl = ev.aggregate([e_spl, e_nospl], requested=2, attempted=2)
    assert agg_spl["spl_n"] == 1 and abs(agg_spl["spl"] - 5.0 / 6.0) < 1e-12, agg_spl
    assert agg_spl["requested_spl"] is None, agg_spl
    e_fail = dict(e_spl, success=False, spl=0.0)
    agg_spl2 = ev.aggregate([e_spl, e_fail], requested=4, attempted=4)
    assert agg_spl2["spl_n"] == 2 and abs(agg_spl2["spl"] - 0.5 * 5.0 / 6.0) < 1e-12, agg_spl2
    assert abs(agg_spl2["requested_spl"] - (5.0 / 6.0) / 4) < 1e-12, agg_spl2
    assert empty["spl"] is None and empty["spl_n"] == 0 and empty["requested_spl"] is None, empty
    # l_i：教师 A* 到验收线（任意 y）不长于到线上任一点；无柱子 / 无目标 -> None。
    pillars = [{"x": 1.5, "y": 0.0, "size": [0.4, 0.4, 3.0]}, {"x": 3.0, "y": 1.6, "size": [0.4, 0.4, 3.0]}]
    l_line = ev.teacher_shortest_path_m(pillars, (-0.5, 0.0), goal_x=4.0, clearance=0.4)
    l_point = ev.path_length_m(ev.teacher.astar_path(pillars, (-0.5, 0.0), (4.0, 0.0), clearance=0.4))
    assert l_line is not None and 4.5 <= l_line <= l_point + 1e-9, (l_line, l_point)
    # 抽捷径后不短于欧氏直线、不长于栅格路径；抽完的折线每段都离柱 >= clearance。
    grid = ev.teacher.astar_path(pillars, (-0.5, 0.0), (4.0, -1.0), clearance=0.4)
    pulled = ev.shortcut_path(grid, ev._pillar_segment_free(pillars, 0.4))
    assert (4.5 ** 2 + 1.0 ** 2) ** 0.5 - 1e-9 <= ev.path_length_m(pulled) <= ev.path_length_m(grid), pulled
    assert all(ev._pillar_segment_free(pillars, 0.4)(a, b) for a, b in zip(pulled, pulled[1:]))
    assert not ev._pillar_segment_free(pillars, 0.4)((-0.5, 0.0), (4.0, 0.0))   # 穿第一根门柱
    l_wp = ev.teacher_shortest_path_m(pillars, (-0.5, 0.0), waypoint={"x": 2.0, "y": -2.0}, clearance=0.4)
    assert l_wp is not None and l_wp >= ((2.5 ** 2 + 2.0 ** 2) ** 0.5) - 0.15, l_wp   # 栅格 0.1 的离散误差
    assert ev.teacher_shortest_path_m([], (0, 0), goal_x=4.0) is None
    assert ev.teacher_shortest_path_m(pillars, (-0.5, 0.0)) is None
    # waypoint 落在柱子膨胀区里：A* 无路 -> None
    assert ev.teacher_shortest_path_m(pillars, (-0.5, 0.0), waypoint={"x": 1.5, "y": 0.0}, clearance=0.4) is None

    print("test_eval_metrics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
