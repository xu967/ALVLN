#!/usr/bin/env python3
"""eval_closed_loop.summarize_episode 自检：contact/stuck 口径（EVAL_STANDARD §5.1）
与 SIM-09 新口径（reach_ever/final_in_radius/stop_action_issued/task_success）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ec

WP = {"x": 2.0, "y": 1.0}


def run(rows, x=6.0, goal_x=5.0):
    return ec.summarize_episode(rows, [{"pose_gt": {"x": x}}], goal_x)


def l1(rows, oracle, stop_radius=0.5, goal_radius=None, timed_out=False):
    return ec.summarize_episode(rows, oracle, 6.0, WP, stop_radius=stop_radius,
                                goal_radius=goal_radius, timed_out=timed_out)


def main():
    # contact：算碰撞、算 false_forward，即使过了线也不算 success
    s = run([{"collision": True, "src": "contact", "action_executed": 1, "infer_ms": 1.0}])
    assert s["collision"] and not s["stuck"] and s["false_forward"] and not s["success"], s
    assert not s["collision_free"] and s["stuck_free"], s

    # stuck-only：不是碰撞，也不进 false_forward（过线仍算 success）
    s = run([{"collision": True, "src": "stuck", "action_executed": 1}])
    assert not s["collision"] and s["stuck"] and not s["false_forward"] and s["success"], s
    assert s["collision_free"] and not s["stuck_free"], s

    # contact+stuck 同一步：按 contact 算
    s = run([{"collision": True, "src": "contact+stuck", "action_executed": 1}])
    assert s["collision"] and s["stuck"] and s["false_forward"] and not s["success"], s

    # 旧 episode 没有 src 字段：保持旧口径（collision 就是碰撞）
    s = run([{"collision": True, "action_executed": 1}])
    assert s["collision"] and not s["stuck"] and s["false_forward"] and not s["success"], s

    # false_forward 只认 contact 碰撞且动作是 FORWARD；撞了但动作不是 FORWARD 不算。
    s = run([{"collision": True, "src": "contact", "action_executed": 3}])
    assert s["collision"] and not s["false_forward"], s
    s = run([{"collision": False, "action_executed": 1}])
    assert not s["false_forward"], s

    # 无碰撞且过线 = success
    s = run([{"collision": False, "action_executed": 1}])
    assert s["success"] and not s["collision"] and not s["stuck"], s

    # L1：经过 waypoint 才算 success；没经过即使过线也不算
    rows = [{"collision": False, "action_executed": 1}]
    oracle = [{"pose_gt": {"x": 6.0, "y": 0.0}}]
    s = ec.summarize_episode(rows, oracle, 5.0, {"x": 6.0, "y": 0.0})
    assert s["success"] and s["waypoint_min_dist"] == 0.0, s
    s = ec.summarize_episode(rows, oracle, 5.0, {"x": 6.0, "y": 2.0})
    assert not s["success"] and s["waypoint_min_dist"] == 2.0, s

    # 空集：字段齐全，不计成功
    s = ec.summarize_episode([], [], 5.0)
    assert s["steps"] == 0 and "stuck" in s and not s["success"], s
    assert s["metric_version"] == ec.METRIC_VERSION, s
    assert s["reach_ever"] is False and s["final_in_radius"] is False, s
    assert s["stop_action_issued"] is False and s["task_success"] is False, s
    assert s["stable_termination"] is None and s["goal_radius_reach"] == 1.5, s

    # L1 goal-reach：只要求无碰撞 + 末态在 REACH_RADIUS(1.5) 内，末动作不参与判定。
    oracle_stop = [{"pose_gt": {"x": 2.1, "y": 1.1}}]
    for last in (0, 1):  # 末动作 SAFE 或 FORWARD，只要到了就算成功
        s = ec.summarize_episode([{"collision": False, "action_executed": last}],
                                 oracle_stop, 6.0, WP, stop_radius=0.5)
        assert s["success"] and s["final_waypoint_dist"] < 0.2, (last, s)
    s = ec.summarize_episode([{"collision": False, "action_executed": 0}],
                             [{"pose_gt": {"x": 5.0, "y": 1.0}}], 6.0, WP,
                             stop_radius=0.5)
    assert not s["success"] and s["final_waypoint_dist"] > 1.5, s
    # 半径边界：1.4 m 算到，1.6 m 不算（REACH_RADIUS=1.5）
    for dy, want in ((1.4, True), (1.6, False)):
        s = ec.summarize_episode([{"collision": False, "action_executed": 0}],
                                 [{"pose_gt": {"x": 2.0, "y": 1.0 + dy}}], 6.0, WP,
                                 stop_radius=0.5)
        assert s["success"] is want, (dy, s)
    # 到位但撞了柱：不算成功
    s = ec.summarize_episode([{"collision": True, "src": "contact", "action_executed": 0}],
                             oracle_stop, 6.0, WP, stop_radius=0.5)
    assert not s["success"] and not s["stop_success"], s
    assert not s["reach_ever"] and not s["final_in_radius"] and not s["task_success"], s

    # 2026-11-02 口径：success 看「轨迹上曾进入 1.5 m」，stop_success 看「末位姿」。
    # 到过 0.1 m 又飞走到 5 m：算 success，不算 stop_success。
    flyby = [{"collision": False, "action_executed": 0},
             {"collision": False, "action_executed": 1}]
    oracle_flyby = [{"pose_gt": {"x": 2.1, "y": 1.1}}, {"pose_gt": {"x": 7.0, "y": 1.0}}]
    s = ec.summarize_episode(flyby, oracle_flyby, 6.0, WP, stop_radius=0.5)
    assert s["success"] and not s["stop_success"], s
    assert s["waypoint_min_dist"] < 0.2 and s["final_waypoint_dist"] >= 5.0, s
    # 新口径：到过但末位姿飞走 => reach_ever 真、final/task 假；SAFE 在最近点窗口内。
    assert s["reach_ever"] and not s["final_in_radius"], s
    assert s["stop_action_issued"] and not s["task_success"], s
    assert s["collision_free"] and s["stuck_free"], s
    assert s["metric_version"] == ec.METRIC_VERSION, s
    assert s["goal_radius_reach"] == 1.5 and s["goal_radius_stop"] == 0.5, s
    assert s["stable_termination"] is None, s

    # 全程在 1.5 m 外：两个口径都不算成功
    s = ec.summarize_episode(flyby, [{"pose_gt": {"x": 5.0, "y": 1.0}},
                                     {"pose_gt": {"x": 5.5, "y": 1.0}}],
                             6.0, WP, stop_radius=0.5)
    assert not s["success"] and not s["stop_success"], s
    assert not s["reach_ever"] and not s["final_in_radius"], s

    # 经过时撞柱：即使进入过 1.5 m 也不算成功（两个口径都要无碰撞）
    s = ec.summarize_episode(
        [{"collision": True, "src": "contact", "action_executed": 1}],
        [{"pose_gt": {"x": 2.1, "y": 1.1}}], 6.0, WP, stop_radius=0.5)
    assert not s["success"] and not s["stop_success"], s
    assert not s["reach_ever"] and not s["final_in_radius"], s

    # final_in_radius 用 scene.meta 的 stop_radius，不是 1.5 m 到达半径。
    actions = [{"collision": False, "action_executed": 0}]
    one_m = [{"pose_gt": {"x": 3.0, "y": 1.0}}]
    s = l1(actions, one_m, stop_radius=0.5)
    assert s["reach_ever"] and not s["final_in_radius"] and not s["task_success"], s
    s = l1(actions, one_m, stop_radius=1.5)
    assert s["final_in_radius"] and s["task_success"], s

    # 完整 task_success：到达 + 末位姿在 0.5 m 内 + 最近点窗口有 SAFE + 无 contact。
    near = [{"collision": False, "action_executed": 0},
            {"collision": False, "action_executed": 0}]
    oracle_near = [{"pose_gt": {"x": 2.05, "y": 1.0}},
                   {"pose_gt": {"x": 2.1, "y": 1.0}}]
    s = l1(near, oracle_near, stop_radius=0.5)
    assert s["reach_ever"] and s["final_in_radius"] and s["stop_action_issued"], s
    assert s["task_success"] and s["stopped"], s
    assert s["stable_termination"] is None, "没有速度数据不得声称稳定终止"
    # 超时不 task_success。
    s = l1(near, oracle_near, stop_radius=0.5, timed_out=True)
    assert s["timed_out"] and not s["task_success"], s

    # stop_action_issued 看最近 waypoint 点前后各 2 步，不只看最后一步。
    oracle_win = [{"pose_gt": {"x": 10.0, "y": 1.0}} for _ in range(6)]
    oracle_win[2] = {"pose_gt": {"x": 2.0, "y": 1.0}}
    rows_win = [{"collision": False, "action_executed": 1} for _ in range(6)]
    rows_win[5] = {"collision": False, "action_executed": 0}
    s = l1(rows_win, oracle_win, stop_radius=0.5)
    assert s["reach_ever"] and not s["stop_action_issued"], s  # 最近点 idx=2，±2 不含 5
    rows_win[5] = {"collision": False, "action_executed": 1}
    rows_win[4] = {"collision": False, "action_executed": 0}
    s = l1(rows_win, oracle_win, stop_radius=0.5)
    assert s["stop_action_issued"], s
    rows_win[4] = {"collision": False, "action_executed": 1}
    rows_win[0] = {"collision": False, "action_executed": 0}
    s = l1(rows_win, oracle_win, stop_radius=0.5)
    assert s["stop_action_issued"] and not s["stopped"], s

    # stuck-only：stuck_free=False，但 collision_free=True；旧 success 仍按原口径。
    s = l1([{"collision": True, "src": "stuck", "action_executed": 1}],
           oracle_stop, stop_radius=0.5)
    assert s["stuck"] and not s["collision"] and s["collision_free"], s
    assert not s["stuck_free"] and not s["false_forward"], s
    assert s["success"] and s["reach_ever"], s
    assert not s["task_success"], s  # 到点后动作是 FORWARD，没发 SAFE

    # 旧 success 不受新字段影响：campus 的 goal_radius=0.5 仍决定 success，
    # 而 reach_ever 按 1.5 m 单独算。
    s = l1(actions, one_m, stop_radius=0.5, goal_radius=0.5)
    assert not s["success"] and s["reach_ever"] and not s["final_in_radius"], s

    # SPL（2026-11-07）：S_i 用的是该口径的 success（L1 = goal_reach_ever）；p_i 量到第一次进入 1.5 m 圆盘。
    # 先绕远再进圆：(0,1)->(0,-3) 全程离 WP >= 2 m 不进圆，(0,-3)->(2,1) 长 √20，在离 WP 1.5 m 处进圆。
    oracle_detour = [{"pose_gt": {"x": 0.0, "y": 1.0}}, {"pose_gt": {"x": 0.0, "y": -3.0}},
                     {"pose_gt": {"x": 2.0, "y": 1.0}}]
    p_detour = 4.0 + (20.0 ** 0.5 - 1.5)
    s = ec.summarize_episode([{"collision": False, "action_executed": 1}] * 3, oracle_detour, 6.0, WP,
                             stop_radius=0.5, shortest_path_m=2.0)
    assert s["success"] and abs(s["path_length_m"] - p_detour) < 1e-9 and abs(s["spl"] - 2.0 / p_detour) < 1e-12, s
    s = ec.summarize_episode([{"collision": True, "src": "contact", "action_executed": 1}] * 3, oracle_detour,
                             6.0, WP, stop_radius=0.5, shortest_path_m=2.0)
    assert not s["success"] and s["spl"] == 0.0 and abs(s["path_length_m"] - p_detour) < 1e-9, s
    # 进圆后飞过头再回来：过头那段不进 p_i（与 success=goal_reach_ever 同口径，不考停的时机）。
    oracle_over = [{"pose_gt": {"x": 0.0, "y": 1.0}}, {"pose_gt": {"x": 4.0, "y": 1.0}},
                   {"pose_gt": {"x": 2.0, "y": 1.0}}]
    s = ec.summarize_episode([{"collision": False, "action_executed": 1}] * 3, oracle_over, 6.0, WP,
                             stop_radius=0.5, shortest_path_m=0.5)
    assert s["success"] and abs(s["path_length_m"] - 0.5) < 1e-12 and abs(s["spl"] - 1.0) < 1e-12, s
    # campus 的 goal_radius 决定 success，spl 跟着它：到过 1.5 m 内但没进 0.5 m -> success 假 -> spl 0。
    s = l1(actions, one_m, stop_radius=0.5, goal_radius=0.5)
    assert s["spl"] is None, "没给 l_i 时不得给 spl"
    s = ec.summarize_episode(actions, one_m, 6.0, WP, stop_radius=0.5, goal_radius=0.5, shortest_path_m=1.0)
    assert s["spl"] == 0.0 and s["reach_ever"], s

    print("test_eval_summary: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
