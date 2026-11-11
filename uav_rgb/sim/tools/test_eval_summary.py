#!/usr/bin/env python3
"""eval_closed_loop.summarize_episode 自检：contact/stuck 口径（EVAL_STANDARD §5.1）
与 SIM-09 新口径（reach_ever/final_in_radius/stop_action_issued/task_success）。
W1-3：帧间隔。actions 行的 rgb_ts（ns）相邻两步之差的最大值写 max_frame_gap_s（秒）；大于 1.2 × hold 时再写
frame_gap_flag: true（不大于则不写这个键）。hold 由调用方给（run_episode 传 --hold）；没给时用行里的 sim_dt。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ec

WP = {"x": 2.0, "y": 1.0}

# 旧代码（W1-3 之前，75e85e2）下 summarize_episode 的真实键集合：一般调用、waypoint 调用、空集三种都是这 28 个。
# actions 行没有 rgb_ts（旧数据）、只有 1 步、空集时，输出必须仍是这 28 个键。
SUMMARY_KEYS_OLD = {
    'collision', 'collision_free', 'corridor_violation', 'false_forward', 'final_in_radius',
    'final_waypoint_dist', 'goal_radius_reach', 'goal_radius_stop', 'infer_ms', 'max_x', 'metric_version',
    'path_length_m', 'reach_ever', 'shortest_path_m', 'spl', 'stable_termination', 'steps', 'steps_to_goal',
    'stop_action_issued', 'stop_success', 'stopped', 'stuck', 'stuck_free', 'success', 'task_success',
    'timed_out', 'waypoint_min_dist', 'y_max_m'
}
GAP_KEYS = {"max_frame_gap_s", "frame_gap_flag"}


def run(rows, x=6.0, goal_x=5.0):
    return ec.summarize_episode(rows, [{"pose_gt": {"x": x}}], goal_x)


def l1(rows, oracle, stop_radius=0.5, goal_radius=None, timed_out=False):
    return ec.summarize_episode(rows, oracle, 6.0, WP, stop_radius=stop_radius,
                                goal_radius=goal_radius, timed_out=timed_out)


def gap_rows(*gaps_s, **extra):
    """rgb_ts 从 10 s 起；第 i 步与第 i-1 步相隔 gaps_s[i-1] 秒（整数 ns，避免浮点误差）。extra 加进每一行。"""
    ts = [10 * 10**9]
    for g in gaps_s:
        ts.append(ts[-1] + int(round(g * 1e9)))
    return [{"collision": False, "action_executed": 1, "rgb_ts": t, **extra} for t in ts]


def gap_summary(rows, goal_x=5.0, **kw):
    """rows 配一条飞过线的 oracle（逐行一一对应）；kw 原样传给 summarize_episode（如 hold）。"""
    return ec.summarize_episode(rows, [{"pose_gt": {"x": 6.0}} for _ in rows], goal_x, **kw)


def gap_run(gaps_s, hold=None, **row_extra):
    """合成一条相邻帧间隔为 gaps_s 的轨迹（无碰撞、飞过线）。hold 给了才传给 summarize_episode；row_extra 进每一行（如 sim_dt）。"""
    return gap_summary(gap_rows(*gaps_s, **row_extra), **({} if hold is None else {"hold": hold}))


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

    frame_gap_checks()
    old_outputs_unchanged_checks()

    print("test_eval_summary: PASS")
    return 0


def frame_gap_checks():
    """W1-3：max_frame_gap_s / frame_gap_flag 的取值与 hold 的来源。"""
    # 相邻帧间隔 2.4 s（hold 1.0）：写最大间隔，frame_gap_flag 为真
    s = gap_run((1.0, 1.0, 2.4, 1.0), hold=1.0)
    assert s["max_frame_gap_s"] == 2.4 and s["frame_gap_flag"] is True, s
    # 相邻帧间隔 1.0 s：写最大间隔，不写 frame_gap_flag
    s = gap_run((1.0, 1.0, 1.0), hold=1.0)
    assert s["max_frame_gap_s"] == 1.0 and "frame_gap_flag" not in s, s
    # 取的是相邻两步之差的最大值，不是首尾之差，也不是平均
    assert gap_run((1.0, 1.1, 1.0), hold=1.0)["max_frame_gap_s"] == 1.1
    # 阈值是严格大于 1.2 × hold：恰好 1.2 s 不打旗，1.201 s 打旗
    assert "frame_gap_flag" not in gap_run((1.0, 1.2), hold=1.0)
    assert gap_run((1.0, 1.201), hold=1.0)["frame_gap_flag"] is True
    # 阈值跟着 hold 走：同样 1.0 s 的间隔，hold 0.5 时算断帧，hold 1.0 时不算
    assert gap_run((1.0, 1.0), hold=0.5)["frame_gap_flag"] is True
    assert "frame_gap_flag" not in gap_run((1.0, 1.0), hold=1.0)

    # 拿不到 hold 时用行里的 sim_dt（3DGS 的行带 sim_dt = hold_s）
    assert gap_run((1.0, 1.0), sim_dt=0.5)["frame_gap_flag"] is True
    assert "frame_gap_flag" not in gap_run((1.0, 1.0), sim_dt=1.0)
    # 给了 hold 就以 hold 为准，不看 sim_dt
    assert "frame_gap_flag" not in gap_run((1.0, 1.0), hold=1.0, sim_dt=0.5)
    # 行里的 sim_dt 取中位数：个别行的离群值不能把阈值抬高
    rows = gap_rows(1.0, 1.5, 1.0, sim_dt=1.0)
    rows[2]["sim_dt"] = 9.0
    s = gap_summary(rows)
    assert s["max_frame_gap_s"] == 1.5 and s["frame_gap_flag"] is True, s
    # hold 与 sim_dt 都没有：只写间隔，没有判据就不打旗
    s = gap_run((1.0, 5.0))
    assert s["max_frame_gap_s"] == 5.0 and "frame_gap_flag" not in s, s

    # 缺 rgb_ts 的行不桥接：桥接会把一个缺行算成 2 倍间隔
    rows = gap_rows(1.0, 1.0, 1.0)
    rows[1]["rgb_ts"] = None
    s = gap_summary(rows, hold=1.0)
    assert s["max_frame_gap_s"] == 1.0 and "frame_gap_flag" not in s, s

    # waypoint 任务（l1 / campus）同样写帧间隔，口径与验收线任务一致
    rows = gap_rows(1.0, 2.4)
    s = ec.summarize_episode(rows, [{"pose_gt": {"x": 2.0, "y": 1.0}}] * len(rows), 6.0, WP,
                             stop_radius=0.5, hold=1.0)
    assert s["max_frame_gap_s"] == 2.4 and s["frame_gap_flag"] is True and s["success"], s

    # 新键只有这两个，且只加不改：其余键与值同没有 rgb_ts 的同一条轨迹逐项相等
    with_ts = gap_run((1.0, 1.0, 2.4, 1.0), hold=1.0)
    without_ts = gap_summary([{k: v for k, v in r.items() if k != "rgb_ts"} for r in gap_rows(1.0, 1.0, 2.4, 1.0)])
    assert set(with_ts) - set(without_ts) == GAP_KEYS and {k: with_ts[k] for k in without_ts} == without_ts, \
        (with_ts, without_ts)


def old_outputs_unchanged_checks():
    """W1-3 不改旧输出：没有可算的相邻时间戳时，键集合与旧代码一字不差。这一组在旧代码上也应通过。"""
    # 旧数据没有 rgb_ts
    one = [{"collision": False, "action_executed": 1}]
    assert set(run(one)) == SUMMARY_KEYS_OLD, set(run(one)) ^ SUMMARY_KEYS_OLD
    assert set(l1(one, [{"pose_gt": {"x": 2.0, "y": 1.0}}])) == SUMMARY_KEYS_OLD
    no_ts = gap_summary([{k: v for k, v in r.items() if k != "rgb_ts"} for r in gap_rows(1.0, 1.0, 2.4, 1.0)])
    assert set(no_ts) == SUMMARY_KEYS_OLD, set(no_ts) ^ SUMMARY_KEYS_OLD
    # 一步集没有相邻两步；空集没有步
    assert set(gap_run(())) == SUMMARY_KEYS_OLD, "一步集"
    assert set(ec.summarize_episode([], [], 5.0)) == SUMMARY_KEYS_OLD, "空集"


if __name__ == "__main__":
    sys.exit(main())
