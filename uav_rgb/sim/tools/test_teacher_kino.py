#!/usr/bin/env python3
"""teacher_kino.py 单测：原语物理、避障闭环、停靠、必撞 fail-closed、确定性、耗时。

不启动 Gazebo，纯 stdlib：
  python3 sim/tools/test_teacher_kino.py
闭环回放用 teacher_kino 自己的 advance 当被控对象（和 compare_teachers.py 的独立
欧拉积分器不同，这里只验证教师逻辑自洽；跨模型表现看 compare_teachers）。
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen                                                       # noqa: E402
import teacher_kino as tk                                              # noqa: E402
from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT    # noqa: E402

TEST_SEEDS = (180, 206, 250, 485, 503, 555, 586, 708, 715, 721,
              799, 855, 932, 992, 1113, 1116, 1123, 1145, 1246, 1282)
GATE = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
REST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'}  {name}{(': ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(name)


def rollout(pillars, start, steps, **kw):
    """教师 + 内部模型闭环；返回 (状态轨迹, 动作, 最小净空, 是否撞)。"""
    boxes = tk._boxes(pillars)
    state, prev = start, None
    states, acts, min_clear = [state], [], math.inf
    for _ in range(steps):
        a = tk.plan_action(state[0], state[1], math.degrees(state[2]), pillars,
                           vx=state[3], vy=state[4], yaw_rate=state[5], prev_action=prev, **kw)
        state, _, _ = tk.primitive(state, a)
        c = tk._box_dist(state[0], state[1], boxes)
        min_clear = min(min_clear, c)
        states.append(state)
        acts.append(a)
        prev = a
        if c < tk.ENVELOPE_R:
            return states, acts, min_clear, True
    return states, acts, min_clear, False


def main():
    # --- 参数来源
    check("e1_mini.json 包络半径 0.102", abs(tk.ENVELOPE_R - 0.102) < 1e-9, f"{tk.ENVELOPE_R}")
    check("巡航 0.3 m/s、偏航 0.6 rad/s、延迟 0.10 s",
          (tk.V_FWD, tk.YAW_RATE, tk.LATENCY_S) == (0.3, 0.6, 0.10))

    # --- 原语物理（对照 E1 SITL 实测）
    s, _, _ = tk.primitive(REST, FORWARD_SLOW)
    check("FORWARD 一窗位移 0.15–0.30 m、末速 ≥0.25", 0.15 <= s[0] <= 0.30 and s[3] >= 0.25,
          f"dx={s[0]:.3f} v={s[3]:.3f}")
    s, _, _ = tk.advance((0.0, 0.0, 0.0, tk.V_FWD, 0.0, 0.0), SAFE_NEUTRAL, 3.0)
    check("SAFE 刹停滑行 0.118±0.012 m", abs(s[0] - 0.118) < 0.012, f"dx={s[0]:.3f}")
    s, _, _ = tk.advance(REST, YAW_LEFT, 0.21)
    check("YAW 0.21 s 到 ≥90% 偏航率", s[5] >= 0.9 * tk.YAW_RATE, f"w={s[5]:.3f}")
    s, _, _ = tk.primitive(REST, YAW_LEFT)
    check("YAW_LEFT 一窗转 +25–36°", 25.0 <= math.degrees(s[2]) <= 36.0, f"{math.degrees(s[2]):.1f}°")
    s, _, _ = tk.primitive(REST, YAW_RIGHT)
    check("YAW_RIGHT 一窗转 -25–-36°", -36.0 <= math.degrees(s[2]) <= -25.0, f"{math.degrees(s[2]):.1f}°")
    # 动量世界系：正以 0.3 m/s 向 +x 飞，左转一窗，仍有 +x 位移且几乎没有 +y 位移
    s, _, _ = tk.primitive((0.0, 0.0, 0.0, tk.V_FWD, 0.0, 0.0), YAW_LEFT)
    check("偏航时动量不随机头转", s[0] > 0.08 and abs(s[1]) < 0.02, f"dx={s[0]:.3f} dy={s[1]:.3f}")

    # --- 无柱子：对准就前进，偏航就纠正
    check("无柱子对准 -> FORWARD", tk.plan_action(0.0, 0.0, 0.0, []) == FORWARD_SLOW)
    check("无柱子偏左 40° -> YAW_RIGHT", tk.plan_action(0.0, 0.0, 40.0, []) == YAW_RIGHT)
    check("无柱子偏右 40° -> YAW_LEFT", tk.plan_action(0.0, 0.0, -40.0, []) == YAW_LEFT)

    # --- 门柱正前方：闭环绕过，不撞，提前转，侧向让够
    states, acts, min_clear, hit = rollout(GATE, (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0), 18)
    first_turn = next((i for i, a in enumerate(acts) if a in (YAW_LEFT, YAW_RIGHT)), None)
    x_turn = states[first_turn][0] if first_turn is not None else None
    y_at_pillar = next((abs(st[1]) for st in states if st[0] >= 1.5), 0.0)
    check("门柱：不撞且最小净空 ≥0.35 m", not hit and min_clear >= 0.35, f"min={min_clear:.2f}")
    check("门柱：x<0.8 m 前已开始转", x_turn is not None and x_turn < 0.8,
          f"x_turn={x_turn} acts={''.join(map(str, acts))}")
    check("门柱：经过柱时 |y| ≥0.4 m", y_at_pillar >= 0.4, f"|y|={y_at_pillar:.2f}")
    check("门柱：绕过后到 x≥3", states[-1][0] >= 3.0, f"x_end={states[-1][0]:.2f}")
    check("门柱：无左右摆", not any({a, b} == {YAW_LEFT, YAW_RIGHT} for a, b in zip(acts, acts[1:])))

    # --- 高速逼近柱面：FORWARD 会撞，教师不能再给 FORWARD
    got = tk.plan_action(0.85, 0.0, 0.0, GATE, vx=0.3)
    check("0.45 m 外 0.3 m/s 逼近柱面 -> 非 FORWARD", got != FORWARD_SLOW, tk.ACTIONS[got])
    # --- 必撞：贴柱面 0.12 m 且 0.3 m/s 冲向它，悬停也停不住 -> SAFE（fail closed）
    got = tk.plan_action(1.18, 0.0, 0.0, GATE, vx=0.3)
    check("必撞状态 -> SAFE", got == SAFE_NEUTRAL, tk.ACTIONS[got])
    # --- 漂进余量但还没撞：要能脱困（非 SAFE）
    got = tk.plan_action(1.5, 0.35, 90.0, GATE)
    check("漂进规划余量(0.15 m)朝外 -> 非 SAFE 脱困", got != SAFE_NEUTRAL, tk.ACTIONS[got])

    # --- 停靠语义（l1 waypoint）
    wp = {"x": 2.0, "y": 0.0}
    check("停靠：已在半径内 -> SAFE",
          tk.plan_action(1.7, 0.1, 0.0, [], waypoint=wp, stop_at_waypoint=True) == SAFE_NEUTRAL)
    check("停靠：半径外 1 m -> 非 SAFE",
          tk.plan_action(1.0, 0.0, 0.0, [], waypoint=wp, stop_at_waypoint=True) != SAFE_NEUTRAL)
    states, acts, _, hit = rollout([], (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0), 14,
                                   waypoint=wp, stop_at_waypoint=True, stop_radius=0.5)
    end = states[-1]
    check("停靠：闭环停在圈里且基本停住",
          math.hypot(end[0] - 2.0, end[1]) <= 0.5 and math.hypot(end[3], end[4]) < 0.05
          and acts[-1] == SAFE_NEUTRAL,
          f"end=({end[0]:.2f},{end[1]:.2f}) v={math.hypot(end[3], end[4]):.3f} acts={''.join(map(str, acts))}")
    # 非停靠 waypoint：到圈不停
    check("非停靠 waypoint 到圈 -> 继续 FORWARD",
          tk.plan_action(1.8, 0.0, 0.0, [], waypoint=wp) == FORWARD_SLOW)

    # --- 20 个 test seed：出生点首步非 SAFE；闭环 40 步到 x≥5.5 不撞
    t_max, worst = 0.0, (math.inf, None)
    for seed in TEST_SEEDS:
        pillars, _, _ = scene_gen.sample_pillars(seed)
        t0 = time.perf_counter()
        a0 = tk.plan_action(scene_gen.SPAWN_X, 0.0, 0.0, pillars)
        t_first = time.perf_counter() - t0
        check(f"seed {seed} 出生点首步非 SAFE", a0 != SAFE_NEUTRAL, tk.ACTIONS[a0])
        states, acts, min_clear, hit = rollout(pillars, (scene_gen.SPAWN_X, 0.0, 0.0, 0.0, 0.0, 0.0), 40)
        t0 = time.perf_counter()
        tk.plan_action(states[3][0], states[3][1], math.degrees(states[3][2]), pillars,
                       vx=states[3][3], vy=states[3][4])
        t_max = max(t_max, time.perf_counter() - t0)
        worst = min(worst, (min_clear, seed))
        check(f"seed {seed} 闭环不撞且 x≥5.5", not hit and states[-1][0] >= 5.5,
              f"x_end={states[-1][0]:.2f} min={min_clear:.2f} first={t_first:.2f}s")
    check("热缓存单步 <0.5 s", t_max < 0.5, f"max={t_max:.3f}s")
    check("20 seed 最小净空 ≥0.30 m", worst[0] >= 0.30, f"min={worst[0]:.2f} @seed {worst[1]}")

    # --- 确定性：同输入同输出
    pillars, _, _ = scene_gen.sample_pillars(485)
    outs = {tk.plan_action(0.6, 0.2, 5.0, pillars, vx=0.2, vy=0.02) for _ in range(3)}
    check("确定性", len(outs) == 1)
    seq = tk.plan_sequence(scene_gen.SPAWN_X, 0.0, 0.0, pillars)
    check("plan_sequence 首动作 == plan_action",
          seq and seq[0] == tk.plan_action(scene_gen.SPAWN_X, 0.0, 0.0, pillars))
    print("test_teacher_kino: 全部通过")


if __name__ == "__main__":
    main()
