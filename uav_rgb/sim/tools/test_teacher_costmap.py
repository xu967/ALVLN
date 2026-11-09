#!/usr/bin/env python3
"""teacher.py 统一代价模型教师（costmap）自检 + legacy / visible 教师数值回归锁。

合成场景（秒级，不需要数据）：
  python3 sim/tools/test_teacher_costmap.py
数据回归（约 3–5 min；集合不存在时 SKIP）：
  python3 sim/tools/test_teacher_costmap.py --data [--workers 4]
    0417 12000 帧：plan_action_global_tiebreak(legacy) 复算 == action_expert（0 不一致）
    sgvis 10659 帧：lr_tie_margin_steps 复算 == 行内 tie_margin_steps（逐位）且 visible 复算 == action_expert

legacy 锁定值（硬编码）取自 main 2899428 未改动前的 teacher.py（仓库解释器 .venv-tools Python 3.12、E1 base 档）：
抽 astar_path 的栅格 helper、把 TIE_* 收进 _rollout/_q_from 之后，这些数必须逐位不变。
换解释器（如 conda 3.11）时 libm 的 hypot/atan2 可能差 1 ulp（相对 1e-16），_same 对浮点放 1e-14 相对 + 1e-13 绝对容差（margin 是两个 Q 之差，抵消后相对误差放大）——
任何模型改动都在 1e-6 以上，不会漏；整数（动作）仍精确比。
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher as t  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
COLLECTION_0417 = f"{DATA_ROOT}/uav_rgb/e1-pipeline-20261107-0417/collection"
COLLECTION_SGVIS = f"{DATA_ROOT}/uav_rgb/e1-pipeline-20261107-1620-sgvis/collection"

BOX = [0.4, 0.4, 3.0]
TIE = [{"x": 1.5, "y": 0.35, "z": 1.5, "size": BOX}, {"x": 2.4, "y": -0.3, "z": 1.5, "size": BOX}]
ONE = [{"x": 1.0, "y": 0.3, "z": 1.5, "size": BOX}]
GATE = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": BOX}]
# (场景, 位姿) → (lr_tie_margin_steps, Q_L, Q_R, plan_action_global)：main 2899428 原始实现的输出，逐位锁定。
LEGACY_LOCK = [
    (TIE, (0, 0, -10.0), 0.21792723167688166, 21.61629066959411, 21.39836343791723, 2),
    (TIE, (0.3, 0.1, 15.0), 0.028427124746190913, 20.45096679918781, 20.479393923934, 3),
    (TIE, (1.0, -0.2, -40.0), 0.510894556366793, 17.41665222413705, 17.927546780503842, 1),
    (ONE, (0, 0, 0.0), 1.4361178260102463, 15.361601225604149, 13.925483399593903, 3),
    (ONE, (0.5, 0.0, 20.0), 0.0, 40.0, 40.0, 3),
    (GATE, (0, 0, 0.0), 1.1914674371664482, 16.588225099390858, 17.779692536557306, 2),
    (GATE, (1.1, 0.48, 0.0), 28.405887450304572, 11.594112549695428, 40.0, 2),
    (GATE, (0.8, -0.3, -30.0), 0.5662359778554507, 14.091567726620926, 13.525331748765476, 1),
]
ASTAR_LOCK = [  # (载具, 场景, start, goal, clearance) → (路径点数, 路径 JSON 的 sha256 前 16 位)
    ("x3", GATE, (0.0, 0.0), (3.5, 0.0), 0.5, 36, "c1946240014b4673"),
    ("e1_mini", TIE, (0.0, 0.0), (4.4, 0.0), 0.40, 45, "fef1621e5e5a9a13"),
]

passed = 0


def check(name, cond, detail=""):
    global passed
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    passed += 1
    print(f"  ok  {name}")


def _same(got, want):
    """逐元素比：整数 / None 精确；浮点精确或 1e-14 相对 / 1e-13 绝对容差（跨解释器 libm 的 ulp 级差异）。"""
    if len(got) != len(want):
        return False
    for a, b in zip(got, want):
        if a == b:
            continue
        if isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-13):
            continue
        return False
    return True


def synthetic():
    t.set_vehicle("e1_mini", "base")
    try:
        # ---- legacy 锁 ----
        for pil, (x, y, yaw), m_ref, ql_ref, qr_ref, act_ref in LEGACY_LOCK:
            m = t.lr_tie_margin_steps(x, y, yaw, pil)
            ql = t._q_after_yaw(x, y, yaw, t.YAW_LEFT, pil, t.ASTAR_CLEARANCE_M, 0.1, 0.7, 25.0)
            qr = t._q_after_yaw(x, y, yaw, t.YAW_RIGHT, pil, t.ASTAR_CLEARANCE_M, 0.1, 0.7, 25.0)
            act = t.plan_action_global(x, y, yaw, pil)
            check(f"legacy 锁 {len(pil)} 柱 {(x, y, yaw)}", _same((m, ql, qr, act), (m_ref, ql_ref, qr_ref, act_ref)),
                  f"got {(m, ql, qr, act)!r} want {(m_ref, ql_ref, qr_ref, act_ref)!r}")
        leg, m_leg = t.plan_action_global_tiebreak(0.0, 0.0, -10.0, TIE, "legacy")
        vis, m_vis = t.plan_action_global_tiebreak(0.0, 0.0, -10.0, TIE, "visible")
        check("tiebreak legacy/visible 不变", _same((leg, m_leg, vis, m_vis), (2, None, 3, 0.21792723167688166)))
        for veh, pil, s, g, cl, n_ref, sha_ref in ASTAR_LOCK:
            t.set_vehicle(veh)
            path = t.astar_path(pil, s, g, clearance=cl)
            sha = hashlib.sha256(json.dumps(path).encode()).hexdigest()[:16]
            check(f"astar_path 锁 {veh}", (len(path), sha) == (n_ref, sha_ref), f"got {(len(path), sha)}")
        t.set_vehicle("e1_mini", "base")
        x0, y0, nx, ny = t._grid_spec(GATE, 0.1)
        check("_grid_spec 与 astar_path 原公式一致", (x0, y0, nx, ny) == (-1.5, -2.8, 86, 57), f"{(x0, y0, nx, ny)}")

        # ---- 测地场 ----
        goal = (3.5, 0.0)
        f = t.geodesic_field(GATE, goal, t.ASTAR_CLEARANCE_M, 0.1)
        check("geodesic_field 缓存命中返回同一对象", t.geodesic_field(GATE, goal, t.ASTAR_CLEARANCE_M, 0.1) is f)
        check("缓存键含目标", t.geodesic_field(GATE, (3.6, 0.0), t.ASTAR_CLEARANCE_M, 0.1) is not f)
        path = t.astar_path(GATE, (0.0, 0.0), goal, clearance=t.ASTAR_CLEARANCE_M)
        length = sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(path, path[1:]))
        d0, cell0 = f.lookup(0.0, 0.0)
        check("D(起点) == A* 路长（浮点舍入内）", abs(d0 - length) < 1e-6, f"D={d0} A*={length}")
        check("目标格 D=0", f.lookup(*goal)[0] == 0.0)
        di, _ = f.lookup(1.5, 0.0)
        check("柱心格不可行：3×3 邻域也在带内 → inf", math.isinf(di))
        d_band, cell_band = f.lookup(1.5, 0.55)   # 距柱面 0.35 < 0.40，在膨胀带内，邻格 (1.5, 0.65) 可行
        check("膨胀带内取邻格 D + 一跳", not math.isinf(d_band) and cell_band is not None and cell_band != f.cell(1.5, 0.55))
        ddir = f.descent_dir(cell0, 0.7)
        # 正对门柱时最短路先斜着离开 0.6 m 膨胀带，0.7 m 内全是对角格 → 恰好 ±45°（A* 的 _path_carrot 同值）
        check("起点下降方向绕柱（0 < |角| ≤ 45°）", ddir is not None and 0.0 < abs(ddir) <= 45.0, f"{ddir}")
        check("目标格下降方向 None", f.descent_dir(f.cell(*goal), 0.7) is None)
        # 八邻域 octile 距离下「直走」与「斜走」每米降得一样多：若按邻格 D 最小取方向，只要有一点横向偏差就永远选斜格，
        # 开阔走廊里 7 cm 的横向偏差会给出 45° 的胡萝卜（0417 的 2297 帧 legacy FWD 被判成 YAW 的来源）。
        # 应按单位长度的下降率取、同率时先直走（与 astar_path 的阶梯顺序一致），方向 ≈ 真实方位 (atan2(−0.07, 0.7) ≈ −6°)。
        side = [{"x": 3.0, "y": 2.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
        f_open = t.geodesic_field(side, (5.0, 0.0), t.ASTAR_CLEARANCE_M, 0.1)
        dd_open = f_open.descent_dir(f_open.cell(0.0, 0.07), 0.7)
        check("开阔走廊横向偏 7 cm：下降方向 |角| < 10°（不是 45°）", dd_open is not None and abs(dd_open) < 10.0, f"{dd_open}")
        dd_diag = f_open.descent_dir(f_open.cell(0.0, 0.7), 0.7)
        check("横向偏 0.7 m 时先直走：|角| ≤ 10°（斜格留到最后，与 A* 阶梯一致）", dd_diag is not None and abs(dd_diag) <= 10.0, f"{dd_diag}")
        check("真需要斜走时仍给 45°：目标正斜向", abs(abs(t.geodesic_field(side, (1.0, -1.0), t.ASTAR_CLEARANCE_M, 0.1)
                                                    .descent_dir(f_open.cell(0.0, 0.0), 0.7)) - 45.0) < 1e-6)

        # ---- plan_costs ----
        q, info = t.plan_costs(-1.0, 1.2, 0.0, GATE)
        check("开阔处对准目标：FWD 最便宜、SAFE 比 FWD 贵约 1 步",
              min(range(4), key=q.__getitem__) == t.FORWARD_SLOW and 0.8 < q[t.SAFE_NEUTRAL] - q[t.FORWARD_SLOW] < 1.2, f"{q}")
        check("info 带目标与当前 D", info["goal"] == (3.5, 0.0) and info["d_now_m"] > 4.0)
        q60, _ = t.plan_costs(-1.0, 1.2, 60.0, GATE)
        check("航向偏左 60°：右转最便宜", min(range(4), key=q60.__getitem__) == t.YAW_RIGHT, f"{q60}")
        qc, _ = t.plan_costs(1.0, 0.0, 0.0, GATE)   # 距柱面 0.3 m 正对柱：前飞一步机身入柱
        check("正对 0.3 m 柱：FWD = TIE_COLLISION_STEPS", qc[t.FORWARD_SLOW] == t.TIE_COLLISION_STEPS, f"{qc}")
        fg = t.geodesic_field(GATE, (3.5, 0.0), t.ASTAR_CLEARANCE_M, 0.1)
        # 栅格格心是 y0 + j*step 的浮点值，±0.6 两行到柱面的距离一边 0.4000000000000005、一边 0.3999999999999996，
        # 膨胀带因此一侧多一格（astar_path 同一张栅格、同一现象）；左右 D 只差一跳以内
        check("左右 D 差在一跳以内（栅格浮点不对称是 A* 原有现象）",
              abs(fg.lookup(0.3, 0.5)[0] - fg.lookup(0.3, -0.5)[0]) < 0.1 and abs(fg.lookup(0.9, 0.3)[0] - fg.lookup(0.9, -0.3)[0]) < 0.1)
        # 正对门柱沿 x 轴逼近：costmap 比 legacy 晚转（1.5 m 处 FWD 仍让 D 降 0.3 m；legacy 在 1.5 m 就 YAW），
        # 0.5–0.8 m 处转向；Q_L ≠ Q_R 只来自下降方向在等长路径间的确定性取舍（legacy A* heap 顺序同一现象：
        # 锁定值 GATE(0,0,0) 的 |Q_L−Q_R| = 1.19 步）。
        acts = [t.plan_action_costmap(x, 0.0, 0.0, GATE)[0] for x in (0.0, 0.25, 0.5, 0.75)]
        check("正对门柱逼近：1.5 m 处 FWD，之后 YAW（晚于 legacy 一步）", acts[0] == t.FORWARD_SLOW
              and all(a in (t.YAW_LEFT, t.YAW_RIGHT) for a in acts[1:]), f"{acts}")
        q30, _ = t.plan_costs(-1.0, 0.7, 30.0, GATE)
        q30_t0, _ = t.plan_costs(-1.0, 0.7, 30.0, GATE, tol=0.0)
        check("tol 是显式参数：tol=0 的 Q 逐动作 ≥ tol=25 的 Q，且 |err|<25° 的动作严格更贵（≤ 25/27 步）",
              all(b >= a for a, b in zip(q30, q30_t0)) and any(b > a + 1e-9 for a, b in zip(q30, q30_t0))
              and all(b - a <= 25.0 / 27.0 + 1e-9 for a, b in zip(q30, q30_t0)), f"{q30} {q30_t0}")
        check("_q_from：1 步 + 路长/v + 航向罚", t._q_from(0.5, 52.0, 25.0) == 1.0 + 2.0 + 1.0 and t._q_from(0.5, 20.0, 25.0) == 3.0)
        q2, info2 = t.plan_costs(-1.0, 1.2, 0.0, GATE, lookahead_steps=2)
        # 第二步取 SAFE 时 Q2(a) = Q1(a) + 1（上界）；第二步再前进 0.25 m 最多再省 1 步（下界）
        check("两步前瞻：Q1−1 ≤ Q2 ≤ Q1+1 逐动作成立、撞柱仍是 TIE_COLLISION_STEPS",
              all(a - 1.0 - 1e-9 <= b <= a + 1.0 + 1e-9 for a, b in zip(q, q2)) and info2["lookahead_steps"] == 2
              and t.plan_costs(1.0, 0.0, 0.0, GATE, lookahead_steps=2)[0][t.FORWARD_SLOW] == t.TIE_COLLISION_STEPS, f"{q2} vs {q}")
        for bad in ({"pillars": []}, {"pillars": GATE, "lookahead_steps": 0},
                    {"pillars": GATE, "waypoint": {"x": 1.0, "y": 1.0}}):
            try:
                t.plan_costs(0.0, 0.0, 0.0, **bad)
            except ValueError:
                check(f"plan_costs 拒绝 {sorted(bad)}", True)
            else:
                check(f"plan_costs 拒绝 {sorted(bad)}", False)
        # 两种 Q 同尺度：路长部分只差浮点舍入（上面 D(起点) 已验）；航向罚可不同——A* 路径在等长阶梯里先直后斜
        # （heap 顺序），测地场最陡下降先斜后直，0.7 m 胡萝卜方向因此可差 45°（≤ 45/27 ≈ 1.7 步）
        qa = [t._q_after_yaw(-1.0, 1.2, 0.0, a, GATE, t.ASTAR_CLEARANCE_M, 0.1, 0.7, 25.0) for a in (2, 3)]
        check("costmap Q_L/Q_R 与 A* 版 _q_after_yaw 同尺度（|Δ| < 2 步）",
              all(abs(a - b) < 2.0 for a, b in zip(qa, (q[2], q[3]))), f"{qa} vs {q[2:]}")

        # ---- plan_action_costmap ----
        a, inf = t.plan_action_costmap(-1.0, 1.2, 0.0, GATE)
        check("costmap 开阔处 FWD，info.q 有 4 个数、无 fallback", a == t.FORWARD_SLOW and len(inf["q"]) == 4 and inf["fallback"] is None)
        # 平局：用 monkeypatch 构造严格平局（真实几何里 |ΔQ| < 1e-6 极少见）
        orig = t.plan_costs
        t.plan_costs = lambda *a_, **k_: ([9.0, 5.0, 3.0, 3.0], {})
        try:
            a_tie, inf_tie = t.plan_action_costmap(0.0, 0.0, -10.0, TIE)   # (1.5,0.35) 在机头左侧
            a_tie0, inf_tie0 = t.plan_action_costmap(0.0, 0.0, 0.0, GATE)  # 柱正对机头 side=0 → 枚举顺序
            a_beh, _ = t.plan_action_costmap(0.0, 0.0, 180.0, GATE)        # 柱在身后：视锥内无柱 → 枚举顺序
        finally:
            t.plan_costs = orig
        check("平局时按可见柱反侧：柱在左 → YAW_RIGHT；柱正对 / 看不见 → 枚举顺序 YAW_LEFT",
              a_tie == t.YAW_RIGHT and inf_tie["tie"] and a_tie0 == t.YAW_LEFT and inf_tie0["tie"] and a_beh == t.YAW_LEFT)
        boxed = [{"x": 0.0, "y": 0.0, "z": 1.5, "size": [0.2, 0.2, 3.0]}]   # 机身已与柱重叠：四个动作全撞
        a_fb, inf_fb = t.plan_action_costmap(0.05, 0.0, 0.0, boxed)
        check("全撞退回 plan_action 并标 fallback", inf_fb["fallback"] == "all_collide"
              and a_fb == t.plan_action(0.05, 0.0, 0.0, boxed, tol=25.0))
        # 门柱正对、带边 x=0.9：SAFE 有限（邻格可行）但三个运动原语全撞 → 必须 fallback 脱困，不能原地保持
        a_edge, inf_edge = t.plan_action_costmap(0.9, 0.0, 0.0, GATE)
        check("带边三运动全撞：fallback 给出转向而不是 SAFE", inf_edge["fallback"] == "all_collide"
              and a_edge in (t.YAW_LEFT, t.YAW_RIGHT) and inf_edge["q"][t.SAFE_NEUTRAL] < t.TIE_COLLISION_STEPS, f"{inf_edge}")
        a_np, inf_np = t.plan_action_costmap(0.0, 0.0, 0.0, [])
        check("无柱退回 plan_action", a_np == t.FORWARD_SLOW and inf_np["fallback"] == "no_pillars")
        wp = {"x": 2.0, "y": 1.0}
        a_stop, inf_stop = t.plan_action_costmap(2.1, 1.05, 0.0, GATE, waypoint=wp, stop_at_waypoint=True)
        a_go, _ = t.plan_action_costmap(0.0, 0.0, 0.0, GATE, waypoint=wp, stop_at_waypoint=True)
        check("goal-stop：半径内 SAFE，半径外非 SAFE", a_stop == t.SAFE_NEUTRAL and inf_stop["fallback"] == "goal_stop"
              and a_go != t.SAFE_NEUTRAL)
        check("版本常量独立于 TEACHER_VERSION", t.TEACHER_COST_VERSION == "uavrgb.teacher_cost.v1"
              and t.TEACHER_COST_VERSION != t.TEACHER_VERSION)
        # L/R 死锁场景（R2-3DGS）：航向已对准、前向 0.4 m 被堵 → legacy 按 err 符号换向；costmap 直接选代价低的一侧
        a_dl, inf_dl = t.plan_action_costmap(0.8, 0.0, 0.0, GATE)   # 距柱面 0.5 m 正对：FWD 撞、两个 YAW 有限
        check("前向被堵时 costmap 直接选代价低的一侧（非 fallback）", a_dl in (t.YAW_LEFT, t.YAW_RIGHT)
              and inf_dl["fallback"] is None and inf_dl["q"][t.FORWARD_SLOW] == t.TIE_COLLISION_STEPS, f"{inf_dl}")
    finally:
        t.set_vehicle("x3")


# ---------------------------------------------------------------- 数据回归（--data）
def _ep_dirs(root):
    return sorted(d for s in ("train", "val", "test") for d in glob.glob(os.path.join(root, s, "*")) if os.path.isdir(d))


def _recompute(args):
    ep_dir, mode = args
    t.set_vehicle("e1_mini", "base")
    scene = json.load(open(os.path.join(ep_dir, "scene.json")))
    man = json.load(open(os.path.join(ep_dir, "episode_manifest.json")))
    pillars = scene["pillars"]
    n = bad_label = bad_margin = 0
    for line in open(os.path.join(ep_dir, "actions.jsonl")):
        r = json.loads(line)
        pb, yb = r["pose_before"], r["yaw_before"]
        n += 1
        if mode == "legacy":
            want, _ = t.plan_action_global_tiebreak(pb["x"], pb["y"], yb, pillars, "legacy")
            bad_label += want != r["action_expert"]
        else:
            want, margin = t.plan_action_global_tiebreak(pb["x"], pb["y"], yb, pillars, man.get("teacher_tiebreak") or "legacy")
            bad_label += want != r["action_expert"]
            bad_margin += margin != r.get("tie_margin_steps")
    return n, bad_label, bad_margin


def data_regression(workers):
    for name, root, mode in (("0417 legacy", COLLECTION_0417, "legacy"), ("sgvis visible", COLLECTION_SGVIS, "visible")):
        dirs = _ep_dirs(root)
        if not dirs:
            print(f"  SKIP {name}: {root} 不存在")
            continue
        with Pool(workers) as pool:
            res = pool.map(_recompute, [(d, mode) for d in dirs], chunksize=4)
        n = sum(r[0] for r in res)
        bad_label = sum(r[1] for r in res)
        bad_margin = sum(r[2] for r in res)
        check(f"{name}: {len(dirs)} 集 {n} 帧，标签不一致 {bad_label}，tie_margin 不一致 {bad_margin}",
              bad_label == 0 and bad_margin == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="另跑 0417 / sgvis 数据回归（约 3–5 min）")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    synthetic()
    if args.data:
        data_regression(args.workers)
    print(f"test_teacher_costmap: PASS ({passed} 项)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
