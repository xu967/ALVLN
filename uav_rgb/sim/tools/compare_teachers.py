#!/usr/bin/env python3
"""compare_teachers：纯 Python 离线对比新旧教师（不起 Gazebo）。

20 个冻结 test seed 的柱阵几何上，用一个简化的 E1 MINI 运动学模型（一阶速度/偏航率
响应 + 指令延迟）闭环执行每个教师，报到达率、碰撞率（按 E1 包络）、步数、动作分布，
并画轨迹 PNG。两个任务：corridor（出生 x=-0.5 飞到 (6,0) 半径 1.5 m）和 l1_stop
（l1.v4 的 waypoint + 0.5 m 停靠，要求停在圈里，不只是碰到圈）。

用法：
  "$UAV_RGB_PYTHON" sim/tools/compare_teachers.py                 # 全量（约 2–4 分钟）
  UAV_RGB_COMPARE_SEEDS=180,206 "$UAV_RGB_PYTHON" sim/tools/compare_teachers.py
守护进程启动（BRIEF §3）：
  setsid nohup "$UAV_RGB_PYTHON" sim/tools/compare_teachers.py > $OUT/run.log 2>&1 < /dev/null &
输出：$OUT/summary.md、summary.json、traj_<task>_<teacher>_<plant>.png
"""
import json
import math
import os
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(TOOLS)), "tools"))
import eval_seeds                                  # noqa: E402  - 冻结评测 seed 的唯一来源（tools/eval_seeds.py）
import landmarks                                   # noqa: E402
import scene_gen                                   # noqa: E402
import teacher                                     # noqa: E402
import teacher_kino                                # noqa: E402
from teacher_predictive import PredictiveTeacher   # noqa: E402

# ---------------------------------------------------------------- 配置区
SEEDS = [int(s) for s in os.environ.get(
    "UAV_RGB_COMPARE_SEEDS", eval_seeds.csv(eval_seeds.EVAL_SEEDS_BASE_20)
).split(",")]                                      # 冻结的 20 个独立几何 test seed（2026-11-09 前是旧的 180,206,… 清单，按 gid 只有 11 个几何）
OUT_DIR = os.environ.get("UAV_RGB_COMPARE_OUT",
                         "/root/rivermind-data/uav_rgb/tmp/agent-fastlab-a/compare")  # 数据盘，不写 /tmp
STEP_S = 1.0           # hold 窗口（BRIEF §4）
ENVELOPE_R = teacher_kino.ENVELOPE_R   # 0.102 m，碰撞判据：机体中心到柱面 < 包络半径
V_FWD = teacher_kino.V_FWD             # 0.3 m/s
YAW_RATE = teacher_kino.YAW_RATE       # 0.6 rad/s
PLANT_DT = 0.01        # 被控对象积分步长（欧拉）；比教师内部模型的 0.05 细，避免同一积分器自证
STOP_SPEED = 0.05      # l1_stop 的「停住」判据（m/s）：rl_teacher RewardConfig.goal_speed 是 0.12，取更严
LOST_Y = 3.5           # |y| 超过走廊边界 3 m 再加 0.5 m 视为迷航
# 被控对象的响应常数：与 teacher_kino 配置区同一组 SITL 实测（0.72 s 到 90%、滑行 0.118 m、
# 偏航 0.21 s 到 90%）；两处故意各写一份，换实测值时一起改。
PLANTS = {
    # 名字: (tau_v_up, tau_v_down, tau_yaw, latency_s)
    "nominal":  (0.313, 0.393, 0.091, 0.10),  # 名义 E1
    "sluggish": (0.47, 0.59, 0.14, 0.20),     # 失配：τ×1.5、延迟 ×2，看教师对模型误差的鲁棒性
    "heavy":    (0.78, 0.98, 0.23, 0.30),     # 失配：τ×2.5、延迟 ×3（滑行 0.29 m，接近 cf_stab 的 0.34）
}
TASKS = {
    # corridor：默认出生点，飞到最后一根柱之外；到圈即算到达（旧 eval 的 reach-ever 口径）
    "corridor": {"goal": (6.0, 0.0), "radius": 1.5, "stop": False, "max_steps": 60},
    # l1_stop：l1.v4 随机出生 + 柱外侧 waypoint；必须停在 0.5 m 圈里（SAFE 且 |v|<STOP_SPEED）
    "l1_stop": {"goal": None, "radius": 0.5, "stop": True, "max_steps": 40},
}
_only = os.environ.get("UAV_RGB_COMPARE_TASKS")   # 例：corridor 或 corridor,l1_stop
if _only:
    TASKS = {k: v for k, v in TASKS.items() if k in _only.split(",")}
# 对比的教师。旧教师用 X3 几何常量原样跑（不改参数，这就是「直接换载具」的效果）；
# local_x3 不支持 waypoint，只跑 corridor。
TEACHERS = ("astar_x3", "local_x3", "predictive_cfstab", "kino_e1")


def box_dist(x, y, pillars):
    best = math.inf
    for p in pillars:
        hx, hy = (v / 2 for v in p["size"][:2])
        dx, dy = abs(x - p["x"]) - hx, abs(y - p["y"]) - hy
        d = math.hypot(max(dx, 0.0), max(dy, 0.0)) if (dx > 0 or dy > 0) else max(dx, dy)
        best = min(best, d)
    return best


class Plant:
    """简化 E1：世界系水平速度一阶跟踪机体前向设定点，偏航率一阶跟踪设定点，指令带延迟。

    YAW 动作的水平设定点为 0（原地偏航），所以转头时在刹车，动量方向不随机头转。
    """

    def __init__(self, cfg, spawn):
        self.tau_up, self.tau_down, self.tau_yaw, self.latency = cfg
        self.x, self.y, self.yaw = spawn["x"], spawn["y"], math.radians(spawn["yaw_deg"])
        self.vx = self.vy = self.w = 0.0

    def speed(self):
        return math.hypot(self.vx, self.vy)

    def run_step(self, action, pillars, goal, radius):
        """执行一个决策窗口：前 latency 秒上一条已过期（悬停），之后 action 生效到窗口末。
        返回 (途中最小净空, 是否撞, 途中是否进过圈, 走过的路程)。"""
        min_clear, hit, touched, path = math.inf, False, False, 0.0
        t = 0.0
        while t < STEP_S - 1e-9:
            act = action if t >= self.latency else teacher.SAFE_NEUTRAL
            v_cmd = V_FWD if act == teacher.FORWARD_SLOW else 0.0
            w_cmd = (YAW_RATE if act == teacher.YAW_LEFT
                     else -YAW_RATE if act == teacher.YAW_RIGHT else 0.0)
            self.w += (w_cmd - self.w) * PLANT_DT / self.tau_yaw
            self.yaw = (self.yaw + self.w * PLANT_DT + math.pi) % (2 * math.pi) - math.pi
            tau = self.tau_up if v_cmd > 0 else self.tau_down
            tx, ty = v_cmd * math.cos(self.yaw), v_cmd * math.sin(self.yaw)
            self.vx += (tx - self.vx) * PLANT_DT / tau
            self.vy += (ty - self.vy) * PLANT_DT / tau
            self.x += self.vx * PLANT_DT
            self.y += self.vy * PLANT_DT
            path += self.speed() * PLANT_DT
            t += PLANT_DT
            c = box_dist(self.x, self.y, pillars)
            min_clear = min(min_clear, c)
            if c < ENVELOPE_R:
                hit = True
                break
            if math.hypot(self.x - goal[0], self.y - goal[1]) <= radius:
                touched = True
        return min_clear, hit, touched, path


def make_policy(name, pillars, goal, task):
    """返回 act(plant, prev_action) -> 动作。旧教师只拿位姿；新教师还拿速度/偏航率。"""
    wp = {"x": goal[0], "y": goal[1]}
    if name == "astar_x3":
        kw = dict(waypoint=wp, stop_at_waypoint=True, stop_radius=task["radius"]) if task["stop"] else {}
        return lambda pl, prev: teacher.plan_action_global(pl.x, pl.y, math.degrees(pl.yaw), pillars, **kw)
    if name == "local_x3":
        return lambda pl, prev: teacher.plan_action(pl.x, pl.y, math.degrees(pl.yaw), pillars)
    if name == "predictive_cfstab":
        pt = PredictiveTeacher(pillars, wp)
        return lambda pl, prev: pt.act(pl.x, pl.y, math.degrees(pl.yaw), pl.vx, pl.vy,
                                       active=prev if prev is not None else 0, latency=0.15)
    if name == "kino_e1":
        kw = dict(waypoint=wp, stop_at_waypoint=True, stop_radius=task["radius"]) if task["stop"] else {}
        return lambda pl, prev: teacher_kino.plan_action(
            pl.x, pl.y, math.degrees(pl.yaw), pillars, vx=pl.vx, vy=pl.vy,
            yaw_rate=pl.w, prev_action=prev, **kw)
    raise ValueError(name)


def scenario(seed, task_name, task):
    pillars, _, _ = scene_gen.sample_pillars(seed)
    if task_name == "corridor":
        return pillars, scene_gen.make_spawn(pillars, seed), task["goal"]
    spawn = scene_gen.make_spawn(pillars, seed, "random")
    _, _, _, wp = landmarks.assign_landmarks(pillars, seed, spawn)
    return pillars, spawn, (float(wp["x"]), float(wp["y"]))


def run_episode(name, plant_cfg, seed, task_name, task):
    pillars, spawn, goal = scenario(seed, task_name, task)
    plant = Plant(plant_cfg, spawn)
    act = make_policy(name, pillars, goal, task)
    traj = [(plant.x, plant.y)]
    actions, prev = [], None
    min_clear, outcome, path, plan_s, touched_ever = math.inf, "timeout", 0.0, 0.0, False
    for _ in range(task["max_steps"]):
        t0 = time.perf_counter()
        a = int(act(plant, prev))
        plan_s += time.perf_counter() - t0
        actions.append(a)
        c, hit, touched, d = plant.run_step(a, pillars, goal, task["radius"])
        traj.append((plant.x, plant.y))
        min_clear, path, prev = min(min_clear, c), path + d, a
        touched_ever = touched_ever or touched
        if hit:
            outcome = "collision"
            break
        inside = math.hypot(plant.x - goal[0], plant.y - goal[1]) <= task["radius"]
        if task["stop"]:
            if inside and a == teacher.SAFE_NEUTRAL and plant.speed() < STOP_SPEED:
                outcome = "reached"
                break
        elif touched:
            outcome = "reached"
            break
        if abs(plant.y) > LOST_Y or plant.x < scene_gen.SPAWN_X - 1.0:
            outcome = "lost"
            break
    # 左右换向次数：相邻决策一个 YAW_LEFT 一个 YAW_RIGHT 算一次摆振
    flips = sum(1 for a, b in zip(actions, actions[1:])
                if {a, b} == {teacher.YAW_LEFT, teacher.YAW_RIGHT})
    return {"seed": seed, "outcome": outcome, "touched": touched_ever, "steps": len(actions),
            "min_clear_m": min_clear, "path_m": path, "actions": actions, "flips": flips,
            "final_goal_dist_m": math.hypot(plant.x - goal[0], plant.y - goal[1]),
            "plan_ms_per_step": 1000.0 * plan_s / max(1, len(actions)), "traj": traj,
            "pillars": pillars, "goal": goal, "radius": task["radius"]}


def summarize(eps):
    n = len(eps)
    reached = [e for e in eps if e["outcome"] == "reached"]
    counts = [0, 0, 0, 0]
    for e in eps:
        for a in e["actions"]:
            counts[a] += 1
    total = max(1, sum(counts))
    return {
        "n": n,
        "reach_rate": len(reached) / n,
        "touch_rate": sum(e["touched"] for e in eps) / n,
        "collision_rate": sum(e["outcome"] == "collision" for e in eps) / n,
        "timeout_rate": sum(e["outcome"] in ("timeout", "lost") for e in eps) / n,
        "steps_mean_reached": (sum(e["steps"] for e in reached) / len(reached)) if reached else None,
        "path_m_mean_reached": (sum(e["path_m"] for e in reached) / len(reached)) if reached else None,
        "min_clear_mean_m": sum(e["min_clear_m"] for e in eps) / n,
        "flips_total": sum(e["flips"] for e in eps),
        "action_dist": {teacher.ACTIONS[i]: counts[i] / total for i in range(4)},
        "plan_ms_per_step": sum(e["plan_ms_per_step"] for e in eps) / n,
    }


def plot(title, eps, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    cols = 5
    rows = math.ceil(len(eps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    colors = {"reached": "tab:green", "collision": "tab:red", "timeout": "tab:orange",
              "lost": "tab:purple"}
    act_colors = ("0.5", "tab:blue", "tab:green", "tab:red")   # SAFE/FORWARD/LEFT/RIGHT
    for ax, e in zip(axes.flat, eps):
        for p in e["pillars"]:
            ax.add_patch(Rectangle((p["x"] - 0.2, p["y"] - 0.2), 0.4, 0.4, color="0.3"))
        ax.add_patch(Circle(e["goal"], e["radius"], fill=False, ls="--", color="0.6"))
        xs, ys = zip(*e["traj"])
        ax.plot(xs, ys, "-", color=colors[e["outcome"]], lw=1.2)
        for (x, y), a in zip(e["traj"][1:], e["actions"]):
            ax.plot(x, y, "o", ms=2.5, color=act_colors[a])
        ax.add_patch(Circle(e["traj"][-1], ENVELOPE_R, fill=False, color=colors[e["outcome"]]))
        ax.set_title(f"seed {e['seed']}: {e['outcome']} {e['steps']}步 "
                     f"min {e['min_clear_m']:.2f} m", fontsize=9)
        ax.set_xlim(-1.0, 7.5)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(eps):]:
        ax.axis("off")
    fig.suptitle(f"{title}  (点色: 灰SAFE 蓝FORWARD 绿LEFT 红RIGHT)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    lines = ["# 新旧教师离线对比（E1 简化运动学闭环）", "",
             f"seeds={SEEDS}", f"envelope_r={ENVELOPE_R} step_s={STEP_S} stop_speed={STOP_SPEED}", ""]
    for task_name, task in TASKS.items():
        lines += [f"## 任务 `{task_name}`：radius={task['radius']} stop={task['stop']} "
                  f"max_steps={task['max_steps']}", ""]
        for plant_name, cfg in PLANTS.items():
            lines += [f"### plant `{plant_name}` tau_up/down/yaw/latency={cfg}", "",
                      "| teacher | 到达率 | 碰到圈 | 碰撞率 | 超时/迷航 | 到达均步数 | 到达均路程 m | "
                      "均最小净空 m | 左右换向 | SAFE/FWD/LEFT/RIGHT | 规划 ms/步 |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for name in TEACHERS:
                if name == "local_x3" and task["stop"]:
                    continue
                t0 = time.time()
                eps = [run_episode(name, cfg, s, task_name, task) for s in SEEDS]
                summ = summarize(eps)
                key = f"{task_name}/{name}@{plant_name}"
                results[key] = {
                    "summary": summ,
                    "episodes": [{k: v for k, v in e.items() if k not in ("traj", "pillars")}
                                 for e in eps]}
                png = os.path.join(OUT_DIR, f"traj_{task_name}_{name}_{plant_name}.png")
                plot(f"{task_name}: {name} on E1 plant '{plant_name}'", eps, png)
                ad = summ["action_dist"]
                lines.append(
                    f"| {name} | {summ['reach_rate']:.2f} | {summ['touch_rate']:.2f} | "
                    f"{summ['collision_rate']:.2f} | {summ['timeout_rate']:.2f} | "
                    f"{_fmt(summ['steps_mean_reached'])} | {_fmt(summ['path_m_mean_reached'], 2)} | "
                    f"{summ['min_clear_mean_m']:.2f} | {summ['flips_total']} | "
                    f"{ad['SAFE_NEUTRAL']:.2f}/{ad['FORWARD_SLOW']:.2f}/{ad['YAW_LEFT']:.2f}/{ad['YAW_RIGHT']:.2f} | "
                    f"{summ['plan_ms_per_step']:.1f} |")
                print(f"{key}: reach {summ['reach_rate']:.2f} coll {summ['collision_rate']:.2f} "
                      f"steps {_fmt(summ['steps_mean_reached'])} ({time.time() - t0:.1f}s) -> {png}",
                      flush=True)
            lines.append("")
            for name in TEACHERS:
                key = f"{task_name}/{name}@{plant_name}"
                if key not in results:
                    continue
                bad = [f"{e['seed']}:{e['outcome']}@{e['steps']}(d={e['final_goal_dist_m']:.2f})"
                       for e in results[key]["episodes"] if e["outcome"] != "reached"]
                lines.append(f"- {key} 未到达: {', '.join(bad) if bad else '无'}")
            lines.append("")
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
