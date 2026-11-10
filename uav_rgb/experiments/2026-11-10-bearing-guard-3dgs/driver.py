#!/usr/bin/env python3
"""bearing-guard-3dgs 驱动：每个场景只加载一次（碰撞代理 / 渲染器 / 占据 A* / PLY 指纹），依次跑 config.json 里的臂 × 路线 × K 集。

基于 11-09 的 xeval_3dgs_driver.py（原因同它：每次调 eval_3dgs / collect_3dgs 都要重读 1.2–1.5 GB 的 PLY 并重建碰撞代理，
一次约 200 s，而一集只要 4–12 s）。逐集调用的仍是同一个 collect_3dgs.run_episode，参数照 eval_3dgs.main：
stop_on_collision=True、split="test"、split_policy="diag-eval-fixed-test"（评测产物，preprocess 默认拒收）。
与 11-09 的差别：
  1. 臂 = 底座（forward / student / teacher）+ 可选的 oracle 护栏（route = 路线前视点方位，goal = waypoint 方位），
     护栏用 sim/3dgs/guard.py，策略对象包一层即可，run_episode 不动；
  2. 每集落 metrics.json（SR 两口径、碰撞、超时、SPL、护栏统计、出生位姿）和 guard.jsonl（逐步方位误差与是否改写），
     汇总只读这些文件，不必再加载场景；
  3. 逐帧 PNG 每集约 4 MB，960 集占 4 GB，超出本任务 3 GB 配额：每集跑完只留首帧 rgb_00000.png，其余删掉
     （actions / oracle / trace / manifest 都在，轨迹可复原）。
每个 (臂, 场景, 路线) 结束写 <OUT>/<臂>/<场景>/r<N>/done.json，再起时据此跳过；半成品改名 *.incomplete-<时刻> 后重跑。
用法：cd <WT> && <3dgs-venv python> driver.py <OUT> <场景> [臂名 ...]     不给臂名 = config.json 里全部（顺序即优先级）
"""
import importlib
import json
import math
import os
import shutil
import sys
import time

OUT, SCENE = os.path.abspath(sys.argv[1]), sys.argv[2]
CFG = json.load(open(os.path.join(OUT, "config.json"), encoding="utf-8"))
ARMS = [a for a in CFG["arms"] if not sys.argv[3:] or a["name"] in sys.argv[3:]]
REPO = CFG["repo"]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
scene_mod = importlib.import_module("sim.3dgs.scene")
coll = importlib.import_module("sim.3dgs.collision")
renderer_mod = importlib.import_module("sim.3dgs.renderer")
planner_mod = importlib.import_module("sim.3dgs.planner")
c3 = importlib.import_module("sim.3dgs.collect_3dgs")
ev = importlib.import_module("sim.3dgs.eval_3dgs")
guard_mod = importlib.import_module("sim.3dgs.guard")
dyn = importlib.import_module("sim.3dgs.dynamics")
from student import StudentPolicy  # noqa: E402  sim/tools；常驻 uavrgb_infer 子进程


def log(msg):
    print(time.strftime("%F %T"), f"[{SCENE}]", msg, flush=True)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def finished(d):
    try:
        j = json.load(open(os.path.join(d, "done.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return not j["failures"] and len(j["episodes"]) >= CFG["k"]


def episode_metrics(ep_dir, sc, occ_planner):
    """单集指标（口径同 11-09 summarize.py）：success_stop = 首次进入 waypoint 半径早于首次碰撞（主列）；
    success_v2 = eval_3dgs 的 3dgs.pilot.v2（过点后再撞也算失败）；timeout = 既没到点也没撞；collide_first = 没到点就撞了。
    spl 沿用 eval_3dgs（成功按 v2），spl_stop 把成功换成 success_stop（l_i、p_i 同一套：A* 抽捷径 / 轨迹截到首次到点）。"""
    actions, oracle = jsonl(os.path.join(ep_dir, "actions.jsonl")), jsonl(os.path.join(ep_dir, "oracle.jsonl"))
    wp, rad = sc["waypoint"], ev.reach_radius_of(sc)
    summ = ev.summarize_3dgs_episode(actions, oracle, sc, timed_out=False,
                                     shortest_path_m=ev.route_shortest_path_m(occ_planner, oracle, sc))
    reach = next((i for i, o in enumerate(oracle)
                  if math.hypot(o["pose_gt"]["x"] - wp["x"], o["pose_gt"]["y"] - wp["y"]) <= rad), None)
    hit = next((i for i, a in enumerate(actions) if a.get("collision")), None)
    success_stop = reach is not None and (hit is None or reach < hit)
    l_i, p_i = summ.get("shortest_path_m"), summ.get("path_length_m")
    last = oracle[-1]["pose_gt"]
    return {"success_stop": success_stop, "success_v2": bool(summ["success"]),
            "collision": hit is not None, "collide_first": hit is not None and not success_stop,
            "timeout": reach is None and hit is None, "steps": len(actions),
            "reach_step": reach, "collision_step": hit,
            "spl": summ.get("spl"),
            "spl_stop": (l_i / max(p_i, l_i) if success_stop and l_i and p_i else 0.0) if l_i is not None else None,
            "final_wp_dist_m": round(math.hypot(last["x"] - wp["x"], last["y"] - wp["y"]), 3),
            "min_wp_dist_m": summ.get("waypoint_min_dist"),
            "start_pose": {"x": actions[0]["pose_before"]["x"], "y": actions[0]["pose_before"]["y"],
                           "yaw_deg": actions[0]["yaw_before"]}}


def drop_frames(ep_dir):
    for f in os.listdir(ep_dir):
        if f.startswith("rgb_") and f.endswith(".png") and f != "rgb_00000.png":
            os.remove(os.path.join(ep_dir, f))


path = os.path.join(CFG["scenes_dir"], SCENE, "scene.json")
raw_routes = json.load(open(path, encoding="utf-8"))["routes"]
t0 = time.time()
base = scene_mod.load_scene(path)
occ = coll.from_scene(base)
renderer = renderer_mod.make_renderer(base, False, None, None, None, "cuda")
occ_planner = planner_mod.OccPlanner(occ)
ply_sha = c3._sha256(base["ply_path"])
params = dyn.CfSitlParams.from_json(None)
log(f"场景加载 {time.time() - t0:.0f}s；dynamics_profile={params.profile()}；臂 {[a['name'] for a in ARMS]}")

rc = 0
for arm in ARMS:
    g = arm["name"]
    student = StudentPolicy(arm["ckpt"], CFG["infer_bin"]) if arm["base"] == "student" else None
    try:
        for r in CFG["routes"]:
            d = os.path.join(OUT, g, SCENE, f"r{r}")
            if finished(d):
                log(f"[{g}] r{r} 跳过（done.json 已完成）")
                continue
            if os.path.isdir(d):
                shutil.move(d, d + ".incomplete-" + time.strftime("%H%M%S"))
            os.makedirs(d)
            sc = scene_mod.load_scene(path)
            scene_mod.apply_route(sc, r)
            wp = (float(sc["waypoint"]["x"]), float(sc["waypoint"]["y"]))
            steps = math.ceil(raw_routes[r]["max_steps"] * CFG["step_frac"]) + CFG["step_pad"]
            identity = c3.scene_identity(sc, ply_sha256=ply_sha)
            eps, fails, tr = [], [], time.time()
            teacher = None
            if arm["base"] == "teacher":
                check = planner_mod.route_check(occ_planner, sc["spawn"], sc["waypoint"])
                if not check["feasible"]:
                    fails.append({"episode": None, "error": f"route_check 不可达 {check}"})
                teacher = planner_mod.OccAstarTeacher(occ_planner, sc["waypoint"])
            for k in range(CFG["k"] if not fails else 0):
                seed = CFG["seed0"] + 100 * r + k
                ep_dir = os.path.join(d, "ep_%05d" % k)
                route = None
                if arm["base"] == "teacher":
                    policy = teacher
                else:
                    inner = c3.ForwardPolicy() if arm["base"] == "forward" else c3.StudentAdapter(student)
                    if arm["guard"] == "route":     # 每集新建：路线在该集第一次调用时从起飞后的实际位姿规划一次
                        route = guard_mod.OracleRouteGuide(occ_planner.plan, wp, CFG["route_lookahead_m"])
                        policy = guard_mod.OracleGuardedPolicy(inner, route.target, CFG["guard_threshold_deg"])
                    elif arm["guard"] == "goal":
                        policy = guard_mod.OracleGuardedPolicy(inner, lambda x, y: wp, CFG["guard_threshold_deg"])
                    else:
                        policy = inner
                try:
                    man, err = c3.run_episode(sc, ep_dir, renderer, occ, params, policy, steps, CFG["hold"],
                                              split="test", seed=seed, episode_id=seed, save_depth=False,
                                              stop_on_collision=True, dynamics_path=None,
                                              spawn_jitter=CFG["jitter"], identity=identity,
                                              split_policy="diag-eval-fixed-test", dynamics="cf_sitl")
                except Exception as e:  # noqa: BLE001 - 单集失败进 done.json，不拖垮整轮
                    fails.append({"episode": k, "seed": seed, "error": f"{type(e).__name__}: {e}"})
                    log(f"[{g}] r{r} ep{k} 失败: {e}")
                    if os.path.isdir(ep_dir):
                        drop_frames(ep_dir)
                    continue
                drop_frames(ep_dir)
                if err or not (man.get("validity") or {}).get("physics_verified", False):
                    fails.append({"episode": k, "seed": seed, "error": err or "physics_verified=false",
                                  "dynamics_anomaly": man.get("dynamics_anomaly")})
                    continue
                m = episode_metrics(ep_dir, sc, occ_planner)
                m.update({"arm": g, "scene": SCENE, "route": r, "episode": k, "seed": seed,
                          "stopped_on_goal": man.get("stopped_on_goal"), "wall_s": man["performance"]["wall_s"],
                          "guard": None})
                if hasattr(policy, "trace"):
                    # 学生 / 盲前进到点后不停，会在 waypoint 附近绕圈，护栏在那里几乎每步都在改写：
                    # 另记到点（或撞）之前的统计（_pre），那才是「导航阶段」护栏的介入强度
                    end = min([x for x in (m["reach_step"], m["collision_step"]) if x is not None] or [m["steps"] - 1])
                    pre = [t for t in policy.trace if t["step"] <= end]
                    m["guard"] = {"n_steps": policy.n_steps, "n_triggered": policy.n_triggered,
                                  "n_rewritten": policy.n_rewritten, "n_steps_pre": len(pre),
                                  "n_triggered_pre": sum(t["triggered"] for t in pre),
                                  "n_rewritten_pre": sum(t["rewritten"] for t in pre),
                                  "plan_failed": None if route is None else route.plan_failed}
                if hasattr(policy, "trace"):
                    with open(os.path.join(ep_dir, "guard.jsonl"), "w", encoding="utf-8") as f:
                        for row in policy.trace:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                with open(os.path.join(ep_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, indent=1)
                eps.append({"episode": k, "seed": seed, "steps": m["steps"], "success_stop": m["success_stop"],
                            "collision": m["collision"], "timeout": m["timeout"], "wall_s": m["wall_s"]})
            with open(os.path.join(d, "done.json"), "w", encoding="utf-8") as f:
                json.dump({"arm": g, "scene": SCENE, "route": r, "steps": steps, "hold": CFG["hold"],
                           "ckpt": arm["ckpt"], "guard": arm["guard"], "dynamics_profile": params.profile(),
                           "episodes": eps, "failures": fails, "wall_s": round(time.time() - tr, 1)},
                          f, ensure_ascii=False, indent=1)
            log(f"[{g}] r{r} {len(eps)}/{CFG['k']} 集 失败 {len(fails)} 到点 {sum(e['success_stop'] for e in eps)} "
                f"撞 {sum(e['collision'] for e in eps)} 超时 {sum(e['timeout'] for e in eps)} {time.time() - tr:.0f}s")
            rc = 1 if fails else rc
    finally:
        if student is not None:
            student.close()
sys.exit(rc)
