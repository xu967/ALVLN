#!/usr/bin/env python3
"""bearing-guard-3dgs 驱动：每个场景只加载一次（碰撞代理 / 渲染器 / 占据 A* / PLY 指纹），依次跑 config.json 里的臂 × 路线 × K 集。

基于 11-09 的 xeval_3dgs_driver.py（原因同它：每次调 eval_3dgs / collect_3dgs 都要重读 1.2–1.5 GB 的 PLY 并重建碰撞代理，
一次约 200 s，而一集只要 4–12 s）。逐集调用的仍是同一个 collect_3dgs.run_episode，参数照 eval_3dgs.main：
stop_on_collision=True、split="test"、split_policy="diag-eval-fixed-test"（评测产物，preprocess 默认拒收；缺省写死，
config 里 split_from_identity = true 才改成取场景身份里按哈希得到的 split，见下面 S1a 一条）。
与 11-09 的差别：
  1. 臂 = 底座（forward / student / teacher）+ 可选的 oracle 护栏（route = 路线前视点方位，goal = waypoint 方位），
     护栏用 sim/3dgs/guard.py，策略对象包一层即可，run_episode 不动；
  2. 每集落 metrics.json（SR 两口径、碰撞、超时、SPL、护栏统计、出生位姿）和 guard.jsonl（逐步方位误差与是否改写），
     汇总只读这些文件，不必再加载场景；
  3. 逐帧 PNG 每集约 4 MB，960 集占 4 GB，超出本任务 3 GB 配额：每集跑完只留首帧 rgb_00000.png，其余删掉
     （actions / oracle / trace / manifest 都在，轨迹可复原）。
每个 (臂, 场景, 路线) 结束写 <OUT>/<臂>/<场景>/r<N>/done.json，再起时据此跳过；半成品改名 *.incomplete-<时刻> 后重跑。
用法：cd <WT> && <3dgs-venv python> driver.py <OUT> <场景> [臂名 ...]     不给臂名 = config.json 里全部（顺序即优先级）
可选的 config.json 键（缺省时行为与原版逐位一致，2026-11-10-sage3d-pilot 共用本驱动）：
  routes    列表 = 所有场景用同一批路线（原写法）；字典 {场景: [路线下标 ...]} = 每个场景各自的路线（场景没有路线则直接退出，不加载）
  step_cap  每路线步数封顶：steps = min(step_cap, ceil(max_steps × step_frac) + step_pad)；缺省不封顶
  teacher_tol_deg  教师航向容差（度），传给 OccAstarTeacher(tol_deg=…)，显式给了就优先于 hold 推导；缺省不传 tol_deg，教师按臂的生效 hold
                   推导（OccAstarTeacher(hold_s=hold)，planner.yaw_tol_deg：hold 1.0 → 18.9°）。2026-11-11 之前缺省是 12°（planner.YAW_TOL_DEG），
                   hold 1.0 下复现旧教师结果要在 config 里显式写 "teacher_tol_deg": 12.0
2026-11-10-rule-oracle-3dgs 再加的可选键（同样缺省时行为与原版逐位一致，test_driver_rule.py 锁定）：
  臂 base = rule-occ / rule-depth   规则层臂（sim/3dgs/rule_oracle.py，用到时才导入）：列距离分别取占据真值 / 渲染深度；
                                    臂的 guard 在这里是航向参照：route = 路线前视点方位，goal = waypoint 方位（其它值启动时就报错，早于加载场景）
  臂 rule_params / 全局 rule_params  覆盖 rule_policy.RULE_V1 的数值（r_safe / d_look / yaw_step_deg …），臂的盖过全局的
  臂 hold         这个臂的动作保持时长（秒），缺省用全局 hold；done.json 里的 hold 记生效值
  臂 step_mult    步数预算乘数（整数），在 step_cap 封顶之后再乘：hold 0.5 的臂同样的仿真时长要 2 倍的步数
  drop_trace      true = 每集跑完删 trace.jsonl（只有视频渲染要它，每集 0.1–0.2 MB）；actions / oracle 里的位姿足够复原轨迹
  规则臂每集多落 rule.jsonl（逐步方位误差 / n / 条目 / 两路列距离），metrics.json 里 guard = None、另有 rule 汇总
2026-11-11-rule-v2-3dgs（任务卡 K3）再加的可选键（缺省时同样与原版逐位一致，test_driver_v2.py 与 K1 的 test_driver_rule.py 锁定）：
  臂 rule_version  "v1" | "v2"，只给规则层臂：传给 RuleOraclePolicy(rule_version=…)；不给 = 不传这个参数（等于 v1，旧臂字节不变）。
                   v2 的 rule_params 键是 rule_policy.RULE_V2 的（levels / order / mem_steps / tabu_* / m_max …），由策略构造时按版本校验；
                   给了这个键，metrics.json 的 rule 与 done.json 的 rule 里多记一个 rule_version（没给就没有，旧产物键不变）
2026-11-11-teacher-clearance-3dgs（任务卡 W1-12）再加的两个臂上的可选键（缺省时同样与原版逐位一致，test_driver_clearance.py 锁定）：
  臂 teacher_clearance  "ray" | "swept"，只给 base = teacher 的臂：传给 OccAstarTeacher(clearance=…)；不给 = 不传这个参数（等于 ray）。
                        hold_s 自 K3 起对所有教师臂都传这个臂的生效 hold（见上面 teacher_tol_deg 一条）。
                        swept 要求生效 hold 为 1.0（位移系数只在 hold 1.0 下量过），启动时就校验，早于加载场景
  臂 spawn_two_layer    true | false，传给 run_episode(spawn_two_layer=…)（出生两层判空，要求全局 jitter > 0）；不给 = 不传。
                        给了任一个键，done.json 末尾多记同名键（没给就没有，旧产物的键不变）
2026-11-12 卡 S1a（审核 D1）再加的全局可选键（缺省时同样与原版逐位一致，test_driver_split.py 锁定）：
  split_from_identity  true | false；不给 = 与原版相同，每集写死 split="test"、split_policy="diag-eval-fixed-test"，场景按几何哈希是 train / val 也一样。
                       true：每条路线把身份交给 eval_3dgs.eval_split(identity, True)，用它给的 (split, split_policy) 传给 run_episode：哈希 test 的场景仍是
                       ("test", "diag-eval-fixed-test")，train / val 的场景写真实 split 与 "diag-train-scene"（诊断评测，preprocess 默认拒收）。
                       给了这个键（true 或 false），done.json 末尾多记 split 与 split_policy（生效值）；没给就没有，旧产物的键不变。值不是布尔时启动就报错，早于加载场景
2026-11-12-3dgs-rebaseline（任务卡 S1b）再加的全局可选键（缺省时同样与原版逐位一致，test_driver_collision_model.py 锁定）：
  collision_model  "v1" | "v2"：传给 collision.from_scene(base, collision_model=…)；不给 = 不传（等于 v1，改动前的判法）。
                   v2 = 审核 B 的发现 1、2、3：结构元素按欧氏距离、高度按层带取并、每步沿 SITL trace 折线扫（出生检查随之统一成层带口径，
                   所以 v2 下任何臂的 spawn_two_layer 为 true 都在启动时报错，早于加载场景）。给了这个键，每集 metrics.json 与 done.json
                   末尾多记 collision_model 与 metric_version（v2 带 +coll.v2 后缀；没给就没有，旧产物的键不变）
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

RULE_BASES = {"rule-occ": "occ", "rule-depth": "depth"}     # 规则层臂的 base -> 决策用的列距离来源（rule_oracle.SOURCES）
for _a in ARMS:         # 配置错在加载场景（约 200 s）之前就报，不要跑完前几个臂才发现
    if _a["base"] in RULE_BASES and _a["guard"] not in ("route", "goal"):
        raise ValueError(f"臂 {_a['name']}：规则层臂的 guard 必须是 route 或 goal（航向参照），得到 {_a['guard']!r}")
    if "rule_version" in _a and (_a["base"] not in RULE_BASES or _a["rule_version"] not in ("v1", "v2")):    # 与 rule_oracle.RULE_VERSIONS 同
        raise ValueError(f"臂 {_a['name']}：rule_version 只给规则层臂（{sorted(RULE_BASES)}），值只能是 v1 或 v2，得到 base={_a['base']!r} "
                         f"rule_version={_a['rule_version']!r}")
    if "teacher_clearance" in _a:       # 值域与 planner.CLEARANCE_MODES 同；swept 的 hold 与 planner.SWEPT_HOLD_S 同
        if _a["base"] != "teacher" or _a["teacher_clearance"] not in ("ray", "swept"):
            raise ValueError(f"臂 {_a['name']}：teacher_clearance 只给 base = teacher 的臂，值只能是 ray 或 swept，得到 base={_a['base']!r} "
                             f"teacher_clearance={_a['teacher_clearance']!r}")
        if _a["teacher_clearance"] == "swept" and abs(float(_a.get("hold", CFG["hold"])) - 1.0) > 1e-9:
            raise ValueError(f"臂 {_a['name']}：teacher_clearance=swept 的位移系数只在 hold 1.0 s 下量过，得到 hold={_a.get('hold', CFG['hold'])!r}")
    if "spawn_two_layer" in _a and (not isinstance(_a["spawn_two_layer"], bool) or (_a["spawn_two_layer"] and not CFG["jitter"] > 0)):
        raise ValueError(f"臂 {_a['name']}：spawn_two_layer 要是 true / false，且为 true 时全局 jitter 必须大于 0，得到 "
                         f"spawn_two_layer={_a['spawn_two_layer']!r} jitter={CFG['jitter']!r}")
if "split_from_identity" in CFG and not isinstance(CFG["split_from_identity"], bool):
    raise ValueError(f"split_from_identity 要是 true / false，得到 {CFG['split_from_identity']!r}")


if "collision_model" in CFG:        # 值域与 collision.COLLISION_MODELS 同；同样在加载场景之前报
    if CFG["collision_model"] not in ("v1", "v2"):
        raise ValueError(f"collision_model 只能是 v1 或 v2，得到 {CFG['collision_model']!r}")
    for _a in ARMS:
        if CFG["collision_model"] == "v2" and _a.get("spawn_two_layer") is True:
            raise ValueError(f"臂 {_a['name']}：collision_model=v2 的出生检查已是层带口径（B7），不与 spawn_two_layer=true（W1-12 的两层判空）叠用")
CM_KW = {"collision_model": CFG["collision_model"]} if "collision_model" in CFG else {}      # 没给键就不传：旧配置的调用与原版逐位相同
METRIC_VERSION = "3dgs.pilot.v2+stop.v1"      # 同 teacher-tol / teacher-clearance 的 analyze.py；v2 口径另加 +coll.v2（collision.metric_version）


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


def drop_trace(ep_dir):
    """drop_trace = true 时删 trace.jsonl（run_episode 写的逐物理步轨迹，只有视频渲染读它）；缺省不动。"""
    p = os.path.join(ep_dir, "trace.jsonl")
    if CFG.get("drop_trace") and os.path.exists(p):
        os.remove(p)


def nav_end(m):
    """导航阶段的末步：首次到点或首次碰撞之前（没有则整集）。护栏与规则臂的「_pre」统计共用这一个定义。"""
    return min([x for x in (m["reach_step"], m["collision_step"]) if x is not None] or [m["steps"] - 1])


def count_entries(rows):
    out = {}
    for t in rows:
        out[t["entry"]] = out.get(t["entry"], 0) + 1
    return out


ROUTES = CFG["routes"][SCENE] if isinstance(CFG["routes"], dict) else CFG["routes"]   # KeyError = 字典里漏了这个场景，直接报
if not ROUTES:
    log("该场景没有要跑的路线，退出（不加载场景）")
    sys.exit(0)
path = os.path.join(CFG["scenes_dir"], SCENE, "scene.json")
raw_routes = json.load(open(path, encoding="utf-8"))["routes"]
t0 = time.time()
base = scene_mod.load_scene(path)
occ = coll.from_scene(base, **CM_KW)
renderer = renderer_mod.make_renderer(base, False, None, None, None, "cuda")
occ_planner = planner_mod.OccPlanner(occ)
ply_sha = c3._sha256(base["ply_path"])
params = dyn.CfSitlParams.from_json(None)
log(f"场景加载 {time.time() - t0:.0f}s；dynamics_profile={params.profile()}；臂 {[a['name'] for a in ARMS]}")

rc = 0
for arm in ARMS:
    g = arm["name"]
    hold = arm.get("hold", CFG["hold"])
    step_mult = int(arm.get("step_mult", 1))
    rule_src = RULE_BASES.get(arm["base"])
    rule_params = {**CFG.get("rule_params", {}), **arm.get("rule_params", {})} if rule_src else None
    rule_kw = {"rule_version": arm["rule_version"]} if "rule_version" in arm else {}     # 没给键就不传：旧臂调用与原版逐位相同
    teacher_kw = {"clearance": arm["teacher_clearance"]} if "teacher_clearance" in arm else {}   # 没给键就不传 clearance（等于 ray）；hold_s 由教师构造处对所有教师臂统一传（K3）
    spawn_kw = {"spawn_two_layer": arm["spawn_two_layer"]} if "spawn_two_layer" in arm else {}                   # 同上：没给键就不传给 run_episode
    student = StudentPolicy(arm["ckpt"], CFG["infer_bin"]) if arm["base"] == "student" else None
    try:
        for r in ROUTES:
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
            if CFG.get("step_cap") is not None:
                steps = min(CFG["step_cap"], steps)
            steps *= step_mult      # 封顶之后再乘：hold 0.5 的臂要同样的仿真时长，封顶 100 步变 200 步
            identity = c3.scene_identity(sc, ply_sha256=ply_sha)
            # 缺省与原版相同（写死 test）；split_from_identity = true 才取身份里按几何哈希得到的 split（D1），train / val 的场景写 diag-train-scene
            split, split_policy = (ev.eval_split(identity, True) if CFG.get("split_from_identity")
                                   else ("test", "diag-eval-fixed-test"))
            eps, fails, tr = [], [], time.time()
            teacher = None
            if arm["base"] == "teacher":
                check = planner_mod.route_check(occ_planner, sc["spawn"], sc["waypoint"])
                if not check["feasible"]:
                    fails.append({"episode": None, "error": f"route_check 不可达 {check}"})
                teacher = planner_mod.OccAstarTeacher(
                    occ_planner, sc["waypoint"], hold_s=hold,
                    **({"tol_deg": CFG["teacher_tol_deg"]} if "teacher_tol_deg" in CFG else {}), **teacher_kw)
            for k in range(CFG["k"] if not fails else 0):
                seed = CFG["seed0"] + 100 * r + k
                ep_dir = os.path.join(d, "ep_%05d" % k)
                route = None
                if arm["base"] == "teacher":
                    policy = teacher
                elif rule_src:
                    ro = importlib.import_module("sim.3dgs.rule_oracle")      # 用到规则臂才导入，其它配置不依赖这个文件
                    if arm["guard"] == "route":      # 每集新建：路线在该集第一次调用时从起飞后的实际位姿规划一次
                        route = guard_mod.OracleRouteGuide(occ_planner.plan, wp, CFG["route_lookahead_m"])
                        target = route.target
                    else:
                        target = lambda x, y, wp=wp: wp      # noqa: E731  goal：恒指向 waypoint，不规划
                    policy = ro.RuleOraclePolicy(occ, target, source=rule_src, params=rule_params, **rule_kw)
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
                    man, err = c3.run_episode(sc, ep_dir, renderer, occ, params, policy, steps, hold,
                                              split=split, seed=seed, episode_id=seed, save_depth=False,
                                              stop_on_collision=True, dynamics_path=None,
                                              spawn_jitter=CFG["jitter"], identity=identity,
                                              split_policy=split_policy, dynamics="cf_sitl", **spawn_kw)
                except Exception as e:  # noqa: BLE001 - 单集失败进 done.json，不拖垮整轮
                    fails.append({"episode": k, "seed": seed, "error": f"{type(e).__name__}: {e}"})
                    log(f"[{g}] r{r} ep{k} 失败: {e}")
                    if os.path.isdir(ep_dir):
                        drop_frames(ep_dir)
                        drop_trace(ep_dir)
                    continue
                drop_frames(ep_dir)
                drop_trace(ep_dir)
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
                    end = nav_end(m)
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
                if rule_src:
                    # 规则臂不是护栏（没有 trace 属性，guard 仍是 None）；到点 / 撞之后的步另列（_pre），那才是导航阶段规则各条目的占比。
                    # 绕圈、悬停的尾巴会把「整集」的占比拉向 V1-fwd 以外的条目。
                    pre = [t for t in policy.rows if t["step"] <= nav_end(m)]
                    m["rule"] = {"source": rule_src, "n_steps": len(policy.rows), "entries": count_entries(policy.rows),
                                 "n_steps_pre": len(pre), "entries_pre": count_entries(pre),
                                 "plan_failed": None if route is None else route.plan_failed, "params": policy.params}
                    if rule_kw:
                        m["rule"].update(rule_kw)      # 新键只在给了 rule_version 的臂上追加，旧臂的 rule 汇总键不变
                    with open(os.path.join(ep_dir, "rule.jsonl"), "w", encoding="utf-8") as f:
                        for row in policy.rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if CM_KW:      # 新键追加在最后，给了 collision_model 才有
                    m["collision_model"] = occ.collision_model
                    m["metric_version"] = coll.metric_version(METRIC_VERSION, occ.collision_model)
                with open(os.path.join(ep_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, indent=1)
                eps.append({"episode": k, "seed": seed, "steps": m["steps"], "success_stop": m["success_stop"],
                            "collision": m["collision"], "timeout": m["timeout"], "wall_s": m["wall_s"]})
            done = {"arm": g, "scene": SCENE, "route": r, "steps": steps, "hold": hold,
                    "ckpt": arm["ckpt"], "guard": arm["guard"], "dynamics_profile": params.profile(),
                    "episodes": eps, "failures": fails, "wall_s": round(time.time() - tr, 1)}
            if rule_src:      # 新键只追加在后面，旧臂的 done.json 键与顺序不变；生效的完整参数每集记在 metrics.json 的 rule.params
                done["rule"] = {"source": rule_src, "overrides": rule_params, **rule_kw}
            if step_mult != 1:
                done["step_mult"] = step_mult
            if teacher_kw:      # 新键只追加在后面，给了才有
                done["teacher_clearance"] = teacher_kw["clearance"]
            if spawn_kw:
                done["spawn_two_layer"] = spawn_kw["spawn_two_layer"]
            if "split_from_identity" in CFG:      # 新键只追加在最后，给了才有；记生效的 split / split_policy（不记开关）
                done["split"], done["split_policy"] = split, split_policy
            if CM_KW:
                done["collision_model"] = occ.collision_model
                done["metric_version"] = coll.metric_version(METRIC_VERSION, occ.collision_model)
            with open(os.path.join(d, "done.json"), "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=1)
            log(f"[{g}] r{r} {len(eps)}/{CFG['k']} 集 失败 {len(fails)} 到点 {sum(e['success_stop'] for e in eps)} "
                f"撞 {sum(e['collision'] for e in eps)} 超时 {sum(e['timeout'] for e in eps)} {time.time() - tr:.0f}s")
            rc = 1 if fails else rc
    finally:
        if student is not None:
            student.close()
sys.exit(rc)
