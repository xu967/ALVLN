#!/usr/bin/env python3
"""干跑用桩驱动（不碰 GPU、不加载场景、不渲染）：命令行与真驱动相同（<OUT_DIR> <场景>），读 <OUT_DIR>/config.json，
对每个 (臂, 路线, 集) 写 summarize.py / select_variant.py / check_k1_match.py 读的全部文件（metrics.json、done.json、actions.jsonl、rule.jsonl），
已评完的路线（done.json 无失败且集数够）跳过，与真驱动的续跑语义一致。数字是伪造的，只用来验流水线接线，不是实验结果。

结局模型（确定性，只由 场景 / 路线 / seed / 臂类型决定）：
  同一个 (场景, 路线, seed) 用同一个均匀数 u 判到点，臂的成功率阈值：forward 0.45，v1 规则臂 0.62，rule-depth 减 0.04，
  v2 臂 = v1 + 0.15 + 0.02·级数（STUB_NO_GAIN=1 时 = v1，结局逐集与 v1 相同，用来走「闸门不过」）。
  v1 臂的结局只与 (场景, 路线, seed) 有关，与臂名无关，所以 `--make-k1` 造的假 K1 与正式 40 条上的 v1 重跑逐集一致；
  STUB_PERTURB_V1=1 时 formal40 类里 v1 的第一集 success_stop 取反（走 F0 对拍不一致）；STUB_FAIL_SCENE=<场景> 时该场景的驱动进程直接非零退出（走「有任务失败」），再给 STUB_FAIL_CLASS=<类别> 就只在该类别（dev / holdout / formal40 / outdoor）失败。
`--make-k1 <根目录> <试点路线 json> <室内场景…> -- <室外场景…>`：造一个假 K1 目录（formal/indoor、formal/outdoor 的 config.json 与 rule-occ+route 臂的结果，dev/indoor/routes_dev.json 由调用方放）。
"""
import hashlib
import json
import os
import shutil
import sys
import time

FWD, YAW_LEFT, YAW_RIGHT = 1, 2, 3
P_V1, P_FWD, GAIN0, GAIN_PER_LEVEL, DEPTH_PENALTY = 0.62, 0.45, 0.15, 0.02, 0.04


def u01(salt, scene, route, seed):
    h = hashlib.sha256(f"{salt}|{scene}|{route}|{seed}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def arm_p(arm):
    base = arm["base"]
    if base == "forward":
        return P_FWD
    p = P_V1
    if arm.get("rule_version") == "v2" and not os.environ.get("STUB_NO_GAIN"):
        p += GAIN0 + GAIN_PER_LEVEL * len(arm["rule_params"]["levels"])
    if base == "rule-depth":
        p -= DEPTH_PENALTY
    return p


def episode(arm, scene, route, k, seed, hold, cls):
    """返回 (metrics 的数字部分, rule.jsonl 行, actions.jsonl 行)。"""
    u, v = u01("succ", scene, route, seed), u01("kind", scene, route, seed)
    v2 = arm.get("rule_version") == "v2"
    succ = u < arm_p(arm)
    if cls == "formal40" and not v2 and arm["base"] == "rule-occ" and os.environ.get("STUB_PERTURB_V1") and route == PERTURB[0] and k == 0 and scene == PERTURB[1]:
        succ = not succ
    cf_prob = 0.15 if v2 else 0.25
    hit_first = (not succ) and v < cf_prob
    steps = 30 + int(40 * u01("len", scene, route, seed))
    reach = steps - 8 if succ else None
    coll_step = (steps - 1) if hit_first else (reach + 2 if succ and u01("late", scene, route, seed) < 0.05 else None)
    if hit_first and u01("early", scene, route, seed) < 0.1:
        steps, coll_step = 1, 0                                # 出生即碰撞
    timeout = (not succ) and not hit_first
    pose = {"x": round(1.0 + 0.1 * route + 0.01 * k, 4), "y": round(2.0 + 0.2 * u01("y", scene, route, seed), 4), "yaw_deg": round(360 * u01("yaw", scene, route, seed) - 180, 3)}
    rows, acts = [], []
    ends_stuck = v2 and timeout and u01("stuck", scene, route, seed) < 0.5
    for i in range(steps):
        w = u01(f"s{i}", scene, route, seed)
        if v2:
            entry = ("V2-stuck" if i == steps - 1 and ends_stuck else "V2-fwd0" if w < 0.6 else "V2-fwd1" if w < 0.65 and len(arm["rule_params"]["levels"]) > 1 else
                     "V2-back0" if w < 0.8 else "V2-seek0")
            action = FWD if "fwd" in entry else YAW_LEFT if w < 0.9 else YAW_RIGHT
        else:
            entry = "V1-fwd" if w < 0.6 else "V1-more" if w < 0.8 else "V1-cap"
            action = FWD if entry == "V1-fwd" else YAW_LEFT if w < 0.9 else YAW_RIGHT
        row = {"step": i, "err_deg": round(20 * (w - 0.5), 2), "n": 0}
        if v2:
            row["e"] = round(w - 0.5, 4)
        row.update({"target": [3.0, 1.0], "action": action, "entry": entry, "mem_n": 0, "clear": [1, 1, 1]})
        if v2:
            row.update({"level": None if entry == "V2-stuck" else int(entry[-1]) if entry[-1].isdigit() else 0, "m": 0, "tabu_n": 0, "ok": [[1, 1, 1]]})
        row.update({"src": "occ" if arm["base"] != "rule-depth" else "depth", "d_occ": [round(1.0 + 0.1 * ((i + j) % 20), 3) for j in range(20)],
                    "d_depth": [round(1.0 + 0.1 * ((i + j) % 20) + 0.05, 3) for j in range(20)]})
        rows.append(row)
        clear = 0.15 if (ends_stuck and i >= steps - 20) else 0.4 + 0.1 * w
        acts.append({"step": i, "action_expert": action, "action_executed": action, "collision": coll_step == i, "pose_before": {"x": round(pose["x"] + 0.05 * i, 4), "y": pose["y"], "z": 1.2},
                     "yaw_before": pose["yaw_deg"], "sim_dt": hold, "clearance_m": clear})
    cnt, cnt_pre = {}, {}
    end_pre = min([x for x in (reach, coll_step) if x is not None] or [steps - 1])
    for r in rows:
        cnt[r["entry"]] = cnt.get(r["entry"], 0) + 1
        if r["step"] <= end_pre:
            cnt_pre[r["entry"]] = cnt_pre.get(r["entry"], 0) + 1
    m = {"success_stop": succ, "success_v2": succ, "collision": coll_step is not None, "collide_first": hit_first, "timeout": timeout, "steps": steps,
         "reach_step": reach, "collision_step": coll_step, "spl": 0.5 if succ else 0.0, "spl_stop": 0.5 if succ else 0.0, "final_wp_dist_m": 0.2 if succ else 3.0,
         "min_wp_dist_m": 0.2 if succ else 2.5, "start_pose": pose, "arm": arm["name"], "scene": scene, "route": route, "episode": k, "seed": seed,
         "stopped_on_goal": succ, "wall_s": 0.1, "guard": None}
    if arm["base"].startswith("rule-"):
        m["rule"] = {"source": "occ" if arm["base"] == "rule-occ" else "depth", "n_steps": steps, "entries": cnt, "n_steps_pre": end_pre + 1, "entries_pre": cnt_pre,
                     "plan_failed": False, "params": arm.get("rule_params", {})}
        if v2:
            m["rule"]["rule_version"] = "v2"
    return m, rows, acts


PERTURB = (None, None)       # (路线, 场景)：STUB_PERTURB_V1 时被取反的那一集，run_scene 里按 config 的第一条路线定


def finished(d, k):
    try:
        with open(os.path.join(d, "done.json"), encoding="utf-8") as f:
            j = json.load(f)
    except (OSError, ValueError):
        return False
    return not j["failures"] and len(j["episodes"]) >= k


def run_scene(out, cfg, scene, only_arms=None):
    global PERTURB
    routes = cfg["routes"][scene] if isinstance(cfg["routes"], dict) else cfg["routes"]
    if routes:
        PERTURB = (routes[0], scene)
    for arm in cfg["arms"]:
        if only_arms and arm["name"] not in only_arms:
            continue
        hold = arm.get("hold", cfg["hold"])
        for r in routes:
            d = os.path.join(out, arm["name"], scene, f"r{r}")
            if os.path.isdir(d) and finished(d, cfg["k"]):
                print(f"[{arm['name']}] r{r} 跳过（done.json 已完成）", flush=True)
                continue
            if os.path.isdir(d):
                shutil.move(d, d + ".incomplete-" + time.strftime("%H%M%S"))
            os.makedirs(d)
            eps = []
            for k in range(cfg["k"]):
                seed = cfg["seed0"] + 100 * r + k
                m, rows, acts = episode(arm, scene, r, k, seed, hold, cfg.get("class"))
                ep = os.path.join(d, "ep_%05d" % k)
                os.makedirs(ep)
                for name, items in (("rule.jsonl", rows), ("actions.jsonl", acts)):
                    if name == "rule.jsonl" and not arm["base"].startswith("rule-"):
                        continue
                    with open(os.path.join(ep, name), "w", encoding="utf-8") as f:
                        for x in items:
                            f.write(json.dumps(x, ensure_ascii=False) + "\n")
                with open(os.path.join(ep, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, indent=1)
                eps.append({"episode": k, "seed": seed, "steps": m["steps"], "success_stop": m["success_stop"], "collision": m["collision"], "timeout": m["timeout"], "wall_s": 0.1})
            done = {"arm": arm["name"], "scene": scene, "route": r, "steps": eps[0]["steps"], "hold": hold, "ckpt": None, "guard": arm.get("guard"),
                    "episodes": eps, "failures": [], "wall_s": 0.1}
            with open(os.path.join(d, "done.json"), "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=1)


def make_k1(root, pilot_routes, indoor_scenes, outdoor_scenes):
    base = {"hold": 1.0, "step_frac": 0.5, "step_pad": 15, "guard_threshold_deg": 45.0, "drop_trace": True, "infer_bin": "unused", "cf_sitl_proc": "stub",
            "k": 3, "seed0": 8000, "rule_params": {"r_safe": 0.25, "d_look": 1.2, "yaw_step_deg": 31.7}}
    arm = {"name": "rule-occ+route", "base": "rule-occ", "ckpt": None, "guard": "route"}
    with open(pilot_routes, encoding="utf-8") as f:
        sel = json.load(f)
    ind = {**base, "class": "indoor", "scenes": indoor_scenes, "routes": {s: [r["route"] for r in sel["scenes"].get(s, [])] for s in indoor_scenes},
           "routes_selected": pilot_routes, "arms": [arm]}
    outd = {**base, "class": "outdoor", "scenes": outdoor_scenes, "routes": [0, 1, 2, 3, 4, 5], "k": 5, "seed0": 7000, "arms": [arm]}
    for sub, cfg, scenes in (("indoor", ind, indoor_scenes), ("outdoor", outd, outdoor_scenes)):
        out = os.path.join(root, "formal", sub)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        for s in scenes:
            run_scene(out, {**cfg, "class": "k1"}, s)          # class 不叫 formal40：STUB_PERTURB_V1 只扰动本次运行，不扰动假 K1
    with open(os.path.join(root, "formal", "DONE"), "w", encoding="utf-8") as f:
        f.write("rc=0 stub\n")


def main(argv):
    if argv[1:2] == ["--make-k1"]:
        root, pilot = argv[2], argv[3]
        i = argv.index("--")
        make_k1(root, pilot, argv[4:i], argv[i + 1:])
        return 0
    out, scene = os.path.abspath(argv[1]), argv[2]
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    print("stub", out, scene, [a["name"] for a in cfg["arms"]], flush=True)
    if os.environ.get("STUB_FAIL_SCENE") == scene and os.environ.get("STUB_FAIL_CLASS", cfg.get("class")) == cfg.get("class"):
        print("stub: 按 STUB_FAIL_SCENE / STUB_FAIL_CLASS 失败", flush=True)
        return 1
    time.sleep(float(os.environ.get("STUB_SLEEP", "0.3")))
    run_scene(out, cfg, scene)
    print("stub done", scene, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
