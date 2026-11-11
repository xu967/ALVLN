#!/usr/bin/env python3
"""变异检查（任务卡 K3 的验证手段）：在一份 git archive 出来的代码副本里每次只改一处关键逻辑，跑对应的测试，必须变红；
不红 = 测试没有盖住这处逻辑，要补测试或者说明这处改动与原行为等价。每个变异先改、跑、改回；结束时核对所有被改文件与原文逐字节相同。

  git archive <提交> | tar -x -C <副本目录>
  python mutation_check.py --tree <副本目录> --logs <日志目录> [--only R0] [--list]

每个变异 = (编号, 文件, 原文片段, 改成, 测试命令列表, 说明)；原文片段在文件里必须恰好出现一次（否则记「定位失败」，不算通过）；
测试命令按顺序试，第一个变红的算杀死它（记下是哪个）。开始前对每条测试命令先跑一遍不变异的对照，对照不绿就整体停下。
一次一个进程，nice 10；单条命令超时 240 s（超时记「红（超时）」）。退出码：0 全部变红（EQUIV 里已说明的等价变异不算）；1 有变异没有变红或定位失败；2 对照不绿。
EQUIV = {编号: 理由}：变异改动的条件被同一处的别的条件蕴含，改了行为不变（等价变异），测试不可能也不应该红；表里的编号照样要跑，结果记「等价（已说明）」。
表里的编号若变红了不算错（说明有人补了测试），照记「红」。
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

D = os.environ.get("DATA_ROOT", "/root/rivermind-data") + "/uav_rgb"
PY = sys.executable
PY3DGS = os.environ.get("PY3DGS", D + "/3dgs-venv/bin/python")
GPU = os.environ.get("GPU", "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740")
EXP = "experiments/2026-11-11-rule-v2-3dgs"

T_RULE = ("rule_v2", [PY, "sim/tools/test_rule_v2.py"])
T_ORACLE = ("oracle_v2", [PY3DGS, "sim/3dgs/test_rule_oracle_v2.py"])
T_DRV = ("driver_v2", [PY, f"{EXP}/test_driver_v2.py"])
T_PARAMS = ("driver_params", [PY, "experiments/2026-11-10-sage3d-pilot/test_driver_params.py"])
T_DRV_K1 = ("driver_rule_k1", [PY, "experiments/2026-11-10-rule-oracle-3dgs/test_driver_rule.py"])
T_ROUTES = ("select_routes", [PY, f"{EXP}/test_select_routes_v2.py"])
T_VARIANT = ("select_variant", [PY, f"{EXP}/test_select_variant.py"])
T_MATCH = ("check_k1_match", [PY, f"{EXP}/test_check_k1_match.py"])
T_BUILD = ("build_config", [PY, f"{EXP}/test_build_config.py"])
T_SUMM = ("summarize", [PY, f"{EXP}/test_summarize.py"])
T_SH = ("run_sh_static", [PY, f"{EXP}/test_run_sh.py"])


def chain(name):
    return (f"chain:{name}", [PY, f"{EXP}/test_run_chain.py", f"TestChain.{name}"])


C_HAPPY = chain("test_happy_path_then_resume_then_changed_config_is_refused")
C_GATE = chain("test_gate_failure_stops_before_formal_with_the_card_message")
C_F0 = chain("test_f0_mismatch_stops_before_formal")
C_DRV = chain("test_driver_failure_is_reported_as_rc7")
C_DRV_FORMAL = chain("test_formal_stage_driver_failure_is_reported_as_rc7")
C_OVR = chain("test_formal_refuses_environment_overrides")
C_GPU = chain("test_no_gpu_exits_rc4_without_work")
C_2ND = chain("test_second_instance_is_refused")
C_SIZE = chain("test_size_watchdog_stops_the_whole_group_with_rc5")

RP = "sim/tools/rule_policy.py"
RO = "sim/3dgs/rule_oracle.py"
RS = "sim/tools/rule_sim.py"
DRV = "experiments/2026-11-10-bearing-guard-3dgs/driver.py"

EQUIV = {
    "C04": "ok 里的 matched == n_expected 被 complete 蕴含：complete 要求 len(new) == n_expected 且没有只在本次出现的键，于是本次与 K1 的交集 matched 必等于 len(new)；去掉这一项行为不变",
    "Z07": "预测（Δ >= 0.10、碰撞 <= 0.10 等全部成立）与证伪（Δ < 0.05 或 碰撞 > 0.15）不可能同时成立，先判哪个结果相同；任务卡写的「先判证伪」没有可观察的差别",
}

# (编号, 文件, 原文片段, 改成, 测试列表, 说明)
M = [
    # ---------------------------------------------------------------- 规则 v2 本体：禁忌
    ("R01", RP, "if (level <= lb and", "if (level >= lb and", [T_RULE], "禁忌的级别判断反向（level <= lb 改成 >=）"),
    ("R02", RP, 'step - step_e <= params["tabu_steps"]', 'step - step_e < params["tabu_steps"]', [T_RULE], "禁忌步龄边界（<= 改成 <）"),
    ("R03", RP, "abs(_wrap180(psi_deg - yaw_e)) <= half", "abs(_wrap180(psi_deg - yaw_e)) <= 2 * half", [T_RULE], "禁忌航向窗口放宽一倍"),
    ("R04", RP, 'and math.hypot(x - xe, y - ye) <= params["tabu_radius"]):', "and True):", [T_RULE], "禁忌不看位置半径"),
    ("R05", RP, "return (deg + 180.0) % 360.0 - 180.0", "return deg", [T_RULE], "航向差不折回 ±180°"),
    ("R06", RP, "(m == 0 or not v2_tabu_hit(tabu, yaw_deg + m * alpha, lv, step, x, y, params))", "(m == 0 or True)", [T_RULE], "候选完全不查禁忌表"),
    ("R07", RP, "(m == 0 or not v2_tabu_hit(", "(not v2_tabu_hit(", [T_RULE], "m = 0（本步直接看到的航向）也查禁忌表"),
    ("R08", RP, "usable = [not (still and prev_action == FORWARD_SLOW) for _r, _d, still in levels]", "usable = [True for _r, _d, still in levels]", [T_RULE],
     "「只许静止起步」的级永远可用"),
    ("R09", RP, "not (still and prev_action == FORWARD_SLOW)", "not (still and prev_action != FORWARD_SLOW)", [T_RULE], "「只许静止起步」的条件取反"),
    ("R10", RP, 'seq = [("a", 0)] + [("b", lv) for lv in range(n_levels)] + [("c", 0)]', 'seq = [("a", 0)] + [("b", 0)] + [("c", 0)]', [T_RULE],
     "near 次序只试第 0 级的前进"),
    ("R11", RP, 'seq += [("a", lv), ("c", lv)]', 'seq += [("c", lv), ("a", lv)]', [T_RULE], "near 次序里更短级的 a / c 顺序对调"),
    ("R12", RP, 'for lv in range(n_levels) for kind in "abc"]', 'for lv in range(n_levels) for kind in "bac"]', [T_RULE], "lexi 次序里 a 与 b 对调"),
    ("R13", RP, 'seq = [(kind, lv) for lv in range(n_levels) for kind in "abc"]', 'seq = [(kind, lv) for kind in "abc" for lv in range(n_levels)]', [T_RULE],
     "lexi 次序的循环嵌套反过来（先按动作再按级）"),
    ("R14", RP, "key=lambda m: (abs(m), abs(e + m),", "key=lambda m: (abs(e + m), abs(m),", [T_RULE], "两侧候选排序：|m| 与 |e+m| 的优先级对调"),
    ("R15", RP, "0 if side == 0 or (m > 0) == (side > 0) else 1,", "0,", [T_RULE], "两侧候选排序：去掉 v1 的左右求和偏向"),
    ("R16", RP, "0 if last == 0 or (m > 0) == (last > 0) else 1,", "0,", [T_RULE], "两侧候选排序：去掉 last_turn 一侧偏好"),
    ("R17", RP, "cap = max(n_cap, abs(n))", "cap = n_cap", [T_RULE], "已偏出 n_cap 时不放宽候选上限"),
    ("R18", RP, "for m in range(-m_max, m_max + 1)", "for m in range(-m_max, m_max)", [T_RULE], "候选范围少一个正侧（+m_max）"),
    ("R19", RP, "abs(n + m) <= cap and ok(m, lv)]", "abs(n + m) < cap and ok(m, lv)]", [T_RULE], "候选的 |n+m| 上限边界（<= 改成 <）"),
    ("R20", RP, "back_m = 0 if n == 0 else (-1 if n > 0 else 1)", "back_m = 0 if n == 0 else (1 if n > 0 else -1)", [T_RULE], "朝回转的方向反了"),
    ("R21", RP, "if n != 0 and ok(back_m, lv):", "if ok(back_m, lv):", [T_RULE], "n = 0 时也走「朝回」"),
    ("R22", RP, 'else YAW_LEFT), "V2-stuck", info(None, None)', 'else YAW_RIGHT), "V2-stuck", info(None, None)', [T_RULE], "V2-stuck 的缺省转向（没有 last_turn 时）改右"),
    ("R23", RP, "e = float(n) if e is None else float(e)", "e = float(n)", [T_RULE], "忽略传入的浮点 e"),
    ("R24", RP, 'return (YAW_LEFT if m > 0 else YAW_RIGHT), f"V2-seek{lv}", info(lv, m)', 'return (YAW_RIGHT if m > 0 else YAW_LEFT), f"V2-seek{lv}", info(lv, m)', [T_RULE],
     "两侧候选的转向方向反了"),
    ("R25", RP, '"clear": [int(clear(1, 0)), int(clear(0, 0)), int(clear(-1, 0))]', '"clear": [int(clear(-1, 0)), int(clear(0, 0)), int(clear(1, 0))]', [T_RULE],
     "info.clear 的左 / 右次序对调（日志与 v1 次序不一致）"),
    ("R26", RP, "tabu.append((yaw_deg, x, y, step, max(i for i, b in enumerate(blocked) if b)))", "tabu.append((yaw_deg, x, y, step, min(i for i, b in enumerate(blocked) if b)))", [T_RULE],
     "禁忌条目的 lb 取被堵的最前一级"),
    ("R27", RP, 'tabu = [t for t in tabu if step - t[3] <= params["tabu_steps"]]', "tabu = list(tabu)", [T_RULE], "不清超龄的禁忌条目"),
    ("R28", RP, "if blocked[0]:", "if blocked[-1]:", [T_RULE], "按最后一级而不是第 0 级是否被堵来记禁忌"),
    ("R29", RP, '    levels = check_v2_params(params)\n    new_world = points_to_world(obstacle_points(d, hfov_deg), x, y, yaw_deg)\n    memory = update_memory(memory, step, new_world, params["mem_steps"])',
     '    levels = check_v2_params(params)\n    new_world = points_to_world(obstacle_points(d, hfov_deg), x, y, yaw_deg)\n    memory = update_memory(memory, step, new_world, params["mem_steps"] // 3)',
     [T_RULE], "v2 的记忆步数只用三分之一"),
    ("R30", RP, "if levels[i][0] > levels[i - 1][0] or levels[i][1] > levels[i - 1][1]:", "if False:", [T_RULE], "不校验「逐级不放宽」"),
    ("R31", RP, 'if params["order"] not in V2_ORDERS:', "if False:", [T_RULE], "不校验 order"),
    ("R32", RP, "r > 0.0 and d_look > 0.0", "r >= 0.0 and d_look >= 0.0", [T_RULE], "r_safe / d_look 允许 0"),
    ("R33", RP, "for lv in range(V2_MAX_LEVELS) for kind in", "for lv in range(V2_MAX_LEVELS - 1) for kind in", [T_RULE, T_ORACLE], "V2_ENTRIES 少一级"),
    ("R34", RP, "side = 1 if left > right else -1", "side = 1 if left < right else -1", [T_RULE], "左右求和偏向的方向反了"),
    ("R35", RP, '"mem_steps": 24,', '"mem_steps": 8,', [T_RULE], "RULE_V2.mem_steps 24 改成 8（任务卡定值）"),
    ("R36", RP, '"levels": ((0.35, 1.2, False),),', '"levels": ((0.30, 1.2, False),),', [T_RULE], "RULE_V2.levels 缺省 r_safe 0.35 改成 0.30"),
    ("R37", RP, '"tabu_radius": 0.30,', '"tabu_radius": 0.50,', [T_RULE], "RULE_V2.tabu_radius 改值"),
    ("R38", RP, '"tabu_steps": 24,', '"tabu_steps": 12,', [T_RULE], "RULE_V2.tabu_steps 改值"),
    ("R39", RP, '"m_max": 6,', '"m_max": 3,', [T_RULE], "RULE_V2.m_max 改值"),
    ("R40", RP, "last = {YAW_LEFT: 1, YAW_RIGHT: -1}.get(last_turn, 0)", "last = {YAW_LEFT: -1, YAW_RIGHT: 1}.get(last_turn, 0)", [T_RULE], "last_turn 一侧的符号反了"),
    ("R41", RP, "        if not usable[lv]:", "        if False:", [T_RULE], "不可用的级不跳过"),
    ("R42", RP, 'if left != right and abs(left - right) >= params["tie_frac"] * max(left, right):',
     'if left != right and abs(left - right) > params["tie_frac"] * max(left, right):', [T_RULE], "v2 左右求和偏向的平局边界（>= 改成 >）"),
    # ---------------------------------------------------------------- 离线模拟
    ("S01", RS, "        prev_action = action", "        prev_action = None", [T_RULE], "离线模拟不记上一步动作"),
    ("S02", RS, "last_turn, prev_action=prev_action,\n", "last_turn, prev_action=None,\n", [T_RULE], "离线模拟的 v2 调用不传 prev_action"),
    # ---------------------------------------------------------------- 3DGS 接线
    ("W01", RO, 'e = -err / self.params["yaw_step_deg"]', 'e = err / self.params["yaw_step_deg"]', [T_ORACLE], "e 的符号与 n 不一致"),
    ("W02", RO, "prev_action=self.last_action, hfov_deg=self.hfov_deg", "prev_action=None, hfov_deg=self.hfov_deg", [T_ORACLE], "不传上一步动作"),
    ("W03", RO, "self.last_action = int(action)", "self.last_action = None", [T_ORACLE], "不记上一步动作"),
    ("W04", RO, "self.tabu, self.last_action = [], None", 'self.tabu, self.last_action = getattr(self, "tabu", []), None', [T_ORACLE], "集边界不清禁忌表"),
    ("W05", RO, 'row["e"] = round(e, 4)', "pass", [T_ORACLE, T_DRV], "v2 日志不写 e"),
    ("W06", RO, 'table = rp.RULE_V1 if rule_version == "v1" else rp.RULE_V2', "table = rp.RULE_V1", [T_ORACLE], "v2 用 v1 的参数表"),
    ("W07", RO, 'self.planner_name, self.version = "rule_v2", POLICY_VERSION_V2', 'self.planner_name, self.version = "rule_v2", POLICY_VERSION', [T_ORACLE], "v2 的版本串沿用 v1"),
    ("W08", RO, '"tabu_n": info["tabu_n"], "ok": info["ok"]}', '"tabu_n": info["tabu_n"]}', [T_ORACLE], "v2 日志不写 ok 位图"),
    ("W09", RO, "rp.check_v2_params(self.params)", "pass", [T_ORACLE], "构造时不校验 v2 参数"),
    ("W10", RO, "if rule_version not in RULE_VERSIONS:", "if False:", [T_ORACLE], "不校验 rule_version"),
    ("W11", RO, "unknown = sorted(set(params or {}) - set(table))", "unknown = sorted(set(params or {}) - set(rp.RULE_V1))", [T_ORACLE], "未知键按 v1 表判"),
    ("W12", RO, 'row.update({"target":', 'row.update({"zz_target":', [T_ORACLE, T_DRV], "日志键改名（target）"),
    # ---------------------------------------------------------------- driver
    ("D01", DRV, 'rule_kw = {"rule_version": arm["rule_version"]} if "rule_version" in arm else {}', "rule_kw = {}", [T_DRV, T_DRV_K1], "driver 不把 rule_version 传给策略"),
    ("D02", DRV, 'occ_planner, sc["waypoint"], hold_s=hold,', 'occ_planner, sc["waypoint"],', [T_PARAMS], "教师不再按臂的生效 hold 推导容差（hold 修复）"),
    ("D03", DRV, 'm["rule"].update(rule_kw)', "pass", [T_DRV], "metrics.json 的 rule 不记 rule_version"),
    ("D04", DRV, 'done["rule"] = {"source": rule_src, "overrides": rule_params, **rule_kw}', 'done["rule"] = {"source": rule_src, "overrides": rule_params}', [T_DRV], "done.json 不记 rule_version"),
    ("D05", DRV, '_a["rule_version"] not in ("v1", "v2")', '_a["rule_version"] not in ("v1", "v2", "v3")', [T_DRV], "driver 放过 rule_version = v3"),
    ("D06", DRV, '_a["base"] not in RULE_BASES or _a["rule_version"]', '_a["rule_version"]', [T_DRV], "driver 放过非规则层臂上的 rule_version"),
    ("D07", DRV, "params=rule_params, **rule_kw)", "params=rule_params)", [T_DRV, T_DRV_K1], "构造策略时不传 rule_kw"),
    # ---------------------------------------------------------------- 选路线
    ("T01", f"{EXP}/select_routes_v2.py", '_merge(pilot_scenes, k1_dev_scenes, new["scenes"]), per_scene=holdout_per_scene', '_merge(pilot_scenes, k1_dev_scenes), per_scene=holdout_per_scene',
     [T_ROUTES], "留出路线不排除 dev-new（三批会重叠）"),
    ("T02", f"{EXP}/select_routes_v2.py", "_merge(pilot_scenes, k1_dev_scenes), per_scene=dev_new_per_scene", "_merge(pilot_scenes), per_scene=dev_new_per_scene", [T_ROUTES],
     "dev-new 不排除 K1 开发路线"),
    ("T03", f"{EXP}/select_routes_v2.py", "if n_hold < a.min_holdout:", "if n_hold <= a.min_holdout:", [T_ROUTES], "留出路线下限边界（< 改成 <=）"),
    ("T04", f"{EXP}/select_routes_v2.py", 'dev_scenes[s] = sorted(tagged, key=lambda r: (r["astar_path_m"], r["route"]))', "dev_scenes[s] = tagged", [T_ROUTES], "开发路线不按长度排序"),
    ("T05", f"{EXP}/select_routes_v2.py", "if bad_repro:", "if False:", [T_ROUTES], "不做「复现 K1 开发路线」的自检"),
    ("T06", f"{EXP}/select_routes_v2.py", "    bad = disjoint_problems(pilot, dev, holdout)\n    if bad:", "    bad = disjoint_problems(pilot, dev, holdout)\n    if False:", [T_ROUTES], "不查三批路线重叠"),
    # ---------------------------------------------------------------- 选变体
    ("V01", f"{EXP}/select_variant.py", 'if table[n]["k_collide_first"] <= max_collide_first]', 'if table[n]["k_collide_first"] < max_collide_first]', [T_VARIANT], "撞在前上限边界（<= 改成 <）"),
    ("V02", f"{EXP}/select_variant.py", 'if table[n]["k_stop"] >= best - tie_slack]', 'if table[n]["k_stop"] > best - tie_slack]', [T_VARIANT], "并列边界（>= 改成 >）"),
    ("V03", f"{EXP}/select_variant.py", 'out["selected"] = out["tied"][0]', 'out["selected"] = out["tied"][-1]', [T_VARIANT], "并列取表里靠后的"),
    ("V04", f"{EXP}/select_variant.py", 'out["gate_passed"] = out["gain_over_v1"] >= gate_min_gain', 'out["gate_passed"] = out["gain_over_v1"] > gate_min_gain', [T_VARIANT], "闸门边界（>= 改成 >）"),
    ("V05", f"{EXP}/select_variant.py", 'best = max(table[n]["k_stop"] for n in eligible)', 'best = min(table[n]["k_stop"] for n in eligible)', [T_VARIANT], "取 SR 最低者"),
    ("V06", f"{EXP}/select_variant.py", 'vr.by_name(variants, dmap[sel])', "vr.by_name(variants, sel)", [T_VARIANT], "depth 臂不按 DEPTH_MAP 换变体"),
    ("V07", f"{EXP}/select_variant.py", 'if not decision["gate_passed"] and not a.no_gate:', "if False:", [T_VARIANT, C_GATE], "闸门不拦"),
    ("V08", f"{EXP}/select_variant.py", 'if done["failures"] or len(done["episodes"]) < k:', "if False:", [T_VARIANT], "有失败集 / 集数不足也照选"),
    ("V09", f"{EXP}/select_variant.py", '            return 2\n    tmp = path + ".tmp"', '            pass\n    tmp = path + ".tmp"', [T_VARIANT], "不拒绝覆盖已冻结的 selected.json"),
    ("V10", f"{EXP}/select_variant.py", 'if len({t["n"] for t in table.values()}) != 1:', "if False:", [T_VARIANT], "各臂集数不同也照选"),
    ("V11", f"{EXP}/select_variant.py", 'if arm_names != ["v1"] + names:', "if False:", [T_VARIANT], "config 的臂与变体表不一致也照选"),
    ("V12", f"{EXP}/select_variant.py", "rate_cap = (a.max_collide_first * n_eps) // 40", "rate_cap = (a.max_collide_first * n_eps) // 20", [T_VARIANT], "对照读法的上限算错（// 40 改成 // 20）"),
    # ---------------------------------------------------------------- F0 对拍
    ("C01", f"{EXP}/check_k1_match.py", '"timeout", "steps", "reach_step", "collision_step")', '"timeout", "reach_step", "collision_step")', [T_MATCH], "对拍不比 steps"),
    ("C02", f"{EXP}/check_k1_match.py", "POSE_TOL = 1e-6", "POSE_TOL = 1.0", [T_MATCH], "出生位姿容差放到 1.0"),
    ("C03", f"{EXP}/check_k1_match.py", 'return all(abs(a[f] - b[f]) <= tol for f in ("x", "y", "yaw_deg"))', 'return any(abs(a[f] - b[f]) <= tol for f in ("x", "y", "yaw_deg"))', [T_MATCH],
     "出生位姿只要一个分量相同就算一致"),
    ("C04", f"{EXP}/check_k1_match.py", 'and res["matched"] == res["n_expected"]', "", [T_MATCH], "不要求全部键都对上"),
    ("C05", f"{EXP}/check_k1_match.py", '    if res["diffs"]:\n        return 1\n', '    if res["diffs"]:\n        return 0\n', [T_MATCH], "有不一致也返回 0"),
    ("C06", f"{EXP}/check_k1_match.py", 'if read_json(os.path.join(d, "done.json")) is None:\n            missing.append((s, r))', 'if False:\n            missing.append((s, r))', [T_MATCH], "没写 done.json 的路线不算缺"),
    ("C07", f"{EXP}/check_k1_match.py", 'res["complete"] = not missing and len(new) == res["n_expected"] and not res["only_new"]', 'res["complete"] = len(new) == res["n_expected"] and not res["only_new"]', [T_MATCH],
     "完整性不看缺的路线"),
    # ---------------------------------------------------------------- 构造 config
    ("B01", f"{EXP}/build_config.py", '"holdout": "SEED0_HOLDOUT", "formal40": "SEED0_FORMAL40"}[cls]', '"holdout": "SEED0_FORMAL40", "formal40": "SEED0_HOLDOUT"}[cls]', [T_BUILD], "留出与正式 40 的 seed 起点对调"),
    ("B02", f"{EXP}/build_config.py", 'return [{"name": "forward+route", **FWD, "guard": "route"}, v1_arm(e), main_arm, depth_arm]', "return [v1_arm(e), main_arm, depth_arm]", [T_BUILD], "留出路线没有 forward+route 臂"),
    ("B03", f"{EXP}/build_config.py", 'v2_arm("rule2-depth+route", "rule-depth", depth)', 'v2_arm("rule2-depth+route", "rule-occ", depth)', [T_BUILD], "depth 臂的 base 写成 rule-occ"),
    ("B04", f"{EXP}/build_config.py", 'depth = vr.by_name(variants, selected["depth_variant"])', 'depth = vr.by_name(variants, selected["variant"])', [T_BUILD], "depth 臂不按 depth_variant"),
    ("B05", f"{EXP}/build_config.py", '"d_look": float(e["V1_D_LOOK"])}}', '"d_look": float(e["V1_D_LOOK"])}, "rule_version": "v1"}', [T_BUILD], "v1 臂带 rule_version 键（与 K1 的臂不再逐项相同）"),
    ("B06", f"{EXP}/build_config.py", 'cfg["rule_params"] = {"yaw_step_deg": float(e["YAW_STEP"])}', 'cfg["rule_params"] = {}', [T_BUILD], "不写全局 yaw_step_deg"),
    ("B07", f"{EXP}/build_config.py", 'cfg["arms"] = [a for a in cfg["arms"] if a["name"] in only]', 'cfg["arms"] = [a for a in cfg["arms"] if a["name"] not in only]', [T_BUILD], "--only 取反"),
    ("B08", f"{EXP}/build_config.py", '"seed0": int(e["SEED0_OUTDOOR"])', '"seed0": int(e["SEED0_HOLDOUT"])', [T_BUILD], "室外 seed 起点取错"),
    ("B09", f"{EXP}/build_config.py", '"holdout": "K_HOLDOUT", "formal40": "K_FORMAL40"}[cls]', '"holdout": "K_FORMAL40", "formal40": "K_HOLDOUT"}[cls]', [T_BUILD], "留出与正式 40 的每路线集数对调"),
    ("B10", f"{EXP}/build_config.py", '"ckpt": None, "guard": "route", "rule_version": "v2"', '"ckpt": None, "guard": "goal", "rule_version": "v2"', [T_BUILD], "v2 臂的导引写成 goal"),
    # ---------------------------------------------------------------- 变体表
    ("VR1", f"{EXP}/variants.py", 'or "v1" in names or not out', "or not out", [T_VARIANT, T_BUILD], "变体表允许叫 v1"),
    ("VR2", f"{EXP}/variants.py", 'still = part.endswith("*")', "still = False", [T_VARIANT, T_BUILD], "解析时丢掉「只许静止起步」的星号"),
    ("VR3", f"{EXP}/variants.py", "m[a] = b", "m[b] = a", [T_VARIANT, T_BUILD], "DEPTH_MAP 方向反了"),
    ("VR4", f"{EXP}/variants.py", 'rp.check_v2_params({**rp.RULE_V2, "levels": v["levels"], "order": v["order"]})', "pass", [T_VARIANT, T_BUILD], "解析变体表时不校验"),
    # ---------------------------------------------------------------- run.sh 整条链
    ("S10", f"{EXP}/run.sh", "if [[ $rc -eq 20 ]]; then", "if [[ $rc -eq 99 ]]; then", [C_GATE], "闸门不过时不写 rc=20"),
    ("S11", f"{EXP}/run.sh", "[[ $rc -ne 1 ]] || fail_done", "[[ $rc -ne 99 ]] || fail_done", [C_F0], "F0 不一致不写 rc=6"),
    ("S12", f"{EXP}/run.sh", '--selected "$SELECTED" --only rule-occ+route', '--selected "$SELECTED" --only rule2-occ+route', [C_HAPPY, C_F0], "F0 先跑的不是 v1 臂"),
    ("S13", f"{EXP}/run.sh", "if [[ ${mb:-0} -gt $SIZE_LIMIT_MB ]]; then", "if [[ ${mb:-0} -gt $((SIZE_LIMIT_MB * 1000)) ]]; then", [C_SIZE], "体积看门狗上限放大 1000 倍"),
    ("S14", f"{EXP}/run.sh", '&& -n "${OVERRIDDEN:-}" ]]', '&& -n "" ]]', [C_OVR], "formal 不拒绝环境变量覆盖"),
    ("S15", f"{EXP}/run.sh", 'flock -n 9 || { echo "已有实例在跑（$OUT_ROOT/.lock），本次退出" >&2; exit 3; }', "flock -n 9 || true", [C_2ND], "不拒绝第二个实例"),
    ("S16", f"{EXP}/run.sh", 'elif ! nvidia-smi -L 2>/dev/null | grep -q "$GPU"; then', "elif false; then", [C_GPU], "不检查 GPU 在不在"),
    ("S17", f"{EXP}/run.sh", 'if [[ ! -e "$OUT_ROOT/dev/.tasks_ok" ]]; then', "if true; then", [C_HAPPY], "续跑时重跑 dev 阶段"),
    ("S18", f"{EXP}/run.sh", 'if [[ -s "$target" ]] && ! cmp -s "$target" "$target.new"; then', "if false; then", [C_HAPPY], "config 与已有的不同也不拒绝"),
    ("S19", f"{EXP}/run.sh", "[[ $rc_all -eq 0 || $rc_all -eq 3 ]] || rc_all=7", "[[ $rc_all -eq 0 || $rc_all -eq 3 ]] || rc_all=1", [C_DRV_FORMAL], "正式阶段驱动失败的退出码不是 7"),
    ("S20", f"{EXP}/run.sh", 'cmp -s "$OUT_ROOT/routes/$f" "$OUT_ROOT/routes.new/$f" || fail_done', 'cmp -s "$OUT_ROOT/routes/$f" "$OUT_ROOT/routes.new/$f" || true; false || fail_done', [C_HAPPY, T_SH],
     "路线与上次不同时不拒绝"),
    # ---------------------------------------------------------------- run.sh：3DGS 池的锁与 cgroup 起止记录（公共规则 §4，2026-11-11 23:55 起）
    ("S21", f"{EXP}/run.sh", "GS3D_LOCK=$D/tmp/phase12-cards/gs3d.lock", "GS3D_LOCK=$D/tmp/phase12-cards/machine.lock", [T_SH], "锁改回 machine.lock"),
    ("S22", f"{EXP}/run.sh", 'exec 8>> "$GS3D_LOCK"; flock 8', 'exec 8>> "$GS3D_LOCK"; flock -n 8 || true', [T_SH], "拿锁改成不阻塞"),
    ("S23", f"{EXP}/run.sh", "cgroup_start() { cgroup_note 起; CG_STARTED=1; }", "cgroup_start() { CG_STARTED=1; }", [C_HAPPY], "起不记 cgroup"),
    ("S24", f"{EXP}/run.sh", 'fail_done() { log "$1"; cgroup_end; echo', 'fail_done() { log "$1"; echo', [T_SH, C_GATE], "失败退出（fail_done）不记「止」"),
    ("S25", f"{EXP}/run.sh", "on_term() { cgroup_end; [[ -s", "on_term() { [[ -s", [T_SH, C_SIZE], "被终止（on_term）不记「止」"),
    ("S26", f"{EXP}/run.sh", 'log "全部结束 rc=$rc_all"\ncgroup_end\n', 'log "全部结束 rc=$rc_all"\n:\n', [T_SH, C_HAPPY], "正常结束不记「止」"),
    ("S27", f"{EXP}/run.sh", '| tee -a "$OUT_ROOT/cgroup.txt"', '| tee "$OUT_ROOT/cgroup.txt"', [C_HAPPY], "cgroup.txt 不追加（续跑时覆盖）"),
    ("S28", f"{EXP}/run.sh", "awk '$1==\"usage_usec\"{print $2}'", "awk '$1==\"usage_usecs\"{print $2}'", [C_HAPPY], "读不到 usage_usec（记成 NA）"),
    # ---------------------------------------------------------------- 汇总
    ("Z01", f"{EXP}/summarize.py", '"coll_le_pred": coll <= PRE["coll_pred"]', '"coll_le_pred": coll < PRE["coll_pred"]', [T_SUMM], "主终点碰撞率边界（<= 改成 <）"),
    ("Z02", f"{EXP}/summarize.py", '"cf_le_pred": cf <= PRE["cf_pred"]', '"cf_le_pred": cf < PRE["cf_pred"]', [T_SUMM], "撞在前边界（<= 改成 <）"),
    ("Z03", f"{EXP}/summarize.py", '"delta_lt_fals": delta < PRE["delta_fals"]', '"delta_lt_fals": delta <= PRE["delta_fals"]', [T_SUMM], "证伪的 Δ 边界（< 改成 <=）"),
    ("Z04", f"{EXP}/summarize.py", '"coll_gt_fals": coll > PRE["coll_fals"]', '"coll_gt_fals": coll >= PRE["coll_fals"]', [T_SUMM], "证伪的碰撞率边界（> 改成 >=）"),
    ("Z05", f"{EXP}/summarize.py", '"p_alpha": 0.05,', '"p_alpha": 0.10,', [T_SUMM], "p 的显著性水平 0.05 改成 0.10"),
    ("Z06", f"{EXP}/summarize.py", '"delta_pred": Fraction(1, 10)', '"delta_pred": Fraction(1, 11)', [T_SUMM], "预测的 Δ 0.10 改成 1/11"),
    ("Z07", f"{EXP}/summarize.py", '"证伪" if out["falsified"] else "达到预测" if out["predicted"] else "介于预测与证伪之间"', '"达到预测" if out["predicted"] else "证伪" if out["falsified"] else "介于预测与证伪之间"', [T_SUMM],
     "判决里先判预测再判证伪"),
    ("Z08", f"{EXP}/summarize.py", 'out["falsified"] = out["delta_lt_fals"] or out["coll_gt_fals"]', 'out["falsified"] = out["delta_lt_fals"] and out["coll_gt_fals"]', [T_SUMM], "证伪条件 or 改成 and"),
    ("Z09", f"{EXP}/summarize.py", 'bc = sum(bool(index[(a, *k)]["success_stop"]) and not index[(b, *k)]["success_stop"] for k in keys)',
     'bc = sum(bool(index[(b, *k)]["success_stop"]) and not index[(a, *k)]["success_stop"] for k in keys)', [T_SUMM], "配对计数 b 与 c 对调"),
    ("Z10", f"{EXP}/summarize.py", 'bc = pc_v1["b_count"] - pc_v1["c_count"]', 'bc = pc_v1["c_count"] - pc_v1["b_count"]', [T_SUMM], "室外 b − c 的符号反了"),
    ("Z11", f"{EXP}/summarize.py", "ix[(k1_name, s, rt, seed)] = m", "pass", [T_SUMM], "室外不把 K1 的 v1 放进配对表"),
    ("Z12", f"{EXP}/summarize.py", 'if f and f["last_entry"] == "V2-stuck":', 'if f and f["last_entry"] == "V2-fwd0":', [T_SUMM], "V2-stuck 收尾列表认错条目"),
    ("Z13", f"{EXP}/summarize.py", 'f["turn"] = sum(a["action_executed"] in (YAW_LEFT, YAW_RIGHT) for a in act)', 'f["turn"] = sum(a["action_executed"] in (YAW_LEFT,) for a in act)', [T_SUMM], "转向步只算左转"),
    ("Z14", f"{EXP}/summarize.py", 'res = {"class": cls, "k": K, "arms": arms, "missing_routes": [list(x) for x in missing], "failed": data["failed"],\n           "complete": not missing and not data["failed"] and',
     'res = {"class": cls, "k": K, "arms": arms, "missing_routes": [list(x) for x in missing], "failed": data["failed"],\n           "complete": not data["failed"] and', [T_SUMM], "完整性不看缺的路线"),
    ("Z15", f"{EXP}/summarize.py", 'f.write(("" if complete_all else "数据不全；") + "；".join(verdict_parts) + "\\n")', 'f.write("；".join(verdict_parts) + "\\n")', [T_SUMM], "数据不全时 verdict.txt 不标注"),
    ("Z16", f"{EXP}/summarize.py", 'if x["collision_step"] == 0)', 'if x["collision_step"] == 1)', [T_SUMM], "出生即碰撞按第 1 步算"),
    ("Z17", f"{EXP}/summarize.py", "    if v1_missing:\n", "    if False:\n", [T_SUMM], "K1 缺键也照判"),
    ("Z18", f"{EXP}/summarize.py", '"sr_ge": sr >= PRE["out_sr"]', '"sr_ge": sr > PRE["out_sr"]', [T_SUMM], "室外 SR 边界（>= 改成 >）"),
    ("Z19", f"{EXP}/summarize.py", '"bc_le": bc <= PRE["out_bc"]', '"bc_le": bc < PRE["out_bc"]', [T_SUMM], "室外 b − c 边界（<= 改成 <）"),
]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run(cmd, tree, log, timeout=240):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": D + "/tmp", "CUDA_VISIBLE_DEVICES": GPU, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.time()
    try:
        p = subprocess.run(["nice", "-n", "10", *cmd], cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True, errors="replace")
        out, rc, to = p.stdout, p.returncode, False
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout
        out = partial if isinstance(partial, str) else (partial or b"").decode("utf-8", "replace")
        rc, to = None, True
    dt = time.time() - t0
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + f"\n[rc={rc} timeout={to} {dt:.1f}s]\n" + out[-6000:])
    summary = [ln for ln in out.splitlines() if ln.startswith(("Ran ", "FAILED", "OK"))]
    return {"rc": rc, "timeout": to, "secs": dt, "summary": " ".join(summary[-3:])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", required=True, help="git archive 出来的代码副本（会被临时改写，改完改回）")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--only", default="", help="只跑编号以此开头的变异")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    sel = [m for m in M if m[0].startswith(a.only)]
    ids = [m[0] for m in M]
    assert len(set(ids)) == len(ids), "变异编号重复"
    if a.list:
        for m in sel:
            print(f"{m[0]:4s} {m[1]:70s} [{','.join(t[0] for t in m[4])}] {m[5]}")
        return 0
    os.makedirs(a.logs, exist_ok=True)
    tree = os.path.abspath(a.tree)
    files = sorted({m[1] for m in sel})
    pristine = {f: sha(os.path.join(tree, f)) for f in files}
    tests = {}
    for m in sel:
        for name, cmd in m[4]:
            tests[name] = cmd
    print(f"变异 {len(sel)} 个，测试命令 {len(tests)} 条，副本 {tree}", flush=True)
    for name, cmd in tests.items():                                   # 对照：不变异时必须全绿
        r = run(cmd, tree, os.path.join(a.logs, f"control-{name.replace(':', '_')}.log"))
        print(f"对照 {name:70s} rc={r['rc']} {r['secs']:.1f}s {r['summary']}", flush=True)
        if r["rc"] != 0:
            print("对照不绿，停（先让副本里的测试全绿）", file=sys.stderr)
            return 2
    rows, bad = [], 0
    for mid, rel, old, new, tlist, note in sel:
        path = os.path.join(tree, rel)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        n = src.count(old)
        if n != 1:
            print(f"{mid:4s} 定位失败：原文片段在 {rel} 里出现 {n} 次（要恰好 1 次）| {note}", flush=True)
            rows.append((mid, rel, "定位失败", "", note))
            bad += 1
            continue
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(src.replace(old, new))
            killed_by, last = None, None
            for name, cmd in tlist:
                r = run(cmd, tree, os.path.join(a.logs, f"{mid}.{name.replace(':', '_')}.log"))
                last = r
                if r["rc"] != 0:
                    killed_by = name + ("（超时）" if r["timeout"] else "")
                    break
        finally:
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
        status = "红" if killed_by else ("等价（已说明）" if mid in EQUIV else "**没变红**")
        bad += killed_by is None and mid not in EQUIV
        rows.append((mid, rel, status, killed_by or (EQUIV[mid] if mid in EQUIV else ""), note))
        print(f"{mid:4s} {status:8s} {killed_by or '':36s} {rel.split('/')[-1]:22s} | {note} | {last['summary'] if last else ''}", flush=True)
    for f in files:
        assert sha(os.path.join(tree, f)) == pristine[f], f"{f} 没有改回原样"
    with open(os.path.join(a.logs, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| 编号 | 文件 | 结果 | 杀死它的测试 | 改了什么 |\n| --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |\n")
    n_red = sum(r[2] == "红" for r in rows)
    n_equiv = sum(r[2].startswith("等价") for r in rows)
    print(f"全部 {len(sel)} 个：变红 {n_red}，等价（已说明）{n_equiv}，没变红或定位失败 {bad}；被改文件已核对与原文一致", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
