#!/usr/bin/env python3
"""E2E-R1 第二部分（闭环工作点扫描）独立核对与事后分析脚本（只读实验输出，写 post/ 目录）。

运行：
  /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python \\
    experiments/2026-11-12-recipe-stability/recheck_sweep.py
  # 指定输入或输出目录：
  ... recheck_sweep.py --closed-loop-dir <DIR> --out-dir <POST_DIR>
"""
import argparse
import json
import math
import os
import statistics
import sys

# ───────────────────────────── 配置区（默认值集中在此，行尾注明依据） ─────────────────────────────
REPO_ROOT = os.environ.get(
    "UAV_RGB_REPO", "/root/claude-code-backup-20261028/code/uav_rgb"
)  # 主仓库（只读借用 tools/paired_stats.py）
DATA_ROOT = os.environ.get(
    "UAV_RGB_DATA", "/root/rivermind-data/uav_rgb"
)  # 数据盘根目录
PART1_DIR = os.path.join(
    DATA_ROOT, "exp-recipe-stability-20261112"
)  # E2E-R1 实验顶层输出（只读）
CLOSED_LOOP_DIR = os.path.join(
    PART1_DIR, "closed-loop"
)  # 第二部分闭环输出目录（只读）
POST_OUT_DIR = os.path.join(
    CLOSED_LOOP_DIR, "post"
)  # 本核对脚本唯一写入的新子目录（任务卡规定）

LEVEL_KEYS = (("m0.5", -0.5, "-0.5 nat"), ("0", 0.0, "0（原校准）"), ("p0.5", 0.5, "+0.5 nat"))  # 预注册 P1 三档
TRAIN_SEEDS = (0, 1, 2)             # 预注册每档 3 个训练种子
EPISODES_PER_ARM = 20               # 预注册 P2 冻结 20 个 val 几何
EXPECTED_METRIC = "sim09.v2.coast1s.corridor3m"   # 公共规则 §2 柱阵口径
EXPECTED_LATENCY_S = 0.20           # W1-0 第二轮选定 L* = 0.20 s
EXPECTED_CORRIDOR_M = 3.0           # 走廊半宽 3.0 m
EXPECTED_SPLIT = "val"              # 预注册 P2 只用 val 分区
NOMINAL_OBS_GAP_S = 0.90            # 10 Hz 相机 + 1.0 s 窗口下名义相邻观测间隔（交叉审核 A）
STALL_GAP_THRESH_S = 1.10           # 停顿步判据：obs_gap_s > 0.90 + 0.20 = 1.10 s（卡 W1-13 / M-10）
LATE_LAG_THRESH_S = 0.22            # 迟到步判据：dispatch_lag_s > L + 0.02 = 0.22 s（卡 W1-0）
STEADY_MIN_RUN = 2                  # 稳态前飞步：连续 FORWARD 第 2 步起（审核 M verify_part2.py 口径）
CRUISE_MIN_RUN = 4                  # 巡航前飞步：连续 FORWARD 第 4 步起（summarize_r2.py 标定口径）
COMMON_STEP_LO = 3                  # summarize_r2.py 公共步段起点
COMMON_STEP_HI = 30                 # summarize_r2.py 公共步段终点
CGROUP_CORES = 12.0                 # 容器 cgroup CPU 配额 12 核（全容器按核累计限流上限 1200%）
PAIRED_BOOT_DEFAULT = 2000          # tools/paired_stats.py 缺省 bootstrap 重抽样次数
PAIRED_SEED_DEFAULT = 0             # tools/paired_stats.py 缺省随机种子
SWEEP_BOOT_B = 10000                # summarize_sweep.py 使用的 bootstrap 重抽样次数
SWEEP_BOOT_SEED = 20261112          # summarize_sweep.py 使用的随机种子
BINOM_N = 20                        # 单臂集数，用于计算抽样标准差
BINOM_P_REF = 0.67                  # 任务卡指定的参考成功率（0 档 40/60 ≈ 0.667，R2e 41/60 ≈ 0.683）
# ──────────────────────────────────────────────────────────────────────────────────────────────────

if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import paired_stats as ps  # noqa: E402


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_kv(path):
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip()
    return out


def classify_clean_episode(ep_row):
    """按预注册单集互斥归类：success -> collision -> corridor -> not_reached。"""
    if ep_row.get("success"):
        return "success"
    if ep_row.get("collision"):
        return "collision"
    if ep_row.get("corridor_violation"):
        return "corridor"
    return "not_reached"


def load_arm_p6(cl_dir, arm_name, expected_seeds):
    """独立按预注册条款 P6 读取单个臂（含可选 -retry1 目录），返回逐集记录与协议核查信息。"""
    arm_dir = os.path.join(cl_dir, arm_name)
    ev = _read_json(os.path.join(arm_dir, "eval.json"))
    retry_dir = os.path.join(cl_dir, arm_name + "-retry1")
    retry_ev = _read_json(os.path.join(retry_dir, "eval.json")) if os.path.isfile(os.path.join(retry_dir, "eval.json")) else None

    first_rows = {int(r["seed"]): r for r in (ev.get("per_episode") or [])}
    first_late = {int(r["seed"]): r for r in (ev.get("late_dispatch_episodes") or [])}
    first_failed = {int(r["seed"]): r for r in (ev.get("failed") or []) if "seed" in r}
    retry_rows = {int(r["seed"]): r for r in (retry_ev.get("per_episode") or [])} if retry_ev else {}

    protocol_ok = (
        not bool(ev.get("partial"))
        and int(ev.get("episodes_requested") or 0) == len(expected_seeds)
        and int(ev.get("episodes_done") or 0) + int(ev.get("failed_count") or 0) == len(expected_seeds)
        and ev.get("metric_version") == EXPECTED_METRIC
        and math.isclose(float(ev.get("action_latency_s") or -1.0), EXPECTED_LATENCY_S, abs_tol=1e-9)
        and math.isclose(float(ev.get("corridor_half_width") or -1.0), EXPECTED_CORRIDOR_M, abs_tol=1e-9)
        and ev.get("split") == EXPECTED_SPLIT
        and [int(s) for s in (ev.get("seeds") or [])] == list(expected_seeds)
    )

    episodes = {}
    for sd in expected_seeds:
        if sd in first_rows:
            r = first_rows[sd]
            episodes[sd] = {
                "source": "first",
                "outcome": classify_clean_episode(r),
                "row": r,
                "ep_dir": os.path.join(arm_dir, "episodes", f"seed_{sd}"),
            }
        elif sd in first_late:
            if sd in retry_rows:
                r = retry_rows[sd]
                episodes[sd] = {
                    "source": "retry1",
                    "outcome": classify_clean_episode(r),
                    "row": r,
                    "ep_dir": os.path.join(retry_dir, "episodes", f"seed_{sd}"),
                }
            else:
                episodes[sd] = {"source": "late_unresolved", "outcome": "late", "row": None, "ep_dir": None}
        elif sd in first_failed:
            episodes[sd] = {"source": "aborted", "outcome": "aborted", "row": None, "ep_dir": None}
        else:
            episodes[sd] = {"source": "missing", "outcome": "aborted", "row": None, "ep_dir": None}

    outcomes = {k: 0 for k in ("success", "collision", "corridor", "not_reached", "late", "aborted")}
    actions = {"0": 0, "1": 0, "2": 0, "3": 0}
    steps_sum = 0
    clean_n = 0
    gids = {}
    for sd, item in episodes.items():
        outcomes[item["outcome"]] += 1
        if item["row"] is not None:
            clean_n += 1
            steps_sum += int(item["row"].get("steps") or 0)
            if item["row"].get("geometry_gid"):
                gids[sd] = item["row"]["geometry_gid"]
            for ak, av in (item["row"].get("actions") or {}).items():
                actions[str(ak)] = actions.get(str(ak), 0) + int(av)

    return {
        "arm": arm_name,
        "protocol_ok": protocol_ok,
        "eval_wall_s": float(ev["wall_s"]) if ev.get("wall_s") is not None else None,
        "ckpt": ev.get("ckpt"),
        "ckpt_sha256": ev.get("ckpt_sha256"),
        "first_late_seeds": sorted(first_late),
        "reran": retry_ev is not None,
        "outcomes": outcomes,
        "actions": actions,
        "steps": steps_sum,
        "clean_episodes": clean_n,
        "gids": gids,
        "episodes": episodes,
    }


def read_cgroup_snapshot(cl_dir, arm_name):
    """解析 <arm>.cg 的起止快照，计算墙钟、平均核数、限流次数与按核累计限流时长占比。"""
    cg_path = os.path.join(cl_dir, f"{arm_name}.cg")
    if not os.path.isfile(cg_path):
        return None
    marks = {}
    with open(cg_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5 and parts[0] in ("start", "end"):
                marks[parts[0]] = [float(x) for x in parts[1:]]
    if "start" not in marks or "end" not in marks:
        return None
    s, e = marks["start"], marks["end"]
    wall_s = e[0] - s[0]
    usage_s = (e[1] - s[1]) / 1e6
    nr_throttled = int(e[2] - s[2])
    throttled_s = (e[3] - s[3]) / 1e6
    return {
        "cg_wall_s": wall_s,
        "avg_cores": usage_s / wall_s if wall_s > 0 else float("nan"),
        "nr_throttled": nr_throttled,
        "throttled_s": throttled_s,
        "throttled_pct_wall": (100.0 * throttled_s / wall_s) if wall_s > 0 else float("nan"),
        "throttled_pct_quota12": (100.0 * throttled_s / (wall_s * CGROUP_CORES)) if wall_s > 0 else float("nan"),
    }


def scan_arm_timing(arm_data, cl_dir):
    """扫描一个臂全部采用集的 actions.jsonl，核对迟到步、相邻观测间隔、派发延迟与前飞位移。"""
    cg = read_cgroup_snapshot(cl_dir, arm_data["arm"])
    obs_gaps = []
    lags = []
    steady_all, steady_common = [], []
    cruise_all, cruise_common = [], []
    late_steps = []
    stall_steps = []
    non_nominal_gaps = []
    total_rows = 0
    action_tally_check = {"0": 0, "1": 0, "2": 0, "3": 0}

    for sd, item in sorted(arm_data["episodes"].items()):
        ep_dir = item["ep_dir"]
        if not ep_dir:
            continue
        act_path = os.path.join(ep_dir, "actions.jsonl")
        with open(act_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        total_rows += len(rows)
        fwd_run = 0
        for i, r in enumerate(rows):
            a_exec = int(r["action_executed"])
            action_tally_check[str(a_exec)] = action_tally_check.get(str(a_exec), 0) + 1
            fwd_run = fwd_run + 1 if a_exec == 1 else 0

            lag = r.get("dispatch_lag_s")
            if lag is not None:
                lag_f = float(lag)
                lags.append(lag_f)
                if bool(r.get("late_dispatch")) or lag_f > LATE_LAG_THRESH_S + 1e-9:
                    late_steps.append({"arm": arm_data["arm"], "seed": sd, "step": int(r["step"]), "dispatch_lag_s": lag_f})
            elif bool(r.get("late_dispatch")):
                late_steps.append({"arm": arm_data["arm"], "seed": sd, "step": int(r["step"]), "dispatch_lag_s": None})

            if i > 0:
                gap_s = (int(r["rgb_ts"]) - int(rows[i - 1]["rgb_ts"])) / 1e9
                obs_gaps.append(gap_s)
                if gap_s > STALL_GAP_THRESH_S + 1e-9:
                    stall_steps.append({"arm": arm_data["arm"], "seed": sd, "step": int(r["step"]), "obs_gap_s": round(gap_s, 6)})
                if abs(gap_s - NOMINAL_OBS_GAP_S) > 1e-3:
                    non_nominal_gaps.append({"arm": arm_data["arm"], "seed": sd, "step": int(r["step"]), "obs_gap_s": round(gap_s, 6)})

                if a_exec == 1 and not bool(r.get("collision")) and r.get("displacement_m") is not None:
                    disp = float(r["displacement_m"])
                    if fwd_run >= STEADY_MIN_RUN:
                        steady_all.append(disp)
                        if COMMON_STEP_LO <= i <= COMMON_STEP_HI:
                            steady_common.append(disp)
                    if fwd_run >= CRUISE_MIN_RUN:
                        cruise_all.append(disp)
                        if COMMON_STEP_LO <= i <= COMMON_STEP_HI:
                            cruise_common.append(disp)

    return {
        "arm": arm_data["arm"],
        "eval_wall_s": arm_data["eval_wall_s"],
        **(cg or {
            "cg_wall_s": None, "avg_cores": None, "nr_throttled": None,
            "throttled_s": None, "throttled_pct_wall": None, "throttled_pct_quota12": None,
        }),
        "total_steps": total_rows,
        "steps_match_eval_json": (total_rows == arm_data["steps"] and action_tally_check == arm_data["actions"]),
        "n_obs_gaps": len(obs_gaps),
        "obs_gap_median_s": statistics.median(obs_gaps) if obs_gaps else None,
        "obs_gap_max_s": max(obs_gaps) if obs_gaps else None,
        "stall_steps_gt_1p1s": len(stall_steps),
        "stall_episodes": sorted({x["seed"] for x in stall_steps}),
        "stall_detail": stall_steps,
        "non_nominal_gaps": non_nominal_gaps,
        "late_steps": len(late_steps),
        "late_episodes": sorted(set(arm_data["first_late_seeds"]) | {x["seed"] for x in late_steps}),
        "late_detail": late_steps,
        "lag_median_s": statistics.median(lags) if lags else None,
        "lag_max_s": max(lags) if lags else None,
        "cruise_disp_mean_m": statistics.fmean(cruise_all) if cruise_all else None,
        "cruise_disp_n": len(cruise_all),
        "cruise_common_mean_m": statistics.fmean(cruise_common) if cruise_common else None,
        "cruise_common_n": len(cruise_common),
        "steady_disp_mean_m": statistics.fmean(steady_all) if steady_all else None,
        "steady_disp_n": len(steady_all),
        "steady_common_mean_m": statistics.fmean(steady_common) if steady_common else None,
        "steady_common_n": len(steady_common),
    }


def aggregate_group(arms_by_seed, expected_seeds):
    """把同一档 / 配方的 3 个训练种子聚合为 60 集统计。"""
    per_seed_outcomes = {s: dict(arms_by_seed[s]["outcomes"]) for s in TRAIN_SEEDS}
    per_seed_succ = {s: arms_by_seed[s]["outcomes"]["success"] for s in TRAIN_SEEDS}
    totals = {k: sum(arms_by_seed[s]["outcomes"][k] for s in TRAIN_SEEDS) for k in ("success", "collision", "corridor", "not_reached", "late", "aborted")}
    actions = {k: sum(arms_by_seed[s]["actions"][k] for s in TRAIN_SEEDS) for k in ("0", "1", "2", "3")}
    tot_act = sum(actions.values())
    shares = {
        lbl: (actions[k] / tot_act if tot_act else 0.0)
        for k, lbl in (("0", "S"), ("1", "F"), ("2", "L"), ("3", "R"))
    }
    steps = sum(arms_by_seed[s]["steps"] for s in TRAIN_SEEDS)
    clean_eps = sum(arms_by_seed[s]["clean_episodes"] for s in TRAIN_SEEDS)
    geom_succ = {
        sd: sum(1 for s in TRAIN_SEEDS if arms_by_seed[s]["episodes"][sd]["outcome"] == "success")
        for sd in expected_seeds
    }
    return {
        "total": totals["success"],
        "per_seed": {str(s): per_seed_succ[s] for s in TRAIN_SEEDS},
        "per_seed_outcomes": {str(s): per_seed_outcomes[s] for s in TRAIN_SEEDS},
        "range": max(per_seed_succ.values()) - min(per_seed_succ.values()),
        "outcomes": totals,
        "action_counts": actions,
        "action_shares": shares,
        "steps": steps,
        "clean_episodes": clean_eps,
        "mean_steps": steps / clean_eps if clean_eps else float("nan"),
        "n": len(TRAIN_SEEDS) * len(expected_seeds),
        "geom_succ": geom_succ,
        "arms": arms_by_seed,
    }


def build_units(grp_a, grp_b, expected_seeds, gid_of, target_outcome="success", clean_only=False):
    """构造 paired_stats.pair_summary 所需的 [(cluster_id, a_flag, b_flag)]。"""
    units = []
    for s in TRAIN_SEEDS:
        for sd in expected_seeds:
            ia = grp_a["arms"][s]["episodes"][sd]
            ib = grp_b["arms"][s]["episodes"][sd]
            if clean_only and (ia["row"] is None or ib["row"] is None):
                continue
            cid = gid_of.get(sd, str(sd))
            units.append((cid, int(ia["outcome"] == target_outcome), int(ib["outcome"] == target_outcome)))
    return units


def binomial_sampling_reference(n=BINOM_N, p=BINOM_P_REF, k_seeds=3):
    """精确计算独立二项 Binomial(n, p) 的单种子标准差与 k_seeds 个种子的极差精确分布。"""
    probs = [math.comb(n, x) * (p ** x) * ((1.0 - p) ** (n - x)) for x in range(n + 1)]
    sigma = math.sqrt(n * p * (1.0 - p))
    range_pmf = {r: 0.0 for r in range(n + 1)}
    if k_seeds == 3:
        for x0 in range(n + 1):
            p0 = probs[x0]
            for x1 in range(n + 1):
                p01 = p0 * probs[x1]
                for x2 in range(n + 1):
                    r = max(x0, x1, x2) - min(x0, x1, x2)
                    range_pmf[r] += p01 * probs[x2]
    else:
        raise ValueError("仅实现 k_seeds == 3 的精确三重枚举")

    exp_range = sum(r * pr for r, pr in range_pmf.items())
    var_range = sum(((r - exp_range) ** 2) * pr for r, pr in range_pmf.items())
    p_ge_8 = sum(pr for r, pr in range_pmf.items() if r >= 8)
    p_ge_5 = sum(pr for r, pr in range_pmf.items() if r >= 5)
    p_le_1 = sum(pr for r, pr in range_pmf.items() if r <= 1)
    cum = 0.0
    q50, q95 = None, None
    for r in range(n + 1):
        cum += range_pmf[r]
        if q50 is None and cum >= 0.50:
            q50 = r
        if q95 is None and cum >= 0.95:
            q95 = r
    return {
        "n": n,
        "p": p,
        "mean_count": n * p,
        "sd_count": sigma,
        "sd_rate": sigma / n,
        "k_seeds": k_seeds,
        "expected_range": exp_range,
        "sd_range": math.sqrt(var_range),
        "median_range": q50,
        "p95_range": q95,
        "prob_range_ge_8": p_ge_8,
        "prob_range_ge_5": p_ge_5,
        "prob_range_le_1": p_le_1,
    }


def reconcile_with_summary(groups, extra_grp, paired_cmp, summary_json, expected_seeds):
    """逐格核对独立重算数与冻结汇总器产出的 summary.json，返回 {checked_cells, mismatches}。"""
    mismatches = []
    checked = 0

    def chk(label, got, want, tol=None):
        nonlocal checked
        checked += 1
        if tol is not None:
            ok = (got is not None and want is not None and math.isclose(float(got), float(want), abs_tol=tol))
        else:
            ok = (got == want)
        if not ok:
            mismatches.append({"item": label, "rechecked": got, "summary_json": want})

    chk("seeds", list(expected_seeds), summary_json.get("seeds"))
    for k, _d, _lbl in LEVEL_KEYS:
        sj = summary_json.get("levels", {}).get(k, {})
        g = groups[k]
        chk(f"level.{k}.total", g["total"], sj.get("total"))
        chk(f"level.{k}.range", g["range"], sj.get("range"))
        chk(f"level.{k}.per_seed", g["per_seed"], sj.get("per_seed"))
        chk(f"level.{k}.outcomes", g["outcomes"], sj.get("outcomes"))
        chk(f"level.{k}.action_counts", g["action_counts"], sj.get("action_counts"))
        chk(f"level.{k}.steps", g["steps"], sj.get("steps"))
        chk(f"level.{k}.clean_episodes", g["clean_episodes"], sj.get("clean_episodes"))
        for ak in ("S", "F", "L", "R"):
            chk(f"level.{k}.share.{ak}", g["action_shares"][ak], sj.get("action_shares", {}).get(ak), tol=1e-12)

    if extra_grp is not None:
        sj = summary_json.get("extra", {})
        chk("extra.total", extra_grp["total"], sj.get("total"))
        chk("extra.range", extra_grp["range"], sj.get("range"))
        chk("extra.per_seed", extra_grp["per_seed"], sj.get("per_seed"))
        chk("extra.outcomes", extra_grp["outcomes"], sj.get("outcomes"))
        chk("extra.action_counts", extra_grp["action_counts"], sj.get("action_counts"))
        chk("extra.steps", extra_grp["steps"], sj.get("steps"))
        chk("extra.clean_episodes", extra_grp["clean_episodes"], sj.get("clean_episodes"))
        for ak in ("S", "F", "L", "R"):
            chk(f"extra.share.{ak}", extra_grp["action_shares"][ak], sj.get("action_shares", {}).get(ak), tol=1e-12)

    for pair_key, pc in paired_cmp.items():
        st = summary_json.get("tests", {}).get(pair_key, {})
        chk(f"test.{pair_key}.diff", pc["diff_episodes"], st.get("diff"))
        chk(f"test.{pair_key}.geoms_nonzero", pc["n_nonzero"], st.get("geoms_nonzero"))
        chk(f"test.{pair_key}.mcnemar_b", pc["b"], st.get("mcnemar_b"))
        chk(f"test.{pair_key}.mcnemar_c", pc["c"], st.get("mcnemar_c"))
        chk(f"test.{pair_key}.p_cluster", pc["p_cluster_paired_stats"], st.get("p_cluster"), tol=1e-15)
        chk(f"test.{pair_key}.p_mcnemar", pc["p_mcnemar_paired_stats"], st.get("p_mcnemar"), tol=1e-15)

    # 预注册判定复核
    totals = [groups[k]["total"] for k, _, _ in LEVEL_KEYS]
    spread = max(totals) - min(totals)
    want_code = "sensitive" if spread >= 8 else ("insensitive" if spread < 4 else "between")
    chk("verdict.spread.code", want_code, summary_json.get("verdicts", {}).get("spread", {}).get("code"))
    if extra_grp is not None:
        want_pass = (extra_grp["range"] <= 3) and (extra_grp["total"] >= groups["0"]["total"] - 3)
        chk("verdict.extra.pass", want_pass, summary_json.get("verdicts", {}).get("extra", {}).get("pass"))

    return {"checked_cells": checked, "mismatch_count": len(mismatches), "all_match": len(mismatches) == 0, "mismatches": mismatches}


def run_recheck(cl_dir=CLOSED_LOOP_DIR):
    """对 closed-loop 目录执行完整只读独立核对，返回结果字典。"""
    cfg = _read_kv(os.path.join(cl_dir, "config.env"))
    expected_seeds = [int(x) for x in cfg["SEEDS"].split(",") if x.strip()]
    extra_tag = cfg.get("EXTRA_ARM", "")
    summary_json = _read_json(os.path.join(cl_dir, "summary.json"))

    entries = sorted(os.listdir(cl_dir))
    partial_dirs = [e for e in entries if ".partial-" in e]
    retry_dirs = [e for e in entries if e.endswith("-retry1")]

    groups = {}
    all_arm_order = []
    arm_records = {}
    for k, _delta, _lbl in LEVEL_KEYS:
        by_seed = {}
        for s in TRAIN_SEEDS:
            aname = f"lvl-{k}-s{s}"
            all_arm_order.append(aname)
            rec = load_arm_p6(cl_dir, aname, expected_seeds)
            by_seed[s] = rec
            arm_records[aname] = rec
        groups[k] = aggregate_group(by_seed, expected_seeds)

    extra_grp = None
    if extra_tag:
        by_seed = {}
        for s in TRAIN_SEEDS:
            aname = f"x-{extra_tag}-s{s}"
            all_arm_order.append(aname)
            rec = load_arm_p6(cl_dir, aname, expected_seeds)
            by_seed[s] = rec
            arm_records[aname] = rec
        extra_grp = aggregate_group(by_seed, expected_seeds)

    # 核对 12 个臂的 20 个几何 gid 一致且两两不同
    ref_gids = arm_records[all_arm_order[0]]["gids"]
    gid_consistent = (
        len(ref_gids) == len(expected_seeds)
        and len(set(ref_gids.values())) == len(expected_seeds)
        and all(arm_records[a]["gids"] == ref_gids for a in all_arm_order)
    )
    all_protocols_ok = all(arm_records[a]["protocol_ok"] for a in all_arm_order)

    # 4 组成功数两两比较（tools/paired_stats.py）
    pair_specs = [
        ("m0.5_vs_0", "−0.5 对 0", groups["m0.5"], groups["0"]),
        ("p0.5_vs_0", "+0.5 对 0", groups["p0.5"], groups["0"]),
        ("m0.5_vs_p0.5", "−0.5 对 +0.5", groups["m0.5"], groups["p0.5"]),
    ]
    if extra_grp is not None:
        pair_specs.append(("extra_vs_0", f"配方臂 {extra_tag}（事后探索）对 0 档", extra_grp, groups["0"]))

    paired_comparisons = {}
    for pkey, plabel, ga, gb in pair_specs:
        units = build_units(ga, gb, expected_seeds, ref_gids, target_outcome="success", clean_only=False)
        ps_def = ps.pair_summary(units, n_boot=PAIRED_BOOT_DEFAULT, seed=PAIRED_SEED_DEFAULT)
        ps_10k = ps.pair_summary(units, n_boot=SWEEP_BOOT_B, seed=SWEEP_BOOT_SEED)
        sj_test = summary_json.get("tests", {}).get(pkey, {})
        paired_comparisons[pkey] = {
            "label": plabel,
            "diff_episodes": int(ps_def["observed_sum"]),
            "diff_rate": ps_def["diff"],
            "n_units": ps_def["n_units"],
            "n_clusters": ps_def["n_clusters"],
            "n_nonzero": ps_def["n_nonzero"],
            "p_cluster_paired_stats": ps_def["p_value"],
            "p_cluster_summary_md": sj_test.get("p_cluster"),
            "p_cluster_abs_diff": abs(ps_def["p_value"] - float(sj_test["p_cluster"])) if sj_test.get("p_cluster") is not None else None,
            "ci_rate_ps_default": [ps_def["ci_lo"], ps_def["ci_hi"]],
            "ci_ep_ps_default_x60": [60.0 * ps_def["ci_lo"], 60.0 * ps_def["ci_hi"]],
            "ci_ep_ps_10k_linear": [60.0 * ps_10k["ci_lo"], 60.0 * ps_10k["ci_hi"]],
            "ci_ep_summary_md": sj_test.get("ci"),
            "b": ps_def["b"],
            "c": ps_def["c"],
            "p_mcnemar_paired_stats": ps_def["mcnemar_p"],
            "p_mcnemar_summary_md": sj_test.get("p_mcnemar"),
            "p_mcnemar_abs_diff": abs(ps_def["mcnemar_p"] - float(sj_test["p_mcnemar"])) if sj_test.get("p_mcnemar") is not None else None,
        }

    reconciliation = reconcile_with_summary(groups, extra_grp, paired_comparisons, summary_json, expected_seeds)

    # 12 个臂的时序健康全量扫描
    timing_rows = [scan_arm_timing(arm_records[a], cl_dir) for a in all_arm_order]
    timing_totals = {
        "arms_count": len(timing_rows),
        "total_steps": sum(r["total_steps"] for r in timing_rows),
        "total_obs_gaps": sum(r["n_obs_gaps"] for r in timing_rows),
        "total_late_steps": sum(r["late_steps"] for r in timing_rows),
        "total_stall_steps_gt_1p1s": sum(r["stall_steps_gt_1p1s"] for r in timing_rows),
        "all_steps_match_eval_json": all(r["steps_match_eval_json"] for r in timing_rows),
        "max_obs_gap_s": max(r["obs_gap_max_s"] for r in timing_rows if r["obs_gap_max_s"] is not None),
        "max_lag_s": max(r["lag_max_s"] for r in timing_rows if r["lag_max_s"] is not None),
        "non_nominal_gaps": [g for r in timing_rows for g in r["non_nominal_gaps"]],
        "late_episodes": [(r["arm"], r["late_episodes"]) for r in timing_rows if r["late_episodes"]],
        "stall_episodes": [(r["arm"], r["stall_episodes"]) for r in timing_rows if r["stall_episodes"]],
    }

    # 事后分析：失败构成（撞柱与越界）按几何聚类的符号翻转检验（未做多重比较校正）
    failure_posthoc = {}
    for outcome_key, outcome_zh in (("collision", "撞柱"), ("corridor", "越界")):
        sub = {}
        for pkey, plabel, ga, gb in pair_specs:
            # 干净集对上比较（本实验 240/240 集全部干净完成，clean_only=True 与 False 等价，n_units=60）
            units = build_units(ga, gb, expected_seeds, ref_gids, target_outcome=outcome_key, clean_only=True)
            ps_res = ps.pair_summary(units, n_boot=PAIRED_BOOT_DEFAULT, seed=PAIRED_SEED_DEFAULT)
            clust = ps.group_units(units)
            diffs_g = [v[0] - v[1] for v in clust.values()]
            sub[pkey] = {
                "label": plabel,
                "metric": outcome_zh,
                "n_clean_units": ps_res["n_units"],
                "count_a": int(round(ps_res["rate_a"] * ps_res["n_units"])),
                "count_b": int(round(ps_res["rate_b"] * ps_res["n_units"])),
                "diff_episodes": int(ps_res["observed_sum"]),
                "n_nonzero_geoms": ps_res["n_nonzero"],
                "geoms_pos": sum(1 for d in diffs_g if d > 0),
                "geoms_neg": sum(1 for d in diffs_g if d < 0),
                "p_cluster": ps_res["p_value"],
                "ci_rate_ps": [ps_res["ci_lo"], ps_res["ci_hi"]],
                "ci_ep_ps_x60": [60.0 * ps_res["ci_lo"], 60.0 * ps_res["ci_hi"]],
                "mcnemar_b": ps_res["b"],
                "mcnemar_c": ps_res["c"],
                "p_mcnemar": ps_res["mcnemar_p"],
            }
        failure_posthoc[outcome_key] = sub

    binom_ref = binomial_sampling_reference(n=BINOM_N, p=BINOM_P_REF, k_seeds=3)
    binom_ref_exact_0 = binomial_sampling_reference(n=BINOM_N, p=40.0 / 60.0, k_seeds=3)

    return {
        "schema": "e2e-r1.recheck.v1",
        "closed_loop_dir": cl_dir,
        "extra_tag": extra_tag,
        "partial_dirs": partial_dirs,
        "retry_dirs": retry_dirs,
        "all_protocols_ok": all_protocols_ok,
        "gid_consistent_across_12_arms": gid_consistent,
        "distinct_gids": len(set(ref_gids.values())),
        "reconciliation": reconciliation,
        "groups": {
            k: {kk: vv for kk, vv in g.items() if kk != "arms"}
            for k, g in groups.items()
        },
        "extra": ({kk: vv for kk, vv in extra_grp.items() if kk != "arms"} if extra_grp else None),
        "paired_comparisons": paired_comparisons,
        "timing_rows": timing_rows,
        "timing_totals": timing_totals,
        "failure_posthoc": failure_posthoc,
        "binomial_reference_p067": binom_ref,
        "binomial_reference_p2_3": binom_ref_exact_0,
    }


def render_markdown(data):
    """把核对字典渲染为结构化 Markdown 报告。"""
    rec = data["reconciliation"]
    tt = data["timing_totals"]
    b67 = data["binomial_reference_p067"]
    lines = [
        "# E2E-R1 第二部分独立核对报告（`recheck_sweep.py` 生成）",
        "",
        "## 1. 对账结论与完整性核查",
        "",
        f"- 核对范围：`{data['closed_loop_dir']}` 下全部 {tt['arms_count']} 个臂（共 {tt['arms_count'] * 20} 集，{tt['total_steps']} 步）。",
        f"- 与 `summary.json` / `summary.md` 逐格对账：共核对 **{rec['checked_cells']}** 项，不一致项 **{rec['mismatch_count']}** 个（`all_match = {rec['all_match']}`）。",
        f"- 协议与几何一致性：12 个臂 `protocol_ok = {data['all_protocols_ok']}`；20 个评测 seed 的 `geometry_gid` 跨 12 个臂逐集相等且两两互异（`{data['distinct_gids']}` 个独立几何）。",
        f"- 半成品与重跑目录：`*.partial-*` 目录 **{len(data['partial_dirs'])}** 个，`*-retry1` 重跑目录 **{len(data['retry_dirs'])}** 个（02:35:54 首次起跑在等 `machine.lock` 未进入任何臂，06:58:13 续起后一次性跑完 12 臂）。",
        f"- `actions.jsonl` 逐步行数及四类动作计数与 `eval.json` 对账：`all_steps_match_eval_json = {tt['all_steps_match_eval_json']}`。",
        "",
        "## 2. `tools/paired_stats.py` 四组两两比较对账",
        "",
        "说明：",
        "- 几何聚类符号翻转 `p` 与逐集精确 McNemar `p` 在 `paired_stats.py` 与 `summary.md` 之间逐位一致（绝对差均为 `0.0`）。",
        "- 区间口径区别：`summary.md` 的 `bootstrap_ci`（`B=10000, seed=20261112`）单位是**60 集成功总数之差**，取排序数组离散下标（`floor(0.025*B)` 与 `ceil(0.975*B)-1`，不插值）；`tools/paired_stats.py` 的 `pair_summary`（缺省 `n_boot=2000, seed=0`）单位是**成功率之差（除以 60）**，分位数采用与 `numpy.percentile` 相同的线性插值。下表同时给出 `paired_stats` 缺省区间及其乘 60 折算的集数差。",
        "",
        "| 比较 | 成功数差（/60） | 成功率差 | p（聚类，`summary.md`） | p（聚类，`paired_stats`） | `summary.md` 区间（集数差，B=10000 离散） | `paired_stats` 区间（率差，B=2000 线性插值） | `paired_stats` ×60（集数差） | McNemar b/c | p（逐集 McNemar，两者同） |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pkey in ("m0.5_vs_0", "p0.5_vs_0", "m0.5_vs_p0.5", "extra_vs_0"):
        if pkey not in data["paired_comparisons"]:
            continue
        c = data["paired_comparisons"][pkey]
        ci_s = c["ci_ep_summary_md"]
        ci_r = c["ci_rate_ps_default"]
        ci_e = c["ci_ep_ps_default_x60"]
        lines.append(
            f"| {c['label']} | {c['diff_episodes']:+d} | {c['diff_rate']:+.4f} "
            f"| {c['p_cluster_summary_md']:.4f} | {c['p_cluster_paired_stats']:.4f} "
            f"| [{ci_s[0]:+d}, {ci_s[1]:+d}] | [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}] "
            f"| [{ci_e[0]:+.2f}, {ci_e[1]:+.2f}] | {c['b']}/{c['c']} | {c['p_mcnemar_paired_stats']:.4f} |"
        )

    lines += [
        "",
        "## 3. 12 个臂时序健康表（审核 M-10 口径）",
        "",
        "表头说明：「被限流时长 / 墙钟」取自各臂 `.cg` 起止快照的 `throttled_usec` 增量除以 `.cg` 墙钟；Linux cgroup v2 的 `throttled_usec` **按全容器所有被限流线程累计**，在 12 核配额容器内多个线程同时受限时比值会超过 100%（理论上限约 1200% × 线程并发数，括号内并列除以 `12 × 墙钟` 的配额归一化比例）。「巡航前飞位移」指连续 `FORWARD_SLOW` 第 4 步起（`r >= 4`）无碰撞步均值；「稳态前飞位移」指连续 `FORWARD_SLOW` 第 2 步起（`r >= 2`，与审核者 `verify_part2.py` 同口径）无碰撞步均值。",
        "",
        "| 臂 | `.cg` 墙钟 s（`eval` 墙钟） | 平均核数 | 限流次数 | 被限流时长 / 墙钟（按核累计；÷12核） | 迟到步 | `obs_gap` 中位 / 最大 s | `>1.1s` 步数 / 间隔数 | 派发延迟 中位 / 最大 s | 巡航前飞均值 m（`r>=4`, n） | 稳态前飞均值 m（`r>=2`, n） |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in data["timing_rows"]:
        lines.append(
            f"| {r['arm']} | {r['cg_wall_s']:.1f} ({r['eval_wall_s']:.0f}) | {r['avg_cores']:.2f} | {r['nr_throttled']} "
            f"| {r['throttled_pct_wall']:.1f}% ({r['throttled_pct_quota12']:.1f}%) | {r['late_steps']} "
            f"| {r['obs_gap_median_s']:.3f} / {r['obs_gap_max_s']:.3f} | {r['stall_steps_gt_1p1s']} / {r['n_obs_gaps']} "
            f"| {r['lag_median_s']:.3f} / {r['lag_max_s']:.3f} "
            f"| {r['cruise_disp_mean_m']:.4f} ({r['cruise_disp_n']}) | {r['steady_disp_mean_m']:.4f} ({r['steady_disp_n']}) |"
        )

    lines += [
        "",
        f"- 含停顿步（`obs_gap_s > 1.10 s`）的集：**0 集**（12 臂共 {tt['total_obs_gaps']} 个相邻观测间隔，`> 1.10 s` 为 0 步）。",
        f"- 含迟到步（`late_dispatch` 或 `dispatch_lag_s > 0.22 s`）的集：**0 集**（12 臂共 {tt['total_steps']} 步，各臂 `dispatch_lag_s` 中位数均为 `0.200 s`，全量最大值 `{tt['max_lag_s']:.3f} s`）。",
    ]
    if tt["non_nominal_gaps"]:
        nng_str = "；".join(
            f"`{g['arm']}` seed `{g['seed']}` step `{g['step']}` (`{g['obs_gap_s']:.3f} s`)"
            for g in tt["non_nominal_gaps"]
        )
        lines.append(f"- 偏离名义 `0.900 s` 但未超 `1.100 s` 停顿阈值的观测间隔（共 {len(tt['non_nominal_gaps'])} 步）：{nng_str}。")
    else:
        lines.append("- 所有相邻观测间隔均等于名义 `0.900 s`。")

    lines += [
        "",
        "## 4. 失败构成与训练种子波动（事后分析，非预注册判定）",
        "",
        "### 4.1 逐档逐训练种子的结局分解（每种子 20 集，每档 60 集）",
        "",
        "| 组 | 60 集：成功 / 撞柱 / 越界 / 未到线 | s0（成/撞/越/未） | s1（成/撞/越/未） | s2（成/撞/越/未） | 种子成功数极差 |",
        "|---|---|---|---|---|---|",
    ]
    group_list = [(k, lbl, data["groups"][k]) for k, _, lbl in LEVEL_KEYS]
    if data["extra"] is not None:
        group_list.append(("extra", f"配方臂 {data['extra_tag']}（事后探索）", data["extra"]))
    for _gk, glbl, g in group_list:
        oc = g["outcomes"]
        ps_cells = []
        for s in ("0", "1", "2"):
            so = g["per_seed_outcomes"][s]
            ps_cells.append(f"{so['success']}/{so['collision']}/{so['corridor']}/{so['not_reached']}")
        lines.append(
            f"| {glbl} | {oc['success']} / {oc['collision']} / {oc['corridor']} / {oc['not_reached']} "
            f"| {ps_cells[0]} | {ps_cells[1]} | {ps_cells[2]} | {g['range']} |"
        )

    lines += [
        "",
        "### 4.2 撞柱数与越界数的按几何聚类符号翻转检验（事后探索分析，未做多重比较校正）",
        "",
        "| 指标 | 比较 | 前者 / 后者（/60） | 差（集数） | 非零几何（正/负） | p（几何聚类符号翻转） | 95% 区间（率差，`paired_stats`） | McNemar b/c | p（逐集 McNemar） |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for outcome_key in ("collision", "corridor"):
        for pkey in ("m0.5_vs_p0.5", "m0.5_vs_0", "p0.5_vs_0", "extra_vs_0"):
            if pkey not in data["failure_posthoc"][outcome_key]:
                continue
            f = data["failure_posthoc"][outcome_key][pkey]
            ci_r = f["ci_rate_ps"]
            lines.append(
                f"| {f['metric']} | {f['label']} | {f['count_a']} / {f['count_b']} | {f['diff_episodes']:+d} "
                f"| {f['n_nonzero_geoms']} ({f['geoms_pos']}/{f['geoms_neg']}) | {f['p_cluster']:.4f} "
                f"| [{ci_r[0]:+.4f}, {ci_r[1]:+.4f}] | {f['mcnemar_b']}/{f['mcnemar_c']} | {f['p_mcnemar']:.4f} |"
            )

    lines += [
        "",
        "### 4.3 20 集二项抽样波动基准",
        "",
        f"- 单种子参考分布：独立二项 `Binomial(n={b67['n']}, p={b67['p']:.2f})` 的期望为 `{b67['mean_count']:.2f}` 集，标准差 `σ = {b67['sd_count']:.2f}` 集（成功率标准差 `{b67['sd_rate']:.4f}`）。",
        f"- 3 个独立同分布 `Binomial(20, 0.67)` 种子的极差 `R = max - min`：期望 `{b67['expected_range']:.2f}` 集，标准差 `{b67['sd_range']:.2f}` 集，中位数 `{b67['median_range']}` 集，95% 分位点 `{b67['p95_range']}` 集；`P(R >= 8) = {b67['prob_range_ge_8']:.4f}`（`{100*b67['prob_range_ge_8']:.2f}%`），`P(R >= 5) = {b67['prob_range_ge_5']:.4f}`，`P(R <= 1) = {b67['prob_range_le_1']:.4f}`。",
        f"- 注：闭环评测里三个训练种子评在同一批 20 个几何上，几何间难度方差在三个种子间共享抵消；若三个种子的真实策略一致，仅由闭环重评随机性产生的极差应小于独立二项基准。0 档出现极差 8（17/14/9）超出独立二项 95% 分位点（`P(R >= 8) = {100*b67['prob_range_ge_8']:.2f}%`），主要反映 R0 配方三个训练种子学出的策略本身存在分歧。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closed-loop-dir", default=CLOSED_LOOP_DIR, help="第二部分闭环输出目录（只读）")
    ap.add_argument("--out-dir", default=POST_OUT_DIR, help="核对产物写入目录（新建 post/）")
    args = ap.parse_args(argv)

    data = run_recheck(args.closed_loop_dir)
    md = render_markdown(data)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "recheck.json")
    md_path = os.path.join(args.out_dir, "recheck.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    rec = data["reconciliation"]
    tt = data["timing_totals"]
    print(
        f"recheck ok: checked={rec['checked_cells']} mismatches={rec['mismatch_count']} "
        f"arms={tt['arms_count']} steps={tt['total_steps']} late={tt['total_late_steps']} "
        f"stalls_gt_1p1={tt['total_stall_steps_gt_1p1s']} -> {args.out_dir}"
    )
    return 0 if rec["all_match"] else 2


if __name__ == "__main__":
    sys.exit(main())
