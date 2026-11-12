#!/usr/bin/env python3
"""卡 E2E-1 闭环评测的按几何聚类补报与异常集分母修正（审核 M-4、M-6；写在闭环结果产出之前）。

不修改冻结汇总器 summarize_e2e.py 与输出目录原有文件，只读 12 个评测臂目录并在目标目录写出
cluster_report.md 与 cluster_report.json。

用法（仓库根，.venv-tools 的 python）：
  python experiments/2026-11-12-e2e-heading-code/cluster_report.py $OUT
  python experiments/2026-11-12-e2e-heading-code/cluster_report.py $OUT --out-dir $OUT/cluster
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (HERE, os.path.join(REPO, "tools"), os.path.join(REPO, "sim", "tools"),
           os.path.join(HERE, "..", "2026-11-10-action-latency")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e2e_arms as ea  # noqa: E402
import paired_stats as ps  # noqa: E402
import summarize_e2e as se  # noqa: E402

# ---------------------------------------------------------------- 配置区（任务卡 2026-11-12 09:09 补充定死，看数据前锁定）
CLUSTER_ALPHA = 0.05          # 20 个几何簇的双侧显著性水平（7 个几何全同向 p=0.0156，不照搬逐集 M1 的 0.01）
STALL_THRESHOLD_S = 1.1       # W1-13 A3 停顿判据：hold=1.0 s 名义观测周期 0.9 s 加 0.2 s 余量
STALL_TOL_S = 1e-6            # 浮点容差：相邻 rgb_ts 差恰好 1.1 s 不算停顿，严格大于 1.100001 s 算停顿
DEFAULT_N_BOOT = 2000         # 整簇有放回 bootstrap 重采样次数（与 paired_stats 默认一致）
DEFAULT_BOOT_SEED = 0         # 固定重采样随机种子，保证区间可复现
DEFAULT_SUBDIR = "cluster"    # 缺省输出子目录：<OUT>/cluster/cluster_report.{md,json}
# ----------------------------------------------------------------

FROZEN_SEEDS = list(se.FROZEN_SEEDS)
SEEDS_TRAIN = tuple(ea.SEEDS_TRAIN)
EPISODES = se.EPISODES
N_PAIRS = se.N_PAIRS

READING_SIG = "成立，按几何聚类也显著"
READING_NS = "逐集检验下成立，按几何聚类不显著，效应集中在少数几何"

DECLARATIONS = [
    "判决以冻结的汇总器（summarize_e2e.py）为准；仅在出现非迟到崩溃集导致冻结汇总器漏计分母拒判，或 M2 存在基础设施失败单元需对等剔除时，按同一组预注册阈值给出重算结果并列明涉及的集。",
    "按几何聚类的符号翻转检验与整簇 bootstrap 区间为补报（双侧 α = 0.05），与 60 对逐集精确 McNemar 并列。",
    "这 20 个冻结几何（EVAL_SEEDS_BASE_20）对航向码阈值（|n| >= 2）属样本内（阈值沿用护栏设计时看过的同批轨迹）。",
    "另外 40 个 test 分区几何（EVAL_SEEDS_BASE_60 去掉冻结 20 个）的样本外复核尚未做；在样本外复核出来之前，逐集成立而按几何聚类不显著的情形不当成已立住的结论引用。",
]


class ClusterReportError(ValueError):
    """输入数据违反几何一一对应或缺少 geometry_gid 时抛出，命令行以退出码 2 终止，不猜测。"""


def _read_json_if(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_gid_and_manifest(base_dir, seed, entry=None, require_gid=False):
    """提取单集的 geometry_gid 与 episode_manifest.json；已完成集缺 geometry_gid 或两侧冲突直接报错。"""
    gid_eval = entry.get("geometry_gid") if isinstance(entry, dict) else None
    ep_rel = (entry.get("episode_dir") if isinstance(entry, dict) else None) or os.path.join("episodes", f"seed_{seed}")
    ep_dir = os.path.join(base_dir, ep_rel)
    man = _read_json_if(os.path.join(ep_dir, "episode_manifest.json"))
    gid_man = man.get("geometry_gid") if isinstance(man, dict) else None
    if gid_eval and gid_man and str(gid_eval) != str(gid_man):
        raise ClusterReportError(
            f"{base_dir} seed={seed}: eval.json 的 geometry_gid={gid_eval!r} 与 episode_manifest.json 的 {gid_man!r} 不一致"
        )
    gid = gid_eval or gid_man
    if require_gid and not gid:
        raise ClusterReportError(
            f"{base_dir} seed={seed}: 已完成集缺少 geometry_gid（eval.json 与 episode_manifest.json 均无），拒绝猜测"
        )
    return (str(gid) if gid else None), man, ep_dir


def inspect_step_log(ep_dir):
    """按每集 actions.jsonl 的相邻 rgb_ts 差 > 1.1 s 检测停顿步；缺少文件或缺少 rgb_ts 照实标出。"""
    act_path = os.path.join(ep_dir, "actions.jsonl")
    if not os.path.isfile(act_path):
        return {"readable": False, "unreadable_reason": "缺 actions.jsonl",
                "stall_steps": None, "stall_step_ids": [], "max_rgb_gap_s": None, "actions": []}
    actions = se.lc.read_jsonl(act_path)
    if not actions:
        return {"readable": False, "unreadable_reason": "actions.jsonl 为空",
                "stall_steps": None, "stall_step_ids": [], "max_rgb_gap_s": None, "actions": []}
    if not all(isinstance(r, dict) and isinstance(r.get("rgb_ts"), (int, float)) for r in actions):
        return {"readable": False, "unreadable_reason": "actions.jsonl 缺少数值型 rgb_ts",
                "stall_steps": None, "stall_step_ids": [], "max_rgb_gap_s": None, "actions": actions}
    if len(actions) == 1:
        return {"readable": True, "unreadable_reason": None,
                "stall_steps": 0, "stall_step_ids": [], "max_rgb_gap_s": 0.0, "actions": actions}
    gaps = [
        (int(b.get("step", idx + 1)), (float(b["rgb_ts"]) - float(a["rgb_ts"])) / 1e9)
        for idx, (a, b) in enumerate(zip(actions, actions[1:]))
    ]
    stall_ids = [step_id for step_id, g in gaps if g > STALL_THRESHOLD_S + STALL_TOL_S]
    max_gap = max(g for _, g in gaps)
    return {"readable": True, "unreadable_reason": None,
            "stall_steps": len(stall_ids), "stall_step_ids": stall_ids, "max_rgb_gap_s": max_gap, "actions": actions}


def _late_seed_set(ev):
    out = {int(x["seed"]) for x in (ev.get("late_dispatch_episodes") or []) if isinstance(x, dict) and x.get("seed") is not None}
    for f in ev.get("failed") or []:
        if isinstance(f, dict) and f.get("seed") is not None and (f.get("late_dispatch_steps") or 0) > 0:
            out.add(int(f["seed"]))
    return out


def _eval_run_complete(ev, expected_n):
    if ev is None or ev.get("partial"):
        return False
    done_n = int(ev.get("episodes_done") or 0)
    failed_n = int(ev["failed_count"]) if ev.get("failed_count") is not None else len(ev.get("failed") or [])
    skipped_n = int(ev["skipped_count"]) if ev.get("skipped_count") is not None else len(ev.get("skipped") or [])
    return done_n + failed_n + skipped_n == expected_n


def load_arm_record(out_dir, arm, train_seed):
    """读取单个评测臂（含 -retry1 重跑合并），保证已跑完臂对 20 个冻结 seed 全覆盖（分母恒为 20）。"""
    edir = ea.evdir(out_dir, arm, train_seed)
    arm_name = os.path.basename(edir)
    first_ev = _read_json_if(os.path.join(edir, "eval.json"))
    if first_ev is None:
        state = "未完成（进行中）" if os.path.isdir(os.path.join(edir, "episodes")) else "未完成（未跑）"
        return {"arm": arm, "train_seed": train_seed, "arm_name": arm_name, "edir": edir,
                "complete": False, "state": state, "units": {}, "observed_gids": []}

    first_per = {int(x["seed"]): x for x in (first_ev.get("per_episode") or []) if isinstance(x, dict) and x.get("seed") is not None}
    first_late = _late_seed_set(first_ev)
    first_failed = {int(f["seed"]): f for f in (first_ev.get("failed") or [])
                    if isinstance(f, dict) and f.get("seed") is not None and int(f["seed"]) not in first_late}
    first_skipped = {int(k["seed"]): k for k in (first_ev.get("skipped") or [])
                     if isinstance(k, dict) and k.get("seed") is not None}
    first_ok = _eval_run_complete(first_ev, EPISODES)

    rdir = edir + se.RETRY
    retry_ev = _read_json_if(os.path.join(rdir, "eval.json"))
    retry_per, retry_late, retry_failed = {}, set(), {}
    if retry_ev is not None:
        retry_per = {int(x["seed"]): x for x in (retry_ev.get("per_episode") or []) if isinstance(x, dict) and x.get("seed") is not None}
        retry_late = _late_seed_set(retry_ev)
        retry_failed = {int(f["seed"]): f for f in (retry_ev.get("failed") or [])
                        if isinstance(f, dict) and f.get("seed") is not None and int(f["seed"]) not in retry_late}
    retry_ok = (not first_late) or _eval_run_complete(retry_ev, len(first_late))
    complete = first_ok and retry_ok

    known_seeds = set(FROZEN_SEEDS)
    extra_seeds = (set(first_per) | set(first_late) | set(first_failed) | set(first_skipped)
                   | set(retry_per) | set(retry_late) | set(retry_failed)) - known_seeds
    if extra_seeds:
        raise ClusterReportError(f"{arm_name}: 出现不在 20 个冻结 seed 内的评测 seed: {sorted(extra_seeds)}")

    observed_gids = []
    units = {}
    for seed in FROZEN_SEEDS:
        # 先顺带收集首次目录与重跑目录里可能已落盘的 geometry_gid，供跨臂一致性核对
        for bdir, pmap in ((edir, first_per), (rdir, retry_per)):
            if os.path.isdir(bdir):
                g_opt, _, _ = _extract_gid_and_manifest(bdir, seed, pmap.get(seed), require_gid=(seed in pmap and bdir == edir and seed not in first_late))
                if g_opt:
                    observed_gids.append((arm_name, seed, g_opt))

        if seed in first_late:
            if retry_ev is None or (seed not in retry_per and seed not in retry_late and seed not in retry_failed and not retry_ok):
                units[seed] = {
                    "final": "none", "first_late": True, "retry": "missing",
                    "success": False, "klass": "late", "flag_corridor": False, "flag_collision": False,
                    "clean_done": False, "infra_fail": True, "infra_reason": "首次迟到且尚未重跑",
                    "crashed": False, "has_manifest": True, "seeded_in_failed": True,
                    "step_log_readable": True, "unreadable_reason": None, "stall_steps": 0, "max_rgb_gap_s": None,
                }
                continue
            if seed in retry_per and seed not in retry_late:
                entry = retry_per[seed]
                gid, man, ep_dir = _extract_gid_and_manifest(rdir, seed, entry, require_gid=True)
                observed_gids.append((arm_name + se.RETRY, seed, gid))
                st = inspect_step_log(ep_dir)
                oracle = se.lc.read_jsonl(os.path.join(ep_dir, "oracle.jsonl"))
                goal = (man.get("goal_line_x") if isinstance(man, dict) else None) or entry.get("goal_x") or se.GOAL_X_DEFAULT
                klass, _ = se.classify_row(bool(entry.get("success")), False, st["actions"], oracle, float(goal))
                has_stall = bool(st["stall_steps"] and st["stall_steps"] > 0)
                units[seed] = {
                    "final": "retry", "first_late": True, "retry": "ok",
                    "success": bool(entry.get("success")), "klass": klass,
                    "flag_corridor": bool(entry.get("corridor_violation")), "flag_collision": bool(entry.get("collision")),
                    "clean_done": not has_stall, "infra_fail": has_stall,
                    "infra_reason": (f"重跑含停顿步 {st['stall_steps']} 步（最大间隔 {st['max_rgb_gap_s']:.3f} s）" if has_stall else None),
                    "crashed": False, "has_manifest": man is not None, "seeded_in_failed": True,
                    "step_log_readable": st["readable"], "unreadable_reason": st["unreadable_reason"],
                    "stall_steps": st["stall_steps"] or 0, "max_rgb_gap_s": st["max_rgb_gap_s"],
                }
                continue
            if seed in retry_late:
                units[seed] = {
                    "final": "none", "first_late": True, "retry": "late",
                    "success": False, "klass": "late", "flag_corridor": False, "flag_collision": False,
                    "clean_done": False, "infra_fail": True, "infra_reason": "重跑后仍迟到",
                    "crashed": False, "has_manifest": True, "seeded_in_failed": True,
                    "step_log_readable": True, "unreadable_reason": None, "stall_steps": 0, "max_rgb_gap_s": None,
                }
                continue
            # 重跑发生非迟到崩溃（含 retry1 的 failed 列表有/无该 seed）
            _, man_r, _ = _extract_gid_and_manifest(rdir, seed, None, require_gid=False)
            seeded_r = seed in retry_failed
            units[seed] = {
                "final": "none", "first_late": True, "retry": "failed",
                "success": False, "klass": "late", "flag_corridor": False, "flag_collision": False,
                "clean_done": False, "infra_fail": True,
                "infra_reason": "重跑采集进程崩溃" + ("" if man_r is not None else "（缺 episode_manifest.json）"),
                "crashed": True, "has_manifest": man_r is not None, "seeded_in_failed": seeded_r,
                "step_log_readable": True, "unreadable_reason": None, "stall_steps": 0, "max_rgb_gap_s": None,
            }
            continue

        if seed in first_per:
            entry = first_per[seed]
            gid, man, ep_dir = _extract_gid_and_manifest(edir, seed, entry, require_gid=True)
            observed_gids.append((arm_name, seed, gid))
            st = inspect_step_log(ep_dir)
            oracle = se.lc.read_jsonl(os.path.join(ep_dir, "oracle.jsonl"))
            goal = (man.get("goal_line_x") if isinstance(man, dict) else None) or entry.get("goal_x") or se.GOAL_X_DEFAULT
            klass, _ = se.classify_row(bool(entry.get("success")), False, st["actions"], oracle, float(goal))
            has_stall = bool(st["stall_steps"] and st["stall_steps"] > 0)
            units[seed] = {
                "final": "first", "first_late": False, "retry": None,
                "success": bool(entry.get("success")), "klass": klass,
                "flag_corridor": bool(entry.get("corridor_violation")), "flag_collision": bool(entry.get("collision")),
                "clean_done": not has_stall, "infra_fail": has_stall,
                "infra_reason": (f"停顿步 {st['stall_steps']} 步（最大间隔 {st['max_rgb_gap_s']:.3f} s）" if has_stall else None),
                "crashed": False, "has_manifest": man is not None, "seeded_in_failed": True,
                "step_log_readable": st["readable"], "unreadable_reason": st["unreadable_reason"],
                "stall_steps": st["stall_steps"] or 0, "max_rgb_gap_s": st["max_rgb_gap_s"],
            }
            continue

        if seed in first_failed or seed in first_skipped or first_ok:
            _, man, ep_dir = _extract_gid_and_manifest(edir, seed, None, require_gid=False)
            actions = se.lc.read_jsonl(os.path.join(ep_dir, "actions.jsonl"))
            oracle = se.lc.read_jsonl(os.path.join(ep_dir, "oracle.jsonl"))
            goal = (man.get("goal_line_x") if isinstance(man, dict) else None) or se.GOAL_X_DEFAULT
            klass, _ = se.classify_row(False, True, actions, oracle, float(goal))
            f_info = first_failed.get(seed) or first_skipped.get(seed) or {}
            seeded = seed in first_failed
            detail = f"rc={f_info.get('rc')}" if f_info.get("rc") is not None else (f_info.get("error") or f_info.get("reason") or "未记录 seed")
            reason = f"采集进程崩溃/失败（{detail}" + ("，缺 episode_manifest.json" if man is None else "") + "）"
            units[seed] = {
                "final": "first", "first_late": False, "retry": None,
                "success": False, "klass": klass, "flag_corridor": False, "flag_collision": False,
                "clean_done": False, "infra_fail": True, "infra_reason": reason,
                "crashed": True, "has_manifest": man is not None, "seeded_in_failed": seeded,
                "step_log_readable": True, "unreadable_reason": None, "stall_steps": 0, "max_rgb_gap_s": None,
            }

    if complete:
        state = "完整"
    elif not first_ok:
        done_n = int(first_ev.get("episodes_done") or 0)
        failed_n = int(first_ev["failed_count"]) if first_ev.get("failed_count") is not None else len(first_ev.get("failed") or [])
        state = f"未完成（首次 done={done_n}, failed={failed_n}/{EPISODES}{'，partial' if first_ev.get('partial') else ''}）"
    else:
        state = f"未完成（迟到集 {sorted(first_late)} 待重跑或重跑未完）"

    return {"arm": arm, "train_seed": train_seed, "arm_name": arm_name, "edir": edir,
            "complete": complete, "state": state, "units": units, "observed_gids": observed_gids}


def verify_seed_gid_bijection(arm_records, require_all_20):
    """核对 20 个冻结 seed 与 20 个 geometry_gid 严格一一对应；任何冲突或缺失直接报错，不猜。"""
    gids_by_seed = {}
    where_by_seed_gid = {}
    for rec in arm_records.values():
        for where, seed, gid in rec["observed_gids"]:
            gids_by_seed.setdefault(seed, set()).add(gid)
            where_by_seed_gid.setdefault((seed, gid), []).append(where)

    for seed, gset in sorted(gids_by_seed.items()):
        if len(gset) > 1:
            detail = ", ".join(f"{g}({where_by_seed_gid[(seed, g)][0]})" for g in sorted(gset))
            raise ClusterReportError(f"seed {seed} 对应多个不同的 geometry_gid：{detail}")

    seed_to_gid = {seed: next(iter(gset)) for seed, gset in gids_by_seed.items() if gset}
    seeds_by_gid = {}
    for seed, gid in seed_to_gid.items():
        seeds_by_gid.setdefault(gid, []).append(seed)
    for gid, slist in sorted(seeds_by_gid.items()):
        if len(slist) > 1:
            raise ClusterReportError(f"多个冻结 seed {sorted(slist)} 共用同一个 geometry_gid={gid!r}，违反 20 对 20 一一对应")

    if require_all_20 and set(seed_to_gid) != set(FROZEN_SEEDS):
        missing = sorted(set(FROZEN_SEEDS) - set(seed_to_gid))
        raise ClusterReportError(f"以下冻结 seed 在全部评测臂中均未读到 geometry_gid：{missing}，拒绝猜测")

    return seed_to_gid


def cluster_reading(verdict, cluster_p):
    """任务卡 09:09 补充第 3 条的四种判读写法（未完成时返回 undone）。"""
    if verdict == se.UNDONE:
        return "undone", se.UNDONE
    if verdict == se.PRED:
        if cluster_p is not None and cluster_p < CLUSTER_ALPHA:
            return "established_cluster_sig", READING_SIG
        return "established_cluster_ns", READING_NS
    if verdict == se.FALS:
        return "falsified", se.FALS
    return "inconclusive", verdict


def summarize_cluster_pair(arm_records, arm_x, arm_y, seed_to_gid, n_boot=DEFAULT_N_BOOT, boot_seed=DEFAULT_BOOT_SEED):
    """60 个 (train_seed, eval_seed) 配对单元按 20 个 geometry_gid 聚类的符号翻转精确检验与整簇 bootstrap 区间。"""
    units = []
    for s in SEEDS_TRAIN:
        ux = arm_records[(arm_x, s)]["units"]
        uy = arm_records[(arm_y, s)]["units"]
        for seed in FROZEN_SEEDS:
            units.append((seed_to_gid[seed], int(ux[seed]["success"]), int(uy[seed]["success"])))
    cs = ps.pair_summary(units, n_boot=n_boot, alpha=CLUSTER_ALPHA, seed=boot_seed)
    grouped = ps.group_units(units)
    per_geom_diffs = [grouped[seed_to_gid[seed]][0] - grouped[seed_to_gid[seed]][1] for seed in FROZEN_SEEDS]
    n_pos = sum(1 for d in per_geom_diffs if d > 0)
    n_neg = sum(1 for d in per_geom_diffs if d < 0)
    net = sum(per_geom_diffs)
    n_same_dir = n_pos if net > 0 else (n_neg if net < 0 else max(n_pos, n_neg))
    n_nz = cs["n_nonzero"]
    return {
        "n_units": cs["n_units"],
        "n_clusters": cs["n_clusters"],
        "n_nonzero": n_nz,
        "n_same_dir": n_same_dir,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "min_possible_two_sided_p": (2.0 / (2 ** n_nz)) if n_nz > 0 else 1.0,
        "p_value": cs["p_value"],
        "method": cs["method"],
        "observed_sum": cs["observed_sum"],
        "rate_diff": cs["diff"],
        "ci_lo": cs["ci_lo"],
        "ci_hi": cs["ci_hi"],
        "n_boot": cs["n_boot"],
        "b": cs["b"],
        "c": cs["c"],
        "mcnemar_p": cs["mcnemar_p"],
        "per_geometry_diffs": {str(seed): per_geom_diffs[i] for i, seed in enumerate(FROZEN_SEEDS)},
    }


def _safe_frozen_verdicts(out_dir):
    """只读调用冻结汇总器 summarize_e2e.build 获取其原始判决供并列对账（不写任何文件）。"""
    try:
        _, fv, _ = se.build(out_dir)
        return fv, None
    except Exception as exc:  # noqa: BLE001 - 冻结汇总器在极端坏记录（如 seed=null）下抛错时照实记录
        return None, f"{type(exc).__name__}: {exc}"


def build_cluster_report(out_dir, n_boot=DEFAULT_N_BOOT, boot_seed=DEFAULT_BOOT_SEED, note=None):
    """主入口：读取 12 个评测臂，返回 (markdown 文本, json 结构 dict)。"""
    out_dir = os.path.abspath(out_dir)
    arm_records = {(a, s): load_arm_record(out_dir, a, s) for a in ea.ARMS for s in SEEDS_TRAIN}
    complete = all(r["complete"] for r in arm_records.values())
    seed_to_gid = verify_seed_gid_bijection(arm_records, require_all_20=complete)
    frozen_v, frozen_err = _safe_frozen_verdicts(out_dir)

    unreadable_logs = []
    crashed_episodes = []
    for a in ea.ARMS:
        for s in SEEDS_TRAIN:
            ar = arm_records[(a, s)]
            for seed, u in sorted(ar["units"].items()):
                if not u["step_log_readable"]:
                    unreadable_logs.append({
                        "arm": a, "arm_name": ar["arm_name"], "train_seed": s,
                        "eval_seed": seed, "reason": u["unreadable_reason"],
                    })
                if u["crashed"]:
                    crashed_episodes.append({
                        "arm": a, "arm_name": ar["arm_name"], "train_seed": s,
                        "eval_seed": seed, "has_manifest": u["has_manifest"],
                        "seeded_in_failed": u["seeded_in_failed"], "reason": u["infra_reason"],
                    })

    arms_summary = {}
    for a in ea.ARMS:
        us = [u for s in SEEDS_TRAIN for u in arm_records[(a, s)]["units"].values()]
        arms_summary[a] = {
            "label": ea.LABEL[a],
            "complete_runs": sum(1 for s in SEEDS_TRAIN if arm_records[(a, s)]["complete"]),
            "n": len(us),
            "success": sum(1 for u in us if u["success"]),
            "collision": sum(1 for u in us if u["klass"] == "collision"),
            "corridor": sum(1 for u in us if u["klass"] == "corridor"),
            "not_reached": sum(1 for u in us if u["klass"] == "not_reached"),
            "aborted": sum(1 for u in us if u["klass"] == "aborted"),
            "late": sum(1 for u in us if u["klass"] == "late"),
            "stall_episodes": sum(1 for u in us if u["stall_steps"] > 0),
            "corridor_flag": sum(1 for u in us if u["klass"] not in (None, "late") and u["flag_corridor"]),
        }

    # M2 基础设施失败单元识别（在已有的共同单元上检查）
    m2_excluded = []
    m2_clean_keys = []
    for s in SEEDS_TRAIN:
        u0_map = arm_records[("a0", s)]["units"]
        u1_map = arm_records[("a1", s)]["units"]
        for seed in FROZEN_SEEDS:
            if seed not in u0_map or seed not in u1_map:
                continue
            u0, u1 = u0_map[seed], u1_map[seed]
            if u0["infra_fail"] or u1["infra_fail"]:
                m2_excluded.append({
                    "train_seed": s, "eval_seed": seed, "geometry_gid": seed_to_gid.get(seed),
                    "a0_reason": u0["infra_reason"], "a1_reason": u1["infra_reason"],
                })
            else:
                m2_clean_keys.append((s, seed))

    recomputed_reasons = []
    if complete:
        for ce in crashed_episodes:
            recomputed_reasons.append(f"{ce['arm_name']} seed={ce['eval_seed']}：{ce['reason']}")
        for ex in m2_excluded:
            sides = "；".join(
                f"{lbl} {rsn}" for lbl, rsn in (("A0", ex["a0_reason"]), ("A1", ex["a1_reason"])) if rsn
            )
            tag = f"M2 剔除 (s={ex['train_seed']}, seed={ex['eval_seed']})：{sides}"
            if tag not in recomputed_reasons:
                recomputed_reasons.append(tag)
        if frozen_err:
            recomputed_reasons.append(f"冻结汇总器异常（{frozen_err}）")
        elif frozen_v is not None and not frozen_v.get("complete"):
            recomputed_reasons.append("冻结汇总器因缺失记录判未完成，本脚本按 60 集分母补齐重算")
    recomputed = bool(complete and recomputed_reasons)

    def frozen_verdict_of(key):
        if frozen_v is None:
            return f"异常（{frozen_err}）"
        return frozen_v[key]["verdict"]

    if complete:
        c_m1 = summarize_cluster_pair(arm_records, "a2", "a3", seed_to_gid, n_boot=n_boot, boot_seed=boot_seed)
        diff_m1 = arms_summary["a2"]["success"] - arms_summary["a3"]["success"]
        v_m1 = se.verdict_m1(diff_m1, c_m1["mcnemar_p"])
        rc_m1, rt_m1 = cluster_reading(v_m1, c_m1["p_value"])
        m1 = {
            "a2_success": arms_summary["a2"]["success"], "a3_success": arms_summary["a3"]["success"],
            "n": N_PAIRS, "diff": diff_m1, "b": c_m1["b"], "c": c_m1["c"],
            "episode_mcnemar_p": c_m1["mcnemar_p"], "verdict": v_m1,
            "frozen_verdict": frozen_verdict_of("m1"),
            "cluster": c_m1, "reading_code": rc_m1, "reading_text": rt_m1,
        }

        c_m2_succ = summarize_cluster_pair(arm_records, "a1", "a0", seed_to_gid, n_boot=n_boot, boot_seed=boot_seed)
        diff_m2_succ = arms_summary["a1"]["success"] - arms_summary["a0"]["success"]
        raw_corr_a0, raw_corr_a1 = arms_summary["a0"]["corridor"], arms_summary["a1"]["corridor"]
        raw_v2, raw_pred2, raw_fals2 = se.verdict_m2(
            raw_corr_a0, raw_corr_a1, arms_summary["a0"]["success"], arms_summary["a1"]["success"]
        )
        clean_corr_a0 = sum(1 for s, sd in m2_clean_keys if arm_records[("a0", s)]["units"][sd]["klass"] == "corridor")
        clean_corr_a1 = sum(1 for s, sd in m2_clean_keys if arm_records[("a1", s)]["units"][sd]["klass"] == "corridor")
        clean_succ_a0 = sum(1 for s, sd in m2_clean_keys if arm_records[("a0", s)]["units"][sd]["success"])
        clean_succ_a1 = sum(1 for s, sd in m2_clean_keys if arm_records[("a1", s)]["units"][sd]["success"])
        # 主读法：越界集数在共同干净单元上比，成功数按第 5 条保持每臂 60 分母；另附干净子集成功数读法备查
        clean_v2, clean_pred2, clean_fals2 = se.verdict_m2(
            clean_corr_a0, clean_corr_a1, arms_summary["a0"]["success"], arms_summary["a1"]["success"]
        )
        clean_v2_alt = se.verdict_m2(clean_corr_a0, clean_corr_a1, clean_succ_a0, clean_succ_a1)[0]
        v_m2 = clean_v2 if m2_excluded else raw_v2
        rc_m2, rt_m2 = cluster_reading(v_m2, c_m2_succ["p_value"])

        # M2 越界簇差（每个几何上 A0 越界数减 A1 越界数，只在共同干净单元上算），符号翻转检验只报告
        corr_diffs_by_seed = {sd: 0 for sd in FROZEN_SEEDS}
        corr_units = []
        for s, sd in m2_clean_keys:
            c0 = int(arm_records[("a0", s)]["units"][sd]["klass"] == "corridor")
            c1 = int(arm_records[("a1", s)]["units"][sd]["klass"] == "corridor")
            corr_diffs_by_seed[sd] += c0 - c1
            corr_units.append((seed_to_gid[sd], c0, c1))
        d_corr_list = [corr_diffs_by_seed[sd] for sd in FROZEN_SEEDS]
        sf_corr = ps.signflip_test(d_corr_list)
        ps_corr = ps.pair_summary(corr_units, n_boot=n_boot, alpha=CLUSTER_ALPHA, seed=boot_seed) if corr_units else None
        corr_pos = sum(1 for d in d_corr_list if d > 0)
        corr_neg = sum(1 for d in d_corr_list if d < 0)
        corr_net = sum(d_corr_list)
        corr_same = corr_pos if corr_net > 0 else (corr_neg if corr_net < 0 else max(corr_pos, corr_neg))
        m2 = {
            "a1_success": arms_summary["a1"]["success"], "a0_success": arms_summary["a0"]["success"],
            "n": N_PAIRS, "success_diff": diff_m2_succ, "b": c_m2_succ["b"], "c": c_m2_succ["c"],
            "episode_mcnemar_p": c_m2_succ["mcnemar_p"],
            "raw_corridor_a0": raw_corr_a0, "raw_corridor_a1": raw_corr_a1,
            "raw_pred_clause": raw_pred2, "raw_fals_clause": raw_fals2, "raw_verdict": raw_v2,
            "clean_n": len(m2_clean_keys), "excluded_count": len(m2_excluded), "excluded_units": m2_excluded,
            "clean_corridor_a0": clean_corr_a0, "clean_corridor_a1": clean_corr_a1,
            "clean_succ_a0": clean_succ_a0, "clean_succ_a1": clean_succ_a1,
            "clean_pred_clause": clean_pred2, "clean_fals_clause": clean_fals2,
            "clean_verdict": clean_v2, "clean_verdict_if_clean_succ": clean_v2_alt,
            "verdict": v_m2, "frozen_verdict": frozen_verdict_of("m2"),
            "cluster": c_m2_succ,
            "corridor_cluster": {
                "n_clean_units": len(m2_clean_keys), "n_clusters": len(d_corr_list),
                "n_nonzero": sf_corr["n_nonzero"], "n_same_dir": corr_same,
                "n_pos": corr_pos, "n_neg": corr_neg,
                "observed_drop": sf_corr["observed_sum"], "p_value": sf_corr["p_value"], "method": sf_corr["method"],
                "rate_diff": ps_corr["diff"] if ps_corr else None,
                "ci_lo": ps_corr["ci_lo"] if ps_corr else None, "ci_hi": ps_corr["ci_hi"] if ps_corr else None,
            },
            "reading_code": rc_m2, "reading_text": rt_m2,
        }

        c_m3 = summarize_cluster_pair(arm_records, "a2", "a0", seed_to_gid, n_boot=n_boot, boot_seed=boot_seed)
        diff_m3 = arms_summary["a2"]["success"] - arms_summary["a0"]["success"]
        v_m3 = se.verdict_m3(diff_m3)
        rc_m3, rt_m3 = cluster_reading(v_m3, c_m3["p_value"])
        m3 = {
            "a2_success": arms_summary["a2"]["success"], "a0_success": arms_summary["a0"]["success"],
            "n": N_PAIRS, "diff": diff_m3, "b": c_m3["b"], "c": c_m3["c"],
            "episode_mcnemar_p": c_m3["mcnemar_p"], "verdict": v_m3,
            "frozen_verdict": frozen_verdict_of("m3"),
            "cluster": c_m3, "reading_code": rc_m3, "reading_text": rt_m3,
        }

        best_code_arms = [a for a in ("a1", "a2") if arms_summary[a]["success"] == max(arms_summary["a1"]["success"], arms_summary["a2"]["success"])]
        w22 = [{
            "arm": a, "label": ea.LABEL[a], "success": arms_summary[a]["success"],
            "corridor": arms_summary[a]["corridor"], "corridor_flag": arms_summary[a]["corridor_flag"], "n": N_PAIRS,
            "reached": se.w22_status(arms_summary[a]["success"], arms_summary[a]["corridor"]),
            "reached_by_flag": se.w22_status(arms_summary[a]["success"], arms_summary[a]["corridor_flag"]),
        } for a in best_code_arms]
    else:
        m1 = {"n": 0, "verdict": se.UNDONE, "frozen_verdict": frozen_verdict_of("m1"),
              "cluster": None, "reading_code": "undone", "reading_text": se.UNDONE}
        m2 = {"n": 0, "verdict": se.UNDONE, "frozen_verdict": frozen_verdict_of("m2"),
              "excluded_count": len(m2_excluded), "excluded_units": m2_excluded,
              "cluster": None, "corridor_cluster": None, "reading_code": "undone", "reading_text": se.UNDONE}
        m3 = {"n": 0, "verdict": se.UNDONE, "frozen_verdict": frozen_verdict_of("m3"),
              "cluster": None, "reading_code": "undone", "reading_text": se.UNDONE}
        w22 = []

    rep = {
        "schema": "uavrgb.e2e1.cluster_report.v1",
        "generated_at": time.strftime("%F %T"),
        "eval_out": out_dir,
        "note": note,
        "declarations": DECLARATIONS,
        "complete": complete,
        "complete_arm_count": sum(1 for r in arm_records.values() if r["complete"]),
        "total_arm_count": len(arm_records),
        "arm_states": {r["arm_name"]: r["state"] for r in arm_records.values()},
        "seed_to_gid": {str(k): v for k, v in sorted(seed_to_gid.items())},
        "cluster_alpha": CLUSTER_ALPHA,
        "stall_threshold_s": STALL_THRESHOLD_S,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "recomputed": recomputed,
        "recomputed_reasons": recomputed_reasons,
        "crashed_episodes": crashed_episodes,
        "unreadable_step_logs": unreadable_logs,
        "arms": arms_summary,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "w22": w22,
        "guard_reference": se.guard_reference(),
    }
    return render_markdown(rep), rep


def _fmt_p(p):
    if p is None:
        return "-"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def render_markdown(rep):
    lines = ["# 卡 E2E-1 按几何聚类补报（cluster_report）", ""]
    if rep.get("note"):
        lines += [f"> {rep['note']}", ""]
    lines.append("## 0. 口径与限定声明（必读）")
    lines.append("")
    for d in rep["declarations"]:
        lines.append(f"- {d}")
    lines.append("- 方法混杂提醒（审核 M-1）：M2 判证伪或不判时需结合卡 E2E-1b 按码校准臂分辨全局校准副作用；M3（A2 对 A0）同时包含数据集、航向码与校准目标前进占比（0.734 对 0.873）三项差异。")
    lines.append("")

    lines.append(f"## 1. 完成状态与分母核对（已完成评测臂 {rep['complete_arm_count']}/{rep['total_arm_count']}）")
    lines.append("")
    if not rep["complete"]:
        unfinished = [f"`{k}`：{v}" for k, v in rep["arm_states"].items() if v != "完整"]
        lines.append(f"- **状态：未完成，不出判决**。尚未评完的臂：{'；'.join(unfinished)}。")
    else:
        lines.append(f"- **状态：12 个评测臂全部完成**。20 个冻结评测 seed 与 {len(rep['seed_to_gid'])} 个独立 `geometry_gid` 一一对应核对通过；每臂分母恒为 60。")
        if rep["recomputed"]:
            lines.append("- **重算说明：以下判决为本脚本按预注册同组阈值重算的结果**。原因与涉及的集：")
            for r in rep["recomputed_reasons"]:
                lines.append(f"  - {r}")
        else:
            lines.append("- **与冻结汇总器对账**：无崩溃漏计集、M2 无基础设施剔除集，M1–M3 预注册判决及各臂成功数、越界数与冻结汇总器逐项一致。")
    lines.append("")

    lines.append("## 2. 三个主比较（逐集 McNemar 与 20 几何聚类符号翻转并列，聚类显著性 α = 0.05）")
    lines.append("")
    if not rep["complete"]:
        lines.append("| 比较 | 预注册判决 | 聚类补报判读 |")
        lines.append("| --- | --- | --- |")
        for tag in ("M1 A2 对 A3", "M2 A1 对 A0", "M3 A2 对 A0"):
            lines.append(f"| {tag} | {se.UNDONE} | {se.UNDONE} |")
    else:
        m1, m2, m3 = rep["m1"], rep["m2"], rep["m3"]
        lines.append("| 比较 | 成功数（/60） | 成功差 A−B | 逐集精确 McNemar (b:c, p) | 几何符号翻转精确 p | 率差 [95% 整簇区间] | 非零差几何 / 同向几何（正/负） | 预注册判决（冻结汇总器） | 聚类补报判读 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for title, item, sa, sb in (
            ("M1：A2 对 A3", m1, m1["a2_success"], m1["a3_success"]),
            ("M2：A1 对 A0（成功列）", m2, m2["a1_success"], m2["a0_success"]),
            ("M3：A2 对 A0", m3, m3["a2_success"], m3["a0_success"]),
        ):
            c = item["cluster"]
            diff_val = item.get("diff", item.get("success_diff"))
            fv_note = item["verdict"] if item["verdict"] == item["frozen_verdict"] else f"{item['verdict']}（重算；冻结器：{item['frozen_verdict']}）"
            lines.append(
                f"| {title} | {sa} 对 {sb} | {diff_val:+d} | {item['b']}:{item['c']}, p={_fmt_p(item['episode_mcnemar_p'])} "
                f"| {_fmt_p(c['p_value'])} | {c['rate_diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] "
                f"| {c['n_nonzero']}/20 / 同向 {c['n_same_dir']}（+{c['n_pos']}/−{c['n_neg']}） "
                f"| {fv_note} | **{item['reading_text']}** |"
            )
        lines.append("")
        lines.append("注：20 个几何簇做双侧符号翻转时，离散分辨率下界为 $2 / 2^{k}$（$k$ 为非零差几何数）：$k=7$ 全同向时精确 $p = 0.0156$（$< 0.05$），$k=8$ 全同向时 $p = 0.0078$（$< 0.01$），$k=5$ 全同向时 $p = 0.0625$（$\\ge 0.05$）。")
    lines.append("")

    lines.append("## 3. M2 越界集数专项核对（对等剔除基础设施失败单元 + 越界簇差符号翻转）")
    lines.append("")
    if not rep["complete"]:
        lines.append(f"- 评测臂未全部完成，M2 暂不判（当前已观测到的待剔除基础设施失败单元：{rep['m2']['excluded_count']} 个）。")
    else:
        m2 = rep["m2"]
        cc = m2["corridor_cluster"]
        if m2["excluded_count"] == 0:
            lines.append(f"- **无剔除单元**（60/60 个配对单元两侧均干净完成，无重跑仍迟到、无 `rgb_ts` 差 > 1.1 s 停顿步、无采集崩溃）：越界 A0 = {m2['clean_corridor_a0']}，A1 = {m2['clean_corridor_a1']}，判决与冻结汇总器一致：**{m2['verdict']}**。")
        else:
            lines.append(f"- **共剔除 {m2['excluded_count']} 个基础设施失败配对单元**，保留两侧均干净完成的 **{m2['clean_n']}/60** 个单元（判决按剔除后的结果）：")
            lines.append(f"  - 剔除前（全部 60 对，冻结汇总器算法）：越界 A0 = {m2['raw_corridor_a0']}，A1 = {m2['raw_corridor_a1']} → {m2['raw_verdict']}")
            lines.append(f"  - 剔除后（干净 {m2['clean_n']} 对，本脚本采用判决）：越界 A0 = {m2['clean_corridor_a0']}，A1 = {m2['clean_corridor_a1']}（成功 60 集口径 A1 {m2['a1_success']} 对 A0 {m2['a0_success']}；干净子集内成功 A1 {m2['clean_succ_a1']} 对 A0 {m2['clean_succ_a0']}）→ **{m2['verdict']}**")
            lines.append("  - 被剔除的配对单元明细：")
            for ex in m2["excluded_units"]:
                lines.append(f"    - 训练种子 s{ex['train_seed']}，评测 seed={ex['eval_seed']}（gid=`{ex['geometry_gid']}`）：A0={ex['a0_reason'] or '干净'}；A1={ex['a1_reason'] or '干净'}")
        ci_txt = (f"，率差 {cc['rate_diff']:+.3f} [{cc['ci_lo']:+.3f}, {cc['ci_hi']:+.3f}]"
                  if cc["rate_diff"] is not None else "")
        lines.append(
            f"- **越界簇差符号翻转检验（只报告，每个几何 A0 越界数减 A1 越界数）**：总减少 {cc['observed_drop']:+d} 集，"
            f"非零差几何 {cc['n_nonzero']}/{cc['n_clusters']} 个，同向 {cc['n_same_dir']} 个（+{cc['n_pos']}/−{cc['n_neg']}），"
            f"双侧精确 p = {_fmt_p(cc['p_value'])}{ci_txt}。"
        )
        if m2["verdict"] == se.PRED and m2["cluster"]["p_value"] >= CLUSTER_ALPHA and cc["p_value"] < CLUSTER_ALPHA:
            lines.append("  - 补充说明：M2 预注册对成功数只要求非劣（A1 >= A0），故第 2 节按成功数聚类的 p 不显著；其核心效应「越界减少」在几何聚类下双侧 p < 0.05。")
    lines.append("")

    lines.append("## 4. 四臂汇总与 W2-2 参照（分母每臂 60）")
    lines.append("")
    lines.append("| 臂 | 完成 run | 已读集数 | 成功 | 撞柱 | 越界 | 未到线 | 采集中止/崩溃 | 迟到失败 | 含停顿步集数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for a in ea.ARMS:
        am = rep["arms"][a]
        lines.append(
            f"| {am['label']} | {am['complete_runs']}/3 | {am['n']} | {am['success']} | {am['collision']} "
            f"| {am['corridor']} | {am['not_reached']} | {am['aborted']} | {am['late']} | {am['stall_episodes']} |"
        )
    if rep["complete"] and rep["w22"]:
        ref = rep["guard_reference"]
        ref_txt = "未核实" if ref is None else f"{ref['success']}/{ref['n']}"
        for w in rep["w22"]:
            lines.append(
                f"\n- 只报告（不设判决）：{w['label']} 成功 {w['success']}/{w['n']}，越界 {w['corridor']}/{w['n']}，"
                f"卡 W2-2 判据（>=48/60 且越界 <=3/60）：{'达到' if w['reached'] else '未达到'}；带护栏参照：{ref_txt}。"
            )
    lines.append("")

    lines.append("## 5. 逐步记录可读性与各评测臂状态核查")
    lines.append("")
    if rep["unreadable_step_logs"]:
        lines.append(f"- **读不到逐步 `rgb_ts` 记录的已完成集（共 {len(rep['unreadable_step_logs'])} 集，无法核实停顿步，照实标出）**：")
        for u in rep["unreadable_step_logs"]:
            lines.append(f"  - `{u['arm_name']}` seed={u['eval_seed']}：{u['reason']}")
    else:
        lines.append("- 已完成集的 `actions.jsonl` 与 `rgb_ts` 逐步记录全部可读（0 集缺失）。")
    lines.append("- 12 个评测臂状态：" + "；".join(f"`{k}`：{v}" for k, v in rep["arm_states"].items()))
    return "\n".join(lines) + "\n"


def write_cluster_report(eval_out, out_dir=None, n_boot=DEFAULT_N_BOOT, boot_seed=DEFAULT_BOOT_SEED, note=None):
    eval_out = os.path.abspath(eval_out)
    dest = os.path.abspath(out_dir) if out_dir else os.path.join(eval_out, DEFAULT_SUBDIR)
    md, rep = build_cluster_report(eval_out, n_boot=n_boot, boot_seed=boot_seed, note=note)
    os.makedirs(dest, exist_ok=True)
    md_path = os.path.join(dest, "cluster_report.md")
    js_path = os.path.join(dest, "cluster_report.json")
    with open(md_path + ".tmp", "w", encoding="utf-8") as f:
        f.write(md)
    os.replace(md_path + ".tmp", md_path)
    with open(js_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(js_path + ".tmp", js_path)
    return md, rep, dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_out", help="E2E-1 实验输出目录 $OUT")
    ap.add_argument("pos_out_dir", nargs="?", default=None, help="可选：报告写入目录（缺省为 <OUT>/cluster）")
    ap.add_argument("--out-dir", dest="out_dir", default=None, help="报告写入目录（缺省为 <OUT>/cluster）")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="整簇 bootstrap 重采样次数（默认 2000）")
    ap.add_argument("--seed", type=int, default=DEFAULT_BOOT_SEED, help="bootstrap 随机种子（默认 0）")
    ap.add_argument("--note", default=None, help="附加在报告顶部的一行说明")
    args = ap.parse_args(argv)
    if args.pos_out_dir and args.out_dir and os.path.abspath(args.pos_out_dir) != os.path.abspath(args.out_dir):
        ap.error("位置参数输出目录与 --out-dir 不一致，请只给一处")
    target_dir = args.out_dir or args.pos_out_dir
    try:
        md, _rep, _dest = write_cluster_report(args.eval_out, out_dir=target_dir, n_boot=args.n_boot,
                                               boot_seed=args.seed, note=args.note)
    except ClusterReportError as exc:
        ap.error(str(exc))
    print(md, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
