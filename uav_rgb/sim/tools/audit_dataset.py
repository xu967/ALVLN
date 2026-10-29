#!/usr/bin/env python3
"""数据集验收闸门：逐条检查采集产物的不变量，跑不过就非零退出。

一次采集是否正确，不靠肉眼看图，靠这几条能机械判定的不变量。每条检查
显式给出「覆盖 / 未覆盖」，缺字段的检查标 SKIP 并说明要补什么，不假装通过。

用法：
  python3 sim/tools/audit_dataset.py --dataset /tmp/ds200
  python3 sim/tools/audit_dataset.py --dataset /tmp/ds200 --out /tmp/audit.json

判定口径（阈值都在下面常量里，改阈值要同步改本文件顶部的说明）：

  manifest_validity  每集 validity 三项全 true      训练集必须是逐集跑满的
  time_pairing       每步 depth/pose 与 RGB 的 gap   > 50 ms 说明观测不同步
  action_window      动作窗口 sim_dt 的变异系数       > 5% 说明 hold 随 RTF 漂移
  forward_motion     稳态 FORWARD_SLOW 步的位移       < 0.4×SPEED_FWD×dt 说明动作没生效
  transition_lag     偏航后第一步的位移中位数         < 50% 标称说明过渡步几乎不动
  yaw_direction      YAW 步的 dyaw 方向              需 actions 里有 dyaw_deg
  class_share        每类样本占比                    < 5% 的类学不出来
  split_geometry     train/test 的 geometry_gid 交集 必须为 0
  label_recompute    用 pillars+动作前位姿复算标签    需 teacher=plan_action 且 actions
                                                     里有 pose_before；不一致率必须为 0
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_manifest import geometry_gid  # noqa: E402
from teacher import plan_action  # noqa: E402

NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
SPEED_FWD = 0.3              # 与 collect.py 一致
MAX_GAP_NS = 50_000_000
MAX_WINDOW_CV = 0.05
FWD_MIN_V = 0.4 * SPEED_FWD
MAX_FWD_FAIL_SHARE = 0.01
MIN_TRANSITION_RATIO = 0.5
YAW_MIN_DEG = 0.5
MIN_CLASS_SHARE = 0.05
MAX_LABEL_MISMATCH = 0.0     # plan_action 是确定性的，同一输入必然同一标签


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dist3(a, b):
    return sum((a[k] - b[k]) ** 2 for k in ("x", "y", "z")) ** 0.5


def episode_facts(ep_dir):
    """读一集，返回逐集事实；缺文件时返回 {"error": 原因}。"""
    try:
        man = load_json(os.path.join(ep_dir, "episode_manifest.json"))
        rows = load_jsonl(os.path.join(ep_dir, man["actions_ref"]))
        oracle = load_jsonl(os.path.join(ep_dir, man["oracle_ref"]))
    except (OSError, KeyError, ValueError) as e:
        return {"dir": ep_dir, "error": f"{type(e).__name__}: {e}"}

    gid = None
    pillars = None
    scene_path = os.path.join(ep_dir, "scene.json")
    if os.path.exists(scene_path):
        try:
            raw = load_json(scene_path)["pillars"]
            # pillars 写成 null 时 geometry_gid(None) 会抛 TypeError，整个 audit
            # 直接崩；isinstance 挡住这种「文件在但内容是空的」。
            if isinstance(raw, list) and raw:
                pillars = raw
                gid = geometry_gid(pillars)
        except (KeyError, OSError, ValueError, TypeError):
            pillars = None

    steps = []
    for i, (row, orc) in enumerate(zip(rows, oracle)):
        dt = (orc["action_end_ns"] - orc["action_start_ns"]) / 1e9
        disp = row.get("displacement_m")
        if disp is None and i > 0 and oracle[i - 1].get("pose_gt") and orc.get("pose_gt"):
            # 旧数据没记 displacement_m：用相邻动作后位姿差近似（误差 = 一个位姿周期）
            disp = _dist3(orc["pose_gt"], oracle[i - 1]["pose_gt"])
        steps.append({
            "action": row["action_executed"],
            "prev_action": rows[i - 1]["action_executed"] if i else None,
            "collision": bool(row.get("collision")),
            "dt": dt,
            "disp": disp,
            "disp_approx": row.get("displacement_m") is None,
            "dyaw": row.get("dyaw_deg"),
            "gap": max(row.get("depth_rgb_gap_ns", 0), row.get("pose_rgb_gap_ns", 0)),
            "has_pose_before": "pose_before" in row,
            "pose_before": row.get("pose_before"),
            "yaw_before": row.get("yaw_before"),
            "action_expert": row.get("action_expert"),
        })

    return {
        "dir": ep_dir,
        "split": man.get("split"),
        "teacher": man.get("teacher"),
        "gid": gid,
        "pillars": pillars,
        "validity": man.get("validity") or {},
        "num_steps": len(steps),
        "requested_steps": man.get("requested_steps"),
        "steps": steps,
        "error": None,
    }


def check(name, status, detail, **extra):
    return {"name": name, "status": status, "detail": detail, **extra}


def audit(facts):
    """把逐集事实折成检查项。status: PASS / FAIL / SKIP。"""
    ok = [f for f in facts if not f.get("error")]
    bad = [f for f in facts if f.get("error")]
    steps = [s for f in ok for s in f["steps"]]
    checks = []

    if bad:
        checks.append(check("read_episodes", "FAIL",
                            f"{len(bad)} 集读不出来", samples=[b["dir"] for b in bad[:5]]))
    if not steps:
        checks.append(check("read_episodes", "FAIL", "没有任何步"))
        return checks

    # 1) manifest validity
    bad_val = [f["dir"] for f in ok if not all(f["validity"].get(k) for k in
               ("camera_verified", "physics_verified", "episode_executed"))]
    if not any(f["validity"] for f in ok):
        checks.append(check("manifest_validity", "SKIP", "manifest 里没有 validity 字段"))
    else:
        checks.append(check("manifest_validity", "FAIL" if bad_val else "PASS",
                            f"{len(bad_val)}/{len(ok)} 集 validity 不全 true",
                            samples=bad_val[:5]))

    # 2) 时间配对
    gaps = [s["gap"] for s in steps]
    over = sum(1 for g in gaps if g > MAX_GAP_NS)
    checks.append(check("time_pairing", "FAIL" if over else "PASS",
                        f"max_gap={max(gaps)/1e6:.1f} ms，超 {MAX_GAP_NS/1e6:.0f} ms 的步 {over}/{len(gaps)}"))

    # 3) 动作窗口稳定
    dts = [s["dt"] for s in steps if s["dt"] > 0]
    if not dts:
        checks.append(check("action_window", "FAIL", "没有有效的 sim_dt"))
    else:
        mean = statistics.fmean(dts)
        cv = statistics.pstdev(dts) / mean if mean else 0.0
        checks.append(check("action_window", "FAIL" if cv > MAX_WINDOW_CV else "PASS",
                            f"sim_dt mean={mean:.3f}s min={min(dts):.3f} max={max(dts):.3f} cv={cv:.1%}",
                            mean_s=round(mean, 4), min_s=round(min(dts), 4),
                            max_s=round(max(dts), 4), cv=round(cv, 4)))

    # 4) 前进动作是否真的动。刚做完偏航的那一步有加速滞后，单独看，否则
    #    两群混在一起会得出「13% 动作没生效」的误判。
    approx = "（位移由位姿差近似，窗口错位 ±60 ms）" if any(
        s["disp_approx"] for s in steps) else ""
    steady = [s for s in steps if s["action"] == 1 and s["prev_action"] == 1
              and s["disp"] is not None]
    if not steady:
        checks.append(check("forward_motion", "SKIP", "没有连续两个前进步，无法判定"))
    else:
        bad_fwd = [s for s in steady if s["disp"] < FWD_MIN_V * s["dt"]]
        share = len(bad_fwd) / len(steady)
        checks.append(check("forward_motion", "FAIL" if share > MAX_FWD_FAIL_SHARE else "PASS",
                            f"{len(bad_fwd)}/{len(steady)} 个稳态前进步位移不足 "
                            f"{FWD_MIN_V:.2f}×dt（{share:.1%}，阈值 {MAX_FWD_FAIL_SHARE:.0%}）"
                            f"{approx}"))

    # 4b) 偏航→前进的过渡步几乎不动
    trans = [s for s in steps if s["action"] == 1 and s["prev_action"] in (2, 3)
             and s["disp"] is not None and s["dt"] > 0]
    if not trans:
        checks.append(check("transition_lag", "SKIP", "没有偏航接前进的过渡步"))
    else:
        ratios = sorted(s["disp"] / (SPEED_FWD * s["dt"]) for s in trans)
        med = ratios[len(ratios) // 2]
        checks.append(check("transition_lag", "FAIL" if med < MIN_TRANSITION_RATIO else "PASS",
                            f"偏航接前进的 {len(trans)} 步，位移中位数为标称的 {med:.0%}"
                            f"（阈值 {MIN_TRANSITION_RATIO:.0%}）{approx}",
                            median_ratio=round(med, 3), n=len(trans)))

    # 5) 偏航方向
    yaw = [s for s in steps if s["action"] in (2, 3)]
    if not yaw or yaw[0]["dyaw"] is None:
        checks.append(check("yaw_direction", "SKIP",
                            "actions.jsonl 没有 dyaw_deg 字段（旧格式），偏航方向无法离线核验"))
    else:
        wrong = [s for s in yaw
                 if (s["action"] == 2 and s["dyaw"] <= YAW_MIN_DEG)
                 or (s["action"] == 3 and s["dyaw"] >= -YAW_MIN_DEG)]
        checks.append(check("yaw_direction", "FAIL" if wrong else "PASS",
                            f"{len(wrong)}/{len(yaw)} 个偏航步转向与标签相反或过小"))

    # 6) 类别分布
    counts = Counter(s["action"] for s in steps)
    total = len(steps)
    dist = {NAMES[k]: counts.get(k, 0) for k in range(4)}
    thin = [NAMES[k] for k in range(4) if counts.get(k, 0) / total < MIN_CLASS_SHARE]
    checks.append(check("class_share", "FAIL" if thin else "PASS",
                        f"每类占比低于 {MIN_CLASS_SHARE:.0%}：{thin or '无'}", distribution=dist))

    # 7) split 几何隔离
    by_split = {}
    for f in ok:
        if f["gid"]:
            by_split.setdefault(f["split"], set()).add(f["gid"])
    if len(by_split) < 2:
        checks.append(check("split_geometry", "SKIP", "少于两个 split 且有 gid，无法核验"))
    else:
        leaks = []
        names = sorted(by_split)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = by_split[a] & by_split[b]
                if common:
                    leaks.append(f"{a}∩{b}={len(common)}")
        checks.append(check("split_geometry", "FAIL" if leaks else "PASS",
                            "; ".join(leaks) if leaks else
                            "交集为 0（" + ", ".join(f"{n}={len(g)}" for n, g in by_split.items()) + "）"))

    # 8) 标签复算：拿采集时记下的动作前位姿 + 场景柱子真值，重跑一遍 plan_action。
    # 只对 teacher=plan_action 的集做：深度专家和盲策略本来就不由柱子真值决定，
    # 拿它复算等于换了个教师，比出来的差异没有意义。
    given = [f for f in ok if f.get("teacher") == "plan_action"]
    others = Counter(f.get("teacher") for f in ok if f.get("teacher") != "plan_action")
    # 逐集/逐步统计覆盖：混了新旧格式时要能看出「验了多少」，不能因为一集是
    # 旧格式就把整条检查降级成 SKIP，也不能把没进 given 的集算作已验。
    no_pose = no_truth = 0          # 因缺字段跳过的步数
    uncovered = sum(len(f["steps"]) for f in ok if f.get("teacher") != "plan_action")
    total = mism = 0
    samples = []
    for f in given:
        if not f["pillars"]:
            no_truth += len(f["steps"])
            continue
        for s in f["steps"]:
            pb, yb = s["pose_before"], s["yaw_before"]
            if not isinstance(pb, dict) or yb is None or "x" not in pb:
                no_pose += 1
                continue
            total += 1
            want = plan_action(pb["x"], pb["y"], yb, f["pillars"])
            if want != s["action_expert"]:
                mism += 1
                if len(samples) < 5:
                    samples.append({"dir": f["dir"], "recorded": s["action_expert"],
                                    "recomputed": want})
    skipped = no_pose + no_truth
    if not ok:
        checks.append(check("label_recompute", "SKIP", "没有可读的集"))
    elif not given:
        checks.append(check("label_recompute", "SKIP",
                            f"没有 teacher=plan_action 的集（其它：{dict(others)}），"
                            "标签不由柱子真值决定，无法复算"))
    elif total == 0:
        # 「一步都没复算成」不等于「全错」。旧格式/缺真值走这里，别报成 100% 不一致。
        checks.append(check("label_recompute", "SKIP",
                            f"没有可复算的步：缺动作前位姿 {no_pose} 步、"
                            f"缺柱子真值 {no_truth} 步（共 {len(given)} 集）",
                            skipped=skipped))
    else:
        share = mism / total
        checks.append(check("label_recompute",
                            "FAIL" if share > MAX_LABEL_MISMATCH else "PASS",
                            f"{mism}/{total} 步复算标签与原标签不一致（{share:.1%}，"
                            f"阈值 {MAX_LABEL_MISMATCH:.0%}；跳过 {skipped} 步"
                            f"（缺位姿 {no_pose}、缺真值 {no_truth}）；"
                            f"另有 {uncovered} 步来自非 plan_action 的集未参与）",
                            samples=samples, mismatch=round(share, 4),
                            covered_steps=total, skipped=skipped))

    return checks


def main(argv=None):
    p = argparse.ArgumentParser(description="数据集验收闸门")
    p.add_argument("--dataset", required=True, help="采集输出根目录（递归找 episode_manifest.json）")
    p.add_argument("--out", help="写 audit.json 的路径")
    args = p.parse_args(argv)

    if not os.path.isdir(args.dataset):
        p.error(f"--dataset 不是目录: {args.dataset}")
    eps = []
    for dirpath, _dirs, files in os.walk(args.dataset):
        if "episode_manifest.json" in files:
            eps.append(dirpath)
    eps.sort()
    if not eps:
        p.error(f"{args.dataset} 下没有 episode_manifest.json")

    facts = [episode_facts(e) for e in eps]
    checks = audit(facts)
    failed = [c for c in checks if c["status"] == "FAIL"]

    result = {
        "schema": "uavrgb.audit.v1",
        "dataset": os.path.abspath(args.dataset),
        "episodes": len(eps),
        "steps": sum(f.get("num_steps", 0) for f in facts if not f.get("error")),
        "passed": not failed,
        "checks": checks,
    }

    print(f"数据集: {result['dataset']}")
    print(f"集数={result['episodes']} 步数={result['steps']}")
    for c in checks:
        print(f"  [{c['status']:4}] {c['name']:18} {c['detail']}")
    print("结论: " + ("PASS" if not failed else f"FAIL（{len(failed)} 项）"))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"已写 {args.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
