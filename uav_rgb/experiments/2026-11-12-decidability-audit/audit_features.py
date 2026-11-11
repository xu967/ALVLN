#!/usr/bin/env python3
"""可判定性审计（卡 E2E-0）第一步：选步、真值特征提取、与打包日志对账。

只读：输入是现成的采集目录，输出写到 AUDIT_OUT（数据盘）。特征都由仿真真值算，只用于分析，不进任何学生输入：
  d_wide  同一位姿上 180° 视场、60 列（3° 一列）、上限 4 m 的列距离；中间 20 列（下标 20–39）就是 60° 视场的 `d`
  n       该步之前已执行的左转数减右转数（原值，分析时截到 ±4）
  hist    上一步执行的动作、最近一次转向的方向、距最近一次转向过了几步（封顶 5）
  priv    横向位置 y、到最远柱的纵向距离 x_rem（学生拿不到）
列距离直接调 `sim/tools/rule_policy.py` 的 column_distances，选步直接调 `tools/preprocess.py` 的 load_episode，不另写一套。

「进训练集的步」的口径：两份数据集当时的打包日志（trunc/preprocess.log 与 full-1109 各轮 ds.log）都是 `--drop-collision`、
没有 `--exclude-collision-episodes`，所以用 load_episode(drop_collision=True)。提取结束后逐集、逐 split、逐类与这几份日志对账，
对不上就停（shortcut: 日志里的 episode_id 与 manifest 的 episode_id 一一对应，已核实 932 集互不重复）。

字段与任务卡的出入（以实际为准）：柱子真值在每集目录的 scene.json 的 pillars，episode_manifest.json 里没有 pillars；
动作前位姿在 actions.jsonl 的 pose_before 与 yaw_before（度）。

用法见 run.sh（守护进程入口）。单独跑：AUDIT_* 环境变量由 run.sh 的配置区导出。
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import preprocess as pp  # noqa: E402  load_episode：进训练集的步
import rule_policy as rp  # noqa: E402  column_distances：真值列距离

SAFE, FWD, LEFT, RIGHT = 0, 1, 2, 3
CLASS_NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
HFOV_WIDE = 180.0        # d_wide 的视场；60° 视场的 d 是它正中间的 20 列
MAX_RANGE = 4.0          # 列距离上限，等于 column_distances 的 max_range 默认值
N_NARROW, N_WIDE = 20, 60
NARROW = slice(20, 40)
N_CLIP = 4               # 卡里规定：n 截到 ±4
SINCE_CAP = 5            # 卡里规定：距最近一次转向的步数封顶 5
EXEC_TEACHER, EXEC_STUDENT, EXEC_EXPERT = 0, 1, 2   # 该步实际由谁执行：教师示范 / 学生 / DAgger β 混合里改执行的教师
SPLITS = pp.SPLITS


def env(name, cast=str):
    """配置只从 run.sh 的配置区进来；缺了就报错，不在这里藏默认值。"""
    if name not in os.environ:
        raise SystemExit("缺环境变量 %s：这个脚本由 run.sh 启动，配置区在 run.sh 顶部" % name)
    return cast(os.environ[name])


def log(msg):
    print("%s %s" % (time.strftime("%F %T"), msg), flush=True)


# ---------------------------------------------------------------- 特征
def column_features(x, y, yaw_deg, pillars):
    """(d, d_wide)：d 是 60° 视场 20 列，d_wide 是 180° 视场 60 列；两者同一位姿、同一组射线（d = d_wide 的下标 20–39）。"""
    wide = rp.column_distances(x, y, yaw_deg, pillars, hfov_deg=HFOV_WIDE)
    return wide[NARROW], wide


def executed_history(executed):
    """由逐步「已执行动作」算每一步之前的历史量。executed 必须是该集全部步（含被丢弃的碰撞步），按 step 升序。

    n_raw：此前 YAW_LEFT 数减 YAW_RIGHT 数；prev：上一步动作（首步按 SAFE_NEUTRAL，与 preprocess 的 prev_action 一致）；
    last_dir：最近一次转向的方向（0 没转过、1 左、2 右）；since：距最近一次转向过了几步（1…SINCE_CAP；没转过记 SINCE_CAP）。
    """
    n_steps = len(executed)
    n_raw = np.zeros(n_steps, np.int16)
    prev = np.zeros(n_steps, np.int8)
    last_dir = np.zeros(n_steps, np.int8)
    since = np.full(n_steps, SINCE_CAP, np.int8)
    n, last_t, last_d = 0, None, 0
    for t in range(n_steps):
        n_raw[t] = n
        prev[t] = executed[t - 1] if t > 0 else SAFE
        if last_t is not None:
            last_dir[t] = last_d
            since[t] = min(SINCE_CAP, t - last_t)
        if executed[t] == LEFT:
            n, last_t, last_d = n + 1, t, 1
        elif executed[t] == RIGHT:
            n, last_t, last_d = n - 1, t, 2
    return {"n_raw": n_raw, "prev": prev, "last_dir": last_dir, "since": since}


def clip_n(n_raw):
    return np.clip(np.asarray(n_raw), -N_CLIP, N_CLIP)


def heading_code(n):
    """W1-9 卡的航向码（取执行前的 n）：|n| ≤ 1 → 0，n ≥ 2 → 2（偏左），n ≤ −2 → 3（偏右）；码 1 不用。"""
    n = int(n)
    return 2 if n >= 2 else (3 if n <= -2 else 0)


# ---------------------------------------------------------------- 选步 + 一集的全部记录
def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _exec_code(row):
    who = row.get("executed_by")
    if who is None:
        return EXEC_TEACHER
    if who == "student":
        return EXEC_STUDENT
    if who == "expert":
        return EXEC_EXPERT
    raise ValueError("step %s 的 executed_by=%r 不认识（本审计只处理教师示范与 DAgger 自采）" % (row.get("step"), who))


def episode_records(ep_dir, drop_collision=True):
    """一集进训练集的步，每步一行特征。选步用 preprocess.load_episode（碰撞步丢不丢由 drop_collision 定，和打包日志一致）；
    历史量（n、hist）按该集全部步算，被丢弃的碰撞步照样算「已执行」：它们确实飞过。"""
    ep = pp.load_episode(ep_dir, drop_collision=drop_collision)
    rows = sorted(_read_jsonl(os.path.join(ep_dir, "actions.jsonl")), key=lambda r: int(r["step"]))
    if [int(r["step"]) for r in rows] != list(range(len(rows))):
        raise ValueError("%s: step 不是从 0 起的连续整数，历史量没法按序算" % ep_dir)
    executed = [int(r.get("action_executed", r["action_expert"])) for r in rows]
    hist = executed_history(executed)
    with open(os.path.join(ep_dir, "scene.json"), encoding="utf-8") as f:
        pillars = json.load(f)["pillars"]
    max_px = max(float(p["x"]) for p in pillars)
    out = {k: [] for k in ("step", "label", "executed", "exec_code", "n_raw", "prev", "last_dir", "since",
                           "x", "y", "yaw", "x_rem")}
    wide_rows = []
    for s in ep["samples"]:
        t = s["step"]
        r = rows[t]
        if s["label"] != int(r["action_expert"]) or s["prev_action"] != int(hist["prev"][t]):
            raise AssertionError("%s step %d: 与 preprocess 的标签或 prev_action 不一致" % (ep_dir, t))
        pb = r["pose_before"]
        if not isinstance(pb, dict) or r.get("yaw_before") is None:
            raise ValueError("%s step %d: 缺 pose_before 或 yaw_before" % (ep_dir, t))
        _, wide = column_features(float(pb["x"]), float(pb["y"]), float(r["yaw_before"]), pillars)
        wide_rows.append(wide)
        out["step"].append(t)
        out["label"].append(s["label"])
        out["executed"].append(executed[t])
        out["exec_code"].append(_exec_code(r))
        for k in ("n_raw", "prev", "last_dir", "since"):
            out[k].append(hist[k][t])
        out["x"].append(float(pb["x"]))
        out["y"].append(float(pb["y"]))
        out["yaw"].append(float(r["yaw_before"]))
        out["x_rem"].append(max_px - float(pb["x"]))
    dtypes = {"step": np.int16, "label": np.int8, "executed": np.int8, "exec_code": np.int8, "n_raw": np.int16,
              "prev": np.int8, "last_dir": np.int8, "since": np.int8, "x": np.float32, "y": np.float32,
              "yaw": np.float32, "x_rem": np.float32}
    rec = {k: np.asarray(v, dtypes[k]) for k, v in out.items()}
    rec["d_wide"] = np.asarray(wide_rows, np.float32).reshape(-1, N_WIDE)
    rec.update(episode_id=str(ep["episode_id"]), split=ep["split"], gid=ep.get("geometry_gid") or "",
               dagger_round=int(ep["dagger_round"]), beta=float(ep["beta"]), dir=ep["dir"],
               n_rows=len(rows), n_dropped=len(rows) - len(ep["samples"]), max_px=max_px)
    return rec


def collection_dirs(root):
    dirs = []
    for split in SPLITS:
        dirs += [os.path.dirname(p) for p in sorted(glob.glob(os.path.join(root, split, "*", "episode_manifest.json")))]
    return dirs


# ---------------------------------------------------------------- 与打包日志对账
_RE_EPISODES = re.compile(r"^episodes:\s+(\d+)\s+samples:\s+(\d+)", re.M)
_RE_SPLITS = re.compile(r"^splits:\s+train=(\d+)\s+val=(\d+)\s+test=(\d+)", re.M)
_RE_CLASSES = re.compile(r"^\s+(train|val|test)\s+\{'SAFE_NEUTRAL': (\d+), 'FORWARD_SLOW': (\d+), "
                         r"'YAW_LEFT': (\d+), 'YAW_RIGHT': (\d+)\}", re.M)
_RE_COLLISION = re.compile(r"^collision: dropped (\d+) of (\d+) steps", re.M)
_RE_EPISODE = re.compile(r"^\s+episode_id=(\S+) scene_id=\S+ split=(\w+) steps=(\d+) sample_start=\d+", re.M)


def parse_pack_log(text):
    """preprocess.py 打包日志里和对账有关的行：集数、样本数、各 split 样本数与类别数、丢弃的碰撞步、逐集 (split, steps)。"""
    m = _RE_EPISODES.search(text)
    s = _RE_SPLITS.search(text)
    c = _RE_COLLISION.search(text)
    if not (m and s and c):
        raise ValueError("不是 preprocess 的打包日志（缺 episodes / splits / collision 行）")
    return {"episodes": int(m.group(1)), "samples": int(m.group(2)),
            "splits": dict(zip(SPLITS, map(int, s.groups()))),
            "class_counts": {sp: [int(v) for v in rest] for sp, *rest in _RE_CLASSES.findall(text)},
            "dropped": int(c.group(1)), "seen": int(c.group(2)),
            "per_episode": {eid: (sp, int(n)) for eid, sp, n in _RE_EPISODE.findall(text)}}


def check_against_log(summary, parsed):
    """summary = {episode_id: {split, retained, total, class_counts}}；对日志里列出的每一集与各项合计逐项比对。"""
    problems = []
    eps = parsed["per_episode"]
    if len(eps) != parsed["episodes"]:
        problems.append("日志自身：逐集行 %d ≠ 集数 %d" % (len(eps), parsed["episodes"]))
    missing = [e for e in eps if e not in summary]
    if missing:
        problems.append("日志里有 %d 集不在提取结果里（前几个 %s）" % (len(missing), missing[:5]))
    mine = {e: summary[e] for e in eps if e in summary}
    for e, (sp, steps) in eps.items():
        if e in mine and (mine[e]["split"] != sp or mine[e]["retained"] != steps):
            problems.append("集 %s：日志 split=%s steps=%d，提取 split=%s retained=%d"
                            % (e, sp, steps, mine[e]["split"], mine[e]["retained"]))
    total = sum(v["retained"] for v in mine.values())
    if total != parsed["samples"]:
        problems.append("样本数：日志 %d，提取 %d" % (parsed["samples"], total))
    for sp in SPLITS:
        got = sum(v["retained"] for v in mine.values() if v["split"] == sp)
        if got != parsed["splits"][sp]:
            problems.append("split=%s 样本数：日志 %d，提取 %d" % (sp, parsed["splits"][sp], got))
        cc = [sum(v["class_counts"][k] for v in mine.values() if v["split"] == sp) for k in range(4)]
        if cc != parsed["class_counts"][sp]:
            problems.append("split=%s 类别数：日志 %s，提取 %s" % (sp, parsed["class_counts"][sp], cc))
    seen = sum(v["total"] for v in mine.values())
    if (seen, seen - total) != (parsed["seen"], parsed["dropped"]):
        problems.append("丢弃的碰撞步：日志 %d / %d 步，提取 %d / %d 步" % (parsed["dropped"], parsed["seen"], seen - total, seen))
    return {"ok": not problems, "problems": problems, "episodes": len(eps), "samples": total,
            "dropped": seen - total, "seen": seen}


# ---------------------------------------------------------------- 全部采集 → features.npz
def parse_pairs(text):
    """'a=/p/a;b=/p/b' → [(a, /p/a), (b, /p/b)]，保持顺序。"""
    return [tuple(kv.split("=", 1)) for kv in text.split(";") if kv.strip()]


def extract_all(collections, logs, out_dir, drop_collision=True):
    """collections = [(名字, 采集根目录)]，logs = [打包日志路径]。写 out_dir/features.npz 与 selection_check.json。"""
    t0 = time.time()
    per_ep, summary, chunks = [], {}, []
    for name, root in collections:
        dirs = collection_dirs(root)
        log("采集 %s：%d 集 %s" % (name, len(dirs), root))
        for i, d in enumerate(dirs):
            rec = episode_records(d, drop_collision)
            if rec["episode_id"] in summary:
                raise ValueError("episode_id %s 重复出现（%s）" % (rec["episode_id"], d))
            summary[rec["episode_id"]] = {
                "split": rec["split"], "retained": len(rec["step"]), "total": rec["n_rows"], "source": name,
                "class_counts": [int((rec["label"] == k).sum()) for k in range(4)]}
            rec["source"] = name
            per_ep.append(rec)
            if (i + 1) % 50 == 0:
                log("  %s %d/%d，累计 %d 集，%.0f s" % (name, i + 1, len(dirs), len(per_ep), time.time() - t0))
    gid_split = {}
    for rec in per_ep:
        if rec["gid"] and gid_split.setdefault(rec["gid"], rec["split"]) != rec["split"]:
            raise ValueError("几何组 %s 同时在 %s 与 %s（同几何必须同 split）" % (rec["gid"], gid_split[rec["gid"]], rec["split"]))
    checks = {}
    for path in logs:
        with open(path, encoding="utf-8") as f:
            res = check_against_log(summary, parse_pack_log(f.read()))
        checks[path] = res
        log("对账 %s：%s（%d 集 %d 样本，丢弃碰撞步 %d）" % (path, "通过" if res["ok"] else "失败", res["episodes"], res["samples"], res["dropped"]))
        for p in res["problems"][:10]:
            log("  不一致：" + p)
    with open(os.path.join(out_dir, "selection_check.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=1)
    if not all(c["ok"] for c in checks.values()):
        raise RuntimeError("选步与打包日志对不上，见 selection_check.json；停")
    names = [n for n, _ in collections]
    cat = lambda key: np.concatenate([r[key] for r in per_ep])  # noqa: E731
    counts = [len(r["step"]) for r in per_ep]
    arrays = {k: cat(k) for k in ("step", "label", "executed", "exec_code", "n_raw", "prev", "last_dir", "since",
                                  "x", "y", "yaw", "x_rem")}
    arrays["d_wide"] = np.concatenate([r["d_wide"] for r in per_ep])
    arrays["ep_idx"] = np.repeat(np.arange(len(per_ep), dtype=np.int32), counts)
    arrays["ep_id"] = np.array([r["episode_id"] for r in per_ep])
    arrays["ep_gid"] = np.array([r["gid"] for r in per_ep])
    arrays["ep_split"] = np.array([SPLITS.index(r["split"]) for r in per_ep], np.int8)
    arrays["ep_source"] = np.array([names.index(r["source"]) for r in per_ep], np.int8)
    arrays["ep_round"] = np.array([r["dagger_round"] for r in per_ep], np.int8)
    arrays["ep_beta"] = np.array([r["beta"] for r in per_ep], np.float32)
    arrays["source_names"] = np.array(names)
    # 抽查：d_wide 的中间 20 列与直接按 60° 视场调库函数的结果逐位相同（切片等价性在真实帧上再核一次）
    rng = np.random.RandomState(0)
    idx = rng.choice(len(arrays["step"]), size=min(100, len(arrays["step"])), replace=False)
    for i in idx:
        ep_dir = per_ep[int(arrays["ep_idx"][i])]["dir"]
        row = {int(r["step"]): r for r in _read_jsonl(os.path.join(ep_dir, "actions.jsonl"))}[int(arrays["step"][i])]
        with open(os.path.join(ep_dir, "scene.json"), encoding="utf-8") as f:
            pillars = json.load(f)["pillars"]
        pb = row["pose_before"]
        narrow = rp.column_distances(pb["x"], pb["y"], row["yaw_before"], pillars, hfov_deg=60.0)
        if [float(np.float32(v)) for v in narrow] != [float(v) for v in arrays["d_wide"][i, NARROW]]:
            raise AssertionError("d_wide 中间 20 列与 60° 视场的列距离不一致：%s step %s" % (ep_dir, row["step"]))
    log("抽查 %d 帧：d_wide[20:40] 与 hfov=60° 的 column_distances 逐位相同" % len(idx))
    np.savez_compressed(os.path.join(out_dir, "features.npz"), **arrays)
    log("写 features.npz：%d 集 %d 帧，%.0f s" % (len(per_ep), len(arrays["step"]), time.time() - t0))
    return arrays


def main():
    out = env("AUDIT_OUT")
    os.makedirs(out, exist_ok=True)
    cols = parse_pairs(env("AUDIT_COLLECTIONS"))
    logs = [p for p in env("AUDIT_PACK_LOGS").split(";") if p.strip()]
    extract_all(cols, logs, out, drop_collision=env("AUDIT_DROP_COLLISION", int) == 1)


if __name__ == "__main__":
    main()
