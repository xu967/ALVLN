"""独立复核：只用 features.npz 与 preds_*.npz，不调 audit_* 里的任何函数，重算表里的整体准确率、macro-F1（F/L/R）与「该转时转了」，并数几个报告里引用的计数。

只读。用法（输出到标准输出）：
    $D/teacher-venv/bin/python recheck_tables.py [输出目录，默认 $D/exp-decidability-audit-20261112]
输出：
  1. pooled_w、pooled_u 的 S1、S3、S5 在三个子集上的整体准确率与该转时转了（均值与种子范围，三位小数）；
  2. 8 个特征组在三个子集上的 macro-F1 与相邻组差值（报告结论段引用的数）；
  3. 解析 tables.md 里 pooled_w（§2）与 pooled_u（§5）的主表，把 2 配置 × 3 子集 × 8 组 × 3 指标的均值逐格与重算值比（三位小数的字符串相同才算一致），不一致的逐格列出；
  4. 几个计数（几何簇数、执行者、|n| ≥ 2 的帧占比、画面空的帧占比）。
「转向帧左右对」用 L、R 两个 logit 比，preds 里只存了 argmax，这里重算不了。
"""
import re
import sys

import numpy as np

OUT = (sys.argv[1] if len(sys.argv) > 1 else "/root/rivermind-data/uav_rgb/exp-decidability-audit-20261112").rstrip("/") + "/"
F = np.load(OUT + "features.npz")
P = {"pooled_w": np.load(OUT + "preds_pooled_w.npz"), "pooled_u": np.load(OUT + "preds_pooled_u.npz")}
y = F["label"]
ep = F["ep_idx"]
split = F["ep_split"][ep]
src = F["ep_source"][ep]
names = [str(s) for s in F["source_names"]]
test = np.nonzero(split == 2)[0]
teacher = np.isin(src, [names.index("trunc"), names.index("more_demos")])
student = np.isin(src, [names.index(n) for n in ("round_1", "round_2", "round_3")])
empty = (F["d_wide"][:, 20:40] >= 4.0).all(1)
SUBSETS = (("teacher", teacher), ("student", student), ("empty", empty))


def f1_frl(pred, true):
    """FORWARD / LEFT / RIGHT 三类 F1 的宏平均（SAFE 不进平均）；F1 = 2tp / (2tp + fp + fn)。"""
    out = []
    for c in (1, 2, 3):
        tp = float(((pred == c) & (true == c)).sum())
        fp = float(((pred == c) & (true != c)).sum())
        fn = float(((pred != c) & (true == c)).sum())
        out.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0)
    return float(np.mean(out))


# 重算：配置 × 子集 × 特征组，每格三个种子各一个值
CALC = {}
for tag, P_ in P.items():
    assert (P_["test_idx"] == test).all()
    groups = [str(g) for g in P_["groups"]]
    for sub, m in SUBSETS:
        mm = m[test]
        yy = y[test][mm]
        turn = np.isin(yy, (2, 3))
        for gi, g in enumerate(groups):
            preds = [P_["pred"][gi, s][mm] for s in range(3)]
            CALC[(tag, sub, g)] = {
                "n": int(mm.sum()),
                "acc": [float((p == yy).mean()) for p in preds],
                "f1": [f1_frl(p, yy) for p in preds],
                "turn": [float(np.isin(p[turn], (2, 3)).mean()) for p in preds],
            }

print("--- 整体准确率与该转时转了（S1、S3、S5；均值 [最小, 最大]）")
for tag in P:
    for g in ("S1", "S3", "S5"):
        for sub, _ in SUBSETS:
            c = CALC[(tag, sub, g)]
            print("%s %s %-7s n=%4d acc=%.3f [%.3f,%.3f]  turn_recall=%.3f [%.3f,%.3f]" % (
                tag, g, sub, c["n"], np.mean(c["acc"]), min(c["acc"]), max(c["acc"]), np.mean(c["turn"]), min(c["turn"]), max(c["turn"])))
print("基线(恒选前进) 教师/学生/无障碍:", [round(float((y[test][m[test]] == 1).mean()), 3) for _, m in SUBSETS])

# macro-F1（F/L/R）：各组三个种子的均值，及报告引用的相邻组差值
print("--- macro-F1（F/L/R，三个种子的均值）与相邻组差值")
PAIRS = (("S3", "S1"), ("S4", "S3"), ("S5", "S4"), ("S6", "S5"), ("S7", "S6"), ("S3c", "S3"))
GROUPS = [str(g) for g in P["pooled_w"]["groups"]]
for tag in P:
    for sub, _ in (SUBSETS[1], SUBSETS[0], SUBSETS[2]):
        f1 = {g: float(np.mean(CALC[(tag, sub, g)]["f1"])) for g in GROUPS}
        print("%s %-7s macro-F1: %s" % (tag, sub, " ".join("%s=%.3f" % (g, f1[g]) for g in GROUPS)))
        print("%s %-7s 差值:     %s" % (tag, sub, " ".join("%s−%s=%+.3f" % (a, b, f1[a] - f1[b]) for a, b in PAIRS)))


# 与 tables.md 逐格对照
def parse_tables(path):
    """tables.md 的主表：{(配置, 子集, 组): (整体准确率, macro-F1, 该转时转了)}，只取单元格里的第一个数（均值）。
    pooled_w 在 §2（### 小节），pooled_u 在 §5 的 `### pooled_u` 下（** 小节）；只认表头以「特征组 | 整体准确率」开头的表，
    每个 (配置, 子集) 的第一张；附加子集、区间表（单元格只有区间）、各类召回表不取。"""
    cfg = sub = None
    main_table = False
    got = {}
    for line in open(path, encoding="utf-8").read().split("\n"):
        if line.startswith("## "):
            cfg = "pooled_w" if line.startswith("## 2.") else None
            sub = None
        elif line.startswith("### pooled_u"):
            cfg, sub = "pooled_u", None
        elif line.startswith("### ") and not line.startswith("### 主表"):
            if cfg == "pooled_u":
                cfg = None
        if line.startswith("### ") or line.startswith("**"):
            if "纯学生" in line or "全部测试帧" in line or "区间" in line:
                sub = None
            elif "教师驱动" in line:
                sub = "teacher"
            elif "学生驱动" in line:
                sub = "student"
            elif "画面" in line:
                sub = "empty"
        if line.startswith("| 特征组 |"):
            main_table = line.startswith("| 特征组 | 整体准确率 |")
        m = re.match(r"\| (S\d+c?)[（ ]", line)
        if m and cfg and sub and main_table:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            nums = [re.match(r"(\d\.\d{3})", c) for c in cells[1:4]]
            if all(nums):
                got.setdefault((cfg, sub, m.group(1)), tuple(n.group(1) for n in nums))
    return got


print("--- 与 tables.md 逐格对照（pooled_w §2、pooled_u §5；整体准确率、macro-F1、该转时转了的均值）")
try:
    tab = parse_tables(OUT + "tables.md")
except FileNotFoundError:
    tab = {}
    print("没有 tables.md，跳过")
if tab:
    bad, checked = [], 0
    for key, c in sorted(CALC.items()):
        if key not in tab:
            bad.append((key, "tables.md 里没有这一格"))
            continue
        for name, idx in (("acc", 0), ("f1", 1), ("turn", 2)):
            checked += 1
            mine = "%.3f" % np.mean(c[name])
            if mine != tab[key][idx]:
                bad.append((key + (name,), "重算 %s，表 %s" % (mine, tab[key][idx])))
    print("检查 %d 个格子（应为 2 × 3 × 8 × 3 = 144），不一致 %d 个" % (checked, len(bad)))
    for item in bad:
        print("  不一致：", item)

print("--- 计数")
print("集数 train/val/test:", np.bincount(F["ep_split"]), "几何簇总数", len(set(F["ep_gid"])),
      "各 split 几何簇", [len(set(F["ep_gid"][F["ep_split"] == k])) for k in range(3)])
print("学生驱动帧的执行者 exec_code（0 教师 1 学生 2 专家）:", np.bincount(F["exec_code"][student], minlength=3),
      "教师驱动帧:", np.bincount(F["exec_code"][teacher], minlength=3))
print("|n_raw|>=2 的帧占比 教师/学生:", round(float((np.abs(F["n_raw"][teacher]) >= 2).mean()), 3), round(float((np.abs(F["n_raw"][student]) >= 2).mean()), 3))
print("|n_raw|>=5（n 被截断）的帧占比 教师/学生:", round(float((np.abs(F["n_raw"][teacher]) >= 5).mean()), 3), round(float((np.abs(F["n_raw"][student]) >= 5).mean()), 3))
print("画面无障碍帧占比 教师/学生:", round(float(empty[teacher].mean()), 3), round(float(empty[student].mean()), 3))
