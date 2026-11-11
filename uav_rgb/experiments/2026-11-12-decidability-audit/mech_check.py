"""机理核对（只读）：转向标签的方向与已执行转向的净次数 n 的符号是什么关系。

n > 0 = 净左转多（偏左），标签 3 = YAW_RIGHT；「朝回转」= n > 0 且标签 RIGHT，或 n < 0 且标签 LEFT。
第一部分用全部 split 的帧数标签；第二部分只用测试帧，按 |n_raw| 分层，用 preds_pooled_w.npz 里的 argmax
（不用 logit），看「判成转向的转向帧里方向对的比例」。这个口径与 tables.md 里的「转向帧左右对」（L、R 两个 logit 比）不同。
用法：$D/teacher-venv/bin/python mech_check.py [输出目录，默认 $D/exp-decidability-audit-20261112]
"""
import sys

import numpy as np

OUT = (sys.argv[1] if len(sys.argv) > 1 else "/root/rivermind-data/uav_rgb/exp-decidability-audit-20261112").rstrip("/") + "/"
F = np.load(OUT + "features.npz")
y, ep = F["label"], F["ep_idx"]
src = F["ep_source"][ep]
names = [str(s) for s in F["source_names"]]
teacher = np.isin(src, [names.index("trunc"), names.index("more_demos")])
student = np.isin(src, [names.index(n) for n in ("round_1", "round_2", "round_3")])
n = F["n_raw"]
for tag, m in (("教师驱动", teacher), ("学生驱动", student)):
    turn = m & np.isin(y, (2, 3))
    big = turn & (np.abs(n) >= 2)
    toward = ((n > 0) & (y == 3)) | ((n < 0) & (y == 2))
    print("%s: 帧 %d，转向帧 %d（%.3f），其中 |n|>=2 的 %d（占转向帧 %.3f）；|n|>=2 的转向帧里标签朝回转的比例 %.3f（%d/%d）" % (
        tag, m.sum(), turn.sum(), turn.sum() / m.sum(), big.sum(), big.sum() / turn.sum(), toward[big].mean(), toward[big].sum(), big.sum()))
    small = turn & (np.abs(n) < 2)
    print("    |n|<2 的转向帧 %d：LEFT %d / RIGHT %d" % (small.sum(), (small & (y == 2)).sum(), (small & (y == 3)).sum()))
    nz = m & (np.abs(n) >= 2)
    print("    |n|>=2 的帧 %d，其中标签为转向的比例 %.3f；|n|<2 的帧里标签为转向的比例 %.3f" % (
        nz.sum(), np.isin(y[nz], (2, 3)).mean(), np.isin(y[m & (np.abs(n) < 2)], (2, 3)).mean()))

print("--- 测试帧：转向帧里判成转向的，方向对的比例，按 |n_raw| 分层（pooled_w；3 个种子均值）")
P = np.load(OUT + "preds_pooled_w.npz")
split = F["ep_split"][ep]
test = np.nonzero(split == 2)[0]
groups = [str(g) for g in P["groups"]]
for tag, m in (("学生驱动", student), ("教师驱动", teacher)):
    mt = m[test]
    yt, nt = y[test][mt], F["n_raw"][test][mt]
    for lab, sel in (("|n|>=2", np.abs(nt) >= 2), ("|n|<2", np.abs(nt) < 2)):
        turn = sel & np.isin(yt, (2, 3))
        row = []
        for g in ("S1", "S3", "S5"):
            gi = groups.index(g)
            vals = []
            for s in range(3):
                pr = P["pred"][gi, s][mt][turn]
                said = np.isin(pr, (2, 3))
                vals.append(float((pr[said] == yt[turn][said]).mean()) if said.any() else float("nan"))
            row.append("%s %.3f" % (g, np.nanmean(vals)))
        print("%s %s：转向帧 %d；%s" % (tag, lab, turn.sum(), "；".join(row)))
