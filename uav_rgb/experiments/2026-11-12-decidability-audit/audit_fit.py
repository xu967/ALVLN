#!/usr/bin/env python3
"""可判定性审计（卡 E2E-0）第二步：按特征组拟合小 MLP，统计三个子集上的混淆矩阵；另做不依赖拟合器的分箱核对。

输入 AUDIT_OUT/features.npz（audit_features.py 写的），输出：
  stats_<配置>.npz      每个（特征组, 种子, 子集, 几何簇）的 4×4 混淆矩阵与「转向帧左右对」计数，report 阶段据此算指标与自助法区间
  fitmeta_<配置>.json   每次拟合的输入维数、训练/验证帧数、停止轮数、验证损失、耗时
  preds_<配置>.npz      测试帧上的预测（只存全量拟合的配置）
  data_summary.json     各 split × 子集的帧数、类别数、几何簇数
  bins.json             分箱核对：每个特征组 × 子集的箱内多数占比与留出准确率

特征组（卡里定的，不改）：S1={d}  S2={d, 上一步动作}  S3={d, n}  S4={d, n, hist}  S5={d_wide, n, hist}  S6=S5+priv。
卡外附加两组，只用于定位，报告里标明：S3c={d, 航向码}（n 换成 W1-9 卡的三值码 0/2/3）；S7=S6+真实航向 sin/cos（天花板参照）。
拟合器：两层各 64 个单元的 ReLU MLP，Adam，类别加权交叉熵（权重 = N/(C·n_c)，缺的类 0，与 src/class_weights.cpp 的 auto 一致），
train 拟合、val 的加权交叉熵最低的那一轮停止（patience 轮没有改善就停）、test 只报数；torch 只用 CPU 1 线程。
子集与指标定义见 README.md。
"""
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_features as af  # noqa: E402

SAFE, FWD, LEFT, RIGHT = af.SAFE, af.FWD, af.LEFT, af.RIGHT
CARD_GROUPS = ("S1", "S2", "S3", "S4", "S5", "S6")
EXTRA_GROUPS = ("S3c", "S7")
ALL_GROUPS = CARD_GROUPS + EXTRA_GROUPS
SUBSET_ORDER = ("teacher", "student", "empty", "student_r23", "all")
CONFIG_DEFS = {
    # 主配置：所有训练帧一起拟合，类别加权；各子集只在测试帧上取数
    "pooled_w": {"weighted": True, "fit_subset": "all", "eval": SUBSET_ORDER, "groups": ALL_GROUPS, "save_preds": True},
    # 敏感性：不加权（分箱核对的多数规则也是不加权的，与它对量级用这一版）
    "pooled_u": {"weighted": False, "fit_subset": "all", "eval": SUBSET_ORDER, "groups": ALL_GROUPS, "save_preds": True},
    # 敏感性：只用该子集的训练帧拟合、该子集的验证帧停止，只在该子集的测试帧上取数（卡没说拟合池，两种读法都报）
    "teacher_w": {"weighted": True, "fit_subset": "teacher", "eval": ("teacher",), "groups": CARD_GROUPS, "save_preds": False},
    "student_w": {"weighted": True, "fit_subset": "student", "eval": ("student", "student_r23"), "groups": CARD_GROUPS,
                  "save_preds": False},
}


# ---------------------------------------------------------------- 特征组矩阵
def class_weights_auto(counts):
    """N / (C·n_c)，没有的类 0；与 src/class_weights.cpp 的 auto 同一个式子，C 固定 4。"""
    counts = np.asarray(counts, np.float64)
    return np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1.0)), 0.0)


def onehot(idx, k):
    idx = np.asarray(idx).astype(np.int64)
    out = np.zeros((len(idx), k), np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def code3_index(n_raw):
    """W1-9 卡的航向码折成 3 列的下标：|n| ≤ 1 → 0（码 0）；n ≥ 2 → 1（码 2，偏左）；n ≤ −2 → 2（码 3，偏右）。"""
    n = np.asarray(n_raw)
    return np.where(n >= 2, 1, np.where(n <= -2, 2, 0))


def build_groups(feats, train_mask):
    """每个特征组的输入矩阵（float32）。距离除以 4 m 上限；n 截到 ±4 后 9 位独热；hist = 上一步动作 4 + 最近转向方向 3 + 距上次转向步数 5；
    priv（横向位置 y、到最远柱的纵向距离）按训练帧的均值与标准差标准化。"""
    dw = feats["d_wide"].astype(np.float32) / af.MAX_RANGE
    d = dw[:, af.NARROW]
    n_oh = onehot(af.clip_n(feats["n_raw"]) + af.N_CLIP, 2 * af.N_CLIP + 1)
    prev_oh = onehot(feats["prev"], 4)
    hist = np.concatenate([prev_oh, onehot(feats["last_dir"], 3), onehot(np.asarray(feats["since"]) - 1, af.SINCE_CAP)], 1)
    tr = np.asarray(train_mask, bool)
    priv_raw = np.stack([feats["y"], feats["x_rem"]], 1).astype(np.float64)
    mu, sd = priv_raw[tr].mean(0), priv_raw[tr].std(0)
    priv = ((priv_raw - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
    yaw = np.deg2rad(np.asarray(feats["yaw"], np.float64))
    sincos = np.stack([np.sin(yaw), np.cos(yaw)], 1).astype(np.float32)
    cat = lambda *a: np.ascontiguousarray(np.concatenate(a, 1), np.float32)  # noqa: E731
    g = OrderedDict()
    g["S1"] = cat(d)
    g["S2"] = cat(d, prev_oh)
    g["S3"] = cat(d, n_oh)
    g["S4"] = cat(d, n_oh, hist)
    g["S5"] = cat(dw, n_oh, hist)
    g["S6"] = cat(g["S5"], priv)
    g["S3c"] = cat(d, onehot(code3_index(feats["n_raw"]), 3))
    g["S7"] = cat(g["S6"], sincos)
    return g


def subset_masks(feats, teacher, student, student_pure):
    """三个子集（卡）与两个附加：教师驱动 = 示范集的帧；学生驱动 = DAgger 各轮的帧；画面里 4 m 内没有障碍 = 60° 视场 20 列全是 4 m 封顶；
    student_r23 = 纯学生执行的轮（β=0）作敏感性；all = 全部帧。名字对不上 source_names 就报错，不静默当空集。"""
    names = [str(s) for s in feats["source_names"]]
    src = np.asarray(feats["ep_source"])[np.asarray(feats["ep_idx"])]

    def inset(group):
        return np.isin(src, [names.index(n) for n in group])

    narrow = np.asarray(feats["d_wide"])[:, af.NARROW]
    return {"teacher": inset(teacher), "student": inset(student), "empty": (narrow >= af.MAX_RANGE).all(1),
            "student_r23": inset(student_pure), "all": np.ones(len(src), bool)}


# ---------------------------------------------------------------- 混淆矩阵与指标
def confusion_by_cluster(y, pred, cluster, n_clusters, mask=None):
    y, pred, cluster = (np.asarray(a, np.int64) for a in (y, pred, cluster))
    if mask is not None:
        mask = np.asarray(mask, bool)
        y, pred, cluster = y[mask], pred[mask], cluster[mask]
    key = cluster * 16 + y * 4 + pred
    return np.bincount(key, minlength=n_clusters * 16).reshape(n_clusters, 4, 4).astype(np.int32)


def lr_ok_by_cluster(y, lr_logit, cluster, n_clusters, mask=None):
    """转向帧里「左右判对」的帧数：只在 L / R 两个 logit 之间比（lr_logit = L 的 logit − R 的 logit），与 4 类 argmax 判没判成转向无关。"""
    y, lr_logit, cluster = np.asarray(y, np.int64), np.asarray(lr_logit), np.asarray(cluster, np.int64)
    ok = ((y == LEFT) & (lr_logit > 0)) | ((y == RIGHT) & (lr_logit < 0))
    if mask is not None:
        ok &= np.asarray(mask, bool)
    return np.bincount(cluster[ok], minlength=n_clusters).astype(np.int32)


def metrics_from_counts(cm, lr_ok):
    """cm[..., 真, 预测] 与 lr_ok[...] → 指标（支持前面任意多个批维度；分母为 0 的给 NaN）。
    acc；recall（四类）；f1（四类）；f1_frl = FORWARD/LEFT/RIGHT 三类宏平均；f1_4 = 四类宏平均（缺类按 0，与 C++ 训练器一致）；
    turn_recall = 教师标签是转向的帧里被判成转向（L 或 R）的比例（该转的时候转没转）；
    lr_acc = 教师标签是转向的帧里 L/R 两个 logit 比出的方向对的比例（转向帧里左右对不对）；lr_cond = 只看判成转向的帧，方向对的比例；
    false_turn = 教师标签是前进的帧里被判成转向的比例。"""
    cm = np.asarray(cm, np.float64)
    lr_ok = np.asarray(lr_ok, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.einsum("...ii->...i", cm)
        rows, cols = cm.sum(-1), cm.sum(-2)
        n = rows.sum(-1)
        f1 = np.where(rows + cols > 0, 2 * tp / (rows + cols), 0.0)
        turn_rows = rows[..., LEFT] + rows[..., RIGHT]
        turn_pred = cm[..., LEFT:RIGHT + 1, LEFT:RIGHT + 1].sum((-1, -2))
        has = n > 0
        return {"acc": tp.sum(-1) / n, "recall": tp / rows, "f1": f1,
                "f1_frl": np.where(has, f1[..., 1:].mean(-1), np.nan), "f1_4": np.where(has, f1.mean(-1), np.nan),
                "turn_recall": turn_pred / turn_rows, "lr_acc": lr_ok / turn_rows,
                "lr_cond": (cm[..., LEFT, LEFT] + cm[..., RIGHT, RIGHT]) / turn_pred,
                "false_turn": (cm[..., FWD, LEFT] + cm[..., FWD, RIGHT]) / rows[..., FWD],
                "n": n, "n_turn": turn_rows}


def majority_baseline_counts(cm):
    """恒选 FORWARD_SLOW 的混淆矩阵：每行的总数全落在第 1 列。"""
    cm = np.asarray(cm)
    out = np.zeros_like(cm)
    out[..., FWD] = cm.sum(-1)
    return out


# ---------------------------------------------------------------- 小 MLP
def fit_mlp(Xtr, ytr, Xva, yva, weights, seed, cfg):
    """输入 → hidden × layers（ReLU）→ 4 类；Adam；加权交叉熵；每轮在 val 上算同一个加权交叉熵，取最低的那一轮（patience 轮没改善就停）。"""
    import torch
    from torch import nn
    torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    dims = [int(Xtr.shape[1])] + [int(cfg["hidden"])] * int(cfg["layers"]) + [4]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    net = nn.Sequential(*mods)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(np.asarray(weights, np.float32)))
    xt = torch.from_numpy(np.ascontiguousarray(Xtr, np.float32))
    yt = torch.from_numpy(np.asarray(ytr, np.int64))
    xv = torch.from_numpy(np.ascontiguousarray(Xva, np.float32))
    yv = torch.from_numpy(np.asarray(yva, np.int64))
    n, bs = len(xt), int(cfg["batch"])
    best, best_epoch, best_state, bad, epochs = float("inf"), 0, None, 0, 0
    for ep in range(1, int(cfg["max_epochs"]) + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(xt[idx]), yt[idx]).backward()
            opt.step()
        epochs = ep
        net.eval()
        with torch.no_grad():
            vloss = float(lossf(net(xv), yv))
        if vloss < best - 1e-6:
            best, best_epoch, bad = vloss, ep, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"best_epoch": best_epoch, "epochs_run": epochs, "best_val_loss": best, "layers": dims,
                 "n_train": int(n), "n_val": int(len(xv))}


def predict_logits(net, X):
    import torch
    with torch.no_grad():
        return net(torch.from_numpy(np.ascontiguousarray(X, np.float32))).numpy()


# ---------------------------------------------------------------- 分箱核对（不依赖拟合器）
def sector_levels(d, n_sectors, edges):
    """把列距离按角度分成 n_sectors 个等宽扇区，每个扇区取最近距离，再按 edges 分档（x < edges[0] → 0，…，x ≥ edges[-1] → 最高档）。"""
    d = np.asarray(d)
    n, c = d.shape
    if c % n_sectors:
        raise ValueError("%d 列分不成 %d 个等宽扇区" % (c, n_sectors))
    mins = d.reshape(n, n_sectors, c // n_sectors).min(2)
    return np.digitize(mins, np.asarray(edges, np.float64)).astype(np.int64)


def mixed_radix(parts, radices):
    out = np.zeros(len(parts[0]), np.int64)
    for p, r in zip(parts, radices):
        p = np.asarray(p, np.int64)
        if p.min() < 0 or p.max() >= r:
            raise ValueError("取值超出进制 %d" % r)
        out = out * r + p
    return out


def bin_check(tr_bin, tr_y, te_bin, te_y):
    """purity_train：训练帧每个箱里教师动作的多数占比（乐观，箱越多越高）；
    heldout_acc：按训练帧的箱内多数类去判测试帧的准确率（箱没见过 → 训练集总体多数类；并列取类号小的）；
    heldout_lr：测试帧里教师标签是转向的，用箱内 L/R 计数比大小判方向的正确率（并列记 0.5）。"""
    tr_bin, tr_y, te_bin, te_y = (np.asarray(a, np.int64) for a in (tr_bin, tr_y, te_bin, te_y))
    nb = int(max(tr_bin.max(initial=-1), te_bin.max(initial=-1))) + 1
    counts = np.bincount(tr_bin * 4 + tr_y, minlength=nb * 4).reshape(nb, 4)
    seen = counts.sum(1) > 0
    glob = int(np.bincount(tr_y, minlength=4).argmax()) if len(tr_y) else FWD
    res = {"n_train": int(len(tr_y)), "n_test": int(len(te_y)), "bins_train": int(seen.sum()),
           "purity_train": float(counts.max(1).sum() / counts.sum()) if len(tr_y) else float("nan"),
           "frames_per_bin": float(len(tr_y) / seen.sum()) if seen.any() else float("nan")}
    if len(te_y):
        pred = np.where(seen[te_bin], counts.argmax(1)[te_bin], glob)
        res["heldout_acc"] = float((pred == te_y).mean())
        res["coverage"] = float(seen[te_bin].mean())
        turn = np.isin(te_y, (LEFT, RIGHT))
        if turn.any():
            l, r = counts[te_bin[turn], LEFT], counts[te_bin[turn], RIGHT]
            y = te_y[turn]
            credit = np.where(l > r, y == LEFT, np.where(r > l, y == RIGHT, 0.5))
            res["heldout_lr"] = float(credit.mean())
        else:
            res["heldout_lr"] = float("nan")
    else:
        res.update(heldout_acc=float("nan"), coverage=float("nan"), heldout_lr=float("nan"))
    return res


def build_bin_ids(feats, train_mask, d_sectors, d_edges, w_sectors, w_edges):
    """各特征组的粗分箱编号。d：4 个扇区取最近距离 × 3 档；d_wide：6 个扇区 × 2 档；n 用三值航向码；
    hist 用「最近转向方向（3）× 距上次转向步数分 3 档（1 / 2–3 / 4–5）」（上一步动作在 since=1 时已由最近转向方向决定，不另分箱）；
    priv 用训练帧的三分位分箱（y 3 档 × 到最远柱的纵向距离 3 档）。"""
    dl = sector_levels(np.asarray(feats["d_wide"])[:, af.NARROW], d_sectors, d_edges)
    wl = sector_levels(np.asarray(feats["d_wide"]), w_sectors, w_edges)
    r_d, r_w = len(d_edges) + 1, len(w_edges) + 1
    code = code3_index(feats["n_raw"])
    last_dir = np.asarray(feats["last_dir"], np.int64)
    since_lv = np.digitize(np.asarray(feats["since"]), [2, 4])
    tr = np.asarray(train_mask, bool)
    q_y = np.quantile(np.asarray(feats["y"])[tr], [1 / 3, 2 / 3])
    q_x = np.quantile(np.asarray(feats["x_rem"])[tr], [1 / 3, 2 / 3])
    y_lv, x_lv = np.digitize(feats["y"], q_y), np.digitize(feats["x_rem"], q_x)
    dparts = [dl[:, j] for j in range(d_sectors)]
    wparts = [wl[:, j] for j in range(w_sectors)]
    rd, rw = [r_d] * d_sectors, [r_w] * w_sectors
    hist_p, hist_r = [last_dir, since_lv], [3, 3]
    out = OrderedDict()
    out["S1"] = mixed_radix(dparts, rd)
    out["S2"] = mixed_radix(dparts + [feats["prev"]], rd + [4])
    out["S3"] = mixed_radix(dparts + [code], rd + [3])
    out["S4"] = mixed_radix(dparts + [code] + hist_p, rd + [3] + hist_r)
    out["S5"] = mixed_radix(wparts + [code] + hist_p, rw + [3] + hist_r)
    out["S6"] = mixed_radix(wparts + [code] + hist_p + [y_lv, x_lv], rw + [3] + hist_r + [3, 3])
    return out


# ---------------------------------------------------------------- 阶段入口
def _load(out_dir):
    z = np.load(os.path.join(out_dir, "features.npz"))
    return {k: z[k] for k in z.files}


def _split_ids(feats):
    sp = np.asarray(feats["ep_split"])[np.asarray(feats["ep_idx"])]
    return sp == af.SPLITS.index("train"), sp == af.SPLITS.index("val"), sp == af.SPLITS.index("test")


def _subsets_from_env(feats):
    csv = lambda name: [s for s in af.env(name).split(",") if s.strip()]  # noqa: E731
    return subset_masks(feats, csv("AUDIT_TEACHER_SOURCES"), csv("AUDIT_STUDENT_SOURCES"), csv("AUDIT_STUDENT_PURE_SOURCES"))


def _cluster_index(feats, frame_mask):
    """几何簇 = geometry_gid（同一几何的集合并作一簇；没有 gid 的集自成一簇）。返回 (簇名列表, 每帧簇下标)，只对 frame_mask 内的帧有效。"""
    ep = np.asarray(feats["ep_idx"])
    key = np.array([g if g else "ep:" + i for g, i in zip(feats["ep_gid"], feats["ep_id"])])
    names = sorted(set(key[np.unique(ep[frame_mask])]))
    pos = {k: i for i, k in enumerate(names)}
    cl = np.array([pos.get(k, -1) for k in key])[ep]
    return names, cl


def data_summary(feats, masks):
    train, val, test = _split_ids(feats)
    y = np.asarray(feats["label"])
    out = {}
    for sp_name, sp in (("train", train), ("val", val), ("test", test)):
        for sub in SUBSET_ORDER:
            m = sp & masks[sub]
            eps = np.unique(np.asarray(feats["ep_idx"])[m])
            gids = {feats["ep_gid"][e] or "ep:" + feats["ep_id"][e] for e in eps}
            out["%s/%s" % (sp_name, sub)] = {"frames": int(m.sum()), "episodes": int(len(eps)), "geometry_clusters": len(gids),
                                            "class_counts": np.bincount(y[m], minlength=4).tolist()}
    return out


def run_fit(out_dir):
    feats = _load(out_dir)
    train, val, test = _split_ids(feats)
    masks = _subsets_from_env(feats)
    y = np.asarray(feats["label"]).astype(np.int64)
    groups = build_groups(feats, train)
    seeds = [int(s) for s in af.env("AUDIT_SEEDS").split()]
    cfg = {"hidden": af.env("AUDIT_HIDDEN", int), "layers": af.env("AUDIT_LAYERS", int), "lr": af.env("AUDIT_LR", float),
           "batch": af.env("AUDIT_BATCH", int), "max_epochs": af.env("AUDIT_MAX_EPOCHS", int),
           "patience": af.env("AUDIT_PATIENCE", int), "threads": af.env("AUDIT_THREADS", int)}
    force = af.env("AUDIT_FORCE", int) == 1
    with open(os.path.join(out_dir, "data_summary.json"), "w", encoding="utf-8") as f:
        json.dump(data_summary(feats, masks), f, ensure_ascii=False, indent=1)
    cl_names, cluster = _cluster_index(feats, test)
    n_cl = len(cl_names)
    test_idx = np.nonzero(test)[0]
    af.log("特征 %d 帧（train %d / val %d / test %d），测试几何簇 %d，拟合参数 %s" % (len(y), train.sum(), val.sum(), test.sum(), n_cl, cfg))
    for name in [c for c in af.env("AUDIT_FIT_CONFIGS").split(",") if c.strip()]:
        spec = CONFIG_DEFS[name]
        stats_path = os.path.join(out_dir, "stats_%s.npz" % name)
        if os.path.exists(stats_path) and not force:
            af.log("配置 %s：%s 已存在，跳过（AUDIT_FORCE=1 重跑）" % (name, stats_path))
            continue
        fit_tr, fit_va = train & masks[spec["fit_subset"]], val & masks[spec["fit_subset"]]
        counts = np.bincount(y[fit_tr], minlength=4)
        weights = class_weights_auto(counts) if spec["weighted"] else np.ones(4)
        af.log("配置 %s：拟合池 %s，训练 %d 帧（类别数 %s）验证 %d 帧，类别权重 %s" % (
            name, spec["fit_subset"], fit_tr.sum(), counts.tolist(), fit_va.sum(), np.round(weights, 4).tolist()))
        gnames, subs = list(spec["groups"]), list(spec["eval"])
        cm = np.zeros((len(gnames), len(seeds), len(subs), n_cl, 4, 4), np.int32)
        lr_ok = np.zeros((len(gnames), len(seeds), len(subs), n_cl), np.int32)
        preds = np.full((len(gnames), len(seeds), len(test_idx)), -1, np.int8)
        meta = []
        for gi, g in enumerate(gnames):
            X = groups[g]
            for si, seed in enumerate(seeds):
                t0 = time.time()
                net, info = fit_mlp(X[fit_tr], y[fit_tr], X[fit_va], y[fit_va], weights, seed, cfg)
                logits = predict_logits(net, X[test_idx])
                pred = logits.argmax(1)
                lr = logits[:, LEFT] - logits[:, RIGHT]
                preds[gi, si] = pred
                for ui, sub in enumerate(subs):
                    sel = masks[sub][test_idx]
                    cm[gi, si, ui] = confusion_by_cluster(y[test_idx], pred, cluster[test_idx], n_cl, sel)
                    lr_ok[gi, si, ui] = lr_ok_by_cluster(y[test_idx], lr, cluster[test_idx], n_cl, sel)
                m_all = None
                if "all" in subs:
                    ai = subs.index("all")
                    m_all = metrics_from_counts(cm[gi, si, ai].sum(0), lr_ok[gi, si, ai].sum())
                info.update(config=name, group=g, seed=seed, in_dim=int(X.shape[1]), wall_s=round(time.time() - t0, 1))
                meta.append(info)
                af.log("  %s %s seed=%d in=%d best_epoch=%d/%d val_loss=%.4f%s %.0f s" % (
                    name, g, seed, X.shape[1], info["best_epoch"], info["epochs_run"], info["best_val_loss"],
                    "" if m_all is None else " 测试全部帧 acc=%.3f f1_frl=%.3f lr_acc=%.3f" % (m_all["acc"], m_all["f1_frl"], m_all["lr_acc"]),
                    info["wall_s"]))
        tmp = stats_path + ".tmp.npz"
        np.savez_compressed(tmp, cm=cm, lr_ok=lr_ok, groups=np.array(gnames), subsets=np.array(subs), seeds=np.array(seeds),
                            clusters=np.array(cl_names), weights=weights, fit_subset=spec["fit_subset"])
        os.replace(tmp, stats_path)
        with open(os.path.join(out_dir, "fitmeta_%s.json" % name), "w", encoding="utf-8") as f:
            json.dump({"cfg": cfg, "weights": weights.tolist(), "fit_counts": counts.tolist(), "fits": meta}, f, ensure_ascii=False, indent=1)
        if spec["save_preds"]:
            np.savez_compressed(os.path.join(out_dir, "preds_%s.npz" % name), pred=preds, test_idx=test_idx, groups=np.array(gnames),
                                seeds=np.array(seeds))
        af.log("配置 %s 完成：%d 次拟合" % (name, len(meta)))


def run_bins(out_dir):
    feats = _load(out_dir)
    train, val, test = _split_ids(feats)
    masks = _subsets_from_env(feats)
    y = np.asarray(feats["label"]).astype(np.int64)
    ids = build_bin_ids(feats, train, af.env("AUDIT_BIN_D_SECTORS", int), [float(v) for v in af.env("AUDIT_BIN_D_EDGES").split(",")],
                        af.env("AUDIT_BIN_W_SECTORS", int), [float(v) for v in af.env("AUDIT_BIN_W_EDGES").split(",")])
    res = {}
    for g, b in ids.items():
        res[g] = {}
        for sub in SUBSET_ORDER:
            m = masks[sub]
            res[g][sub] = {
                "pooled": bin_check(b[train], y[train], b[test & m], y[test & m]),            # 全部训练帧的箱表，去判该子集的测试帧
                "own": bin_check(b[train & m], y[train & m], b[test & m], y[test & m])}       # 只用该子集的训练帧
        af.log("分箱 %s：占用 %d 箱，训练帧/箱 %.1f" % (g, res[g]["all"]["pooled"]["bins_train"], res[g]["all"]["pooled"]["frames_per_bin"]))
    with open(os.path.join(out_dir, "bins.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    out = af.env("AUDIT_OUT")
    if stage == "fit":
        run_fit(out)
    elif stage == "bins":
        run_bins(out)
    else:
        raise SystemExit("用法：audit_fit.py fit|bins（环境变量见 run.sh 配置区）")


if __name__ == "__main__":
    main()
