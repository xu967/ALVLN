#!/usr/bin/env python3
"""零样本基线的离线核对（不起 Gazebo）：转向符号、碰撞概率分布、动作占比、DroNet 前向独立复核。

解释器：$D/teacher-venv/bin/python（torch + numpy + PIL；只读使用）。步骤（每步一条命令，结果落 OUT）：
    python offline_check.py meta                 # 枚举 train 分区的帧，算「最近可见柱」的距离与方位角，分组 L/R，写 meta.npz
    python offline_check.py infer pulp_v3        # 逐帧跑基线（原图 + 左右翻转），写 inf_<基线>_<取法>.npz；三个基线可并行起
    python offline_check.py analyze              # 镜像检验 / 绝对符号 / 与教师一致率 / p_coll 分布 / 动作占比，写 offline_check.{json,md}
    python offline_check.py dronet-verify        # DroNet：逐层 NumPy 解释器（按 model_struct.json 通用解释）对 wrapper 的 torch 前向
只用 train 分区：评测用的 20 个 seed 在 test 分区（几何按 gid 哈希分区，不重叠），这里不读 test / val。
"""
import json
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "sim", "tools", "baselines"))

# ---------------------------------------------------------------- 配置（改这里；环境变量 OFFLINE_* 可覆盖 EP_STRIDE / OUT）
D = "/root/rivermind-data/uav_rgb"
TRAIN_DIR = f"{D}/e1-pipeline-20261107-0417/collection/train"   # 教师（特权 A*）采集的 train 分区：255 集 × ≤40 步，带 scene.json 柱子真值与位姿
OUT = os.environ.get("OFFLINE_OUT", f"{D}/tmp/phase12-20261110/baselines-run/offline")
EP_STRIDE = int(os.environ.get("OFFLINE_EP_STRIDE", "1"))        # 每 N 集取 1 集；1 = 全取。推理耗时与它成反比
FOV_DEG = 60.0                                                   # 学生相机水平视场（sim/tools/teacher.py STUDENT_HFOV_DEG）
CROP_DEG = 19.0                                                  # native 取法只保留中心 200 像素宽：atan(100/277.1) = 19.8°，取 19 留余量
NEAR_M = 3.0                                                     # 只看 3 m 内的最近可见柱：再远的柱在 200x200 灰度里不足 40 像素宽
SIDE_DEG = 10.0                                                  # 「明显在左/右半」：柱中心方位角绝对值不小于 10°
CLOSE_M = 1.2                                                    # 碰撞头诊断的正例：视锥中心 ±15° 内 1.2 m 以内有柱（教师 emergency 0.8 m、转向 1.5 m 之间）
N_BOOT = 2000                                                    # 符号判定的分集自助法次数；同一集的帧相关，按集重采样
SEED = 0
DRONET_VERIFY_FRAMES = 60                                        # 逐层复核用多少帧
YAW_SCALE = {"pulp_v3": 90.0, "pulp_tiny": 90.0, "dronet": 20.0}  # 忠实映射的 °/s：与 launch_*.sh 一致，出处见 sim/tools/baselines/README.md
P_CRIT = 0.7
YAW_STEP_DEG = 25.0
# ----------------------------------------------------------------------------------------------------------------


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def build_meta():
    """每帧：集号、步、帧路径、教师动作、最近可见柱的距离/方位角、分组。方位角 > 0 = 柱在机头左侧（逆时针为正）。"""
    eps = sorted(e for e in os.listdir(TRAIN_DIR) if os.path.isdir(f"{TRAIN_DIR}/{e}"))[::EP_STRIDE]
    rows = []
    for ei, ep in enumerate(eps):
        d = f"{TRAIN_DIR}/{ep}"
        pillars = json.load(open(f"{d}/scene.json"))["pillars"]
        for line in open(f"{d}/actions.jsonl"):
            a = json.loads(line)
            if not a.get("rgb_ok") or not a.get("pose_before") or a.get("yaw_before") is None:
                continue
            x, y, yaw = a["pose_before"]["x"], a["pose_before"]["y"], a["yaw_before"]
            best = None                                    # 最近的「可见」柱：外接圆有一部分落在 ±FOV/2 内（与 teacher.visible_pillar_side 同口径）
            for p in pillars:
                dx, dy = p["x"] - x, p["y"] - y
                dist = math.hypot(dx, dy)
                outer = math.hypot(p["size"][0] / 2, p["size"][1] / 2)
                half = 90.0 if dist <= outer else math.degrees(math.asin(outer / dist))
                b = wrap(math.degrees(math.atan2(dy, dx)) - yaw)
                if abs(b) - half > FOV_DEG / 2:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, b)
            dist, b = best if best else (np.nan, np.nan)
            in_frame = best is not None and abs(b) <= FOV_DEG / 2 and dist <= NEAR_M
            side = (0 if not in_frame or abs(b) < SIDE_DEG else (1 if b > 0 else -1))   # +1 = 柱在左半，-1 = 右半
            rows.append((ei, int(ep), int(a["step"]), f"{d}/{a['frame']}", int(a["action_expert"]), dist, b, side,
                         1 if (in_frame and abs(b) <= CROP_DEG and abs(b) >= SIDE_DEG) else 0))
    arr = lambda i, dt: np.array([r[i] for r in rows], dtype=dt)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/meta.npz", ep_idx=arr(0, np.int32), ep=arr(1, np.int32), step=arr(2, np.int32), path=arr(3, object),
             teacher=arr(4, np.int8), dist=arr(5, np.float64), bearing=arr(6, np.float64), side=arr(7, np.int8),
             in_crop=arr(8, np.int8))
    m = np.load(f"{OUT}/meta.npz", allow_pickle=True)
    n_l, n_r = int((m["side"] == 1).sum()), int((m["side"] == -1).sum())
    n_lc = int(((m["side"] == 1) & (m["in_crop"] == 1)).sum())
    n_rc = int(((m["side"] == -1) & (m["in_crop"] == 1)).sum())
    print(f"episodes={len(eps)} (stride {EP_STRIDE}) frames={len(rows)} | 柱在左半 {n_l} 右半 {n_r} | 其中在 native 裁剪视场内 左 {n_lc} 右 {n_rc}")
    print(f"teacher 动作占比 SAFE/FWD/L/R = {[round(float((m['teacher'] == i).mean()), 3) for i in range(4)]}")


def load_frame(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def infer(name, prep):
    os.environ["BASELINE_PREP"] = prep                       # wrapper 在 import 时读这个变量
    import time
    import baseline_infer as bi
    import torch
    torch.set_num_threads(int(os.environ.get("OFFLINE_THREADS", "2")))
    ad = bi.REGISTRY[name]()
    m = np.load(f"{OUT}/meta.npz", allow_pickle=True)
    n = len(m["path"])
    res = {k: np.full(n, np.nan) for k in ("steer", "p", "steer_flip", "p_flip")}
    t0 = time.time()
    for i, p in enumerate(m["path"]):
        rgb = load_frame(p)
        _, raw = ad.step([rgb], 0)
        _, raw_f = ad.step([np.ascontiguousarray(rgb[:, ::-1])], 0)
        res["steer"][i], res["p"][i] = raw["steer"], raw["p_coll"]
        res["steer_flip"][i], res["p_flip"][i] = raw_f["steer"], raw_f["p_coll"]
        if i % 500 == 0:
            print(f"[{name}/{prep}] {i}/{n} {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{OUT}/inf_{name}_{prep}.npz", **res)
    print(f"[{name}/{prep}] 完成 {n} 帧 {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 分析
def cluster_boot_delta(steer, side, ep_idx, mask_l, mask_r, rng):
    """delta = mean(steer | 右半) − mean(steer | 左半) 及按集自助法的 95% 区间。"""
    eps = np.unique(ep_idx)
    index = {e: i for i, e in enumerate(eps)}
    pos = np.array([index[e] for e in ep_idx])
    nl, nr = np.zeros(len(eps)), np.zeros(len(eps))
    sl, sr = np.zeros(len(eps)), np.zeros(len(eps))
    np.add.at(nl, pos[mask_l], 1)
    np.add.at(sl, pos[mask_l], steer[mask_l])
    np.add.at(nr, pos[mask_r], 1)
    np.add.at(sr, pos[mask_r], steer[mask_r])
    point = sr.sum() / nr.sum() - sl.sum() / nl.sum()
    deltas = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(eps), len(eps))
        if nl[pick].sum() == 0 or nr[pick].sum() == 0:
            continue
        deltas.append(sr[pick].sum() / nr[pick].sum() - sl[pick].sum() / nl[pick].sum())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def auroc(score, pos):
    """秩和法 AUROC：pos 为 True 的样本得分高于负例的概率。"""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # 并列取平均秩
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def map_actions(steer, p, scale, sign, p_crit, clamp):
    import baseline_infer as bi
    s = np.clip(steer, -1, 1) if clamp else steer
    return np.array([bi.map_steer_coll(float(a), float(b), scale, sign, p_crit, YAW_STEP_DEG, 1.0) for a, b in zip(s, p)])


def frac(acts):
    return [round(float((acts == i).mean()), 3) for i in range(4)]


def analyze():
    m = np.load(f"{OUT}/meta.npz", allow_pickle=True)
    rng = np.random.default_rng(SEED)
    rep = {"frames": int(len(m["path"])), "episodes": int(len(np.unique(m["ep_idx"]))),
           "teacher_fractions_safe_fwd_l_r": frac(m["teacher"]), "baselines": {}}
    side, in_crop, teacher = m["side"], m["in_crop"], m["teacher"]
    near = (side != 0) | ((m["dist"] <= NEAR_M) & (np.abs(m["bearing"]) < SIDE_DEG))
    close_ahead = (m["dist"] <= CLOSE_M) & (np.abs(m["bearing"]) <= 15.0)
    no_pillar = np.isnan(m["dist"]) | (m["dist"] > NEAR_M)
    for name in ("pulp_v3", "pulp_tiny", "dronet"):
        path = f"{OUT}/inf_{name}_native.npz"
        if not os.path.exists(path):
            rep["baselines"][name] = "未跑 infer"
            continue
        z = np.load(path)
        s, p, sf, pf = z["steer"], z["p"], z["steer_flip"], z["p_flip"]
        r = {}
        # 1) 镜像检验：翻转输入，steer 应变号（奇部 = 图像相关的转向信号，偶部 = 与图像左右无关的偏置）
        odd, even = (s - sf) / 2, (s + sf) / 2
        big = (np.abs(s) > 0.02) & (np.abs(sf) > 0.02)
        r["mirror"] = {"corr_steer_vs_flipped": float(np.corrcoef(s, sf)[0, 1]),
                       "frac_sign_flipped(|steer|>0.02)": float((np.sign(s[big]) == -np.sign(sf[big])).mean()),
                       "n_big": int(big.sum()),
                       "std_odd": float(odd.std()), "std_even": float(even.std()), "mean_even": float(even.mean()),
                       "corr_p_vs_flipped": float(np.corrcoef(p, pf)[0, 1])}
        # 2) 绝对符号：柱在左半 / 右半的帧上 steer 的均值；delta = 右半 − 左半（避障网络应朝远离柱的一侧转）
        sg = {}
        for tag, ml, mr in (("画面内", side == 1, side == -1), ("native 裁剪视场内", (side == 1) & (in_crop == 1), (side == -1) & (in_crop == 1))):
            d_pt, lo, hi = cluster_boot_delta(s, side, m["ep_idx"], ml, mr, rng)
            sg[tag] = {"n_left": int(ml.sum()), "n_right": int(mr.sum()),
                       "mean_steer_pillar_left": float(s[ml].mean()), "mean_steer_pillar_right": float(s[mr].mean()),
                       "delta_right_minus_left": d_pt, "ci95": [lo, hi],
                       "sign": ("+1" if lo > 0 else "-1" if hi < 0 else "undetermined"),
                       "odd_part_delta": cluster_boot_delta(odd, side, m["ep_idx"], ml, mr, rng)[0]}
        r["sign_test"] = sg
        # 3) 与教师转向方向一致率：教师转向的帧上 sign(sign·steer) 与教师方向相同的比例（0.5 = 无信息）
        ty = (teacher == 2) | (teacher == 3)
        tdir = np.where(teacher[ty] == 2, 1, -1)
        ag = {f"sign{sg_:+d}": float((np.sign(sg_ * s[ty]) == tdir).mean()) for sg_ in (1, -1)}
        ag["n_teacher_yaw_frames"] = int(ty.sum())
        r["teacher_agreement"] = ag
        # 4) p_coll 分布与是否带障碍信号（AUROC：近障碍帧 vs 无柱帧）
        r["p_coll"] = {"median": float(np.median(p)), "p10": float(np.percentile(p, 10)), "p90": float(np.percentile(p, 90)),
                       "frac_above_0.7": float((p > P_CRIT).mean()),
                       "median_close_ahead": float(np.median(p[close_ahead])), "median_no_pillar": float(np.median(p[no_pillar])),
                       "auroc_close_vs_no_pillar": auroc(p[close_ahead | no_pillar], close_ahead[close_ahead | no_pillar]),
                       "n_close": int(close_ahead.sum()), "n_no_pillar": int(no_pillar.sum())}
        r["steer_stats"] = {"mean": float(s.mean()), "std": float(s.std()), "p10": float(np.percentile(s, 10)),
                            "p90": float(np.percentile(s, 90))}
        # 5) 动作占比（单帧，不是闭环）：忠实映射，符号取 +1 与 −1 各算一遍；dronet 另算 recal（S=90、阈值=中位数）
        sc, clamp = YAW_SCALE[name], name == "dronet"
        r["actions_faithful"] = {f"sign{sg_:+d}": frac(map_actions(s, p, sc, sg_, P_CRIT, clamp)) for sg_ in (1, -1)}
        if name == "dronet":
            med = float(np.median(p))
            r["actions_recal"] = {"p_crit": med, **{f"sign{sg_:+d}": frac(map_actions(s, p, 90.0, sg_, med, clamp)) for sg_ in (1, -1)}}
        # 与教师转向帧的动作对账（忠实映射、符号 +1）：教师转向时基线给出同向转 / 反向转 / 不转
        a1 = map_actions(s, p, sc, 1, P_CRIT, clamp)
        r["on_teacher_yaw_frames_sign+1"] = {"same_dir": float((a1[ty] == teacher[ty]).mean()),
                                              "opposite_dir": float(((a1[ty] == 2) | (a1[ty] == 3)).mean() - (a1[ty] == teacher[ty]).mean()),
                                              "no_yaw": float(((a1[ty] == 0) | (a1[ty] == 1)).mean())}
        rep["baselines"][name] = r
    json.dump(rep, open(f"{OUT}/offline_check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- DroNet 独立复核
def np_forward_from_struct(w, struct, x):
    """按 Keras model_struct.json 通用解释：只实现该网络出现的层类型；填充按 TF 'same'（多出的一格补在右/下）。x: [H,W,1]。"""
    from numpy.lib.stride_tricks import sliding_window_view
    cfg = struct["config"]

    def same_pad(t, k, s):
        def tot(n):
            return max((-(-n // s) - 1) * s + k - n, 0)
        ph, pw = tot(t.shape[0]), tot(t.shape[1])
        return np.pad(t, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))

    acts = {}
    out = {}
    for layer in cfg["layers"]:
        name, cls, c = layer["name"], layer["class_name"], layer["config"]
        src = [n[0] for n in layer["inbound_nodes"][0]] if layer["inbound_nodes"] else []
        if cls == "InputLayer":
            acts[name] = x.astype(np.float64)
        elif cls == "Conv2D":
            t = acts[src[0]]
            kh, kw = c["kernel_size"]
            s = c["strides"][0]
            assert c["padding"] == "same" and c["kernel_size"][0] == c["kernel_size"][1] and c["strides"][0] == c["strides"][1]
            tp = same_pad(t, kh, s)
            win = sliding_window_view(tp, (kh, kw), axis=(0, 1))[::s, ::s]          # [H',W',C,kh,kw]
            ker = w[f"{name}/kernel"].astype(np.float64)                              # [kh,kw,Cin,Cout]
            acts[name] = np.einsum("hwcij,ijco->hwo", win, ker) + w[f"{name}/bias"].astype(np.float64)
        elif cls == "MaxPooling2D":
            t = acts[src[0]]
            k, s = c["pool_size"][0], c["strides"][0]
            assert c["padding"] == "valid"
            acts[name] = sliding_window_view(t, (k, k), axis=(0, 1))[::s, ::s].max(axis=(-1, -2))
        elif cls == "BatchNormalization":
            t = acts[src[0]]
            g, b, mu, var = (w[f"{name}/{k}"].astype(np.float64) for k in ("gamma", "beta", "moving_mean", "moving_variance"))
            acts[name] = (t - mu) / np.sqrt(var + c["epsilon"]) * g + b
        elif cls == "Activation":
            t = acts[src[0]]
            acts[name] = np.maximum(t, 0) if c["activation"] == "relu" else 1 / (1 + np.exp(-t))
        elif cls == "Add":
            acts[name] = acts[src[0]] + acts[src[1]]
        elif cls == "Flatten":
            acts[name] = acts[src[0]].reshape(-1)                                     # HWC 展平 = Keras channels_last
        elif cls == "Dropout":
            acts[name] = acts[src[0]]                                                 # 推理时恒等
        elif cls == "Dense":
            assert c["activation"] == "linear"
            acts[name] = acts[src[0]] @ w[f"{name}/kernel"].astype(np.float64) + w[f"{name}/bias"].astype(np.float64)
        else:
            raise NotImplementedError(cls)
    out.update(acts)
    return out


def dronet_verify():
    os.environ["BASELINE_PREP"] = "native"
    import baseline_infer as bi
    import torch
    base = bi.BASE
    z = np.load(f"{base}/dronet/dronet_best_weights.npz")
    # 独立读键：不用 wrapper 的键拼接逻辑
    w = {}
    for k in z.files:
        layer, _, tail = k.split("|")
        w[f"{layer}/{tail.replace(':0', '')}"] = z[k]
    struct = json.load(open(f"{base}/dronet/model/dronet/model_struct.json"))
    ad = bi.DroNet()
    m = np.load(f"{OUT}/meta.npz", allow_pickle=True)
    idx = np.linspace(0, len(m["path"]) - 1, DRONET_VERIFY_FRAMES).astype(int)
    pairs = [("conv2d_1", "conv2d_1"), ("add_1", "conv2d_4"), ("add_2", "conv2d_7"), ("add_3", "conv2d_10")]   # (Keras 层名, wrapper.forward_layers 的键)
    worst = {name: 0.0 for name, _ in pairs}
    worst.update({"steer": 0.0, "coll_logit": 0.0, "p_coll": 0.0})
    scale = {name: 0.0 for name, _ in pairs}
    scale.update({"steer": 0.0, "coll_logit": 0.0})
    for i in idx:
        gray = bi._gray(load_frame(m["path"][i]), "native", bottom_aligned=True)           # [200,200]
        ref = np_forward_from_struct(w, struct, gray[:, :, None].astype(np.float64))
        got = ad.forward_layers(torch.from_numpy(gray)[None, None])
        for np_name, torch_name in pairs:
            g = got[torch_name][0].permute(1, 2, 0).numpy().astype(np.float64)               # torch NCHW → HWC
            worst[np_name] = max(worst[np_name], float(np.abs(g - ref[np_name]).max()))
            scale[np_name] = max(scale[np_name], float(np.abs(ref[np_name]).max()))
        for tn, nn_ in (("steer", "dense_1"), ("coll_logit", "dense_2")):
            worst[tn] = max(worst[tn], abs(got[tn] - float(ref[nn_][0])))
            scale[tn] = max(scale[tn], abs(float(ref[nn_][0])))
        p_t = float(torch.sigmoid(torch.tensor(got["coll_logit"])))
        worst["p_coll"] = max(worst["p_coll"], abs(p_t - float(ref["activation_8"][0])))
    res = {"frames": int(len(idx)), "max_abs_diff": worst, "max_abs_value": scale,
           "覆盖": "conv2d_1 全部输出、三个残差块的和 add_1/2/3、steer、碰撞 logit、碰撞概率；NumPy 解释器按 model_struct.json 的层图与 TF same 填充独立实现，权重键独立解析",
           "未覆盖": "与 Keras/TF 原模型的对拍（本机无 TF 1.x）；npz 与 h5 原文件的逐键比对见 h5_vs_npz.txt（venv-h5 执行）"}
    json.dump(res, open(f"{OUT}/dronet_verify.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "meta":
        build_meta()
    elif cmd == "infer":
        infer(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "native")
    elif cmd == "analyze":
        analyze()
    elif cmd == "dronet-verify":
        dronet_verify()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
