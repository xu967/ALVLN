#!/usr/bin/env python3
"""任务卡 W1-8 验收 2：解析深度分带（preprocess.py --depth-source analytic）与 Gazebo 深度图分带（file）逐格对拍。

输入：同一集 episode 目录（采集产物，含 scene.json、actions.jsonl、--save-depth 的深度图），以及它分别用
`--depth-source file` 和 `--depth-source analytic` 打出的两个包（默认参数，包络膨胀同为 0.102 m）。
口径与门槛是 run.sh 文件头的预注册，这里照写，不改：
  主比较 = 柱子所在单元的分带一致率（侧车里的最终分带，膨胀之后）。柱子所在单元 = 稠密解析渲染（同一位姿，每格 16×16 个像素中心各一条射线）
          里至少有 1 个柱面像素的格；分母 = 全集这类格逐步汇总的总数。P1：≥ 95%。
  不一致格的归类（每格一类，次序固定，只看稠密渲染里该格 16×16 个像素属于哪根柱 / 地面 / 天空与远处）：
    地面   该格内有某一列，柱面像素的正下方紧邻地面像素（柱脚的轮廓线经过这格；邻接只在格内算）；
    柱边缘 不是「地面」，且格内像素不全属于同一根柱；
    其它   格内像素全是同一根柱的柱面。
  另报（不设门）：全部格的一致率；不一致格的分带差（analytic − file）直方图；不一致格在地平线以上 / 地平线行 / 以下的分布；每步一致率。
「事后诊断」一节在预注册之外：膨胀前的每格最小深度（file 取深度图的格内最小，analytic 取 3×3 射线的格内最小）的相对误差分位数、
膨胀前的分带一致率。它们只用来解释主比较，不参与判定。

  python compare.py --episode EP --file DS_FILE --analytic DS_ANALYTIC --out compare.json
输出：--out 的 JSON、同名 .md（报告），同时把 .md 打到标准输出。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import preprocess as pre  # noqa: E402

DENSE = 16            # 稠密渲染每格每个方向的射线数：320×240 源图、15×20 网格时就是逐像素中心
GATE = 0.95           # P1：柱子所在单元的一致率门槛（任务卡验收 2）
CLASSES = ("柱边缘", "地面", "其它")


def read_depth_bands(ds_dir):
    """<ds_dir>/dataset.aux 的深度分带 (n, dh, dw) uint8。"""
    blob = (Path(ds_dir) / "dataset.aux").read_bytes()
    magic, n, flags, dh, dw, _k, _unk, _res = pre.AUX_HEADER.unpack_from(blob, 0)
    if magic != pre.AUX_MAGIC or not flags & 1:
        sys.exit("%s: 不是带深度分带的 UAVRAUX1 侧车" % ds_dir)
    return np.frombuffer(blob, dtype=np.uint8, count=n * dh * dw, offset=pre.AUX_HEADER_SIZE + 5 * n).reshape(n, dh, dw)


def read_meta(ds_dir):
    raw = (Path(ds_dir) / "dataset.uavr").read_bytes()
    off = int.from_bytes(raw[36:44], "little")
    ln = int.from_bytes(raw[44:48], "little")
    return json.loads(raw[off:off + ln])


def classify(block):
    """block：一格的 16×16 像素归属（≥ 0 柱，−1 地面，−2 天空 / 远处）。"""
    if ((block[:-1, :] >= 0) & (block[1:, :] == -1)).any():
        return "地面"
    if np.unique(block).size > 1:
        return "柱边缘"
    return "其它"


def zone_of(row, horizon_row):
    return "地平线以上" if row < horizon_row else ("地平线行" if row == horizon_row else "地平线以下")


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="采集产物的 episode 目录（含 scene.json、actions.jsonl、深度图）")
    ap.add_argument("--file", required=True, help="--depth-source file 打出的包目录")
    ap.add_argument("--analytic", required=True, help="--depth-source analytic 打出的包目录")
    ap.add_argument("--out", required=True, help="输出 JSON 路径（同名 .md 一并写）")
    args = ap.parse_args()

    bf, ba = read_depth_bands(args.file), read_depth_bands(args.analytic)
    mf, ma = read_meta(args.file), read_meta(args.analytic)
    if bf.shape != ba.shape:
        sys.exit("两个包的深度分带形状不同：%r 对 %r" % (bf.shape, ba.shape))
    if mf["aux_sidecar"].get("depth_source") not in (None, "file") or ma["aux_sidecar"].get("depth_source") != "analytic":
        sys.exit("包的 depth_source 不对：file 包 %r，analytic 包 %r" % (
            mf["aux_sidecar"].get("depth_source"), ma["aux_sidecar"].get("depth_source")))
    if mf["aux_sidecar"]["envelope_radius_m"] != ma["aux_sidecar"]["envelope_radius_m"]:
        sys.exit("两个包的包络膨胀半径不同，不可比")
    if mf["episodes"][0]["episode_id"] != ma["episodes"][0]["episode_id"] or mf["num_samples"] != ma["num_samples"]:
        sys.exit("两个包不是同一集")

    ep = pre.load_episode(args.episode)
    if len(ep["samples"]) != len(bf):
        sys.exit("包里 %d 个样本，episode 读出 %d 个（用了 --drop-collision？）" % (len(bf), len(ep["samples"])))
    mount, pillars = pre.analytic_episode_inputs(ep)
    focal = float(pre.episode_focal_px(ep))
    src_wh = (ep["src_w"], ep["src_h"])
    dh, dw = bf.shape[1:]
    grid = (dh, dw)
    horizon_row = int((src_wh[1] / 2.0) // (src_wh[1] / dh))     # 主点所在的行（15 行时 = 7）

    pillar_cells = agree_pillar = total_cells = agree_all = unknown_in_file = 0
    cls = Counter()
    delta = Counter()
    zone_dis = Counter()
    zone_all = Counter()
    per_step = []
    raw_rel = []
    raw_band_agree = raw_band_total = 0
    have_depth = all(s["depth"] is not None for s in ep["samples"])
    for k, s in enumerate(ep["samples"]):
        pose = pre._analytic_pose(ep, s)
        dense, pid = pre.analytic_ray_depth(pose, pillars, focal, src_wh, grid, mount, DENSE, want_pid=True)
        blocks = pid.reshape(dh, DENSE, dw, DENSE)
        is_pillar = (blocks >= 0).any(axis=(1, 3))
        same = bf[k] == ba[k]
        total_cells += same.size
        agree_all += int(same.sum())
        unknown_in_file += int((bf[k] == pre.AUX_UNKNOWN).sum())
        n_p = int(is_pillar.sum())
        a_p = int((same & is_pillar).sum())
        pillar_cells += n_p
        agree_pillar += a_p
        per_step.append({"step": s["step"], "pillar_cells": n_p, "agree": a_p,
                         "rate": (a_p / n_p) if n_p else None})
        for r, c in zip(*np.nonzero(is_pillar)):
            z = zone_of(int(r), horizon_row)
            zone_all[z] += 1
            if not same[r, c]:
                name = classify(blocks[r, :, c, :])
                cls[name] += 1
                zone_dis[z] += 1
                delta[int(ba[k][r, c]) - int(bf[k][r, c])] += 1
        if have_depth:
            raw_file = pre.cell_min_depth(pre.DEPTH_LOADERS[s["depth_format"]](s["depth"]), grid)
            raw_an = pre.analytic_cell_depth(pose, pillars, focal, src_wh, grid, mount)
            bands_f, bands_a = pre.depth_bands(raw_file), pre.depth_bands(raw_an)
            raw_band_total += int(is_pillar.sum())
            raw_band_agree += int((is_pillar & (bands_f == bands_a)).sum())
            both = is_pillar & np.isfinite(raw_file) & np.isfinite(raw_an) & (raw_file > 0)
            raw_rel.extend(((raw_an[both] - raw_file[both]) / raw_file[both]).tolist())
    dis = pillar_cells - agree_pillar
    rate = agree_pillar / pillar_cells if pillar_cells else float("nan")
    rates = [p["rate"] for p in per_step if p["rate"] is not None]
    result = {
        "episode_dir": str(Path(args.episode).resolve()),
        "steps": len(per_step),
        "grid": [dh, dw],
        "pillars_in_scene": len(pillars),
        "pillar_cells": pillar_cells,
        "pillar_cells_agree": agree_pillar,
        "pillar_cells_rate": rate,
        "gate": GATE,
        "p1_pass": bool(rate >= GATE),
        "disagree_classes": {name: cls[name] for name in CLASSES},
        "all_cells": total_cells,
        "all_cells_agree": agree_all,
        "all_cells_rate": agree_all / total_cells,
        "file_unknown_cells": unknown_in_file,
        "delta_band_hist_analytic_minus_file": {str(d): n for d, n in sorted(delta.items())},
        "pillar_cells_by_zone": dict(zone_all),
        "disagree_by_zone": dict(zone_dis),
        "per_step_rate": {"min": min(rates) if rates else None, "median": float(np.median(rates)) if rates else None,
                          "max": max(rates) if rates else None},
        "per_step": per_step,
    }
    if have_depth:
        rel = np.asarray(raw_rel)
        result["post_hoc"] = {
            "note": "预注册之外：膨胀前的诊断，不参与判定",
            "raw_band_agree_pillar_cells": raw_band_agree,
            "raw_band_total_pillar_cells": raw_band_total,
            "raw_band_rate": raw_band_agree / raw_band_total if raw_band_total else None,
            "raw_depth_rel_err_cells": int(rel.size),
            "raw_depth_rel_err_pct_analytic_minus_file_over_file": (
                {str(q): float(np.percentile(rel, q)) for q in (5, 25, 50, 75, 95)} if rel.size else None),
        }
    out = Path(args.out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    lines = ["# 解析深度对拍（compare.py）", "",
             "- episode：`%s`，%d 步，网格 %dx%d，scene 柱 / 树干 %d 根" % (result["episode_dir"], result["steps"], dh, dw, len(pillars)),
             "- file 包：`%s`；analytic 包：`%s`" % (args.file, args.analytic), "",
             "## 主比较（预注册 P1）", "",
             "| 项 | 值 |", "|---|---|",
             "| 柱子所在单元总数（逐步汇总） | %d |" % pillar_cells,
             "| 分带一致 | %d |" % agree_pillar,
             "| 一致率 | **%.2f%%**（门槛 %.0f%%）：%s |" % (100 * rate, 100 * GATE, "通过" if result["p1_pass"] else "不通过"),
             "| 每步一致率 最小 / 中位 / 最大 | %.1f%% / %.1f%% / %.1f%% |" % tuple(
                 100 * result["per_step_rate"][k] for k in ("min", "median", "max")), "",
             "## 不一致格归类（%d 格；次序 地面 → 柱边缘 → 其它）" % dis, "",
             "| 类 | 格数 | 占不一致 |", "|---|---|---|"]
    for name in CLASSES:
        lines.append("| %s | %d | %.1f%% |" % (name, cls[name], pct(cls[name], dis)))
    lines += ["", "## 另报（不设门）", "",
              "- 全部格一致率：%.2f%%（%d / %d）；file 包里未知（255）的格：%d" % (
                  100 * result["all_cells_rate"], agree_all, total_cells, unknown_in_file),
              "- 不一致的柱子格的分带差（analytic − file）：%s" % (
                  "，".join("%+d：%d" % (d, n) for d, n in sorted(delta.items())) or "无"),
              "- 柱子格按行区分（一致率）：" + "；".join(
                  "%s %d 格，不一致 %d（%.1f%%）" % (z, zone_all[z], zone_dis[z], pct(zone_dis[z], zone_all[z]))
                  for z in ("地平线以上", "地平线行", "地平线以下") if zone_all[z])]
    if have_depth:
        ph = result["post_hoc"]
        q = ph["raw_depth_rel_err_pct_analytic_minus_file_over_file"]
        lines += ["", "## 事后诊断（预注册之外，不参与判定）", "",
                  "- 膨胀前的分带一致率（柱子格）：%.2f%%（%d / %d）" % (
                      100 * ph["raw_band_rate"], raw_band_agree, raw_band_total),
                  "- 膨胀前每格最小深度的相对误差（analytic − file）/ file，柱子格 %d 个，分位数 5/25/50/75/95：%s" % (
                      ph["raw_depth_rel_err_cells"],
                      " / ".join("%+.1f%%" % (100 * q[str(p)]) for p in (5, 25, 50, 75, 95)) if q else "无")]
    text = "\n".join(lines) + "\n"
    out.with_suffix(".md").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
