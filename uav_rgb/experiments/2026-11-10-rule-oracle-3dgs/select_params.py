#!/usr/bin/env python3
"""开发路线上的 r_safe / d_look 选取（任务卡 K1）：读 dev 阶段的输出，按预先写死的规则排序，打印并落 dev_selection.json / .md。

输入 <DEV>（= <OUT>/dev/indoor）：config.json、<臂>/<场景>/r<N>/{done.json, ep_*/metrics.json}（driver 落的）。
候选 = config.json 里 rule_params 同时含 r_safe 与 d_look 的规则臂（hold 0.5 的测速臂、forward 臂不是候选）。
规则（起 dev 之前定死，不看结果改）：
  1. SR(stop)（到点早于首次碰撞的集占比；分母 = 路线数 × K，没跑的 / 失败的集算失败）降序；
  2. 并列取 r_safe 大的（任务卡）；
  3. 仍并列（任务卡没写，起 dev 之前补的）：撞在前（没到点就撞了）的集占比低的；
  4. 仍并列：d_look 大的。
任一候选没跑满（路线的 done.json 缺、有失败集、或集数不够）就拒绝选：拿残缺数据选出的参数不可复现。--allow-partial 只为调试。
用法：python select_params.py --dev <DEV> [--allow-partial]；最后一行是 R_SAFE=… D_LOOK=…，写进 run.sh 配置区后冻结。
"""
import argparse
import glob
import json
import os
import sys


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def rank_key(c):
    return (-c["sr_stop"], -c["r_safe"], c["collide_first_rate"], -c["d_look"])


def collect(dev):
    """返回候选列表（未排序）。每个候选：arm / r_safe / d_look / n_req / n_done / k_stop / sr_stop / collide_first_rate / timeout_rate / complete / per_scene。"""
    cfg = read_json(os.path.join(dev, "config.json"))
    if cfg is None:
        raise SystemExit(f"读不到 {dev}/config.json")
    routes = cfg["routes"]
    n_routes = sum(len(v) for v in routes.values()) if isinstance(routes, dict) else len(routes) * len(cfg["scenes"])
    out = []
    for arm in cfg["arms"]:
        rp = arm.get("rule_params") or {}
        if "r_safe" not in rp or "d_look" not in rp or not arm["base"].startswith("rule-"):
            continue
        rows, complete, per_scene = [], True, {}
        for s in cfg["scenes"]:
            rs = routes.get(s, []) if isinstance(routes, dict) else routes
            for r in rs:
                d = os.path.join(dev, arm["name"], s, f"r{r}")
                done = read_json(os.path.join(d, "done.json"))
                if done is None or done["failures"] or len(done["episodes"]) < cfg["k"]:
                    complete = False
                for p in sorted(glob.glob(os.path.join(d, "ep_*", "metrics.json"))):
                    m = read_json(p)
                    if m is not None:
                        rows.append(m)
                        ps = per_scene.setdefault(s, {"n": 0, "k_stop": 0})
                        ps["n"] += 1
                        ps["k_stop"] += int(m["success_stop"])
        n_req = cfg["k"] * n_routes
        out.append({"arm": arm["name"], "r_safe": rp["r_safe"], "d_look": rp["d_look"], "n_req": n_req, "n_done": len(rows),
                    "k_stop": sum(int(m["success_stop"]) for m in rows),
                    "sr_stop": sum(int(m["success_stop"]) for m in rows) / n_req if n_req else 0.0,
                    "collide_first_rate": sum(int(m["collide_first"]) for m in rows) / n_req if n_req else 0.0,
                    "timeout_rate": sum(int(m["timeout"]) for m in rows) / n_req if n_req else 0.0,
                    "complete": complete and len(rows) == n_req, "per_scene": per_scene})
    return out


def table(ranked, chosen):
    lines = ["| 名次 | r_safe | d_look | 到点 SR(stop) | 撞在前 | 超时 | 完成集数 / 应跑 | 选中 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, c in enumerate(ranked, 1):
        lines.append(f"| {i} | {c['r_safe']:.2f} | {c['d_look']:.1f} | {c['k_stop']}/{c['n_req']} = {c['sr_stop']:.2f} "
                     f"| {c['collide_first_rate']:.2f} | {c['timeout_rate']:.2f} | {c['n_done']}/{c['n_req']} | {'是' if c is chosen else ''} |")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev", required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args(argv)
    cands = collect(a.dev)
    if not cands:
        print("config.json 里没有候选臂（rule_params 含 r_safe 与 d_look 的规则臂）", file=sys.stderr)
        return 2
    bad = [c["arm"] for c in cands if not c["complete"]]
    if bad and not a.allow_partial:
        print(f"拒绝选参数：这些候选没跑满 {bad}（done.json 缺 / 有失败集 / 集数不够）；跑完再来，或 --allow-partial（只为调试）", file=sys.stderr)
        return 3
    ranked = sorted(cands, key=rank_key)
    chosen = ranked[0]
    top = [c for c in ranked if c["sr_stop"] == chosen["sr_stop"]]
    lines = ["# dev 选参结果（只用开发路线）", "",
             "规则：SR(stop) 降序；并列取 r_safe 大的；再并列取撞在前率低的；再并列取 d_look 大的。", ""] + table(ranked, chosen) + [
            "", f"选定 `r_safe = {chosen['r_safe']:.2f}`、`d_look = {chosen['d_look']:.1f}`（臂 `{chosen['arm']}`）；"
                f"SR(stop) 并列第一的有 {len(top)} 个。"]
    out = {"rule": "SR(stop) 降序 -> r_safe 降序 -> collide_first 率升序 -> d_look 降序", "partial_allowed": bool(a.allow_partial and bad),
           "ranked": ranked, "chosen": {k: chosen[k] for k in ("arm", "r_safe", "d_look", "sr_stop")}, "n_tied_at_top": len(top)}
    with open(os.path.join(a.dev, "dev_selection.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.dev, "dev_selection.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"R_SAFE={chosen['r_safe']:.2f} D_LOOK={chosen['d_look']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
