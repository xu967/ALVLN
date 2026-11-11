#!/usr/bin/env python3
"""开发路线上的变体自动选取（任务卡 K3 的选取规则，机械执行，不做人工判断）：

  1. 候选 = 变体表里的 v2 变体（v1 是 K1 冻结的对照臂，不参选）；撞在前（没到点就撞）的集数多于 MAX_COLLIDE_FIRST（3，绝对集数，按任务卡「3/40」的字面读成 3 集）的变体出局；
     开发路线实际不足 40 条时（场景 sage3d_841150 的合格路线只剩 2 条）「3/40」还可以读成比率 7.5%：对照读法（上限取 floor(3·n/40) 集）的选取结果只写进
     dev_selection.json 的 decision_rate_reading 与 dev_selection.md 的末行，不参与选取；
  2. 其余里 SR(stop) 最高者为最优；与最优相差不超过 TIE_SLACK（1 集）的算并列；并列取变体表里靠前的；
  3. 关口：选中的变体必须比 v1 在同一批路线上多到点至少 GATE_MIN_GAIN（3）集，否则退出码 20，DONE 写「开发路线上 v2 没有比 v1 好」；
  4. 过关才写 <out-dir>/selected.json（正式阶段只读它）；dev_selection.json / dev_selection.md 总是写。
depth 臂用的变体：选中的是 DEPTH_MAP 里被映射的变体（v2e / v2f）就换成映射目标（v2a / v2d，同结构的 0.25 版本），否则同一个。
dev 不完整（有路线没写 done.json、有失败集、集数不够）退出码 2，不选。
用法：python select_variant.py --dev <OUT_ROOT>/dev --out-dir <OUT_ROOT> --variants "<变体表文本>" --depth-map "v2e=v2a v2f=v2d" \
          [--max-collide-first 3] [--tie-slack 1] [--gate-min-gain 3] [--no-gate]
--no-gate 只给冒烟用（路线太少，关口必然不过；照样写 selected.json 让链条往下走，selected.json 里记 gate_bypassed）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import variants as vr  # noqa: E402


class Incomplete(Exception):
    pass


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_arm(dev_dir, name, scenes, routes, k, origin):
    """一个臂的全部集：[{scene, route, k, success_stop, collide_first, collision, timeout, origin}]；不完整抛 Incomplete。"""
    eps = []
    for s in scenes:
        rs = routes[s] if isinstance(routes, dict) else routes
        for r in rs:
            d = os.path.join(dev_dir, name, s, f"r{r}")
            try:
                done = read_json(os.path.join(d, "done.json"))
            except (OSError, ValueError):
                raise Incomplete(f"{name}/{s}/r{r} 没有 done.json") from None
            if done["failures"] or len(done["episodes"]) < k:
                raise Incomplete(f"{name}/{s}/r{r} 有失败集或集数不足（{len(done['episodes'])}/{k}，失败 {len(done['failures'])}）")
            for kk in range(k):
                try:
                    m = read_json(os.path.join(d, "ep_%05d" % kk, "metrics.json"))
                except (OSError, ValueError):
                    raise Incomplete(f"{name}/{s}/r{r}/ep{kk} 没有 metrics.json") from None
                eps.append({"scene": s, "route": r, "k": kk, "success_stop": bool(m["success_stop"]), "collide_first": bool(m["collide_first"]),
                            "collision": bool(m["collision"]), "timeout": bool(m["timeout"]), "origin": origin.get((s, r), "?")})
    return eps


def stats(eps):
    def cnt(key, origin=None):
        return sum(e[key] for e in eps if origin is None or e["origin"] == origin)
    return {"n": len(eps), "k_stop": cnt("success_stop"), "k_collide_first": cnt("collide_first"), "k_collision": cnt("collision"),
            "k_timeout": cnt("timeout"),
            "by_origin": {o: {"n": sum(e["origin"] == o for e in eps), "k_stop": cnt("success_stop", o)} for o in sorted({e["origin"] for e in eps})}}


def choose(table, variant_names, *, max_collide_first, tie_slack, gate_min_gain):
    """table = {臂名: stats}；返回决策 dict（纯函数，单测直接调）。"""
    eligible = [n for n in variant_names if table[n]["k_collide_first"] <= max_collide_first]
    out = {"eligible": eligible, "excluded": [n for n in variant_names if n not in eligible], "selected": None, "tied": [], "best": None,
           "gain_over_v1": None, "gate_passed": False}
    if not eligible:
        return out
    best = max(table[n]["k_stop"] for n in eligible)
    out["best"] = best
    out["tied"] = [n for n in eligible if table[n]["k_stop"] >= best - tie_slack]
    out["selected"] = out["tied"][0]
    out["gain_over_v1"] = table[out["selected"]]["k_stop"] - table["v1"]["k_stop"]
    out["gate_passed"] = out["gain_over_v1"] >= gate_min_gain
    return out


def render_md(cfg, variants, table, decision, rule, params, origin_counts):
    lines = ["# 开发路线上的变体选取", "",
             f"路线 {sum(len(v) for v in cfg['routes'].values())} 条（K1 开发 {origin_counts.get('k1', 0)} + 新选 {origin_counts.get('new', 0)}）× {cfg['k']} 集，"
             f"hold {cfg['hold']}，seed {cfg['seed0']} + 100·路线；到点成功 = SR(stop)。", "",
             f"选取规则（机械执行）：{rule}", "",
             "| 臂 | 级别（r_safe:d_look，* = 只许静止起步） | 次序 | 到点 | 撞在前 | 整集有撞 | 超时 | K1 开发路线到点 | 新路线到点 | 状态 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    by_name = {v["name"]: v for v in variants}
    for arm in ["v1"] + [v["name"] for v in variants]:
        t, bo = table[arm], table[arm]["by_origin"]
        if arm == "v1":
            lv, order, state = "r_safe 0.25、d_look 1.2（K1 冻结，规则 v1）", "—", "对照"
        else:
            lv = ", ".join(f"{r}:{d}{'*' if st else ''}" for r, d, st in by_name[arm]["levels"])
            order = by_name[arm]["order"]
            if arm in decision["excluded"]:
                state = f"出局（撞在前 {t['k_collide_first']} > {params['max_collide_first']}）"
            elif arm == decision["selected"]:
                state = "**选中**"
            elif arm in decision["tied"]:
                state = "并列（表里靠后）"
            else:
                state = ""
        lines.append(f"| {arm} | {lv} | {order} | {t['k_stop']}/{t['n']} | {t['k_collide_first']} | {t['k_collision']} | {t['k_timeout']} | "
                     f"{bo.get('k1', {}).get('k_stop', '—')}/{bo.get('k1', {}).get('n', '—')} | "
                     f"{bo.get('new', {}).get('k_stop', '—')}/{bo.get('new', {}).get('n', '—')} | {state} |")
    lines.append("")
    sel = decision["selected"]
    if sel is None:
        lines.append("没有候选：全部变体的撞在前集数都超过上限，不进正式阶段。")
    else:
        gain = decision["gain_over_v1"]
        lines.append(f"选中 **{sel}**：到点 {table[sel]['k_stop']}/{table[sel]['n']}，比 v1（{table['v1']['k_stop']}）"
                     f"{'多' if gain >= 0 else '少'} {abs(gain)} 集；关口（至少多 {params['gate_min_gain']} 集）{'通过' if decision['gate_passed'] else '未通过'}。")
        lines.append(f"并列组（与最优 {decision['best']} 相差不超过 {params['tie_slack']} 集）：{decision['tied']}。")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev", required=True, help="dev 目录（<OUT_ROOT>/dev，含 config.json 与各臂结果）")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants", required=True)
    p.add_argument("--depth-map", default="")
    p.add_argument("--max-collide-first", type=int, default=3)
    p.add_argument("--tie-slack", type=int, default=1)
    p.add_argument("--gate-min-gain", type=int, default=3)
    p.add_argument("--no-gate", action="store_true", help="只给冒烟用：关口不过也写 selected.json")
    a = p.parse_args(argv)
    variants = vr.parse_variants(a.variants)
    names = [v["name"] for v in variants]
    dmap = vr.parse_depth_map(a.depth_map, variants)
    cfg = read_json(os.path.join(a.dev, "config.json"))
    routes_sel = read_json(cfg["routes_selected"])
    origin = {(s, r["route"]): r.get("origin", "?") for s, rs in routes_sel["scenes"].items() for r in rs}
    arm_names = [x["name"] for x in cfg["arms"]]
    if arm_names != ["v1"] + names:
        print(f"config.json 的臂 {arm_names} 与变体表 {['v1'] + names} 不一致", file=sys.stderr)
        return 2
    try:
        table = {n: stats(load_arm(a.dev, n, cfg["scenes"], cfg["routes"], cfg["k"], origin)) for n in arm_names}
    except Incomplete as exc:
        print(f"dev 不完整，不选：{exc}", file=sys.stderr)
        return 2
    if len({t["n"] for t in table.values()}) != 1:
        print(f"各臂集数不同 {[(n, t['n']) for n, t in table.items()]}", file=sys.stderr)
        return 2
    decision = choose(table, names, max_collide_first=a.max_collide_first, tie_slack=a.tie_slack, gate_min_gain=a.gate_min_gain)
    n_eps = table["v1"]["n"]
    rate_cap = (a.max_collide_first * n_eps) // 40          # 对照读法：「3/40」读成比率 7.5%，上限取 floor(3·n/40)；只记录，不参与选取
    decision_rate = choose(table, names, max_collide_first=rate_cap, tie_slack=a.tie_slack, gate_min_gain=a.gate_min_gain)
    rule = (f"撞在前多于 {a.max_collide_first} 集的变体出局；其余里 SR(stop) 最高者为最优，相差不超过 {a.tie_slack} 集算并列，并列取表里靠前的；"
            f"选中的比 v1 多到点至少 {a.gate_min_gain} 集才进正式阶段")
    in_dev = {(s, r) for s in cfg["scenes"] for r in cfg["routes"][s]}
    origin_counts = {o: sum(1 for key, oo in origin.items() if oo == o and key in in_dev) for o in set(origin.values())}
    params = {"max_collide_first": a.max_collide_first, "tie_slack": a.tie_slack, "gate_min_gain": a.gate_min_gain}
    md = render_md(cfg, variants, table, decision, rule, params, origin_counts)
    md += (f"\n对照读法（不参与选取）：任务卡的「3/40」读成比率 7.5%，开发路线共 {n_eps} 集时撞在前上限取 {rate_cap} 集，出局 {decision_rate['excluded']}，"
           f"选中 {decision_rate['selected']}，比 v1 {'多' if (decision_rate['gain_over_v1'] or 0) >= 0 else '少'} {abs(decision_rate['gain_over_v1'] or 0)} 集，"
           f"关口{'通过' if decision_rate['gate_passed'] else '未通过'}；与字面读法{'相同' if decision_rate['selected'] == decision['selected'] else '不同'}。\n")
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {"rule": rule, "params": params, "arms": table, "decision": decision, "decision_rate_reading": {"max_collide_first": rate_cap, **decision_rate},
               "variants": variants, "depth_map": dmap, "n_routes": len(in_dev), "k": cfg["k"]}
    with open(os.path.join(a.out_dir, "dev_selection.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out_dir, "dev_selection.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md, flush=True)
    sel = decision["selected"]
    if sel is None:
        return 20
    if not decision["gate_passed"] and not a.no_gate:
        print(f"开发路线上 v2 没有比 v1 好：选中 {sel}，比 v1 多 {decision['gain_over_v1']} 集 < {a.gate_min_gain}", file=sys.stderr)
        return 20
    v, dv = vr.by_name(variants, sel), vr.by_name(variants, dmap[sel])
    selected = {"variant": sel, "levels": v["levels"], "order": v["order"], "depth_variant": dv["name"], "depth_levels": dv["levels"],
                "depth_order": dv["order"], "gain_over_v1": decision["gain_over_v1"], "gate_bypassed": bool(a.no_gate and not decision["gate_passed"]),
                "dev_k_stop": {n: table[n]["k_stop"] for n in arm_names}, "rule": rule}
    path = os.path.join(a.out_dir, "selected.json")
    if os.path.exists(path):
        old = read_json(path)
        if {k: old.get(k) for k in ("variant", "levels", "order", "depth_variant")} != {k: selected[k] for k in ("variant", "levels", "order", "depth_variant")}:
            print(f"已有的 {path} 与本次选取不同（旧 {old.get('variant')}，新 {sel}）；正式阶段已冻结，拒绝覆盖", file=sys.stderr)
            return 2
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
