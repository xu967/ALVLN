#!/usr/bin/env python3
"""变体表的解析（run.sh 配置区是变体表的唯一来源；build_config.py、select_variant.py、summarize.py 共用本文件）。

一行一个变体：  名字|级别列表|次序
  级别列表 = 逗号分隔的 r_safe:d_look，末尾带 * 表示「只许静止起步」，例如 0.25:1.2,0.25:0.8,0.20:0.55*
  次序     = lexi | near（规则 v2 的两种决策次序，见 rule_policy.rule_v2_entry）
解析后每个变体 = {"name", "levels": [[r_safe, d_look, 只许静止起步], ...], "order"}，并按 rule_policy.check_v2_params 校验
（levels 逐级不放宽、次序合法），不合格抛 ValueError，起跑前就报错。
depth 臂的变体映射 DEPTH_MAP = 空格分隔的「选中=用于 depth」，例如 v2e=v2a v2f=v2d；没列出的变体 depth 臂用同一个。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim", "tools"))
import rule_policy as rp  # noqa: E402


def parse_variant(spec):
    try:
        name, levels, order = spec.strip().split("|")
        out = []
        for part in levels.split(","):
            part = part.strip()
            still = part.endswith("*")
            r, d = part.rstrip("*").split(":")
            out.append([float(r), float(d), still])
    except ValueError as exc:
        raise ValueError(f"变体行格式应为 名字|r:d[*],r:d[*]|lexi 或 near，得到 {spec!r}（{exc}）") from None
    v = {"name": name.strip(), "levels": out, "order": order.strip()}
    rp.check_v2_params({**rp.RULE_V2, "levels": v["levels"], "order": v["order"]})
    return v


def parse_variants(text):
    """text = 多行（或列表），空行忽略；名字不许重复，不许叫 v1（v1 是 K1 冻结的对照臂）。"""
    lines = text.splitlines() if isinstance(text, str) else list(text)
    out = [parse_variant(x) for x in lines if x.strip()]
    names = [v["name"] for v in out]
    if len(set(names)) != len(names) or "v1" in names or not out:
        raise ValueError(f"变体名不能重复、不能叫 v1、至少要有一个：{names}")
    return out


def parse_depth_map(text, variants):
    """DEPTH_MAP 文本 -> {变体名: depth 臂用的变体名}；没列出的映射到自己。"""
    names = {v["name"] for v in variants}
    m = {n: n for n in names}
    for item in (text or "").split():
        a, b = item.split("=")
        if a not in names or b not in names:
            raise ValueError(f"DEPTH_MAP 项 {item!r} 引用了不存在的变体（{sorted(names)}）")
        m[a] = b
    return m


def by_name(variants, name):
    for v in variants:
        if v["name"] == name:
            return v
    raise KeyError(name)


def rule_params_of(v):
    """变体 -> 臂的 rule_params（只含 levels 与 order，其余规则参数用 RULE_V2 的值，yaw_step_deg 走全局 rule_params）。"""
    return {"levels": [list(x) for x in v["levels"]], "order": v["order"]}
