#!/usr/bin/env python3
"""训练配置：只允许已注册的模型 profile，避免 YAML 变成任意代码入口。"""
import hashlib
import json
import math
import os
import re
import copy


class ConfigError(ValueError):
    pass


PROFILES = {
    "nano_proto_v6": {"backend": "cpp", "model": "proto_cnn", "export": True},
    "base_torch_v1": {"backend": "torch", "model": "base_cnn", "export": False},
}

DEFAULTS = {
    "schema_version": 1,
    "model": {"profile": "nano_proto_v6", "backend": "cpp", "input": {"height": 120, "width": 160, "channels": 3},
               "language": {"enabled": False, "text_dim": 0, "max_tokens": 9},
               "history": {"frames": 1, "prev_action": False}},
    "train": {"epochs": 30, "batch": 128, "lr": 0.001, "beta1": 0.9, "beta2": 0.999,
              "eps": 1e-8, "weight_decay": 0.0, "max_grad_norm": 5.0,
              "drop_collision": True, "distill_alpha": 0.5, "device": "cuda", "seed": 0},
    "eval": {"split": "all", "stop_mode": "explicit"},
}

_KEYS = {
    "": {"schema_version", "model", "train", "eval"},
    "model": {"profile", "backend", "input", "language", "history"},
    "model.input": {"height", "width", "channels"},
    "model.language": {"enabled", "text_dim", "max_tokens"},
    "model.history": {"frames", "prev_action"},
    "train": {"data", "out", "epochs", "batch", "lr", "beta1", "beta2", "eps", "weight_decay",
              "max_grad_norm", "drop_collision", "distill_alpha", "seed", "device"},
    "eval": {"split", "stop_mode"},
}


def _scalar(value):
    value = value.strip()
    if not value:
        return {}
    if value in ("{}", "[]"):
        return json.loads(value)
    if (value[0:1] in ("'", '"') and value[-1:] == value[0]):
        return value[1:-1]
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ConfigError(f"配置内联值不是 JSON：{value}") from None
    try:
        return float(value) if any(c in value for c in ".eE") else int(value)
    except ValueError:
        return value


def _simple_yaml(text):
    """支持本项目配置需要的 YAML 子集；复杂 YAML 请安装 PyYAML。"""
    lines = []
    for raw in text.splitlines():
        raw = raw.split(" #", 1)[0].rstrip()
        if raw.strip() and not raw.lstrip().startswith("#"):
            lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    root = {}
    stack = [(-1, root)]
    for i, (indent, line) in enumerate(lines):
        if ":" not in line or line.startswith("-"):
            raise ConfigError(f"仅支持映射式 YAML，第 {i + 1} 行无法解析")
        key, raw = line.split(":", 1)
        key = key.strip()
        if not key or re.search(r"[^A-Za-z0-9_.-]", key):
            raise ConfigError(f"配置键非法：{key!r}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict) or key in parent:
            raise ConfigError(f"配置缩进或重复键错误：{key}")
        value = _scalar(raw)
        parent[key] = value
        if raw.strip() == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
    return root


def load(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # optional; teacher-venv intentionally does not require it
        except ImportError:
            value = _simple_yaml(text)
        else:
            value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ConfigError("配置根必须是对象/映射")
    return value


def _merge(base, extra):
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _check_keys(value, path=""):
    if not isinstance(value, dict):
        return
    allowed = _KEYS.get(path)
    if allowed is None:
        raise ConfigError(f"不支持的配置段：{path}")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{path or '根'} 含未知字段：{', '.join(sorted(unknown))}")
    for key, child in value.items():
        if isinstance(child, dict):
            _check_keys(child, f"{path}.{key}" if path else key)


def resolve(raw=None, *, cli=None):
    raw = raw or {}
    for section in ("model", "train", "eval"):
        if section in raw and not isinstance(raw[section], dict):
            raise ConfigError(f"配置段 {section} 必须是映射")
    _check_keys(raw)
    # A resolved config is mutable while CLI overrides and profile defaults are
    # applied.  Copy the defaults so one run cannot silently change the next.
    cfg = _merge(copy.deepcopy(DEFAULTS), raw)
    cli = cli or {}
    for key, value in cli.items():
        if value is None:
            continue
        if key in ("seed", "data", "out", "epochs", "batch", "device"):
            cfg["train"][key] = value
    model = cfg["model"]
    profile = model.get("profile")
    if profile not in PROFILES:
        raise ConfigError(f"未知 model.profile：{profile!r}；可选：{', '.join(PROFILES)}")
    spec = PROFILES[profile]
    configured_backend = raw.get("model", {}).get("backend")
    if configured_backend is not None and configured_backend != spec["backend"]:
        raise ConfigError(f"profile {profile} 的 backend 必须是 {spec['backend']}")
    model["backend"] = spec["backend"]
    if profile == "nano_proto_v6":
        if model["input"].get("channels", 3) != 3 or model["history"].get("frames", 1) != 1:
            raise ConfigError("nano_proto_v6 训练入口当前只支持单帧 RGB")
        if model["language"].get("enabled") or model["language"].get("text_dim", 0):
            raise ConfigError("nano_proto_v6 当前训练器未接语言标签，language.enabled/text_dim 必须关闭")
        if model["history"].get("prev_action"):
            raise ConfigError("nano_proto_v6 当前训练器未接 prev-action 标签")
    elif profile == "base_torch_v1" and model["input"].get("channels", 3) != 3:
        raise ConfigError("base_torch_v1 当前只支持 RGB 输入")
    if model["language"].get("enabled") or model["language"].get("text_dim", 0):
        raise ConfigError(f"{profile} 当前模型没有语言输入实现，language.enabled/text_dim 必须关闭")
    if model["history"].get("frames", 1) != 1 or model["history"].get("prev_action"):
        raise ConfigError(f"{profile} 当前模型没有历史输入实现，history 必须是单帧且不含 prev_action")
    for key in ("height", "width", "channels"):
        value = model["input"].get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"model.input.{key} 必须是正整数")
    if int(cfg["train"]["epochs"]) <= 0 or int(cfg["train"]["batch"]) <= 0:
        raise ConfigError("train.epochs 和 train.batch 必须为正数")
    positive = ("lr", "eps", "max_grad_norm")
    nonnegative = ("weight_decay",)
    for key in positive:
        value = float(cfg["train"][key])
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"train.{key} 必须是有限正数")
    for key in nonnegative:
        value = float(cfg["train"][key])
        if not math.isfinite(value) or value < 0:
            raise ConfigError(f"train.{key} 必须是有限非负数")
    for key in ("beta1", "beta2"):
        value = float(cfg["train"][key])
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ConfigError(f"train.{key} 必须在 [0, 1) 内")
    alpha = float(cfg["train"]["distill_alpha"])
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ConfigError("train.distill_alpha 必须在 [0, 1] 内")
    return cfg


def canonical_json(cfg):
    return json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(cfg):
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()


def write_resolved(run_dir, cfg):
    value = {"config_hash": digest(cfg), "config": cfg}
    with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
    return value["config_hash"]
