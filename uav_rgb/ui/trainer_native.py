#!/usr/bin/env python3
"""把 uavrgb.train.v1 配置翻译成真实 trainer 的 CLI，然后 exec 它。

控制台用「训练命令」框里填本脚本启动：`python3 ui/trainer_native.py --data <dataset.uavr>`，
控制台会自动追加 `--config <run_dir>.monitor/train_config.json`。其它参数
（--resume / --class-weights / --max-grad-norm / --drop-collision）也写在命令框里，
原样追加到 trainer 参数之后；同名参数以命令框里的为准。

用法：
  python3 ui/trainer_native.py --data /tmp/bench_ds.uavr
  UAVRGB_TRAIN_BIN=/path/to/uavrgb_train python3 ui/trainer_native.py --config cfg.json
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAINER = os.path.join(REPO_ROOT, "src", "build", "uavrgb_train")
SCHEMA = "uavrgb.train.v1"


class ConfigError(Exception):
    """配置不合法（对应退出码 2，与 uavrgb_train 的用法错误一致）。"""


def load_config(path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            cfg = json.load(f)
    except (OSError, ValueError, RecursionError) as exc:
        raise ConfigError(f"无法读取配置 {path}: {exc}") from exc
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
        raise ConfigError(f"配置 schema 必须是 {SCHEMA}")
    for key in ("run_dir", "epochs", "batch_size", "lr", "seed"):
        if key not in cfg:
            raise ConfigError(f"配置缺少 {key}")
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="uavrgb.train.v1 -> uavrgb_train CLI")
    p.add_argument("--config", required=True, help="控制台写下的 train_config.json")
    args, extra = p.parse_known_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"trainer_native: {exc}", file=sys.stderr)
        return 2
    # 数据集既可以写在配置里（data 字段），也可以直接写在控制台的「训练命令」里
    data = cfg.get("data") or ""
    if not data and "--data" not in extra:
        print("trainer_native: 缺数据集；请在控制台「训练命令」里写 --data <dataset.uavr>",
              file=sys.stderr)
        return 2
    trainer = os.environ.get("UAVRGB_TRAIN_BIN", DEFAULT_TRAINER)
    if not os.access(trainer, os.X_OK):
        print(f"trainer_native: 找不到可执行的 trainer: {trainer}（先构建 src/）",
              file=sys.stderr)
        return 2
    cmd = [trainer] + (["--data", str(data)] if data else []) + [
           "--epochs", str(cfg["epochs"]),
           "--batch", str(cfg["batch_size"]),
           "--lr", repr(float(cfg["lr"])),
           "--seed", str(cfg["seed"]),
           "--out", str(cfg["run_dir"])] + list(extra)
    os.execv(trainer, cmd)   # 替换自身：pid 不变，控制台的停止/信号照常生效


if __name__ == "__main__":
    sys.exit(main())
