#!/usr/bin/env python3
"""uav_rgb 演示训练器：纯 Python 标准库，生成模拟训练曲线，不做真实训练。

真实 C++/CUDA trainer 尚未实现；本脚本只按 uavrgb.train.v1 配置在 run_dir
写出 results.csv / metrics.jsonl，供 ui/monitor.py 端到端演示和自检。所有
损失/准确率都是确定性模拟值（由 seed 决定），不代表任何模型效果。

用法：python3 ui/trainer_demo.py --config PATH
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time

SCHEMA = "uavrgb.train.v1"
STEP_SLEEP = 0.05   # 每个上报事件后短暂停顿；8×50 步约 4.8 s，便于 1 s 轮询的 UI 看到进度
_ETA_PER_STEP = 0.01  # 纯展示的 ETA 估算（秒/步）

_interrupted = False
_current_epoch = 1


class ConfigError(Exception):
    """配置不合法（对应退出码 2）。"""


class _Interrupted(Exception):
    """SIGINT 标志被置位后，在主循环检查点抛出。"""


def _on_sigint(_signum, _frame):
    # 只置标志，不抛异常：保证不会在写 JSON/CSV 的中途留下半行。
    global _interrupted
    _interrupted = True


def _check_stop():
    if _interrupted:
        raise _Interrupted


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_config(path):
    """读取并校验 uavrgb.train.v1 配置；非法时抛 ConfigError。"""
    try:
        with open(path, encoding="utf-8-sig") as f:
            cfg = json.load(f)
    except (OSError, ValueError, RecursionError) as exc:
        raise ConfigError(f"无法读取配置 {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError("配置必须是 JSON 对象")
    if cfg.get("schema") != SCHEMA:
        raise ConfigError(f"schema 必须是 {SCHEMA!r}")
    run_dir = cfg.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ConfigError("run_dir 必须是非空字符串")
    for key in ("epochs", "steps_per_epoch", "batch_size"):
        if not _is_int(cfg.get(key)) or cfg[key] < 1:
            raise ConfigError(f"{key} 必须是 >=1 的整数")
    lr = cfg.get("lr")
    try:
        lr_ok = (not isinstance(lr, bool) and isinstance(lr, (int, float))
                 and math.isfinite(float(lr)) and lr > 0)
    except (OverflowError, ValueError):
        lr_ok = False
    if not lr_ok:
        raise ConfigError("lr 必须是有限正数")
    if not _is_int(cfg.get("seed")):
        raise ConfigError("seed 必须是整数")
    return cfg


def _lr_factor(lr):
    """把学习率映射到 0..1 的学习进度；0.001 视为基准，过小则几乎学不动。"""
    return min(1.0, lr / 0.001)


def _train_loss(progress, lr):
    """模拟训练损失：从 ln(4) 按 lr 和步数进度指数衰减。"""
    factor = _lr_factor(lr)
    floor = 0.20 + 0.75 * (1.0 - factor)
    return floor + (math.log(4.0) - floor) * math.exp(-3.0 * factor * progress)


def _accuracy(progress, lr):
    """模拟动作准确率：从四分类随机水平 0.25 上升。"""
    factor = _lr_factor(lr)
    return 0.25 + 0.62 * (1.0 - math.exp(-2.8 * factor * progress))


def _noise_scale(batch_size):
    """batch 越大噪声越小。"""
    return math.sqrt(16.0 / batch_size)


def _clamp01(value):
    return min(1.0, max(0.0, value))


def _sample_steps(steps_per_epoch):
    """每个 epoch 采样约 10 个步上报，必含首步、中点附近和末步。"""
    stride = max(1, steps_per_epoch // 8)
    steps = list(range(1, steps_per_epoch + 1, stride))
    if steps[-1] != steps_per_epoch:
        steps.append(steps_per_epoch)
    return steps


def _emit(stream, event):
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def _write_row(stream, epoch, train_ce, dev_ce, dev_acc):
    stream.write(f"{epoch},{train_ce:.6f},{dev_ce:.6f},{dev_acc:.6f}\n")
    stream.flush()


def run(cfg):
    """执行模拟训练，返回进程退出码（0 正常 / 130 被 SIGINT 中断）。"""
    global _interrupted, _current_epoch
    _interrupted = False
    _current_epoch = 1
    run_dir = cfg["run_dir"]
    epochs = cfg["epochs"]
    steps_per_epoch = cfg["steps_per_epoch"]
    batch_size = cfg["batch_size"]
    lr = float(cfg["lr"])
    total_steps = epochs * steps_per_epoch
    rnd = random.Random(cfg["seed"])
    noise_scale = _noise_scale(batch_size)

    print(f"[DEMO] 模拟训练（非真实训练）开始，run_dir={run_dir}", flush=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    csv_path = os.path.join(run_dir, "results.csv")
    with open(metrics_path, "w", encoding="utf-8") as metrics, \
            open(csv_path, "w", encoding="utf-8") as results:
        results.write("epoch,train_ce,dev_ce,dev_accuracy\n")
        results.flush()
        _emit(metrics, {"event": "status", "message": "demo trainer start"})
        time.sleep(STEP_SLEEP)
        try:
            for epoch in range(1, epochs + 1):
                _current_epoch = epoch
                _check_stop()
                steps = _sample_steps(steps_per_epoch)
                for i, local in enumerate(steps):
                    _check_stop()
                    step = (epoch - 1) * steps_per_epoch + local
                    progress = step / total_steps
                    loss = max(0.01, _train_loss(progress, lr)
                               + rnd.gauss(0.0, 0.012 * noise_scale))
                    acc = _clamp01(_accuracy(progress, lr)
                                   + rnd.gauss(0.0, 0.008 * noise_scale))
                    _emit(metrics, {
                        "event": "train_step", "step": step,
                        "total_steps": total_steps, "loss": round(loss, 6),
                        "action_accuracy": round(acc, 6), "lr": lr,
                        "eta_sec": round(_ETA_PER_STEP * (total_steps - step), 2),
                    })
                    if i == 0:
                        print(f"[DEMO] epoch {epoch}/{epochs} 开始："
                              f"step {step}/{total_steps} loss={loss:.4f}", flush=True)
                    elif i == len(steps) // 2:
                        print(f"[DEMO] epoch {epoch}/{epochs} 进度："
                              f"step {step}/{total_steps} loss={loss:.4f} "
                              f"action_acc={acc:.3f}", flush=True)
                    time.sleep(STEP_SLEEP)

                _check_stop()
                epoch_progress = epoch / epochs
                train_ce = max(0.01, _train_loss(epoch_progress, lr)
                               + rnd.gauss(0.0, 0.004 * noise_scale))
                # dev 略高于 train，accuracy 与训练同源但加一点噪声
                dev_ce = (train_ce * 1.04 + 0.02
                          + abs(rnd.gauss(0.0, 0.01 * noise_scale)))
                dev_acc = _clamp01(_accuracy(epoch_progress, lr)
                                   + rnd.gauss(0.0, 0.008 * noise_scale))
                # validation 与 CSV 行连续写入，中间不设中断检查，避免只写一半。
                _emit(metrics, {
                    "event": "validation", "step": epoch * steps_per_epoch,
                    "loss": round(dev_ce, 6), "action_accuracy": round(dev_acc, 6),
                    "split": "val",
                })
                _write_row(results, epoch, train_ce, dev_ce, dev_acc)
                print(f"[DEMO] epoch {epoch}/{epochs} 完成：train_ce={train_ce:.4f} "
                      f"dev_ce={dev_ce:.4f} dev_acc={dev_acc:.3f}", flush=True)
                time.sleep(STEP_SLEEP)
            _check_stop()
            _emit(metrics, {"event": "status", "message": "demo trainer done"})
            print("[DEMO] 模拟训练结束（非真实训练）", flush=True)
            return 0
        except _Interrupted:
            _emit(metrics, {"event": "status",
                            "message": f"demo trainer interrupted at epoch {_current_epoch}"})
            print(f"[DEMO] 已中断（非真实训练）：epoch {_current_epoch}", flush=True)
            return 130


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="uav_rgb 演示训练器（纯模拟，非真实训练）")
    parser.add_argument("--config", required=True, help="uavrgb.train.v1 配置 JSON")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"trainer_demo: 配置错误: {exc}", file=sys.stderr)
        return 2
    try:
        os.makedirs(cfg["run_dir"], exist_ok=True)
    except OSError as exc:
        print(f"trainer_demo: 无法创建 run_dir: {exc}", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        return run(cfg)
    except OSError as exc:
        print(f"trainer_demo: 写文件失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
