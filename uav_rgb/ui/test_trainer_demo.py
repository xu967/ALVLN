#!/usr/bin/env python3
"""trainer_demo.py 自检：正常训练 / 非法配置 / SIGINT 中断。

纯标准库，真实启动子进程验证退出码与落盘文件。
用法：python3 ui/test_trainer_demo.py
"""
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "trainer_demo.py")
PY = sys.executable or "python3"

sys.dont_write_bytecode = True  # 不在 ui/ 里留下 __pycache__
sys.path.insert(0, HERE)
import trainer_demo
from trainer_demo import _train_loss

passed = 0


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    passed += 1
    print(f"  ok  {name}")


def base_config(run_dir):
    return {"schema": "uavrgb.train.v1", "run_dir": run_dir, "epochs": 8,
            "steps_per_epoch": 50, "batch_size": 16, "lr": 0.001, "seed": 42}


def write_config(path, cfg):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)


def run_trainer(config_path):
    return subprocess.run(
        [PY, TRAINER, "--config", config_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace", timeout=30)


def read_events(path):
    """读 JSONL；任何一行的解析失败都直接抛异常（即半行/坏行）。"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    check(os.path.basename(path) + " 以换行结尾", raw.endswith("\n"))
    lines = raw.splitlines()
    events = [json.loads(line) for line in lines]
    check(os.path.basename(path) + " 无空行", all(line.strip() for line in lines))
    return raw, events


# 1. 正常配置：退出码、stdout、CSV、JSONL
with tempfile.TemporaryDirectory() as tmp:
    run_dir = os.path.join(tmp, "run", "nested")  # 目录不存在，训练器应创建
    cfg_path = os.path.join(tmp, "train.json")
    write_config(cfg_path, base_config(run_dir))

    started = time.monotonic()
    proc = run_trainer(cfg_path)
    elapsed = time.monotonic() - started

    check("正常配置 exit 0", proc.returncode == 0)
    check("正常配置无 traceback", "Traceback" not in proc.stderr)
    out_lines = proc.stdout.splitlines()
    check("stdout 每行带 [DEMO]",
          out_lines and all(line.startswith("[DEMO]") for line in out_lines))
    check("stdout 标明非真实训练", "非真实训练" in proc.stdout)
    check("stdout 打印 run_dir", run_dir in proc.stdout)
    check("有模拟训练停顿（>2 s）", elapsed > 2.0)
    for epoch in range(1, 9):
        n = sum(1 for line in out_lines if f"epoch {epoch}/8" in line)
        check(f"epoch {epoch} 有 2-3 条进度行", 2 <= n <= 3)

    check("run_dir 被创建", os.path.isdir(run_dir))
    check("run_dir 只有两个输出文件",
          sorted(os.listdir(run_dir)) == ["metrics.jsonl", "results.csv"])

    with open(os.path.join(run_dir, "results.csv"), encoding="utf-8") as f:
        csv_lines = f.read().splitlines()
    check("CSV 表头精确匹配",
          csv_lines[0] == "epoch,train_ce,dev_ce,dev_accuracy")
    check("CSV 共 8 行数据", len(csv_lines) == 9)
    rows = []
    for line in csv_lines[1:]:
        cells = line.split(",")
        check("CSV 行 4 列完整", len(cells) == 4)
        rows.append([int(cells[0])] + [float(x) for x in cells[1:]])
    check("CSV epoch 为 1..8", [r[0] for r in rows] == list(range(1, 9)))
    check("CSV 损失有限", all(math.isfinite(v) for r in rows for v in r[1:3]))
    check("CSV dev_accuracy 在 [0,1]",
          all(0.0 <= r[3] <= 1.0 for r in rows))
    check("CSV dev_ce 略高于 train_ce", all(r[2] > r[1] for r in rows))
    check("CSV train_ce 下降", rows[-1][1] < rows[0][1])
    check("CSV dev_accuracy 上升", rows[-1][3] > rows[0][3])

    raw, events = read_events(os.path.join(run_dir, "metrics.jsonl"))
    kinds = {e.get("event") for e in events}
    check("JSONL 事件类型齐全", {"train_step", "validation", "status"} <= kinds)
    check("首事件为 start", events[0] == {"event": "status",
                                        "message": "demo trainer start"})
    check("末事件为 done", events[-1] == {"event": "status",
                                        "message": "demo trainer done"})
    train_events = [e for e in events if e["event"] == "train_step"]
    val_events = [e for e in events if e["event"] == "validation"]
    check("每个 epoch 至少 3 个 train_step 且覆盖首末步",
          all(len([e for e in train_events
                   if (e["step"] - 1) // 50 == ep]) >= 3 for ep in range(8)))
    for ep in range(1, 9):
        steps = [e["step"] for e in train_events if (e["step"] - 1) // 50 == ep - 1]
        check(f"epoch {ep} train_step 首末步正确",
              steps[0] == (ep - 1) * 50 + 1 and steps[-1] == ep * 50)
    check("每 epoch 一个 validation",
          len(val_events) == 8 and [e["step"] for e in val_events]
          == [50 * e for e in range(1, 9)])
    check("validation 标记 split=val", all(e["split"] == "val" for e in val_events))
    check("train_step total_steps=400",
          all(e["total_steps"] == 400 for e in train_events))
    check("train_step 字段齐全",
          all(all(k in e for k in ("step", "total_steps", "loss",
                                   "action_accuracy", "lr", "eta_sec"))
              for e in train_events))
    check("train_step loss 有限",
          all(math.isfinite(e["loss"]) and e["loss"] > 0 for e in train_events))
    check("train_step accuracy 在 [0,1]",
          all(0.0 <= e["action_accuracy"] <= 1.0 for e in train_events))
    check("train_step lr/eta 有限",
          all(math.isfinite(e["lr"]) and e["lr"] > 0
              and math.isfinite(e["eta_sec"]) for e in train_events))
    check("validation loss/accuracy 合法",
          all(math.isfinite(e["loss"]) and e["loss"] > 0
              and 0.0 <= e["action_accuracy"] <= 1.0 for e in val_events))

# 2. 模拟曲线：初始 ln4、小 lr 收敛更差
check("初始损失为 ln4",
      abs(_train_loss(0.0, 0.001) - math.log(4.0)) < 1e-9)
check("极小 lr 的最终损失更差",
      _train_loss(1.0, 0.001) < _train_loss(1.0, 1e-9))

# 3. 非法配置：exit 2、stderr、无输出文件
bad_cases = [
    ("schema 错误", {"schema": "uavrgb.train.v2"}),
    ("epochs=0", {"epochs": 0}),
    ("lr 为负", {"lr": -0.001}),
    ("lr=0", {"lr": 0}),
    ("缺少 run_dir", None),
]
for name, override in bad_cases:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run")
        cfg = base_config(run_dir)
        if override is None:
            del cfg["run_dir"]
        else:
            cfg.update(override)
        cfg_path = os.path.join(tmp, "bad.json")
        write_config(cfg_path, cfg)

        proc = run_trainer(cfg_path)
        check(f"{name}: exit 2", proc.returncode == 2)
        check(f"{name}: stderr 有说明", bool(proc.stderr.strip()))
        check(f"{name}: 未创建 run_dir", not os.path.exists(run_dir))
        check(f"{name}: 无 CSV/JSONL", sorted(os.listdir(tmp)) == ["bad.json"])

# 4. SIGINT：干净中断、130、JSONL 无半行、CSV 行数不超过已完成 epoch
with tempfile.TemporaryDirectory() as tmp:
    run_dir = os.path.join(tmp, "run")
    cfg_path = os.path.join(tmp, "long.json")
    write_config(cfg_path, {**base_config(run_dir), "epochs": 50})

    proc = subprocess.Popen(
        [PY, TRAINER, "--config", cfg_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace")
    try:
        time.sleep(0.5)
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:  # 兜底：不能让子进程残留
            proc.kill()
            proc.communicate()

    check("SIGINT exit 130", proc.returncode == 130)
    check("SIGINT 后子进程已退出", proc.poll() is not None)
    check("SIGINT 无 traceback", "Traceback" not in err)
    check("SIGINT stdout 每行带 [DEMO]",
          all(line.startswith("[DEMO]") for line in out.splitlines()))

    _, events = read_events(os.path.join(run_dir, "metrics.jsonl"))
    interrupted = [e for e in events if e.get("event") == "status"
                   and str(e.get("message", "")).startswith(
                       "demo trainer interrupted at epoch ")]
    check("SIGINT 有且仅有一条 interrupted 状态", len(interrupted) == 1)
    check("SIGINT interrupted 是最后一条", events[-1] == interrupted[0])
    epoch_x = int(interrupted[0]["message"].rsplit(" ", 1)[1])
    check("SIGINT 中断 epoch 在 1..50", 1 <= epoch_x <= 50)

    with open(os.path.join(run_dir, "results.csv"), encoding="utf-8") as f:
        csv_lines = f.read().splitlines()
    check("SIGINT CSV 表头完整",
          csv_lines[0] == "epoch,train_ce,dev_ce,dev_accuracy")
    rows = [line.split(",") for line in csv_lines[1:]]
    check("SIGINT CSV 无半行", all(len(r) == 4 for r in rows))
    check("SIGINT CSV epoch 连续",
          [int(r[0]) for r in rows] == list(range(1, len(rows) + 1)))
    check("SIGINT CSV 行数 <= 已完成 epoch", len(rows) <= epoch_x)

print(f"test_trainer_demo: {passed} 项通过")
