#!/usr/bin/env python3
"""monitor.py 自检：解析层 + 增量 tail + 曲线绘制 + GUI 冒烟。

GUI 部分需要 tkinter 和可用的 DISPLAY；缺失时跳过并返回 0。
用法：python3 test_monitor.py   （本机可加 DISPLAY=:20）
"""
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import monitor

DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
passed = 0
skipped = 0


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    passed += 1
    print(f"  ok  {name}")


def skip(name, why):
    global skipped
    skipped += 1
    print(f"  skip  {name}（{why}）")


class FakeCanvas:
    """只实现 _line_chart 用到的 Canvas 方法，用于无显示环境下测试绘制。"""

    def __init__(self, w=400, h=300):
        self.w, self.h = w, h
        self.items = []

    def delete(self, *_args):
        self.items.clear()

    def winfo_width(self):
        return self.w

    def winfo_height(self):
        return self.h

    def create_line(self, *args, **kwargs):
        self.items.append(("line", args, kwargs))

    def create_text(self, *args, **kwargs):
        self.items.append(("text", args, kwargs))

    def create_oval(self, *args, **kwargs):
        self.items.append(("oval", args, kwargs))

    def create_rectangle(self, *args, **kwargs):
        self.items.append(("rectangle", args, kwargs))


# 1. demo 数据
rows = monitor.load_metrics(DEMO)
check("demo results.csv 8 行", len(rows) == 8)
check("demo epoch 序列 1..8", [r["epoch"] for r in rows] == list(range(1, 9)))
check("demo 首行 train_ce=1.386", abs(rows[0]["train_ce"] - 1.386) < 1e-9)
check("demo 末行 dev_acc=0.790", abs(rows[-1]["dev_accuracy"] - 0.790) < 1e-9)

ev = monitor.load_events(DEMO)
check("demo metrics.jsonl 4 事件", len(ev) == 4)
check("train_step 事件文本", monitor.fmt_event(ev[0]).startswith("[train] step=100"))
check("error 事件文本含 code", "TRAIN_NONFINITE_METRIC" in monitor.fmt_event(ev[3]))

# 2. 缺失目录 -> 空
check("缺失目录 metrics 空", monitor.load_metrics("/nonexistent/x") == [])
check("缺失目录 events 空", monitor.load_events("/nonexistent/x") == [])

# 3. 损坏/非 UTF-8/BOM/空 epoch -> 不崩溃，坏单元格为 None
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,train_ce\nx,1.0\ny,\n")
    bad = monitor.load_metrics(d)
    check("损坏 csv 保留 2 行", len(bad) == 2)
    check("损坏 csv epoch 为 None", [r["epoch"] for r in bad] == [None, None])
    check("损坏 csv 数值行保留", bad[0]["train_ce"] == 1.0)
    canvas = FakeCanvas()
    monitor._line_chart(canvas, [{"name": "x", "color": "#000",
                                  "points": [(r["epoch"], r["train_ce"]) for r in bad]}], "bad")
    check("空 epoch 曲线不崩溃",
          any(i[0] == "text" and i[2].get("text") == "(无数据)" for i in canvas.items))

    with open(os.path.join(d, "results.csv"), "wb") as f:
        f.write(b"epoch,train_ce,dev_ce,dev_accuracy\n1,\xff,0.9,0.5\n")
    non_utf8 = monitor.load_metrics(d)
    check("非 UTF-8 csv 不崩溃", len(non_utf8) == 1 and non_utf8[0]["epoch"] == 1)

    with open(os.path.join(d, "results.csv"), "w", encoding="utf-8-sig") as f:
        f.write("epoch,train_ce,dev_ce,dev_accuracy\n1,1.0,0.9,0.5\n")
    bom = monitor.load_metrics(d)
    check("BOM 头正常解析", len(bom) == 1 and bom[0]["epoch"] == 1)

    with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,train_ce,dev_ce,dev_accuracy\n,1.0,1.0,0.5\n2,0.5,0.4,0.7\n")
    blank = monitor.load_metrics(d)
    check("空 epoch 排到最后", [r["epoch"] for r in blank] == [2, None])

    with open(os.path.join(d, "metrics.jsonl"), "wb") as f:
        f.write(b'\xff\xfe\n{"event":"status","message":"ok"}\n[1,2,3]\nnull\n')
    ev2 = monitor.load_events(d)
    check("非 UTF-8/非对象行跳过", len(ev2) == 1 and ev2[0]["event"] == "status")

    # csv.Error（超长字段）时保留已解析的行，而不是整文件清空。
    with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,train_ce,dev_ce,dev_accuracy\n1,1.0,0.9,0.5\n2," + "x" * 200000 + ",0.4,0.6\n")
    partial = monitor.load_metrics(d)
    check("csv.Error 保留已解析行", len(partial) == 1 and partial[0]["epoch"] == 1)

# 4. fmt_event 边界
check("非对象事件可显示", monitor.fmt_event([1, 2]) == "[1, 2]")
check("长路径缩写", monitor._short_path(
    "/root/claude-code-backup-20261028/code/uav_rgb/ui/demo") == "…/ui/demo")
check("短路径不缩写", monitor._short_path("/tmp/run1") == "/tmp/run1")
check("数据年龄秒", monitor._fmt_age(59) == "59s 前")
check("数据年龄分钟", monitor._fmt_age(120) == "2m 前")
check("数据年龄小时", monitor._fmt_age(7200) == "2.0h 前")
train = {"event": "train_step", "step": 100, "total_steps": 800, "loss": 1.0,
         "action_accuracy": 0.4, "lr": 0.001, "eta_sec": 87.4}
check("train_step 含进度/ETA", "/800" in monitor.fmt_event(train)
      and "eta=87s" in monitor.fmt_event(train))
status0 = {"event": "status", "step": 0, "message": "saved"}
check("step=0 不丢", "step=0" in monitor.fmt_event(status0))
fatal = {"event": "error", "severity": "FATAL", "code": "X", "message": "bad"}
check("error 带 severity", monitor.fmt_event(fatal).startswith("[error/FATAL]"))
sur = monitor._safe_text('{"message":"\ud800"}')
check("孤立代理字符可安全显示", "\ud800" not in sur and "?" in sur)

# 5. 增量 tail：追加/半行/截断/替换
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "metrics.jsonl")
    state = {}
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"event":"a"}\n{"event":"b"}\n')
    got, reset = monitor.tail_events(d, state)
    check("tail 首次 2 条且 reset", len(got) == 2 and reset)
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"event":"c"}\n')
    got, reset = monitor.tail_events(d, state)
    check("tail 追加 1 条", len(got) == 1 and got[0]["event"] == "c" and not reset)
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"event":"d"')
    check("tail 半行不解析", monitor.tail_events(d, state)[0] == [])
    with open(p, "a", encoding="utf-8") as f:
        f.write('}\n')
    got, _ = monitor.tail_events(d, state)
    check("tail 半行补全后解析", len(got) == 1 and got[0]["event"] == "d")
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"event":"new"}\n')
    got, reset = monitor.tail_events(d, state)
    check("tail 截断后 reset", reset and len(got) == 1 and got[0]["event"] == "new")
    tmp = p + ".new"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write('{"event":"replaced"}\n')
    os.replace(tmp, p)
    got, reset = monitor.tail_events(d, state)
    check("tail 替换后 reset", reset and len(got) == 1 and got[0]["event"] == "replaced")

    with open(p, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"event": "e", "step": i}) + "\n")
    got = monitor.load_events(d, limit=2)
    check("load_events limit 取最新", [e["step"] for e in got] == [3, 4])

    # 同 inode 截断后写回更大文件：不能按旧 offset 切进新文件中间。
    with open(p, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"event": "old", "step": i, "pad": "x" * 40}) + "\n")
    state = {}
    monitor.tail_events(d, state)
    old_size = os.path.getsize(p)
    with open(p, "w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({"event": "new", "step": i, "pad": "y" * 60}) + "\n")
    check("tail 原地写回更大文件", os.path.getsize(p) > old_size)
    got, reset = monitor.tail_events(d, state)
    check("tail 原地写回更大文件 reset", reset and len(got) == 6
          and [e["event"] for e in got] == ["new"] * 6)
    os.remove(p)
    got, reset = monitor.tail_events(d, state)
    check("tail 缺失文件要求清空", got == [] and reset)
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"event":"x"}\n{"event":"y"}\n')
    check("load_events limit=0 返回空", monitor.load_events(d, limit=0) == [])

# 6. 曲线：断点分段 + 单点圆点
canvas = FakeCanvas()
monitor._line_chart(canvas, [{"name": "x", "color": "#000",
                              "points": [(1, 1.0), (2, 1.5), (3, None),
                                         (4, 3.0), (5, 4.0)]}], "gap")
data_lines = [i for i in canvas.items if i[0] == "line" and i[2].get("fill") == "#000"]
check("断点分成两段", len(data_lines) == 2)
canvas = FakeCanvas()
monitor._line_chart(canvas, [{"name": "x", "color": "#000", "points": [(1, 1.0)]}], "one")
check("单点画圆点", len([i for i in canvas.items if i[0] == "oval"]) == 1)
labels = [i[2].get("text") for i in canvas.items
          if i[0] == "text" and i[2].get("anchor") == "n"
          and i[2].get("font") == ("TkDefaultFont", 8)]
check("单点只画一个 x 刻度", labels == ["1"])

# 8. 训练控制：纯函数（无需 GUI）
argv = monitor.build_train_argv(f"{sys.executable} ui/trainer_demo.py",
                                "/tmp/x/train_config.json")
check("训练 argv 追加 --config", argv[-2:] == ["--config", "/tmp/x/train_config.json"])
check("训练命令含空格不拆",
      monitor.build_train_argv("'/a b/trainer' --x", "/c")[0] == "/a b/trainer")
for bad in ("", "   ", "'unterminated"):
    try:
        monitor.build_train_argv(bad, "/c")
    except ValueError:
        pass
    else:
        raise AssertionError(f"FAIL: 坏训练命令 {bad!r} 未拒绝")
check("坏训练命令被拒绝", True)
cfg = monitor.validate_train_config("/tmp/run_x", "8", "50", "16", "0.001", "42")
check("训练配置字段", cfg["schema"] == monitor.TRAIN_SCHEMA and cfg["epochs"] == 8
      and cfg["steps_per_epoch"] == 50 and cfg["batch_size"] == 16
      and abs(cfg["lr"] - 0.001) < 1e-12 and cfg["seed"] == 42)
for args in (("/tmp/r", "0", "1", "1", "0.001", "0"),
             ("/tmp/r", "1", "1", "1", "nan", "0"),
             ("/tmp/r", "1", "1", "1", "1.1", "0"),
             ("/tmp/r", "1", "1", "1", "0.001", "-1"),
             ("/tmp/r", "1.5", "1", "1", "0.001", "0"),
             (monitor.DEFAULT_RUN_DIR, "1", "1", "1", "0.001", "0")):
    try:
        monitor.validate_train_config(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(f"FAIL: 非法训练配置 {args!r} 未拒绝")
check("非法训练配置被拒绝", True)
with tempfile.TemporaryDirectory() as d:
    cfg = monitor.validate_train_config(d, "2", "3", "4", "0.01", "7")
    argv = monitor.build_train_argv(f"{sys.executable} ui/trainer_demo.py",
                                    os.path.join(d, "train_config.json"))
    monitor.write_train_config(d, cfg, argv)
    with open(os.path.join(d, "train_config.json"), encoding="utf-8") as f:
        back = json.load(f)
    check("训练配置可往返", back == cfg)
    with open(os.path.join(d, "train_command.txt"), encoding="utf-8") as f:
        cmd = f.read()
    check("训练命令已落盘", "--config" in cmd and cmd.endswith("\n"))
    check("无 .tmp 残留", not any(n.endswith(".tmp") for n in os.listdir(d)))

    logp = os.path.join(d, "train.log")
    state = {}
    with open(logp, "w", encoding="utf-8") as f:
        f.write("a\nb\n")
    text, reset = monitor.tail_text(logp, state)
    check("tail_text 首次", text == "a\nb\n" and reset)
    with open(logp, "a", encoding="utf-8") as f:
        f.write("c\n")
    text, reset = monitor.tail_text(logp, state)
    check("tail_text 追加", text == "c\n" and not reset)
    with open(logp, "w", encoding="utf-8") as f:
        f.write("z\n")
    text, reset = monitor.tail_text(logp, state)
    check("tail_text 截断 reset", reset and text == "z\n")

    line = "epoch 0001 loss=0.12345 action_acc=0.678\n"
    logq = os.path.join(d, "big.log")
    with open(logq, "w", encoding="utf-8") as f:
        f.write(line * 3000)                                   # 123 KB，必然分块
    st2 = {}
    first, _ = monitor.tail_text(logq, st2)
    second, _ = monitor.tail_text(logq, st2)
    check("tail_text 按行边界切", first.endswith("\n") and second.startswith("epoch 0001")
          and first.count("\n") + second.count("\n") == 3000)
    zhq = os.path.join(d, "zh.log")
    with open(zhq, "w", encoding="utf-8") as f:
        f.write("中文日志行：epoch 完成\n" * 6000)          # 186 KB 短行日志
    st3 = {}
    zh1, _ = monitor.tail_text(zhq, st3)
    zh2, _ = monitor.tail_text(zhq, st3)
    check("tail_text 中文不被切坏", "�" not in zh1 + zh2 and zh1.endswith("\n"))

    # 文件末尾还没有换行：半行要留到下一次追加后整行显示。
    half = os.path.join(d, "half.log")
    with open(half, "w", encoding="utf-8") as f:
        f.write("abc")
    st4 = {}
    h1, _ = monitor.tail_text(half, st4)
    with open(half, "a", encoding="utf-8") as f:
        f.write("def\n")
    h2, _ = monitor.tail_text(half, st4)
    check("tail_text 末尾半行不提前显示", h1 == "" and h2 == "abcdef\n")

    # 多字节字符被块边界切开：半行缓存必须按字节保留，不能 decode 成替换符。
    mb = os.path.join(d, "mb.log")
    raw = "中文\n".encode("utf-8")
    with open(mb, "wb") as f:
        f.write(raw[:1])
    st5 = {}
    m1, _ = monitor.tail_text(mb, st5)
    with open(mb, "ab") as f:
        f.write(raw[1:])
    m2, _ = monitor.tail_text(mb, st5)
    check("tail_text 多字节不被切开", m1 == "" and m2 == "中文\n")

# 9. 真实 trainer 封装：uavrgb.train.v1 -> uavrgb_train CLI
NATIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer_native.py")
with tempfile.TemporaryDirectory() as d:
    fake_bin = os.path.join(d, "fake_train")
    dump = os.path.join(d, "argv.txt")
    with open(fake_bin, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {dump}\n")
    os.chmod(fake_bin, 0o755)
    cfp = os.path.join(d, "train_config.json")
    cfg = {"schema": monitor.TRAIN_SCHEMA, "run_dir": os.path.join(d, "run"),
           "epochs": 3, "steps_per_epoch": 9, "batch_size": 16,
           "lr": 0.001, "seed": 7, "data": "/tmp/bench_ds.uavr"}
    with open(cfp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    env = dict(os.environ, UAVRGB_TRAIN_BIN=fake_bin)
    rc = subprocess.call([sys.executable, NATIVE, "--config", cfp,
                          "--resume", "/x/ckpt.bin"], env=env)
    with open(dump, encoding="utf-8") as f:
        got = f.read().split()
    check("native 封装转 CLI", rc == 0 and got[:2] == ["--data", "/tmp/bench_ds.uavr"]
          and got[got.index("--batch") + 1] == "16"
          and got[got.index("--out") + 1] == os.path.join(d, "run")
          and "--config" not in got)
    check("native 封装透传额外参数", got[-2:] == ["--resume", "/x/ckpt.bin"])
    del cfg["data"]
    with open(cfp, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    rc = subprocess.call([sys.executable, NATIVE, "--config", cfp,
                          "--data", "/tmp/other.uavr"], env=env)
    with open(dump, encoding="utf-8") as f:
        got = f.read().split()
    check("native 命令框给数据集", rc == 0 and got.count("--data") == 1
          and got[got.index("--data") + 1] == "/tmp/other.uavr")
    rc = subprocess.call([sys.executable, NATIVE, "--config", cfp], env=env,
                         stderr=subprocess.DEVNULL)
    check("native 缺 data 报错", rc == 2)

# 10. 评测结果（uavrgb.eval.v1）
EVAL = {"schema": "uavrgb.eval.v1", "ckpt": "/x/runs/r1/ckpt.bin", "split": "test",
        "episodes_requested": 3, "episodes_done": 2, "steps": 60, "hold_s": 0.5,
        "wall_s": 12.0, "success_rate": 0.5, "collision_rate": 0.5,
        "false_forward_rate": 0.0,
        "inference_ms": {"p50": 3.2, "p95": 5.1, "n_warm": 58,
                         "cold_first_ms": [300.0, 322.9]},
        "action_distribution": {"SAFE_NEUTRAL": 20, "FORWARD_SLOW": 30,
                                "YAW_LEFT": 5, "YAW_RIGHT": 0},
        "failed": [{"seed": 7, "rc": 1, "stderr_tail": "boom"}],
        "per_episode": [{"seed": 1, "max_x": 12.0, "goal_x": 20.0,
                         "collision": False, "success": True},
                        {"seed": 2, "max_x": 5.0, "goal_x": 20.0,
                         "collision": True, "success": False}]}
with tempfile.TemporaryDirectory() as d:
    check("缺失 eval.json 返回 None", monitor.load_eval(d) is None)
    with open(os.path.join(d, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(EVAL, f)
    parsed = monitor.load_eval(d)
    lines = monitor.fmt_eval(parsed)
    check("eval.json 可解析", parsed is not None and parsed["split"] == "test")
    check("评测摘要含 ckpt/split/SR",
          any("ckpt: ckpt.bin" in ln for ln in lines)
          and any("SR: 50.0%" in ln and "碰撞: 50.0%" in ln for ln in lines))
    check("评测摘要含失败集与 warm 延迟",
          any("失败 1" in ln for ln in lines)
          and any("3.20 ms / 5.10 ms" in ln and "warm" in ln for ln in lines))
    check("评测摘要含冷启动首帧",
          any("冷启动首帧: 300.0–322.9 ms（2 集）" in ln for ln in lines))
    check("动作次数为 0 照常显示",
          any("YAW_RIGHT 0" in ln for ln in lines))
    with open(os.path.join(d, "eval.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "x"}, f)
    check("别的 schema 拒绝", monitor.load_eval(d) is None)
nulls = dict(EVAL, success_rate=None, collision_rate=None, false_forward_rate=None)
check("rate 全 null 显示无有效集", any("无有效集" in ln for ln in monitor.fmt_eval(nulls)))
canvas = FakeCanvas()
monitor._eval_chart(canvas, EVAL["per_episode"])
rects = [i for i in canvas.items if i[0] == "rectangle"]
check("每集柱状图 2 根且颜色不同", len(rects) == 2
      and rects[0][2]["fill"] != rects[1][2]["fill"])
check("柱状图画 goal 参照线",
      any(i[0] == "line" and i[2].get("dash") for i in canvas.items))
canvas = FakeCanvas()
monitor._eval_chart(canvas, [])
check("空 per_episode 提示", any(i[0] == "text" and "无 per_episode" in i[2].get("text", "")
                                for i in canvas.items))

check("fmt_exit", monitor.fmt_exit(0, False) == "训练完成 rc=0"
      and monitor.fmt_exit(3, False) == "异常退出 rc=3"
      and monitor.fmt_exit(0, True) == "已停止 rc=0")
check("elapsed 格式", monitor._fmt_elapsed(3661) == "01:01:01")

# 7. GUI 冒烟（需要 tkinter + DISPLAY）
gui_ok = True
if monitor.tk is None:
    skip("GUI 冒烟", "无 tkinter")
    gui_ok = False
else:
    try:
        root = monitor.tk.Tk()
    except monitor.tk.TclError as exc:
        skip("GUI 冒烟", f"无显示: {exc}")
        gui_ok = False
if gui_ok:
    try:
        app = monitor.MonitorApp(root, DEMO, 60000)
        root.update_idletasks()
        root.update()
        app.refresh(force=True)
        root.update_idletasks()
        root.update()
        check("GUI 表格 8 行", len(app.table.get_children()) == 8)
        kids = app.table.get_children()
        app.table.selection_set(kids[2])
        app._set_table(app._rows)
        new_kids = app.table.get_children()
        check("GUI 表格重建保留选中行", app.table.selection() == (new_kids[2],))
        check("GUI 事件 4 条", app.log.size() == 4)
        check("GUI DEMO 标识", "DEMO" in app.demo_tag.cget("text"))
        app._set_charts(app._rows)
        check("GUI loss 曲线有内容", len(app.canvas_loss.find_all()) > 0)
        check("GUI acc 曲线有内容", len(app.canvas_acc.find_all()) > 0)

        redraws = []
        orig_redraw = app._redraw_charts
        app._redraw_charts = lambda: (redraws.append(1), orig_redraw())[1]
        root.geometry("760x520")
        end = time.time() + 0.3
        while time.time() < end:
            root.update()
            time.sleep(0.01)
        check("GUI 缩放触发重画", len(redraws) >= 1)
        check("GUI 缩放后曲线有内容", len(app.canvas_loss.find_all()) > 0)
        app._redraw_charts = orig_redraw

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
                f.write("epoch,train_ce,dev_ce,dev_accuracy\n,1.0,1.0,0.5\n2,0.5,0.4,0.7\n")
            app._set_run_dir(d)
            root.update_idletasks()
            root.update()
            check("GUI 空 epoch 不崩溃", len(app.table.get_children()) == 2)
            check("GUI 非 DEMO 标识隐藏", app.demo_tag.cget("text") == "")
            check("GUI 状态含 epochs", "epochs" in app.status.cget("text"))

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
                f.write("epoch,train_ce,dev_ce,dev_accuracy\n1,1.0,1.0,0.5\n")
            app._set_run_dir(d)
            calls = []
            orig_lm, orig_te = monitor.load_metrics, monitor.tail_events
            monitor.load_metrics = lambda rd: (calls.append("metrics"), orig_lm(rd))[1]
            monitor.tail_events = lambda rd, st: (calls.append("tail"), orig_te(rd, st))[1]
            try:
                app.refresh()
                check("GUI 签名未变时不读文件", calls == [])
                with open(os.path.join(d, "results.csv"), "a", encoding="utf-8") as f:
                    f.write("2,0.5,0.5,0.7\n")
                app.refresh()
                check("GUI 签名变化后重新读取", calls == ["metrics", "tail"])
            finally:
                monitor.load_metrics, monitor.tail_events = orig_lm, orig_te

        app._toggle_pause()
        check("GUI 暂停", app._paused and app.pause_btn.cget("text") == "继续")
        app._toggle_pause()
        check("GUI 继续", not app._paused and app.pause_btn.cget("text") == "暂停")

        app._set_run_dir("/tmp/\udcff")   # 真实不可解码文件名：fsencode 可还原
        root.update_idletasks()
        root.update()
        check("GUI 代理字符路径不崩溃", "\udcff" not in app.run_label.cget("text"))

        # 训练控制：用临时假 trainer 验证启动/停止/崩溃（不依赖真的 DEMO 训练器）
        app.train_cmd_var.set(monitor.default_train_command())
        check("GUI 默认命令 DEMO 标识", "DEMO" in app.train_demo_tag.cget("text"))
        with tempfile.TemporaryDirectory() as td:
            fake = os.path.join(td, "fake_trainer.py")
            with open(fake, "w", encoding="utf-8") as f:
                f.write(
                    "import argparse, json, os, sys, time\n"
                    "p = argparse.ArgumentParser(); p.add_argument('--config', required=True)\n"
                    "a = p.parse_args(); cfg = json.load(open(a.config))\n"
                    "d = cfg['run_dir']; os.makedirs(d, exist_ok=True)\n"
                    "mode = os.environ.get('FAKE_MODE', 'ok')\n"
                    "if mode == 'crash':\n"
                    "    print('boom', file=sys.stderr); sys.exit(3)\n"
                    "open(os.path.join(d, 'results.csv'), 'w').write("
                    "'epoch,train_ce,dev_ce,dev_accuracy\\n1,1.0,1.1,0.5\\n')\n"
                    "open(os.path.join(d, 'metrics.jsonl'), 'w').write("
                    "json.dumps({'event': 'status', 'message': 'fake'}) + '\\n')\n"
                    "print('fake stdout')\n"
                    "if mode == 'sleep':\n"
                    "    print('READY', flush=True)\n"
                    "    try: time.sleep(60)\n"
                    "    except KeyboardInterrupt: print('SIGINT', flush=True)\n")
            app.train_cmd_var.set(f"{shlex.quote(sys.executable)} {shlex.quote(fake)}")

            os.environ["FAKE_MODE"] = "ok"
            app.train_dir_var.set(os.path.join(td, "run1"))
            app._start_training()
            check("GUI 训练启动", app.proc is not None and app._train_state == "running")
            end = time.time() + 5
            while app.proc is not None and time.time() < end:
                app._poll_train()
                root.update()
                time.sleep(0.02)
            check("GUI 训练正常退出", app.proc is None and app._last_rc == 0)
            check("GUI 训练输出",
                  "fake stdout" in app.train_log.get("1.0", "end"))
            check("GUI 训练指标", len(app.table.get_children()) >= 1)
            check("GUI 控制台文件在 run 目录外",
                  not os.path.exists(os.path.join(td, "run1", "train.log"))
                  and os.path.exists(os.path.join(td, "run1.monitor", "train.log")))
            app._set_run_dir(os.path.join(td, "run1"))   # 重新打开已跑完的目录
            app._refresh_train_log()
            check("GUI 历史 run 日志可见",
                  "fake stdout" in app.train_log.get("1.0", "end"))
            with tempfile.TemporaryDirectory() as legacy:   # 旧布局：日志写在 run 目录里
                with open(os.path.join(legacy, "train.log"), "w", encoding="utf-8") as f:
                    f.write("legacy log\n")
                app._set_run_dir(legacy)
                check("GUI 旧布局 train.log 仍可见",
                      "legacy log" in app.train_log.get("1.0", "end"))

            with tempfile.TemporaryDirectory() as evd:
                with open(os.path.join(evd, "eval.json"), "w", encoding="utf-8") as f:
                    json.dump(EVAL, f)
                app._set_run_dir(evd)
                root.update()
                txt = app.eval_text.get("1.0", "end")
                check("GUI 评测页显示 SR 与 ckpt",
                      "SR: 50.0%" in txt and "ckpt: ckpt.bin" in txt)
                check("GUI 状态栏显示评测", "评测 test" in app.status.cget("text"))
            with tempfile.TemporaryDirectory() as evd:
                app._set_run_dir(evd)
                root.update()
                check("GUI 无 eval.json 提示",
                      "没有 eval.json" in app.eval_text.get("1.0", "end"))

            os.environ["FAKE_MODE"] = "sleep"
            app.train_dir_var.set(os.path.join(td, "run2"))
            app._start_training()
            check("GUI 训练运行中", app.proc is not None and app._train_state == "running")
            app._stop_training()
            end = time.time() + 8
            while app.proc is not None and time.time() < end:
                app._poll_train()
                root.update()
                time.sleep(0.02)
            check("GUI 训练停止", app.proc is None and app._last_rc is not None)
            check("GUI 停止状态", "已停止" in app.train_status.cget("text"))

            os.environ["FAKE_MODE"] = "crash"
            app.train_dir_var.set(os.path.join(td, "run3"))
            app._start_training()
            end = time.time() + 5
            while app.proc is not None and time.time() < end:
                app._poll_train()
                root.update()
                time.sleep(0.02)
            check("GUI 训练崩溃",
                  app._last_rc == 3 and "异常退出" in app.train_status.cget("text"))
            check("GUI 训练页签",
                  app.notebook.index("current") == app.notebook.index(app.tab_output))
            check("GUI 训练控件", len(app.train_vars) == 5
                  and str(app.stop_btn.cget("state")) == "disabled")

            # 启动失败要回滚本次产物，否则改好命令后同一目录也报“不是空目录”
            app.train_cmd_var.set("/nonexistent/trainer")
            app.train_dir_var.set(os.path.join(td, "run4"))
            app._start_training()
            check("GUI 启动失败提示", "启动失败" in app.train_status.cget("text"))
            check("GUI 启动失败不留产物",
                  (not os.path.exists(os.path.join(td, "run4")) or
                   os.listdir(os.path.join(td, "run4")) == [])
                  and os.listdir(os.path.join(td, "run4.monitor")) == [])
            app.train_cmd_var.set(f"{shlex.quote(sys.executable)} {shlex.quote(fake)}")
            os.environ["FAKE_MODE"] = "ok"
            app._start_training()
            check("GUI 失败后同目录可重试",
                  app.proc is not None and app._train_state == "running")
            end = time.time() + 5
            while app.proc is not None and time.time() < end:
                app._poll_train()
                root.update()
                time.sleep(0.02)
            check("GUI 重试训练完成", app.proc is None and app._last_rc == 0)

            # 结束后默认目录必须是新的：快跑（同整秒内起停）不能算回刚用过的目录
            check("GUI 结束后默认目录换新",
                  app.train_dir_var.get() != os.path.join(td, "run4")
                  and not os.path.exists(app.train_dir_var.get()))
            made = app._default_run_dir()
            os.makedirs(made)
            try:
                check("GUI 默认目录跳过已存在", app._default_run_dir() != made)
            finally:
                os.rmdir(made)

            # 写盘中途失败留下的 .tmp 也要回滚，否则目录永远“不是空的”
            def bad_write(rd, _cfg, _argv):
                with open(os.path.join(rd, "train_config.json.tmp"), "w"):
                    pass
                raise OSError("disk full")

            orig_write = monitor.write_train_config
            monitor.write_train_config = bad_write
            try:
                app.train_dir_var.set(os.path.join(td, "run5"))
                app._start_training()
            finally:
                monitor.write_train_config = orig_write
            check("GUI 写盘失败回滚 .tmp",
                  os.listdir(os.path.join(td, "run5")) == []
                  and os.listdir(os.path.join(td, "run5.monitor")) == [])

        # 刷新异常也必须重排 after，否则自动刷新永久停掉。
        calls = []
        orig_after, orig_refresh = root.after, app.refresh

        def boom():
            raise RuntimeError("boom")

        app.refresh = boom
        root.after = lambda ms, fn: calls.append((ms, fn))
        with contextlib.redirect_stderr(io.StringIO()):
            app._tick()
        check("GUI 刷新异常后仍重排", len(calls) == 1)
        check("GUI 异常写入状态栏", "刷新失败" in app.status.cget("text"))
        app.refresh, root.after = orig_refresh, orig_after
    finally:
        root.destroy()

print(f"test_monitor: {passed} 项通过" + (f"，{skipped} 项跳过" if skipped else ""))
