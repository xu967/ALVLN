#!/usr/bin/env python3
"""并行场景采集编排：N 个 GZ_PARTITION 隔离的 gz sim + N 个并发 collect.py。

用法：
  python3 sim/tools/collect_parallel.py --instances 6 --episodes-per-instance 2 \
      --steps 40 --out /tmp/dataset

流程：
  1. 为每个 (instance, episode) 派生 scene_id = seed = base_seed + i*E + e，
     调 scene_gen.py 生成各自世界（先全部生成，失败则不起 sim）。
  2. 每个 instance 一个 worker：顺序跑 E 个 episode；每个 episode 都新起一个
     `gz sim -s -r <world>`（collect.py 没有 episode reset，world 必须全新），
     setsid 独立进程组，每个 episode 独立 GZ_PARTITION、DISPLAY=:20、LP_NUM_THREADS=1
     （软件渲染限线程：6 实例并发实测 RTF 0.21→0.50；起飞 timeout 已从
     30 s 提到 120 s，RTF 0.2 时爬升 1.95 m 要 ~60 s 墙钟）。
     某个 episode 失败会重启 sim 重试 1 次（collect.py 发布层偶发丢首条指令）。
  3. 并发跑 collect.py（子进程，各自 --scene-id/--seed/--split）。
  4. 全部结束后把 staging 汇总到 --out/<split>/<scene_id>/。
  5. 无论成功、失败、Ctrl-C：kill -- -PGID 杀光自己起的 gz sim / collect 进程组，
     并清掉自己的临时 world 目录。

split 策略（写入 collection_manifest.json）：按几何组而不是 scene_id 分。
  世界生成后算出几何组 ID，再由 split_manifest.py 的固定 salt hash 决定
  train / val / test。同一几何永远同一 split，跨 split 交集为 0。
  旧策略 `scene_id % 10` 会让不同 scene_id 生成同一几何并跨 split（seed
  0..1999 实测 241 组），已废弃。

纯 stdlib + Gazebo CLI；不碰 ROS 2 / gz-transport 直连。
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SIM_DIR)
SCENE_GEN = os.path.join(SIM_DIR, "tools", "scene_gen.py")
COLLECT = os.path.join(SIM_DIR, "tools", "collect.py")
DEFAULT_TEMPLATE = os.path.join(SIM_DIR, "worlds", "uav_rgb.sdf")
DISPLAY_VALUE = ":20"
XDG_RUNTIME_DIR = "/tmp/xdg"
SIM_SETTLE_S = 3.0   # sim-ready 后再稳 3 s 才发首条起飞指令（发布层偶发丢首包）
SPLIT_POLICY = ("geometry group: sha256(uavrgb.geom.v3:<gid>) % 100 -> "
                "<80 train; <90 val; else test")

_REG_LOCK = threading.Lock()
_PROC_GROUPS = {}   # pgid -> Popen，收尾时兜底 kill


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import split_manifest as sm  # noqa: E402  （路径插在常量之后，所以放这里）


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path, base=REPO_ROOT):
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gz_env(partition):
    env = dict(os.environ)
    env.update({"DISPLAY": DISPLAY_VALUE,
                "XDG_RUNTIME_DIR": XDG_RUNTIME_DIR,
                "GZ_PARTITION": partition,
                # 软件渲染（llvmpipe）默认每进程开满线程，多实例并发时互相抢核，
                # 实测 6 实例 RTF 0.21；限成 1 线程后 0.50（起飞 30 s 超时才够用）。
                "LP_NUM_THREADS": "1"})
    return env


def _register(proc):
    with _REG_LOCK:
        _PROC_GROUPS[proc.pid] = proc


def _unregister(proc):
    with _REG_LOCK:
        _PROC_GROUPS.pop(proc.pid, None)


def stop_group(proc, label, term_timeout=8.0):
    """对自己 setsid 起的进程组发 SIGTERM，超时 SIGKILL；返回是否已清空。"""
    if proc is None:
        return True
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=term_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    clean = False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            clean = True
            break
        time.sleep(0.1)
    if not clean:
        print(f"警告: {label} 进程组 {pgid} 收尾后仍有残留", flush=True)
    _unregister(proc)
    return clean


def _sweep_orphan_topics():
    """杀掉父进程已死的 gz-transport-topic 订阅者。

    collect.py 被 SIGKILL 时它起的 `gz topic -e` 子进程会掉到 init（ppid=1）并
    继续订阅；实测 30+ 个残留把 4 实例 RTF 压到 0.1（10 s/步），清掉后 1.4 s/步。
    只认 /uav/ 与 /model/x3 话题，避免误杀别的 gz topic。
    """
    try:
        r = subprocess.run(["ps", "-eo", "pid,ppid,args", "--no-headers"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != "1":
            continue
        pid_s, _ppid, cmd = parts
        if "gz-transport-topic" not in cmd:
            continue
        if "/uav/" not in cmd and "/model/x3" not in cmd:
            continue
        try:
            os.kill(int(pid_s), signal.SIGKILL)
        except (ValueError, ProcessLookupError, PermissionError):
            pass


def cleanup_all():
    with _REG_LOCK:
        procs = list(_PROC_GROUPS.values())
    for proc in procs:
        stop_group(proc, "cleanup")
    _sweep_orphan_topics()


def _tail(path, n=20):
    try:
        with open(path, "r", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def _start_sim(world, partition, log_path):
    log = open(log_path, "ab")  # 追加：重试时保留第一次失败的日志
    try:
        proc = subprocess.Popen(
            ["gz", "sim", "-s", "-r", world],
            env=gz_env(partition), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    finally:
        log.close()
    _register(proc)
    return proc


def _topic_list(partition):
    try:
        r = subprocess.run(["gz", "topic", "-l"], env=gz_env(partition),
                           capture_output=True, text=True, timeout=20)
        return r.stdout.split()
    except subprocess.TimeoutExpired:
        return []


def _has_frame(topic, partition):
    try:
        r = subprocess.run(
            ["gz", "topic", "-e", "-t", topic, "-n", "1", "--json-output"],
            env=gz_env(partition), capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def wait_ready(sim, partition, sim_log, timeout):
    """等 /uav/camera + /uav/depth 都真出过一帧，避免 collect 第一步读空。"""
    deadline = time.monotonic() + timeout
    last = "no topics"
    while time.monotonic() < deadline:
        rc = sim.poll()
        if rc is not None:
            raise RuntimeError(f"gz sim 提前退出 rc={rc}，日志 {sim_log}:\n{_tail(sim_log)}")
        topics = _topic_list(partition)
        if "/uav/camera" in topics and "/uav/depth" in topics:
            if _has_frame("/uav/camera", partition) and _has_frame("/uav/depth", partition):
                return
            last = "topics up but no frame yet"
        else:
            last = f"topics={len(topics)} (camera missing)"
        time.sleep(0.5)
    raise RuntimeError(f"gz sim 在 {timeout}s 内未就绪（{last}），日志 {sim_log}:\n"
                       f"{_tail(sim_log)}")


def _pump(stream, prefix, log_path):
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        for line in stream:
            log.write(line)
            log.flush()
            print(prefix + line.rstrip("\n"), flush=True)


@dataclass
class Task:
    instance: int
    episode: int
    scene_id: int
    seed: int
    split: str
    world: str
    scene_meta_path: str
    partition: str = field(default_factory=lambda: "uavrgb_" + uuid.uuid4().hex)
    world_sha256: str = ""
    scene_meta: dict = field(default_factory=dict)
    spawn_mode: str = "default"
    staging: str = ""
    final_dir: str = ""
    status: str = "pending"
    exit_code: int = None
    error: str = None
    steps_completed: int = 0
    validity: dict = None
    attempts: int = 0
    duration_s: float = 0.0
    ok: bool = False


def _popen_collect(task, args, env, log_path):
    staging = task.staging
    cmd = [sys.executable, COLLECT,
           "--steps", str(args.steps), "--hold", str(args.hold),
           "--out", staging,
           "--scene-id", str(task.scene_id), "--seed", str(task.seed),
           "--episode-id", str(task.scene_id), "--split", task.split,
           "--scene-meta", task.scene_meta_path]
    if args.policy != "teacher":
        cmd += ["--policy", args.policy]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    _register(proc)
    pump = threading.Thread(target=_pump,
                            args=(proc.stdout, f"[inst {task.instance}] ", log_path),
                            daemon=True)
    pump.start()
    return proc, pump


def _run_episode_once(task, args, staging_root, logs_dir):
    partition = task.partition   # 独立运行和不同 episode 也不能复用 transport 分区
    env = gz_env(partition)
    sim_log = os.path.join(logs_dir, f"sim_scene_{task.scene_id}.log")
    collect_log = os.path.join(logs_dir, f"collect_scene_{task.scene_id}.log")
    task.staging = os.path.join(staging_root, f"scene_{task.scene_id}")
    sim = collect = None
    pump = None
    try:
        print(f"[inst {task.instance}] scene {task.scene_id} ({task.split}) "
              f"启动 gz sim (partition={partition})", flush=True)
        sim = _start_sim(task.world, partition, sim_log)
        wait_ready(sim, partition, sim_log, args.sim_timeout)
        time.sleep(SIM_SETTLE_S)   # 等速度控制器就绪，降低首条起飞指令被丢的概率
        print(f"[inst {task.instance}] scene {task.scene_id} sim ready pid={sim.pid}；"
              f"开始 collect {args.steps} 步 hold={args.hold}s", flush=True)
        collect, pump = _popen_collect(task, args, env, collect_log)
        try:
            task.exit_code = collect.wait(timeout=args.episode_timeout)
        except subprocess.TimeoutExpired:
            task.error = f"collect 超时 {args.episode_timeout}s"
            stop_group(collect, f"collect scene {task.scene_id}")
            collect = None
            task.exit_code = -9
        if pump is not None:
            pump.join(timeout=10)
        if collect is not None:
            stop_group(collect, f"collect scene {task.scene_id}")
            collect = None
        manifest_path = os.path.join(task.staging, "episode_manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            task.steps_completed = int(m.get("num_steps", 0))
            task.validity = m.get("validity")
            if m.get("error") and not task.error:
                task.error = str(m["error"])
            # 三个 validity 是 AND：相机没出图时 rgb 写盘会返回 False，但行照写、步数照满，
            # 只看 episode_executed 会把它当成功搬进 train/（preprocess 随后在缺帧上 die）。
            validity = m.get("validity") or {}
            task.ok = (task.exit_code == 0 and task.steps_completed == args.steps
                       and all(validity.get(k) for k in
                               ("episode_executed", "camera_verified", "physics_verified")))
        else:
            if not task.error:
                task.error = "episode_manifest.json 缺失"
            task.ok = False
    except Exception as e:  # noqa: BLE001 - 单个 episode 失败不拖垮其它实例
        task.error = f"{type(e).__name__}: {e}"
        task.ok = False
    finally:
        stop_group(collect, f"collect scene {task.scene_id}")
        stop_group(sim, f"sim scene {task.scene_id}")


def run_episode(task, args, staging_root, logs_dir):
    """跑一个 episode；失败重启 sim 重试一次（world 由 seed 确定性生成，不换场景）。

    当前 collect.py 的常驻 publisher 偶发丢首条起飞指令（6 实例实测 2 次），
    属发布层抖动；重试只重启 sim，数据语义不变。
    """
    t0 = time.monotonic()
    for attempt in (1, 2):
        task.attempts = attempt
        _run_episode_once(task, args, staging_root, logs_dir)
        if task.ok or attempt == 2:
            break
        print(f"[inst {task.instance}] scene {task.scene_id} 失败，重启 sim 重试 "
              f"({attempt}/1)", flush=True)
        shutil.rmtree(task.staging, ignore_errors=True)
        task.error = None
        task.exit_code = None
        task.steps_completed = 0
        task.validity = None
    task.duration_s = time.monotonic() - t0
    task.status = "ok" if task.ok else "failed"
    line = (f"[inst {task.instance}] scene {task.scene_id} {task.status} "
            f"rc={task.exit_code} steps={task.steps_completed}/{args.steps} "
            f"attempts={task.attempts} {task.duration_s:.1f}s")
    if task.error:
        line += f" error={task.error}"
    print(line, flush=True)
    return task


def run_instance(args, tasks, staging_root, logs_dir):
    return [run_episode(t, args, staging_root, logs_dir) for t in tasks]


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _ruby_pids():
    r = subprocess.run(["ps", "-eo", "pid,comm", "--no-headers"],
                       capture_output=True, text=True)
    pids = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "ruby":
            pids.append(int(parts[0]))
    return pids


def _sweep_world_ruby(worlds_dir):
    """兜底：只杀命令行里带本次临时 world 路径的 gz sim，绝不误杀别的会话。"""
    r = subprocess.run(["ps", "-eo", "pid,args", "--no-headers"],
                       capture_output=True, text=True)
    killed = []
    for line in r.stdout.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        if "gz sim" in cmd and worlds_dir in cmd:
            try:
                pid = int(pid_s)
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    return killed


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances", type=int, default=4, help="并行实例数（默认 4）")
    p.add_argument("--episodes-per-instance", type=int, default=1,
                   help="每个实例顺序采的 episode 数（每个 episode 重启一次 sim）")
    p.add_argument("--steps", type=int, default=sm.DEFAULT_STEPS,
                   help="每 episode 步数（默认见 split_manifest.DEFAULT_STEPS=40）")
    p.add_argument("--hold", type=float, default=1.0,
                   help="每动作仿真时间秒（走满即发零速，与 RTF 无关）")
    p.add_argument("--out", required=True, help="汇总输出目录（必须为空/不存在）")
    p.add_argument("--base-seed", type=int, default=0,
                   help="scene_id = seed = base_seed + 扫描序号；给 --split 时只取该 split 的 seed")
    p.add_argument("--seeds", default="",
                   help="固定 seed 列表（逗号分隔或 a:b），优先于 base-seed/实例数派生")
    p.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="只采命中该 split 的几何组；seed 从 base-seed 起向后扫描")
    p.add_argument("--spawn-mode", choices=("default", "random"), default="default",
                   help="普通 episode 的出生位姿（random = y/yaw 随机化）")
    p.add_argument("--safe-spawn-fraction", type=float, default=0.0,
                   help="每 N 个 episode 有 1 个用 SAFE 出生位姿（T2 的 SAFE 数据来源）")
    p.add_argument("--policy", choices=("teacher", "forward"), default="teacher")
    p.add_argument("--template", default=DEFAULT_TEMPLATE)
    p.add_argument("--keep-worlds", action="store_true",
                   help="把生成的 SDF 保留到 <out>/worlds/ 便于复现")
    p.add_argument("--episode-timeout", type=float, default=600.0)
    p.add_argument("--sim-timeout", type=float, default=120.0)
    args = p.parse_args()

    if args.instances < 1:
        p.error("--instances must be >= 1")
    if args.episodes_per_instance < 1:
        p.error("--episodes-per-instance must be >= 1")
    if args.steps < 1:
        p.error("--steps must be >= 1")
    # NaN <= 0 是 False，只判 <=0 会让 --hold nan 一路跑进 sleep 里挂死。
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold must be finite and > 0")
    if args.episode_timeout <= 0 or args.sim_timeout <= 0:
        p.error("timeouts must be > 0")
    if not (0.0 <= args.safe_spawn_fraction <= 1.0):
        p.error("--safe-spawn-fraction 必须在 [0,1]")
    # 实现是「每 N 集 1 个」，不是任意比例；0.4/0.6 等会静默错配成 1/2。
    if args.safe_spawn_fraction > 0:
        n = 1.0 / args.safe_spawn_fraction
        if abs(n - round(n)) > 1e-9:
            p.error("--safe-spawn-fraction 必须是 1/N（0.5/0.333/0.25/0.2/0.1…）")
    for path, name in ((SCENE_GEN, "scene_gen.py"), (COLLECT, "collect.py"),
                       (args.template, "template")):
        if not os.path.isfile(path):
            p.error(f"{name} not found: {path}")

    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        p.error(f"--out {out} 非空；请用新目录（或先清空）")
    os.makedirs(out, exist_ok=True)
    _sweep_orphan_topics()   # 上一轮被 SIGKILL 的订阅者会拖垮本轮 RTF
    try:
        os.makedirs(XDG_RUNTIME_DIR, mode=0o700, exist_ok=True)
        os.chmod(XDG_RUNTIME_DIR, 0o700)
    except OSError as e:
        print(f"警告: 无法准备 XDG_RUNTIME_DIR {XDG_RUNTIME_DIR}: {e}", flush=True)

    worlds_dir = (os.path.join(out, "worlds") if args.keep_worlds
                  else tempfile.mkdtemp(prefix="uavrgb_worlds_"))
    os.makedirs(worlds_dir, exist_ok=True)
    logs_dir = os.path.join(out, "logs")
    staging_root = os.path.join(out, ".staging")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(staging_root, exist_ok=True)

    target = args.instances * args.episodes_per_instance
    scene_ids = []
    if args.seeds:
        for part in args.seeds.split(","):
            part = part.strip()
            if not part:
                continue
            a, _, b = part.partition(":")
            scene_ids.extend(range(int(a), int(b) if b else int(a) + 1))
        if args.split:
            scene_ids = [s for s in scene_ids
                         if sm.split_for_gid(sm.geometry_gid(
                             sm.sg.sample_pillars(s)[0])) == args.split]
    elif args.split:
        # 只取目标 split：test 只占几何组的 ~10%，顺序取 base-seed 起的前 N 个
        # 会全是 train。这里按几何 gid 的固定 split 过滤，扫描上限防止写错 split
        # 后无限循环。
        seed = args.base_seed
        while len(scene_ids) < target:
            if seed > args.base_seed + 100_000:
                p.error(f"从 base-seed={args.base_seed} 扫了 100000 个 seed 仍凑不齐 "
                        f"{target} 个 split={args.split} 的场景")
            pillars, _, _ = sm.sg.sample_pillars(seed)
            if sm.split_for_gid(sm.geometry_gid(pillars)) == args.split:
                scene_ids.append(seed)
            seed += 1
    else:
        scene_ids = list(range(args.base_seed,
                               args.base_seed + target))
    if not scene_ids:
        p.error("没有可采的 scene_id（--seeds 为空或与 --split 无交集）")
    # SAFE 出生按固定 stride 铺开，保证 train/val/test 都能分到（T3 四类 recall 需要）。
    safe_stride = (max(1, round(1.0 / args.safe_spawn_fraction))
                   if args.safe_spawn_fraction > 0 else 0)
    tasks = []
    for k, scene_id in enumerate(scene_ids):
        i = k % args.instances
        spawn_mode = ("safe" if safe_stride and k % safe_stride == 0 else args.spawn_mode)
        tasks.append(Task(
            instance=i, episode=k // args.instances, scene_id=scene_id, seed=scene_id,
            split="",  # 世界生成后按几何组填，见下
            spawn_mode=spawn_mode,
            world=os.path.join(worlds_dir, f"scene_{scene_id}.sdf"),
            scene_meta_path=os.path.join(worlds_dir, f"scene_{scene_id}.json")))
    if len({t.scene_id for t in tasks}) != len(tasks):
        p.error("scene_id 冲突（base-seed 派生异常）")

    started_at = _utc_now()
    t_start = time.monotonic()
    try:
        # 1) 先全部生成世界；任何失败都不起 sim。
        for t in tasks:
            r = subprocess.run(
                [sys.executable, SCENE_GEN, "--seed", str(t.seed),
                 "--scene-id", str(t.scene_id), "--out", t.world,
                 "--spawn-mode", t.spawn_mode, "--template", args.template],
                capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                print(r.stdout, flush=True)
                print(r.stderr, file=sys.stderr, flush=True)
                print(f"scene_gen 失败 scene_id={t.scene_id}，中止（未启动任何 sim）",
                      file=sys.stderr, flush=True)
                return 1
            with open(t.scene_meta_path, "r", encoding="utf-8") as f:
                t.scene_meta = json.load(f)
            t.world_sha256 = sha256_file(t.world)
            # split 只能在世界生成后定：它取决于几何，不取决于 scene_id。
            t.split = sm.split_for_gid(sm.geometry_gid(t.scene_meta["pillars"]))
            print(f"[scene] scene_id={t.scene_id} split={t.split} "
                  f"pillars={t.scene_meta['pillar_count']} "
                  f"sha256={t.world_sha256[:16]}…", flush=True)
        t_worlds = time.monotonic()

        # 2) N 个 worker 并发；每个 worker 顺序跑自己的 episodes。
        by_instance = [[t for t in tasks if t.instance == i]
                       for i in range(args.instances)]
        results = []
        executor = ThreadPoolExecutor(max_workers=args.instances)
        futures = [executor.submit(run_instance, args, group, staging_root, logs_dir)
                   for group in by_instance]
        try:
            for fut in futures:
                results.extend(fut.result())
        except KeyboardInterrupt:
            print("\n中断：清理所有 gz sim / collect 进程组…", flush=True)
            cleanup_all()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            cleanup_all()
        t_collect = time.monotonic()

        # 3) 汇总：成功 -> <out>/<split>/<scene_id>，失败 -> <out>/failed/scene_<id>
        ok_count = 0
        total_steps = 0
        for t in tasks:
            if t.ok:
                final = os.path.join(out, t.split, str(t.scene_id))
                os.makedirs(os.path.dirname(final), exist_ok=True)
                os.replace(t.staging, final)
                shutil.copyfile(t.scene_meta_path,
                                os.path.join(final, "scene.json"))
                t.final_dir = _rel(final, out)
                ok_count += 1
                total_steps += t.steps_completed
            else:
                if os.path.isdir(t.staging):
                    dest = os.path.join(out, "failed", f"scene_{t.scene_id}")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.replace(t.staging, dest)
                    t.final_dir = _rel(dest, out)
        try:
            os.rmdir(staging_root)
        except OSError:
            pass
        t_end = time.monotonic()

        # 4) manifest
        splits = {"train": [], "val": [], "test": []}
        for t in tasks:
            if t.ok:
                splits[t.split].append(t.scene_id)
        collection_wall = max(t_collect - t_worlds, 1e-9)
        end_to_end_wall = max(t_end - t_start, 1e-9)
        per_instance = [t.duration_s for t in tasks if t.ok and t.duration_s > 0]
        manifest = {
            "schema": "navlab.parallel_collection.v1",
            "tool": "sim/tools/collect_parallel.py",
            "command": sys.argv,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "base_seed": args.base_seed,
            "instances": args.instances,
            "episodes_per_instance": args.episodes_per_instance,
            "steps": args.steps,
            "hold_s": args.hold,
            "policy": args.policy,
            "split": args.split,
            "seeds": args.seeds,
            "spawn_mode": args.spawn_mode,
            "safe_spawn_fraction": args.safe_spawn_fraction,
            "safe_episodes_requested": sum(1 for t in tasks if t.spawn_mode == "safe"),
            "unique_geometry": len({sm.geometry_gid(t.scene_meta["pillars"])
                                    for t in tasks if t.scene_meta}),
            "split_policy": SPLIT_POLICY,
            "template": _rel(os.path.abspath(args.template)),
            "keep_worlds": bool(args.keep_worlds),
            "splits": splits,
            "episodes_requested": len(tasks),
            "episodes_ok": ok_count,
            "episodes_failed": len(tasks) - ok_count,
            "steps_requested_total": len(tasks) * args.steps,
            "steps_completed_total": total_steps,
            "world_gen_s": round(t_worlds - t_start, 3),
            "collection_wall_s": round(collection_wall, 3),
            "end_to_end_wall_s": round(end_to_end_wall, 3),
            "collection_throughput_steps_per_s": round(
                total_steps / collection_wall, 4) if total_steps else 0.0,
            "end_to_end_throughput_steps_per_s": round(
                total_steps / end_to_end_wall, 4) if total_steps else 0.0,
            "per_instance_rate_steps_per_s": round(
                sum(t.steps_completed / t.duration_s for t in tasks
                    if t.ok and t.duration_s > 0) / len(per_instance), 4)
            if per_instance else 0.0,
            "throughput_12_instances_linear": round(
                (total_steps / collection_wall) * 12.0 / args.instances, 4)
            if total_steps else 0.0,
            "throughput_note": ("按本次实测端到端吞吐线性外推到 12 实例，未实测；"
                                "hold 与步数不同时不能直接与历史数字比较"),
            "tasks": [{
                "instance": t.instance,
                "episode": t.episode,
                "partition": t.partition,
                "scene_id": t.scene_id,
                "seed": t.seed,
                "split": t.split,
                "spawn_mode": t.spawn_mode,
                "status": t.status,
                "ok": t.ok,
                "attempts": t.attempts,
                "world": (os.path.join("worlds", f"scene_{t.scene_id}.sdf")
                          if args.keep_worlds else None),
                "world_sha256": t.world_sha256,
                "scene": t.scene_meta,
                "steps_requested": args.steps,
                "steps_completed": t.steps_completed,
                "validity": t.validity,
                "exit_code": t.exit_code,
                "duration_s": round(t.duration_s, 3),
                "out_dir": t.final_dir,
                "collect_log": _rel(os.path.join(
                    logs_dir, f"collect_scene_{t.scene_id}.log"), out),
                "sim_log": _rel(os.path.join(
                    logs_dir, f"sim_scene_{t.scene_id}.log"), out),
                "error": t.error,
            } for t in tasks],
        }
        _write_json(os.path.join(out, "collection_manifest.json"), manifest)

        # 5) 收尾检查：本次 world 路径相关的 ruby + 全局 ruby 清单
        killed = _sweep_world_ruby(worlds_dir)
        if killed:
            print(f"[cleanup] 兜底杀掉带本次 world 路径的 ruby: {killed}", flush=True)
        time.sleep(0.5)
        ruby = _ruby_pids()
        print(f"[cleanup] 残留 ruby 进程: {len(ruby)}"
              + (f" {ruby}" if ruby else "（无）"), flush=True)

        print("", flush=True)
        print(f"== 采集完成: {ok_count}/{len(tasks)} episodes, "
              f"{total_steps}/{len(tasks) * args.steps} steps", flush=True)
        print(f"   并行窗口 {args.instances} 实例 x {args.episodes_per_instance} ep: "
              f"world_gen={t_worlds - t_start:.1f}s collect={collection_wall:.1f}s "
              f"end_to_end={end_to_end_wall:.1f}s", flush=True)
        print(f"   吞吐: {manifest['collection_throughput_steps_per_s']:.3f} 步/s "
              f"（{args.instances} 实例端到端）；"
              f"单实例均值 {manifest['per_instance_rate_steps_per_s']:.3f} 步/s", flush=True)
        print(f"   12 实例线性外推: "
              f"{manifest['throughput_12_instances_linear']:.3f} 步/s（未实测）", flush=True)
        print(f"   输出: {out}", flush=True)
        for split in ("train", "val", "test"):
            if splits[split]:
                print(f"     {split}: scene_id={splits[split]}", flush=True)
        if manifest["episodes_failed"]:
            print(f"     失败: {manifest['episodes_failed']} 个，见 {out}/failed/",
                  flush=True)
        print(f"   manifest: {os.path.join(out, 'collection_manifest.json')}",
              flush=True)
        return 0 if manifest["episodes_failed"] == 0 else 1
    finally:
        cleanup_all()
        if not args.keep_worlds:
            shutil.rmtree(worlds_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
