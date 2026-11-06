#!/usr/bin/env python3
"""CfSitlSystem 真机 Gazebo 冒烟：起 gz（headless、软件渲染）→ 直接走话题验证插件行为 → 写 JSON 报告。

检查项（每种相机配置各跑一次）：插件加载、1 ms 步长、state 200 Hz、arm_takeoff→READY、
FORWARD 3 s 的 +x 机体位移（0.3 m/s）、YAW_LEFT 的 z 正向转角、撞柱 contact、相机帧率、
RTF；并把两种配置（aux 相机 / 无 aux 相机）的 RTF 与「hold=1 s 一步的墙钟估计」并排写出。

长跑守护模式（任务书）：
  setsid nohup $UAV_RGB_PYTHON sim/cf_sitl/gz/tests/smoke_gz.py --out <out> \
      > <out>/run.log 2>&1 < /dev/null & echo $! > <out>/run.pid
结束时写 <out>/DONE；中途停止用 kill -- -<PGID>。gz 子进程都在各自新会话里，退出时逐个 stop。

依赖：scene_gen.py（生成 E1 世界）、collect_parallel.gz_env/_start_sim/wait_ready（与采集完全同一套
环境与启动方式）、gz_io.TopicSub（gz topic -e --json-output）、cf_sitl_exec（发布/状态索引）。
参数文件：sim/dynamics/e1_mini.json 不存在时，把硬件草案复制进 <out>/fake_root/sim/dynamics/
并把 UAVRGB_REPO_ROOT 指过去（插件与 collect 两边同一条解析规则）。
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TOOLS = os.path.join(REPO, "sim", "tools")
sys.path.insert(0, TOOLS)

# ---------------------------------------------------------------- 配置（改这里，别改函数体）
RENDER_BACKEND = "software"   # 本机 gpu/EGL 路径会 segfault（C 报告），采集也用软件渲染
SCENE_SEED = 3                # scene_gen seed：pillar_0 在 (1.5, 0.02) 正前方 2 m，撞柱测试用
CAMERA_RATE_HZ = 10.0         # 与 E1 模板默认一致：hold 1 s 只消费 1 帧，RGB/深度同频配对
# (名字, --no-aux-cameras, 解除 RTF 上限)。模板把 real_time_factor 锁在 1.0（与 X3 模板同口径）；
# 第三种配置只为测物理 1 kHz + 10 Hz 渲染的 RTF 余量，不改模板。
CONFIGS = (("aux", False, False), ("noaux", True, False),
           ("aux_unthrottled", False, True), ("noaux_unthrottled", True, True))
AIM_TOL_DEG = 3.0             # 撞柱前对准 pillar_0 的航向容差：2 m 外 3° 偏差 = 0.1 m，仍在 0.3 m 命中带内
AIM_PULSE_S = 0.1             # 对准用的偏航脉冲：0.6 rad/s × 0.1 s ≈ 3.4°
STATE_WINDOW_S = 2.0          # 测 state 频率/步长的仿真时间窗：200 Hz × 2 s = 400 帧，足够估计
STEP_TOL = 1e-6               # 1 ms 步长判据：相邻 state 的 gz 时间差 ≈ 5 ms（每 5 步发一次）
READY_TIMEOUT_S = 90.0        # arm_takeoff→READY 墙钟上限：RTF 0.05 下 2 m 爬升约 80 s
TAKEOFF_RETRIES = 3           # 起飞 FAILSAFE 的 reset 重试上限；只为拿到后续指标，次数记进 takeoff_failsafes（期望 0）
SETTLE_SIM_S = 3.0            # 重试前等机体落地的仿真秒：FAILSAFE 开环漂升最多 0.5 s，自由落体 <1 s
FORWARD_S = 3.0               # FORWARD 时长：0.3 m/s × 3 s = 0.9 m，位移远大于悬停漂移
FORWARD_RANGE_M = (0.5, 1.3)  # 位移容差：起步加速 + 到期刹车 → 名义 0.9 m 上下各留 0.4
YAW_S = 2.0                   # YAW_LEFT 时长：0.6 rad/s × 2 s ≈ 69°
YAW_MIN_DEG = 30.0            # 转角下限：起始滞后也至少转过一半
CONTACT_TIMEOUT_SIM_S = 20.0  # 朝 pillar_0 前进直到 contact 的仿真时间上限（2 m / 0.3 m/s ≈ 7 s）
TOPIC_WAIT_S = 120.0          # 等 /uav/camera、state 首帧：软件渲染冷启动可达 60 s
SIM_START_TIMEOUT_S = 180.0   # collect_parallel.wait_ready 的上限
HOLD_S = 1.0                  # 「每步墙钟」估计用的 hold
INFER_OVERHEAD_S = 0.25       # 每步除动作窗口外的 I/O + 解码 + teacher 规划开销（collect 实测量级）


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_params(out):
    """参数单源缺失时用草案顶上，返回要设置的 UAVRGB_REPO_ROOT。"""
    real = os.path.join(REPO, "sim", "dynamics", "e1_mini.json")
    if os.path.isfile(real):
        return REPO, real, "repo"
    draft = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"),
                         "uav_rgb", "tmp", "research-hw", "e1_mini_params_draft.json")
    root = os.path.join(out, "fake_root")
    dst = os.path.join(root, "sim", "dynamics", "e1_mini.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(draft, dst)
    return root, dst, "draft"


def gen_world(out, name, no_aux, unthrottled=False):
    world = os.path.join(out, f"world_{name}.sdf")
    cmd = [sys.executable, os.path.join(TOOLS, "scene_gen.py"), "--vehicle", "e1_mini",
           "--seed", str(SCENE_SEED), "--out", world, "--appearance", "v1",
           "--camera-rate", f"{CAMERA_RATE_HZ:g}"]
    if no_aux:
        cmd.append("--no-aux-cameras")
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    if unthrottled:
        with open(world, encoding="utf-8") as f:
            text = f.read()
        assert text.count("<real_time_factor>1.0</real_time_factor>") == 1
        with open(world, "w", encoding="utf-8") as f:
            f.write(text.replace("<real_time_factor>1.0</real_time_factor>",
                                 "<real_time_factor>0</real_time_factor>"))
    with open(world[:-4] + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    return world, meta


def cpu_seconds(pid):
    """gz 进程组（含子进程）的累计 CPU 秒：/proc/<pid>/stat 的 utime+stime+cutime+cstime。"""
    total = 0.0
    for p in subprocess.run(["pgrep", "-g", str(pid)], capture_output=True, text=True).stdout.split():
        try:
            with open(f"/proc/{p}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            total += sum(int(fields[i]) for i in (11, 12)) / os.sysconf("SC_CLK_TCK")
        except (OSError, IndexError, ValueError):
            pass
    return total


class Sim:
    """一个 gz 实例 + 订阅 + 发布；with 退出时无论如何 stop_group。"""

    def __init__(self, world, out, name):
        import collect_parallel as cp
        import gz_io
        import cf_sitl_exec as ce
        self.cp, self.gz_io, self.ce = cp, gz_io, ce
        self.partition = f"uavrgb_smoke_{os.getpid()}_{name}"
        self.log_path = os.path.join(out, f"sim_{name}.log")
        self.env = cp.gz_env(self.partition)
        # 本进程里的 gz topic 订阅/发布子进程要和 sim 同一个 partition
        os.environ.update({k: self.env[k] for k in ("GZ_PARTITION", "XDG_RUNTIME_DIR")})
        self.proc = cp._start_sim(world, self.partition, self.log_path)
        self.subs = {}
        self.pub = None

    def __enter__(self):
        try:
            t0 = time.monotonic()
            self.cp.wait_ready(self.proc, self.partition, self.log_path, SIM_START_TIMEOUT_S)
            self.boot_s = time.monotonic() - t0
            for name, topic, kw in (("state", self.ce.STATE_TOPIC, {"stamp_fn": self.ce.state_stamp_ns}),
                                    ("pose", "/model/e1_mini/pose", {}),
                                    ("rgb", "/uav/camera", {}), ("depth", "/uav/depth", {}),
                                    ("contact", "/uav/contact", {})):
                self.subs[name] = self.gz_io.TopicSub(topic, buf=800, **kw)
            deadline = time.monotonic() + TOPIC_WAIT_S
            while time.monotonic() < deadline and (self.subs["state"].latest() is None
                                                   or self.subs["rgb"].latest() is None):
                time.sleep(0.1)
            if self.subs["state"].latest() is None:
                raise RuntimeError("state 话题无帧：插件没加载或 cf_sitl_init 失败，见 " + self.log_path)
            self.pub = self.ce.open_publisher()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        try:
            if self.pub:
                self.pub.action(0, self.sim_t() + 3600)
                self.pub.close()
        finally:
            for s in self.subs.values():
                s.close()
            self.cp.stop_group(self.proc, "smoke sim")

    # -- 读数
    def state(self):
        m = self.subs["state"].latest()
        return m["data"] if m else None

    def sim_t(self):
        st = self.state()
        return float(st[self.ce.IDX_SIM_T]) if st else 0.0

    def pos_yaw(self):
        st = self.state()
        x, y, z = st[self.ce.IDX_POS:self.ce.IDX_POS + 3]
        w, qx, qy, qz = st[self.ce.IDX_QUAT:self.ce.IDX_QUAT + 4]
        yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        return (x, y, z), yaw

    def wait_sim(self, seconds):
        """等仿真时间走过 seconds（墙钟兜底 20×）。返回 (sim_dt, wall_dt)。"""
        t0, w0 = self.sim_t(), time.monotonic()
        while self.sim_t() - t0 < seconds:
            if time.monotonic() - w0 > 20 * seconds + 10:
                raise RuntimeError(f"仿真时间 {seconds}s 在墙钟 {20 * seconds + 10:.0f}s 内没走完")
            time.sleep(0.01)
        return self.sim_t() - t0, time.monotonic() - w0

    def action(self, a, seconds):
        self.pub.action(a, self.sim_t() + seconds)
        return self.wait_sim(seconds)


def check_state_rate(sim):
    frames = [m["data"][sim.ce.IDX_SIM_T] for _, m in sim.subs["state"].frames()]
    t_end = frames[-1]
    window = [t for t in frames if t >= t_end - STATE_WINDOW_S]
    dts = [b - a for a, b in zip(window, window[1:])]
    hz = (len(window) - 1) / (window[-1] - window[0]) if len(window) > 2 else None
    # 插件每 state_every_=5 步发一次 → 相邻帧 5 ms；dt 的余数检验 1 ms 整步
    step_ok = all(abs(d - round(d / 0.001) * 0.001) < STEP_TOL for d in dts) and dts and min(dts) > 0
    return {"state_hz": hz, "state_dt_ms_min": min(dts) * 1e3 if dts else None,
            "state_dt_ms_max": max(dts) * 1e3 if dts else None, "step_1ms_consistent": bool(step_ok)}


def check_camera_rate(sim, seconds=3.0):
    import gz_io
    n0 = len(sim.subs["rgb"].frames())
    t0 = sim.sim_t()
    sim.wait_sim(seconds)
    fr = sim.subs["rgb"].frames()
    stamps = sorted(gz_io.header_stamp_ns(m) / 1e9 for _, m in fr if gz_io.header_stamp_ns(m) / 1e9 >= t0)
    dstamps = sorted(gz_io.header_stamp_ns(m) / 1e9 for _, m in sim.subs["depth"].frames()
                     if gz_io.header_stamp_ns(m) / 1e9 >= t0)
    hz = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 else None
    pair = max((min(abs(s - d) for d in dstamps) for s in stamps), default=None) if dstamps else None
    return {"camera_hz": hz, "rgb_frames": len(stamps), "new_frames_in_buffer": len(fr) - n0,
            "depth_pair_max_ms": pair * 1e3 if pair is not None else None}


def aim_at(sim, target_xy):
    """原地偏航到朝向 target_xy（容差 AIM_TOL_DEG），用短脉冲避免来回越过。"""
    w0 = time.monotonic()
    while time.monotonic() - w0 < 120:
        (x, y, _), yaw = sim.pos_yaw()
        bearing = math.degrees(math.atan2(target_xy[1] - y, target_xy[0] - x))
        err = (bearing - yaw + 180) % 360 - 180
        if abs(err) <= AIM_TOL_DEG:
            return err
        sim.action(2 if err > 0 else 3, AIM_PULSE_S if abs(err) < 15 else 0.3)
    raise RuntimeError("对准超时")


def run_config(out, name, no_aux, unthrottled, repo_root):
    world, meta = gen_world(out, name, no_aux, unthrottled)
    os.environ["UAVRGB_REPO_ROOT"] = repo_root
    os.environ["UAVRGB_RENDER_BACKEND"] = RENDER_BACKEND
    r = {"config": name, "no_aux_cameras": no_aux, "rtf_unthrottled": unthrottled, "world": world,
         "camera_rate_hz": meta.get("camera_rate_hz")}
    with Sim(world, out, name) as sim:
        r["boot_s"] = round(sim.boot_s, 1)
        r.update(check_state_rate(sim))
        with open(sim.log_path, encoding="utf-8", errors="replace") as f:
            simlog = f.read()
        # 默认 verbosity 下 gzmsg 不打印；state 有帧本身就证明 Configure 成功且 PreUpdate 在跑
        r["plugin_init_error"] = "cf_sitl_init 失败" in simlog or "CfSitlSystem] 模型" in simlog
        r["plugin_loaded"] = sim.state() is not None and not r["plugin_init_error"]
        r["plugin_errors"] = [ln for ln in simlog.splitlines() if "[CfSitlSystem]" in ln][:5]
        st = sim.state()
        r["exec_state_initial"] = int(st[sim.ce.IDX_EXEC])
        # reset + arm_takeoff → READY；起飞阶段进 FAILSAFE 则 reset 重试并计数（期望 0 次）
        r["takeoff_failsafes"] = 0
        for attempt in range(1 + TAKEOFF_RETRIES):
            sim.pub.cmd("reset 0")
            time.sleep(0.3)
            if attempt:   # 等 reset 停桨后机体落回地面
                sim.wait_sim(SETTLE_SIM_S)
            t_arm = sim.sim_t()
            sim.pub.cmd(f"arm_takeoff {sim.ce.TAKEOFF_HEIGHT_M:g}")
            w0 = time.monotonic()
            while int(sim.state()[sim.ce.IDX_EXEC]) not in (sim.ce.EXEC_READY, sim.ce.EXEC_FAILSAFE):
                if time.monotonic() - w0 > READY_TIMEOUT_S:
                    raise RuntimeError(f"{READY_TIMEOUT_S}s 内未 READY，exec={sim.state()[sim.ce.IDX_EXEC]}")
                time.sleep(0.02)
            if int(sim.state()[sim.ce.IDX_EXEC]) == sim.ce.EXEC_READY:
                break
            r["takeoff_failsafes"] += 1
            dump = os.path.join(out, f"state_{name}_failsafe{attempt + 1}.jsonl")   # 最近 800 帧 state，供排查
            with open(dump, "w", encoding="utf-8") as f:
                for _, m in sim.subs["state"].frames():
                    f.write(json.dumps(m["data"]) + "\n")
            log(f"配置 {name}: 起飞进入 FAILSAFE（第 {attempt + 1} 次），state 帧见 {dump}")
            if attempt == TAKEOFF_RETRIES:
                raise RuntimeError(f"起飞连续 {attempt + 1} 次进入 FAILSAFE")
        (x0, y0, z0), yaw0 = sim.pos_yaw()
        r["takeoff"] = {"sim_s": round(sim.sim_t() - t_arm, 2), "wall_s": round(time.monotonic() - w0, 1),
                        "z_at_ready": round(z0, 3)}
        sim.wait_sim(1.0)   # 悬停稳定
        # YAW_LEFT 2 s：z 正向
        (_, _, _), yaw_a = sim.pos_yaw()
        sim.action(2, YAW_S)
        sim.wait_sim(0.5)
        (_, _, _), yaw_b = sim.pos_yaw()
        dyaw = (yaw_b - yaw_a + 180) % 360 - 180
        r["yaw_left"] = {"dyaw_deg": round(dyaw, 1), "ok": dyaw > YAW_MIN_DEG}
        # 转到朝 -x（背对柱阵），FORWARD 3 s 测位移
        w0 = time.monotonic()
        while abs(abs(sim.pos_yaw()[1]) - 180) > 12 and time.monotonic() - w0 < 90:
            sim.action(2, 0.3)   # 每次约 10°，容差 12° 不会来回越过
        sim.action(0, 1.0)
        (x1, y1, z1), yaw1 = sim.pos_yaw()
        cpu0 = cpu_seconds(sim.proc.pid)
        sim_dt, wall_dt = sim.action(1, FORWARD_S)
        cpu1 = cpu_seconds(sim.proc.pid)
        sim.wait_sim(0.5)
        (x2, y2, z2), yaw2 = sim.pos_yaw()
        c, s = math.cos(math.radians(yaw1)), math.sin(math.radians(yaw1))
        fwd = (x2 - x1) * c + (y2 - y1) * s
        lat = -(x2 - x1) * s + (y2 - y1) * c
        r["forward"] = {"body_x_m": round(fwd, 3), "lateral_m": round(lat, 3),
                        "dz_m": round(z2 - z1, 3), "speed_mps": round(fwd / FORWARD_S, 3),
                        "ok": FORWARD_RANGE_M[0] <= fwd <= FORWARD_RANGE_M[1] and abs(lat) < 0.3,
                        "rtf": round(sim_dt / wall_dt, 3),
                        # gz 进程在这 3 s 仿真里用掉的 CPU 核数（CPU 秒 / 墙钟秒）
                        "gz_cpu_cores": round((cpu1 - cpu0) / wall_dt, 2)}
        r["rtf_forward"] = r["forward"]["rtf"]
        r["gz_cpu_cores"] = r["forward"]["gz_cpu_cores"]
        r.update(check_camera_rate(sim))
        # 对准 pillar_0，前进直到 contact
        r["aim_err_deg"] = round(aim_at(sim, (meta["pillars"][0]["x"], meta["pillars"][0]["y"])), 2)
        sim.action(0, 1.0)
        # 按帧序号而不是长度做基线：地面接触会以 100 Hz 刷满 deque(maxlen)，长度基线会失效
        seq0 = max((seq for seq, _ in sim.subs["contact"].frames()), default=0)
        t0, w0 = sim.sim_t(), time.monotonic()
        sim.pub.action(1, t0 + CONTACT_TIMEOUT_SIM_S)
        hit = False
        while sim.sim_t() - t0 < CONTACT_TIMEOUT_SIM_S:
            fr = sim.subs["contact"].frames()
            if any(seq > seq0 and isinstance(m, dict) and m.get("contact") for seq, m in fr):
                hit = True
                break
            time.sleep(0.02)
        sim.pub.action(0, sim.sim_t() + 3600)
        (xc, yc, zc), _ = sim.pos_yaw()
        r["contact"] = {"fired": hit, "sim_s": round(sim.sim_t() - t0, 2),
                        "pos": [round(xc, 3), round(yc, 3), round(zc, 3)],
                        "pillar_0": [meta["pillars"][0]["x"], meta["pillars"][0]["y"]],
                        "rtf": round((sim.sim_t() - t0) / (time.monotonic() - w0), 3)}
        st = sim.state()
        r["late_actions"] = int(st[sim.ce.IDX_LATE])
        r["step_errors"] = int(st[sim.ce.IDX_ERRORS])
        r["exec_state_final"] = int(st[sim.ce.IDX_EXEC])
    rtf = r["rtf_forward"]
    r["est_step_wall_s_hold1"] = round(HOLD_S / rtf + INFER_OVERHEAD_S, 2) if rtf else None
    r["ok"] = bool(r["plugin_loaded"] and r["step_1ms_consistent"] and r["yaw_left"]["ok"]
                   and r["forward"]["ok"] and r["contact"]["fired"] and r["step_errors"] == 0
                   and r["takeoff_failsafes"] == 0
                   and r["state_hz"] and abs(r["state_hz"] - 200) < 10)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    p.add_argument("--repeat", type=int, default=1, help="每种配置重复几次（查偶发问题，如起飞 FAILSAFE）")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    repo_root, params, src = ensure_params(args.out)
    log(f"参数: {params}（{src}）  UAVRGB_REPO_ROOT={repo_root}")
    import collect_parallel as cp
    plugin_dir = cp.find_plugin_dir()
    log(f"插件目录: {plugin_dir}")
    if not plugin_dir:
        raise SystemExit("找不到 libCfSitlSystem.so")
    report = {"params": params, "params_source": src, "plugin_dir": plugin_dir, "configs": []}
    want = set(args.configs.split(","))
    runs = [(f"{name}_r{i + 1}" if i else name, no_aux, unthrottled)
            for name, no_aux, unthrottled in CONFIGS if name in want for i in range(args.repeat)]
    for name, no_aux, unthrottled in runs:
        log(f"== 配置 {name}（no_aux={no_aux} unthrottled={unthrottled}）")
        try:
            r = run_config(args.out, name, no_aux, unthrottled, repo_root)
        except Exception as e:  # noqa: BLE001 - 一种配置失败不影响另一种的结论
            r = {"config": name, "ok": False, "error": f"{type(e).__name__}: {e}"}
            log(f"配置 {name} 失败: {r['error']}")
        report["configs"].append(r)
        log(json.dumps(r, ensure_ascii=False))
        with open(os.path.join(args.out, "smoke.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    left = subprocess.run(["pgrep", "-af", "gz sim|ruby"], capture_output=True, text=True).stdout.strip()
    report["leftover_processes"] = left
    with open(os.path.join(args.out, "smoke.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write("ok\n" if all(c.get("ok") for c in report["configs"]) else "fail\n")
    log("DONE")
    return 0 if all(c.get("ok") for c in report["configs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
