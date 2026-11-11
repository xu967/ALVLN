#!/usr/bin/env python3
"""Gazebo 插件读参数的 A/B 探针（任务卡 W1-10）：起真的 gz sim，用遥测看 CfSitlSystem 实际按哪份参数在算。

背景：插件只读世界 SDF 的 <params>（sim/cf_sitl/gz/CfSitlSystem.cc Configure）。collect.py --cf-sitl-params 以前只改 Python 侧
的指纹（manifest 的 dynamics_profile / params_sha256），物理仍按 sim/dynamics/e1_mini.json（电量 50%），两边会静默不一致。
修复（提交 b436c90）：scene_gen.py --cf-sitl-params 把 <params> 写成副本的绝对路径，collect_parallel / eval_closed_loop 透传。
本探针不靠读代码，用遥测证实这一点。

做法：两种世界各起 N 个 gz sim（每个用一个 reset episode 号）：
  default  不给 --cf-sitl-params 生成（= 第二轮对照臂的世界：<params> 是仓库默认路径，电量 50%）
  copy     给满电副本生成
每个 gz sim：
  ① gz sim -s -r -v 3 起世界，-v 3 让插件的 gzmsg「[CfSitlSystem] 就绪: ... params=<路径>」进日志；
  ② 订阅 /e1_mini/state，读首帧 = 静止电压与悬停油门（DISARMED，还没解锁，不受负载影响）；
  ③ CfSitlExecutor.start()（reset → arm_takeoff → READY），读 READY 时刻的 V_bus、悬停油门、油门杆量与 SITL 时钟
     （SITL 时钟从 reset 归零算起，所以 READY 时刻的读数 = 解锁到就绪的用时）。
判据（预先定，见 judge_probe）：
  G1 每个 gz sim 都起飞到 READY；
  G2 copy 的静止电压中位数比 default 高 MIN_REST_V_GAP 以上（预期约 4.20 − 3.8x ≈ 0.38 V；电量 50% → 100%）；
  G3 gz 日志里读到 params= 行时：copy 的必须是副本路径、default 的必须是仓库默认路径；读不到该行记「未核实」，不单独判败；
  G4 state[39] 的库版本与副本指纹里的 lib<N> 一致。
READY 用时与悬停油门只报告，不进判据（没有预注册的预期；CPU 试探里 50% READY 约 7.3–8.1 s、悬停 0.89，100% 约 9.6–10.3 s、0.79，未核实可复现）。

用法（需要机器锁，由 run.sh 在拿到锁之后调用；环境同 run.sh：UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg
  UAVRGB_CF_SITL_PLUGIN_DIR=<插件目录>）：
  <.venv-tools 解释器> experiments/2026-11-11-soc100-parity/probe_plugin_params.py --params <副本> --out <目录> \
      [--episodes 0,1,2] [--expect-profile cf_sitl_v1:06e5f017:lib31] [--expect-sha256 <整文件 sha256>]
  --check-only 只核对副本的指纹与 sha256（不起 gz，不需要机器锁）。
退出码：0 通过；1 判据没过；2 基础设施失败（scene_gen / gz 没起来 / 没有 state）；3 副本的指纹或文件 sha256 与预期不符。
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
sys.path.insert(0, os.path.join(REPO, "tools"))

# ---------------------------------------------------------------- 配置
EPISODES = "0,1,2"            # 每种世界各起几个 gz sim，数字是 reset 的 episode 号（噪声种子）；3 个看得出波动，又只花约 4 分钟
SCENE_SEED = 1                # 世界几何的 seed：任取固定值，几何不影响起飞读数
MIN_REST_V_GAP = 0.25         # G2：满电与 50% 的静止电压差下限（V）；预期约 0.38，留 0.13 的余量给电压曲线与噪声
FIRST_FRAME_WALL_S = 120.0    # 等 state 首帧的墙钟上限（秒）：软件渲染下世界起来要 5–10 s，负载高时留余量
HOLD_S = 1.0                  # CfSitlExecutor 的 hold，与评测一致
GZ_VERBOSITY = "3"            # 插件「就绪」行是 gzmsg，默认 verbosity 不打印
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")   # gz sim -v 3 的输出里每个 token 前后都带颜色码（2026-11-12 01:43 探针首次真跑发现）
READY_RE = re.compile(r"\[CfSitlSystem\] 就绪: .*?\bparams=(\S+)")
PARAMS_TAG_RE = re.compile(r"<params>([^<]*)</params>")
IDX_BATTERY, IDX_HOVER = 10, 21   # sim/cf_sitl/gz/README.md 的 state 索引表：10 battery_v，21 hover_thrust；油门杆量 38、库版本 39 取自 cf_sitl_exec


class ProbeInfra(RuntimeError):
    """基础设施失败（不是判据没过）：scene_gen 失败、gz 没起来、没有 state 首帧。"""


# ---------------------------------------------------------------- 纯函数（有单测）
def parse_params_log(text):
    """gz 日志里最后一条插件「就绪」行的 params 路径；没有返回 None。"""
    found = READY_RE.findall(ANSI_RE.sub("", text or ""))
    return found[-1] if found else None


def world_params_path(sdf_text):
    """世界 SDF 里 CfSitlSystem 插件的 <params> 内容；不是恰好一个返回 None。"""
    found = PARAMS_TAG_RE.findall(sdf_text or "")
    return found[0] if len(found) == 1 else None


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def judge_probe(runs, copy_path, default_path, lib_version, min_gap=MIN_REST_V_GAP):
    """探针判据（纯函数）。runs = [{variant, episode, error, rest: {battery_v, hover_thrust}, ready: {...}, params_log, lib_version}]。
    返回 {ok, reasons[], rest_v_gap, per_variant: {variant: {n, rest_v_median, ...}}, log_checked}。"""
    reasons = []
    per = {}
    for variant in ("default", "copy"):
        rs = [r for r in runs if r["variant"] == variant]
        good = [r for r in rs if not r.get("error")]
        per[variant] = {
            "n": len(rs), "n_ok": len(good),
            "rest_v_median": median([r["rest"]["battery_v"] for r in good]),
            "rest_hover_median": median([r["rest"]["hover_thrust"] for r in good]),
            "ready_v_median": median([r["ready"]["battery_v"] for r in good]),
            "ready_hover_median": median([r["ready"]["hover_thrust"] for r in good]),
            "ready_throttle_median": median([r["ready"]["throttle"] for r in good]),
            "ready_sitl_s_median": median([r["ready"]["sitl_t_s"] for r in good]),
            "ready_sitl_s_all": [r["ready"]["sitl_t_s"] for r in good],
        }
        if not rs:
            reasons.append(f"G1 {variant}：没有任何一次运行")
        for r in rs:
            if r.get("error"):
                reasons.append(f"G1 {variant} episode {r['episode']} 没到 READY：{r['error']}")
    gap = per["copy"]["rest_v_median"] - per["default"]["rest_v_median"]
    if gap != gap or gap < min_gap:
        reasons.append(f"G2 静止电压 copy − default = {gap:.3f} V < {min_gap} V：插件没有按满电副本在算"
                       if gap == gap else "G2 静止电压缺数据")
    for r in runs:
        in_world = r.get("params_in_world")
        if r["variant"] == "copy" and (in_world is None or os.path.normpath(in_world) != os.path.normpath(copy_path)):
            reasons.append(f"G0 copy episode {r['episode']} 世界里的 <params> = {in_world}，应为 {copy_path}（scene_gen 没把副本写进世界）")
        if r["variant"] == "default" and in_world is not None and os.path.normpath(in_world) == os.path.normpath(copy_path):
            reasons.append(f"G0 default episode {r['episode']} 世界里的 <params> 是副本（default 世界不该带副本）")
    checked = 0
    for r in runs:
        if r.get("error") or r.get("params_log") is None:
            continue
        want = copy_path if r["variant"] == "copy" else default_path
        checked += 1
        if os.path.normpath(r["params_log"]) != os.path.normpath(want):
            reasons.append(f"G3 {r['variant']} episode {r['episode']} 插件日志 params={r['params_log']}，应为 {want}")
    for r in runs:
        if not r.get("error") and r.get("lib_version") is not None and lib_version is not None and int(r["lib_version"]) != int(lib_version):
            reasons.append(f"G4 {r['variant']} episode {r['episode']} 插件库版本 {r['lib_version']} ≠ 副本指纹里的 lib{lib_version}")
    return {"ok": not reasons, "reasons": reasons, "rest_v_gap": gap, "min_rest_v_gap": min_gap,
            "per_variant": per, "log_lines_checked": checked, "log_lines_total": sum(1 for r in runs if not r.get("error"))}


def fingerprint_problems(params_path, expect_profile=None, expect_sha256=None, lib_version="auto"):
    """副本与预期不符的地方（空 = 都符合）。返回 (problems, info)。"""
    import cf_sitl_exec
    import make_params_variant as mpv
    info = {"path": params_path}
    problems = []
    if not os.path.isfile(params_path):
        return [f"副本不存在：{params_path}"], info
    p = cf_sitl_exec.CfSitlParams.from_json(params_path)
    lib = mpv.read_lib_version() if lib_version == "auto" else lib_version
    info.update(sha256=p.sha256, values_sha256=p.values_sha256, lib_version=lib,
                profile=cf_sitl_exec.format_profile(p.values_sha256, lib))
    if expect_profile and info["profile"] != expect_profile:
        problems.append(f"指纹 {info['profile']} ≠ 预期 {expect_profile}")
    if expect_sha256 and p.sha256 != expect_sha256:
        problems.append(f"文件 sha256 {p.sha256} ≠ 预期 {expect_sha256}")
    return problems, info


# ---------------------------------------------------------------- 起 gz 与读遥测
def start_gz(cp, world, env, log_path):
    """与 collect_parallel._start_sim 同一组环境与参数，多一个 -v 3；自己 setsid，按进程组停。"""
    cmd = ["gz", "sim", "-s", "-r", "-v", GZ_VERBOSITY]
    if env["UAVRGB_RENDER_BACKEND"] == "gpu":
        cmd.append("--headless-rendering")
    cmd.append(world)
    log = open(log_path, "wb")
    try:
        log.write((json.dumps({"command": cmd, "GZ_PARTITION": env["GZ_PARTITION"],
                               "UAVRGB_REPO_ROOT": env.get("UAVRGB_REPO_ROOT"),
                               "GZ_SIM_SYSTEM_PLUGIN_PATH": env.get("GZ_SIM_SYSTEM_PLUGIN_PATH")}) + "\n").encode())
        log.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    cp._register(proc)
    return proc


def first_state_frame(cx, gz_io, proc, timeout_s):
    """订阅 state，等到第一帧完整数据；返回 (帧, 订阅对象)。gz 进程先退出或超时抛 ProbeInfra。"""
    sub = gz_io.TopicSub(cx.STATE_TOPIC, buf=4, stamp_fn=cx.state_stamp_ns)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            sub.close()
            raise ProbeInfra(f"gz sim 提前退出 rc={rc}")
        msg = sub.latest()
        data = msg.get("data") if isinstance(msg, dict) else None
        if isinstance(data, list) and len(data) >= cx.STATE_FIELDS:
            return list(data), sub
        time.sleep(0.05)
    sub.close()
    raise ProbeInfra(f"{timeout_s:.0f} s 内没有 /e1_mini/state 首帧")


def snapshot(cx, st):
    return {"battery_v": float(st[IDX_BATTERY]), "hover_thrust": float(st[IDX_HOVER]), "throttle": float(st[cx.IDX_STICKS + 3]),
            "sitl_t_s": float(st[cx.IDX_T]), "gz_t_s": float(st[cx.IDX_SIM_T]), "height_est_m": float(st[cx.IDX_HEIGHT]),
            "exec_state": int(st[cx.IDX_EXEC])}


def run_one(cp, cx, gz_io, variant, params_path, default_path, episode, workdir, scene_seed):
    """起一个 gz sim 读一组遥测；返回一条 runs 记录（没到 READY 时 error 非空，不抛）。基础设施失败抛 ProbeInfra。"""
    tag = f"{variant}_e{episode}"
    world = os.path.join(workdir, f"world_{tag}.sdf")
    cmd = [sys.executable, os.path.join(REPO, "sim", "tools", "scene_gen.py"), "--seed", str(scene_seed), "--scene-id", "0",
           "--out", world, "--vehicle", "e1_mini", "--no-aux-cameras"]
    if variant == "copy":
        cmd += ["--cf-sitl-params", params_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise ProbeInfra(f"scene_gen 失败 rc={r.returncode}：{(r.stderr or r.stdout)[-300:]}")
    with open(world, encoding="utf-8") as f:
        in_world = world_params_path(f.read())
    partition = f"uavrgb_probe_{uuid.uuid4().hex[:8]}"
    env = cp.gz_env(partition)
    gz_log = os.path.join(workdir, f"gz_{tag}.log")
    saved = dict(os.environ)
    os.environ.update(env)
    proc = ex = sub = None
    lines = []
    rec = {"variant": variant, "episode": episode, "world": world, "params_in_world": in_world, "gz_log": gz_log,
           "error": None, "rest": None, "ready": None, "params_log": None, "lib_version": None, "profile_python": None}
    try:
        proc = start_gz(cp, world, env, gz_log)
        frame, sub = first_state_frame(cx, gz_io, proc, FIRST_FRAME_WALL_S)
        rec["rest"] = snapshot(cx, frame)
        rec["lib_version"] = int(frame[cx.IDX_LIB_VERSION])
        sub.close()
        sub = None
        params = cx.CfSitlParams.from_json(params_path if variant == "copy" else default_path)
        ex = cx.CfSitlExecutor(params, HOLD_S, episode, log=lines.append)
        t0 = time.monotonic()
        try:
            ex.start()
            rec["ready"] = snapshot(cx, ex.state())
            rec["ready"]["wall_s"] = time.monotonic() - t0
            rec["profile_python"] = params.profile()
        except Exception as e:  # noqa: BLE001 - 起飞失败是一条判据结果，不是探针自身故障
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["executor_log"] = lines
    finally:
        if ex is not None:
            ex.stop()
        if sub is not None:
            sub.close()
        if proc is not None:
            cp.stop_group(proc, f"probe gz {tag}")
        os.environ.clear()
        os.environ.update(saved)
    try:
        with open(gz_log, encoding="utf-8", errors="replace") as f:
            rec["params_log"] = parse_params_log(f.read())
    except OSError:
        pass
    return rec


# ---------------------------------------------------------------- 输出
def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def report_md(info, runs, verdict):
    pv = verdict["per_variant"]
    lines = ["# Gazebo 插件读参数的 A/B 探针", "",
             f"- 副本：`{info['path']}`，指纹 `{info.get('profile')}`，文件 sha256 `{info.get('sha256')}`",
             f"- 世界 default 的 <params>（仓库默认）：`{info['default_params']}`", "",
             f"**判决：{'通过' if verdict['ok'] else '未通过'}**"
             f"（静止电压 copy − default = {fmt(verdict['rest_v_gap'])} V，门 ≥ {verdict['min_rest_v_gap']} V；"
             f"插件日志的 params= 行读到 {verdict['log_lines_checked']}/{verdict['log_lines_total']} 个）", ""]
    for x in verdict["reasons"]:
        lines.append(f"- 未过：{x}")
    lines += ["", "| 世界 | episode | <params> 写的路径 | 插件日志 params= | Python 侧指纹 | 静止 V_bus | 静止悬停油门 | READY V_bus | READY 悬停油门 | READY 油门杆量 | READY 用时 SITL s | 墙钟 s | 备注 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        rd, rs = r.get("ready") or {}, r.get("rest") or {}
        lines.append(f"| {r['variant']} | {r['episode']} | `{r.get('params_in_world')}` | `{r.get('params_log') or '未核实'}` | `{r.get('profile_python') or '-'}` | "
                     f"{fmt(rs.get('battery_v'))} | {fmt(rs.get('hover_thrust'))} | {fmt(rd.get('battery_v'))} | {fmt(rd.get('hover_thrust'))} | "
                     f"{fmt(rd.get('throttle'))} | {fmt(rd.get('sitl_t_s'), 2)} | {fmt(rd.get('wall_s'), 1)} | {r.get('error') or ''} |")
    lines += ["", "| 世界 | 次数（到 READY） | 静止 V_bus 中位 | READY V_bus 中位 | READY 悬停油门 中位 | READY 油门杆量 中位 | READY 用时 SITL s 中位 |", "|---|---|---|---|---|---|---|"]
    for v in ("default", "copy"):
        p = pv[v]
        lines.append(f"| {v} | {p['n_ok']}/{p['n']} | {fmt(p['rest_v_median'])} | {fmt(p['ready_v_median'])} | {fmt(p['ready_hover_median'])} | "
                     f"{fmt(p['ready_throttle_median'])} | {fmt(p['ready_sitl_s_median'], 2)} |")
    lines += ["", "「default 世界 + 副本的 Python 侧指纹」就是修复前给了 `--cf-sitl-params` 时的状态：manifest 写副本的指纹，插件却按 default 在算。"
              "两行的静止 V_bus 差就是这个静默不一致的大小。", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--params", required=True, help="满电副本（绝对路径）")
    ap.add_argument("--out", help="输出目录：probe.json、probe.md、各 gz 日志与世界；--check-only 时可省")
    ap.add_argument("--episodes", default=EPISODES, help="reset episode 号，逗号分隔（缺省 %(default)s）")
    ap.add_argument("--scene-seed", type=int, default=SCENE_SEED)
    ap.add_argument("--expect-profile", help="副本应有的 dynamics_profile，如 cf_sitl_v1:06e5f017:lib31")
    ap.add_argument("--expect-sha256", help="副本应有的文件字节 sha256")
    ap.add_argument("--check-only", action="store_true", help="只核对副本的指纹与 sha256，不起 gz")
    args = ap.parse_args(argv)
    params_path = os.path.abspath(args.params)

    problems, info = fingerprint_problems(params_path, args.expect_profile, args.expect_sha256)
    print(f"副本 {params_path}\n  指纹 {info.get('profile')}\n  sha256 {info.get('sha256')}")
    if problems:
        for p in problems:
            print(f"FAIL 指纹核对：{p}", file=sys.stderr)
        return 3
    print("指纹核对：通过")
    if args.check_only:
        return 0
    if not args.out:
        ap.error("缺 --out")
    os.makedirs(args.out, exist_ok=True)

    import collect_parallel as cp
    import cf_sitl_exec as cx
    import gz_io
    default_path = os.path.join(os.environ.get("UAVRGB_REPO_ROOT") or cp.REPO_ROOT, "sim", "dynamics", "e1_mini.json")
    info["default_params"] = default_path

    def _term(_sig, _frame):
        cp.cleanup_all()
        sys.exit(143)
    signal.signal(signal.SIGTERM, _term)

    runs = []
    try:
        for episode in (int(x) for x in args.episodes.split(",") if x.strip()):
            for variant in ("default", "copy"):    # 交替跑：慢性漂移（机器负载）对两种世界一视同仁
                rec = run_one(cp, cx, gz_io, variant, params_path, default_path, episode, args.out, args.scene_seed)
                runs.append(rec)
                print(f"[probe] {variant} e{episode}: " + (rec["error"] or
                      f"rest {rec['rest']['battery_v']:.3f} V，READY sitl {rec['ready']['sitl_t_s']:.2f} s V {rec['ready']['battery_v']:.3f} "
                      f"hover {rec['ready']['hover_thrust']:.3f}") + f"，插件日志 params={rec['params_log'] or '未核实'}", flush=True)
    except ProbeInfra as e:
        print(f"FAIL 基础设施：{e}", file=sys.stderr)
        cp.cleanup_all()
        with open(os.path.join(args.out, "probe.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "soc100-probe.v1", "infra_error": str(e), "info": info, "runs": runs}, f, ensure_ascii=False, indent=1)
        return 2
    verdict = judge_probe(runs, params_path, default_path, info.get("lib_version"))
    with open(os.path.join(args.out, "probe.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "soc100-probe.v1", "info": info, "runs": runs, "verdict": verdict}, f, ensure_ascii=False, indent=1)
    text = report_md(info, runs, verdict)
    with open(os.path.join(args.out, "probe.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(("PASS" if verdict["ok"] else "FAIL") + f" 插件读参数探针：静止电压差 {fmt(verdict['rest_v_gap'])} V")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
