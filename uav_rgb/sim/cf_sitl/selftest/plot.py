"""画 cf_sitl_selftest 的 CSV：frame / hover / actuator / steps / battery / longrun 各一张 PNG，写回同一目录。
用法：$UAV_RGB_PYTHON sim/cf_sitl/selftest/plot.py <out_dir>
图上用英文标签（matplotlib 默认字体没有中文字形）。缺哪个 CSV 就跳过哪张图。
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 配置区 ----
DPI = 110             # 与 spike plot.py 一致，单图 < 300 KB
TARGET_H = 2.0        # BRIEF §4 目标高度，画参考线
FWD_TARGET = 0.3      # FORWARD 目标速度
YAW_TARGET = 0.6      # YAW_LEFT 目标角速度（绕 z 正向）
BAT_L1, BAT_L3 = 3.5, 3.0   # safety.ino 阈值，画参考线
STEP_PITCH_DEG = 15.0       # 杆量 0.5 × tiltMax 30°
STEP_YAW_DPS = -150.0       # 杆量 0.5 × maxRate.z 300°/s，正杆 = 顺时针 = 绕 z 负向
STEP_TOL = 0.05             # ±5% 判据带

out = sys.argv[1] if len(sys.argv) > 1 else "out"


def rd(name):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        print("skip", name)
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def finish(fig, name):
    for a in fig.axes:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=DPI)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


# (1) 坐标系：真值 vs 固件估计欧拉角 + 姿态误差；phase 1-3 陀螺段、4-5 静止倾斜段
d = rd("frame.csv")
if d is not None:
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    t = d["t"]
    for i, k in enumerate(("roll", "pitch", "yaw")):
        ax[i].plot(t, d[f"{k}_deg"], label=f"{k} true")
        ax[i].plot(t, d[f"est_{k}_deg"], "--", label=f"{k} fw est")
        ax[i].set_ylabel("deg")
    ax[3].plot(t, d["att_err_deg"], label="attitude error (quat angle)")
    ax[3].set_ylabel("deg")
    ax[3].set_xlabel("sim time [s]")
    for a in ax:
        for ph in range(1, 6):
            m = d["phase"] == ph
            if m.any():
                a.axvspan(t[m].min(), t[m].max(), color="0.9" if ph <= 3 else "0.8", zorder=0)
    ax[0].set_title("(1) frame consistency: injected world rates (grey) and static tilts (dark grey) vs firmware attitude")
    finish(fig, "frame.png")

# (2) 悬停：高度、倾角、水平轨迹、油门
d = rd("hover.csv")
if d is not None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    t = d["t"]
    ax[0, 0].plot(t, d["z"], label="z true")
    ax[0, 0].plot(t, d["height_est"], "--", label="executor height est")
    ax[0, 0].axhline(TARGET_H, color="k", lw=0.5)
    ax[0, 0].set_ylabel("m")
    ax[0, 0].set_title("(2) hover 10 s: height")
    ax[0, 1].plot(t, d["tilt_deg"], label="tilt")
    ax[0, 1].set_ylabel("deg")
    ax[0, 1].set_title("tilt")
    ax[1, 0].plot(d["x"], d["y"], lw=0.8)
    ax[1, 0].plot(d["x"][0], d["y"][0], "go", label="start")
    ax[1, 0].plot(d["x"][-1], d["y"][-1], "rs", label="end")
    ax[1, 0].set_aspect("equal")
    ax[1, 0].set_xlabel("x [m]")
    ax[1, 0].set_ylabel("y [m]")
    ax[1, 0].set_title("horizontal drift")
    for i in range(4):
        ax[1, 1].plot(t, d[f"m{i}"], lw=0.7, label=f"motor {i}")
    ax[1, 1].plot(t, d["hover_thrust"], "k--", label="hover stick")
    ax[1, 1].set_title("motors / hover stick")
    ax[1, 1].set_xlabel("sim time [s]")
    finish(fig, "hover.png")

# (3) 执行器：前向速度、偏航角速度、高度、轨迹；phase 1 高度保持 2 FORWARD 3 SAFE 停 4 YAW_LEFT 5 SAFE
d = rd("actuator.csv")
if d is not None:
    fig, ax = plt.subplots(3, 2, figsize=(13, 10))
    t = d["t"]

    def shade(a):
        for ph, c in ((2, "#dde7f5"), (4, "#f5e6dd")):
            m = d["phase"] == ph
            if m.any():
                a.axvspan(t[m].min(), t[m].max(), color=c, zorder=0)

    ax[0, 0].plot(t, d["vx_b"], label="body vx true")
    ax[0, 0].plot(t, d["vel_est_x"], "--", label="executor vx est")
    ax[0, 0].plot(t, d["vy_b"], label="body vy true", lw=0.7)
    ax[0, 0].axhline(FWD_TARGET, color="k", lw=0.5)
    ax[0, 0].set_ylabel("m/s")
    ax[0, 0].set_title("(3) FORWARD (blue span) / YAW_LEFT (orange span)")
    ax[1, 0].plot(t, d["yaw_rate"], label="yaw rate true (0.1 s diff)")
    ax[1, 0].axhline(YAW_TARGET, color="k", lw=0.5)
    ax[1, 0].set_ylabel("rad/s")
    ax[2, 0].plot(t, d["z"], label="z true")
    ax[2, 0].plot(t, d["height_est"], "--", label="height est")
    ax[2, 0].axhline(TARGET_H, color="k", lw=0.5)
    ax[2, 0].set_ylabel("m")
    ax[2, 0].set_xlabel("sim time [s]")
    ax[0, 1].plot(t, d["roll_deg"], label="roll")
    ax[0, 1].plot(t, d["pitch_deg"], label="pitch")
    ax[0, 1].set_ylabel("deg")
    ax[0, 1].set_title("attitude")
    ax[1, 1].plot(t, d["hover_thrust"], "k--", label="hover stick")
    ax[1, 1].plot(t, d["throttle_stick"], label="throttle stick")
    for i in range(4):
        ax[1, 1].plot(t, d[f"m{i}"], lw=0.6, label=f"motor {i}")
    ax[1, 1].set_title("sticks / motors")
    ax[2, 1].plot(d["x"], d["y"], lw=0.8)
    ax[2, 1].plot(d["x"][0], d["y"][0], "go", label="start")
    ax[2, 1].plot(d["x"][-1], d["y"][-1], "rs", label="end")
    ax[2, 1].set_aspect("equal")
    ax[2, 1].set_xlabel("x [m]")
    ax[2, 1].set_ylabel("y [m]")
    ax[2, 1].set_title("ground track")
    for a in (ax[0, 0], ax[1, 0], ax[2, 0], ax[0, 1], ax[1, 1]):
        shade(a)
    finish(fig, "actuator.png")

# (3.5) 固件级打杆阶跃：phase 1 俯仰 +0.5、2 松杆、3 执行器接管、4 偏航 +0.5、5 松杆、6 接管
d = rd("steps.csv")
if d is not None:
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    t = d["t"]
    raw = d["raw"] > 0.5
    ax[0].plot(t, d["pitch_deg"], label="pitch true")
    ax[0].plot(t, d["est_pitch_deg"], "--", label="pitch fw est (controlled)")
    ax[0].plot(t, d["roll_deg"], lw=0.7, label="roll true")
    for s in (1, -1):
        ax[0].axhspan(STEP_PITCH_DEG * (1 - s * STEP_TOL), STEP_PITCH_DEG * (1 + s * STEP_TOL), color="g", alpha=0.08)
    ax[0].axhline(STEP_PITCH_DEG, color="k", lw=0.5)
    ax[0].set_ylabel("deg")
    ax[0].set_title("(3.5) raw stick steps: pitch +0.5 (span 1), release (2), executor (3), yaw +0.5 (4), release (5), executor (6)")
    ax[1].plot(t, d["yaw_rate_dps"], label="yaw rate true (0.1 s diff)")
    ax[1].axhspan(STEP_YAW_DPS * (1 + STEP_TOL), STEP_YAW_DPS * (1 - STEP_TOL), color="g", alpha=0.08)
    ax[1].axhline(STEP_YAW_DPS, color="k", lw=0.5)
    ax[1].set_ylabel("deg/s")
    ax[2].plot(t, d["z"], label="z true")
    ax[2].plot(t, d["vx_b"], lw=0.8, label="body vx")
    ax[2].axhline(TARGET_H, color="k", lw=0.5)
    ax[2].set_ylabel("m, m/s")
    ax[3].plot(t, np.where(raw, d["stick_pitch"], np.nan), label="raw pitch stick")
    ax[3].plot(t, np.where(raw, d["stick_yaw"], np.nan), label="raw yaw stick")
    ax[3].plot(t, d["throttle"], lw=0.8, label="throttle (raw: executor hover stick)")
    ax[3].plot(t, d["hover_thrust"], "k--", lw=0.8, label="executor hover stick")
    ax[3].plot(t, d["exec_state"] / 4.0, ":", label="exec_state/4 (READY=0.5, FAILSAFE=1)")
    ax[3].set_ylabel("0..1")
    ax[3].set_xlabel("sim time [s]")
    for a in ax:
        for ph, c in ((1, "#dde7f5"), (2, "#eef2f8"), (4, "#f5e6dd"), (5, "#f8efe8")):
            m = d["phase"] == ph
            if m.any():
                a.axvspan(t[m].min(), t[m].max(), color=c, zorder=0)
    finish(fig, "steps.png")

# (5) 电池：母线电压、悬停油门杆、高度、固件模式
d = rd("battery.csv")
if d is not None:
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    t = d["t"]
    ax[0].plot(t, d["battery_v"], label="V_bus (firmware reads this)")
    ax[0].axhline(BAT_L1, color="orange", lw=0.7, label="L1 3.5 V")
    ax[0].axhline(BAT_L3, color="r", lw=0.7, label="L3 3.0 V")
    ax[0].set_ylabel("V")
    ax[0].set_title("(5) battery discharge in hover from SoC 1.0")
    ax[1].plot(t, d["hover_thrust"], "k--", label="hover stick (ff + integral)")
    ax[1].plot(t, d["throttle_stick"], label="throttle stick")
    ax[1].plot(t, d["motor_mean"], label="mean motor duty")
    ax[1].set_ylabel("0..1")
    ax[2].plot(t, d["z"], label="z true")
    ax[2].plot(t, d["fw_mode"] / 4.0 * TARGET_H, ":", label="fw_mode/4 (AUTO=4 -> 2.0)")
    ax[2].plot(t, d["exec_state"] / 4.0 * TARGET_H, "-.", label="exec_state/4 (FAILSAFE=4 -> 2.0)")
    ax[2].set_ylabel("m")
    ax[2].set_xlabel("sim time [s]")
    finish(fig, "battery.png")

# (6) 长航时：phase 1 SAFE 保持 30 s，phase 2 FORWARD/YAW 交替；看姿态估计误差（真值 − 估计）线性增长、
#     速度环积分（杆×30°）跟着涨、机体 vx 是否保持为正、高度/油门/电压
d = rd("longrun.csv")
if d is not None:
    fig, ax = plt.subplots(4, 2, figsize=(14, 12))   # 不共享 x：左下是 x-y 航迹
    t = d["t"]
    fwd = d["action"] == 1
    for a in ax.flat:
        if a is ax[3, 0]:
            continue
        m = d["phase"] == 1
        if m.any():
            a.axvspan(t[m].min(), t[m].max(), color="#e8f5e9", zorder=0)
        for lo, hi in zip(np.flatnonzero(np.diff(fwd.astype(int)) == 1), np.flatnonzero(np.diff(fwd.astype(int)) == -1)):
            a.axvspan(t[lo + 1], t[hi], color="#dde7f5", zorder=0)
    ax[0, 0].plot(t, d["vx_b"], label="body vx true")
    ax[0, 0].plot(t, d["vel_est_x"], "--", lw=0.7, label="executor vx est")
    ax[0, 0].plot(t, d["vy_b"], lw=0.7, label="body vy true")
    ax[0, 0].axhline(FWD_TARGET, color="k", lw=0.5)
    ax[0, 0].axhline(0, color="r", lw=0.5)
    ax[0, 0].set_ylabel("m/s")
    ax[0, 0].set_title("(6) long run: SAFE hold (green) then FORWARD (blue) / YAW alternating")
    ax[1, 0].plot(t, d["pitch_deg"], label="pitch true")
    ax[1, 0].plot(t, d["est_pitch_deg"], "--", label="pitch fw est")
    ax[1, 0].plot(t, d["roll_deg"], lw=0.7, label="roll true")
    ax[1, 0].plot(t, d["est_roll_deg"], "--", lw=0.7, label="roll fw est")
    ax[1, 0].set_ylabel("deg")
    ax[1, 0].set_title("attitude: firmware integrates gyro only in flight -> estimate drifts")
    ax[2, 0].plot(t, d["err_pitch_deg"], label="pitch est - true")
    ax[2, 0].plot(t, d["err_roll_deg"], label="roll est - true")
    ax[2, 0].plot(t, d["pitch_stick"] * 30, "--", label="pitch stick x 30 deg (~ velocity-loop integral)")
    ax[2, 0].plot(t, -d["roll_stick"] * 30, "--", label="-roll stick x 30 deg")
    ax[2, 0].set_ylabel("deg")
    ax[2, 0].set_title("estimate error vs. executor integral (must stay under tilt_int_max)")
    ax[3, 0].plot(d["x"], d["y"], lw=0.8)
    ax[3, 0].plot(d["x"][0], d["y"][0], "go", label="start")
    ax[3, 0].plot(d["x"][-1], d["y"][-1], "rs", label="end")
    ax[3, 0].set_xlabel("x [m]  (ground track, not time)")
    ax[3, 0].set_ylabel("y [m]")
    ax[3, 0].set_title("ground track")
    ax[0, 1].plot(t, d["z"], label="z true")
    ax[0, 1].plot(t, d["height_est"], "--", label="executor height est")
    ax[0, 1].axhline(TARGET_H, color="k", lw=0.5)
    ax[0, 1].set_ylabel("m")
    ax[0, 1].set_title("height")
    ax[1, 1].plot(t, d["yaw_deg"], label="yaw true")
    ax[1, 1].set_ylabel("deg")
    ax[1, 1].set_title("yaw")
    ax[2, 1].plot(t, d["throttle_stick"], label="throttle stick")
    ax[2, 1].plot(t, d["hover_thrust"], "k--", label="hover stick (ff + integral)")
    ax[2, 1].plot(t, d["yaw_stick"], lw=0.7, label="yaw stick")
    ax[2, 1].set_ylabel("0..1")
    ax[2, 1].set_title("sticks")
    ax[3, 1].plot(t, d["battery_v"], label="V_bus")
    ax[3, 1].axhline(BAT_L1, color="orange", lw=0.7, label="L1 3.5 V")
    ax[3, 1].axhline(BAT_L3, color="r", lw=0.7, label="L3 3.0 V")
    ax[3, 1].set_ylabel("V")
    ax[3, 1].set_xlabel("sim time [s]")
    ax[3, 1].set_title("battery")
    ax[3, 0].set_aspect("equal")
    finish(fig, "longrun.png")
