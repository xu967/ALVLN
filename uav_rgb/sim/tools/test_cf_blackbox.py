#!/usr/bin/env python3
"""cf_blackbox 纯函数自检：悬停/前飞/滑行/偏航/延迟/固定 z/参数校验。

不需要 Gazebo。用法：python3 sim/tools/test_cf_blackbox.py
"""
import math
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import cf_blackbox as bb
from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT

passed = 0
BASE = {
    "profile_version": "cf_blackbox_v1",
    "calibrated": False,
    "source": "test placeholder",
    "forward_accel_mps2": 1.0,
    "drag_k": 1.0,
    "latency_s": 0.0,
    "yaw_rate_rad_s": 0.5,
    "yaw_tau_s": 0.0,
    "z_target_m": 2.0,
    "max_tilt_deg": 30.0,
}


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    passed += 1
    print(f"  ok  {name}")


def params(**overrides):
    data = dict(BASE)
    data.update(overrides)
    return bb.CfBlackboxParams.from_dict(data)


def raises(fn, needle=""):
    try:
        fn()
    except ValueError as e:
        return needle in str(e)
    return False


def default_file():
    p = bb.CfBlackboxParams.from_json(bb.DEFAULT_PARAMS_PATH)
    return {
        "profile": p.profile().startswith(bb.PROFILE_VERSION + ":"),
        "sha8": len(p.sha8) == 8 and p.sha256.startswith(p.sha8),
        "uncalibrated": p.calibrated is False,
        "z": p.z_target_m == 2.0,
        "source": bool(p.source),
        "terminal": abs(p.terminal_speed_mps
                        - p.forward_accel_mps2 / p.drag_k) < 1e-12,
        "manifest": (p.manifest()["profile"] == p.profile()
                     and p.manifest()["altitude_assumption"] == "fixed_z"),
    }


def hover():
    m = bb.CfBlackbox(params())
    s = m.step(SAFE_NEUTRAL, 2.0)
    return {
        "still": abs(s["x"]) < 1e-12 and abs(s["y"]) < 1e-12,
        "vel_zero": abs(s["vx"]) < 1e-12 and abs(s["vy"]) < 1e-12,
        "z_fixed": s["z"] == 2.0 and s["vz"] == 0.0,
        "yaw_zero": s["yaw"] == 0.0 and s["yaw_rate"] == 0.0,
        "pose_tuple": s["pose"] == (0.0, 0.0, 2.0)
                      and s["velocity"] == (0.0, 0.0, 0.0),
    }


def forward():
    m = bb.CfBlackbox(params())
    early = m.step(FORWARD_SLOW, 0.1)
    ran = m.step(FORWARD_SLOW, 10.0)
    return {
        "starts": early["vx"] > 0.0 and early["x"] > 0.0,
        "terminal": abs(ran["vx"] - 1.0) < 1e-3,
        "along_x": abs(ran["vy"]) < 1e-9,
        "z_fixed": ran["z"] == 2.0 and ran["vz"] == 0.0,
        "path": ran["path_m"] > 0.0,
    }


def forward_heading():
    m = bb.CfBlackbox(params(), yaw_deg=90.0)
    s = m.step(FORWARD_SLOW, 1.0)
    return abs(s["vx"]) < 1e-9 and s["vy"] > 0.0


def coast():
    m = bb.CfBlackbox(params())
    a = m.step(FORWARD_SLOW, 2.0)
    b = m.step(SAFE_NEUTRAL, 1.0)
    return {
        "decays": abs(b["vx"] - a["vx"] * math.exp(-1.0)) < 1e-9,
        "not_braked": b["vx"] > 0.0,
        "keeps_moving": b["x"] > a["x"],
        "yaw_hold": b["yaw"] == 0.0 and b["yaw_rate"] == 0.0,
    }


def yaw():
    left = bb.CfBlackbox(params()).step(YAW_LEFT, 0.5)
    right = bb.CfBlackbox(params()).step(YAW_RIGHT, 0.5)
    return {
        "left": abs(left["yaw"] - 0.25) < 1e-12 and left["yaw_rate"] == 0.5,
        "right": abs(right["yaw"] + 0.25) < 1e-12 and right["yaw_rate"] == -0.5,
        "no_translation": abs(left["x"]) < 1e-12 and abs(left["y"]) < 1e-12,
        "z_fixed": left["z"] == 2.0,
    }


def yaw_lag():
    m = bb.CfBlackbox(params(yaw_tau_s=0.5))
    early = m.step(YAW_LEFT, 0.1)
    late = m.step(YAW_LEFT, 5.0)
    return {
        "lags": 0.0 < early["yaw_rate"] < 0.5 and early["yaw"] > 0.0,
        "settles": abs(late["yaw_rate"] - 0.5) < 1e-3,
    }


def latency():
    m = bb.CfBlackbox(params(latency_s=0.2))
    first = m.step(FORWARD_SLOW, 0.1)
    second = m.step(FORWARD_SLOW, 0.1)
    third = m.step(FORWARD_SLOW, 0.1)
    return {
        "not_before": first["vx"] == 0.0 and second["vx"] == 0.0,
        "after": third["vx"] > 0.0,
        "same_action_no_restack": abs(third["vx"] - (1.0 - math.exp(-0.1))) < 1e-9,
    }


def delay_queue():
    m = bb.CfBlackbox(params(latency_s=0.1))
    m.dispatch(YAW_LEFT, 0.0)    # 0.10 生效
    m.dispatch(YAW_RIGHT, 0.05)  # 0.15 生效（后派发，乱序入队也要按时间排）
    s = m.step(None, 0.2)
    return {
        "last_active": s["active_action"] == YAW_RIGHT,
        "yaw_cancel": abs(s["yaw"]) < 1e-9,
        "queue_drained": m.pending == [],
    }


def fixed_z():
    m = bb.CfBlackbox(params())
    for action in (FORWARD_SLOW, YAW_LEFT, SAFE_NEUTRAL, YAW_RIGHT):
        s = m.step(action, 1.0)
        if s["z"] != 2.0 or s["vz"] != 0.0:
            return False
    return True


def bad_params():
    return {
        "missing_source": raises(
            lambda: bb.CfBlackboxParams.from_dict({k: v for k, v in BASE.items()
                                                   if k != "source"}), "source"),
        "bad_profile": raises(
            lambda: params(profile_version="cf_sil_v1"), "profile_version"),
        "accel_zero": raises(lambda: params(forward_accel_mps2=0.0), "forward_accel"),
        "accel_nan": raises(lambda: params(forward_accel_mps2=float("nan")), "有限数"),
        "drag_zero": raises(lambda: params(drag_k=0.0), "drag_k"),
        "latency_neg": raises(lambda: params(latency_s=-0.1), "latency_s"),
        "yaw_rate_zero": raises(lambda: params(yaw_rate_rad_s=0.0), "yaw_rate"),
        "tau_neg": raises(lambda: params(yaw_tau_s=-0.1), "yaw_tau"),
        "z_zero": raises(lambda: params(z_target_m=0.0), "z_target"),
        "tilt_90": raises(lambda: params(max_tilt_deg=90.0), "max_tilt"),
        "accel_over_tilt": raises(lambda: params(forward_accel_mps2=6.0), "超过"),
        "missing_key": raises(
            lambda: bb.CfBlackboxParams.from_dict({k: v for k, v in BASE.items()
                                                   if k != "drag_k"}), "drag_k"),
    }


def bad_actions():
    m = bb.CfBlackbox(params())
    return {
        "dispatch_int": raises(lambda: m.dispatch(4), "动作"),
        "step_bool": raises(lambda: m.step(True, 0.1), "动作"),
        "dt_negative": raises(lambda: m.step(SAFE_NEUTRAL, -0.1), "dt"),
        "dt_nan": raises(lambda: m.step(SAFE_NEUTRAL, float("nan")), "dt"),
        "dispatch_time": raises(lambda: m.dispatch(YAW_LEFT, float("inf")), "时间"),
    }


def profile_changes():
    a = params()
    b = params(drag_k=2.0)
    return a.profile() != b.profile() and len(a.profile().split(":")[-1]) == 8


def mixed_api():
    """显式 dispatch 与 step(action) 混用：新动作不能被上一步的去重吞掉。"""
    m = bb.CfBlackbox(params())
    m.step(FORWARD_SLOW, 0.1)
    m.dispatch(SAFE_NEUTRAL)          # 显式改回 SAFE
    s = m.step(FORWARD_SLOW, 0.1)     # 下一步又要求 FORWARD，必须重新入队
    return s["active_action"] == FORWARD_SLOW and s["vx"] > 0.0


def main():
    for k, v in default_file().items():
        check(f"默认参数: {k}", v)
    for group, checks in (
            ("悬停", hover()), ("前飞", forward()),
            ("固定 z", {"all_steps": fixed_z()}), ("滑行", coast()),
            ("左右偏航", yaw()), ("偏航滞后", yaw_lag()),
            ("延迟", latency()), ("延迟队列", delay_queue()),
            ("参数校验", bad_params()), ("动作/dt 校验", bad_actions()),
            ("profile", {"changes": profile_changes()})):
        for k, v in checks.items():
            check(f"{group}: {k}", v)
    check("前飞方向随机头", forward_heading())
    check("显式 dispatch 与 step(action) 混用", mixed_api())
    print(f"test_cf_blackbox: {passed} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
