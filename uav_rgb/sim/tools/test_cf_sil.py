#!/usr/bin/env python3
"""cf_sil 纯函数自检：参数加载、悬停推力、饱和、yaw 差动、FORWARD 倾角、
延迟队列、动作越界、飞行日志 CSV 占位。

不需要 Gazebo。用法：python3 sim/tools/test_cf_sil.py
"""
import copy
import json
import math
import os
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import cf_sil

PARAMS_PATH = os.path.join(os.path.dirname(TOOLS), "dynamics", "cf_sil_v1.json")
passed = 0


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    passed += 1
    print(f"  ok  {name}")


def _data(**overrides):
    with open(PARAMS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data = copy.deepcopy(data)
    for key, value in overrides.items():
        data[key] = value
    return data


def params(**overrides):
    return cf_sil.CfSilParams.from_dict(_data(**overrides))


def odom(z=2.1, v=(0.0, 0.0, 0.0), w=(0.0, 0.0, 0.0),
         e=(0.0, 0.0, 0.0), t_ns=0):
    """合成 gz.msgs.Odometry JSON（linear/angular 机体系）。"""
    roll, pitch, yaw = e
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return {
        "header": {"stamp": {"sec": t_ns // 10**9, "nsec": t_ns % 10**9}},
        "pose": {"position": {"x": 0.0, "y": 0.0, "z": z},
                 "orientation": {"x": qx, "y": qy, "z": qz, "w": qw}},
        "twist": {"linear": {"x": v[0], "y": v[1], "z": v[2]},
                  "angular": {"x": w[0], "y": w[1], "z": w[2]}},
    }


def thrust(motors, p):
    return sum(p.kf * w * w for w in motors)


def firmware_defaults():
    p = cf_sil.CfSilParams.from_json(PARAMS_PATH)
    d = p.pid
    fw = p.data["firmware_default"]

    def g(section):
        return (float(section["kp"]), float(section.get("ki", 0)),
                float(section.get("kd", 0)), abs(float(section.get("i_lim", 0))))

    out = {
        "json_matches_firmware": all(
            g(fw["rate_pid"][a]) == d["rate"][a]
            and g(fw["att_pid"][a]) == d["att"][a] for a in cf_sil.AXES),
        "rate_roll": d["rate"]["roll"] == (0.06, 0.1, 0.001, 0.3),
        "rate_yaw": d["rate"]["yaw"] == (0.3, 0.01, 0.01, 0.3),
        "att_rp": d["att"]["roll"] == (6.0, 0.0, 0.0, 0.0)
                 and d["att"]["pitch"] == (6.0, 0.0, 0.0, 0.0),
        "att_yaw": d["att"]["yaw"] == (3.0, 0.0, 0.0, 0.0),
        "rate_max_deg": [round(math.degrees(v), 3) for v in p.max_rate]
                        == [360.0, 360.0, 300.0],
        "tilt": abs(math.degrees(p.max_tilt) - 30.0) < 1e-9,
        "throttle": p.throttle_min == 0.10 and p.throttle_max == 0.90
                    and p.arm_throttle_limit == 0.05,
        "imu": p.imu_hz == 1000 and 0.5 < p.rate_d_alpha < 0.7,
        "source": "control.ino" in p.firmware_source,
        "calibrated_false": p.calibrated is False,
        "wheelbase": p.hardware.get("wheelbase_m") == 0.118,
        "hardware_unknown": all(p.hardware.get(k) is None for k in
                                ("mass_kg", "inertia_kg_m2", "props", "motors",
                                 "thrust_curve", "battery", "hover_throttle")),
        "altitude": p.manifest()["altitude_assumption"] == "companion_hold",
        "profile": p.profile().startswith("cf_sil_v1:")
                   and len(p.profile().split(":")[1]) == 8,
    }
    p2 = params(mass=p.mass + 0.1)
    out["profile_changes"] = p2.profile() != p.profile()
    return out


def hover():
    p = params(latency_s=0.0)
    c = cf_sil.CfSilController(p)
    c.dispatch(cf_sil.SAFE_NEUTRAL, 0)
    motors, telem = c.step(odom(z=p.altitude_target))
    mg = p.mass * cf_sil.G
    out = {
        "thrust_mg": abs(thrust(motors, p) - mg) < 1e-6 * mg,
        "within_limits": all(0.0 < w < p.max_omega for w in motors),
        "hover_throttle": p.throttle_min <= p.hover_throttle_norm <= p.throttle_max,
        "not_saturated": not telem["saturated"],
    }
    low = cf_sil.CfSilController(p)
    low.dispatch(cf_sil.SAFE_NEUTRAL, 0)
    m_low, _ = low.step(odom(z=1.0, t_ns=50_000_000))
    high = cf_sil.CfSilController(p)
    high.dispatch(cf_sil.SAFE_NEUTRAL, 0)
    m_high, _ = high.step(odom(z=2.5, t_ns=50_000_000))
    out["climb_thrust"] = thrust(m_low, p) > mg
    out["descend_thrust"] = thrust(m_high, p) < mg
    return out


def saturation():
    p = params(latency_s=0.0, max_rpm=1000.0)
    c = cf_sil.CfSilController(p)
    c.dispatch(cf_sil.SAFE_NEUTRAL, 0)
    motors, telem = c.step(odom(z=p.altitude_target))
    return {
        "throttle_clamped": telem["throttle_clamped"] and telem["saturated"],
        "at_max": all(abs(w - p.throttle_max * p.max_omega) < 1e-6 for w in motors),
        "in_range": all(0.0 <= w <= p.max_omega for w in motors),
    }


def yaw_differential():
    p = params(latency_s=0.0)
    out = {}
    for name, action, sign in (("left", cf_sil.YAW_LEFT, 1.0),
                               ("right", cf_sil.YAW_RIGHT, -1.0)):
        c = cf_sil.CfSilController(p)
        c.dispatch(action, 0)
        motors, telem = c.step(odom(z=p.altitude_target, t_ns=50_000_000))
        u = [w / p.max_omega for w in motors]
        out[name + "_rate"] = abs(telem["rate_target_rad_s"][2] - sign * 0.6) < 1e-9
        # rotor 0/1 ccw，2/3 cw；正偏航力矩 = cw 加速、ccw 减速。
        out[name + "_diff"] = ((u[2] + u[3] > u[0] + u[1]) if sign > 0
                               else (u[2] + u[3] < u[0] + u[1]))
    return out


def forward_tilt():
    p = params(latency_s=0.0)
    c = cf_sil.CfSilController(p)
    c.dispatch(cf_sil.FORWARD_SLOW, 0)
    motors, telem = c.step(odom(z=p.altitude_target, t_ns=50_000_000))
    u = [w / p.max_omega for w in motors]
    return {
        "vx_cmd": telem["velocity_cmd_mps"][0] == p.forward_speed,
        "pitch_positive": telem["pitch_target_rad"] > 0.0,
        "accel_forward": telem["accel_cmd_mps2"][0] > 0.0,
        "rate_pitch_positive": telem["rate_target_rad_s"][1] > 0.0,
        # 正俯仰力矩：后侧（rotor 1/3）加速、前侧（rotor 0/2）减速。
        "rear_faster": u[1] + u[3] > u[0] + u[2],
    }


def latency_queue():
    p = params(latency_s=0.1)
    c = cf_sil.CfSilController(p)
    c.dispatch(cf_sil.FORWARD_SLOW, 0)
    _, early = c.step(odom(z=p.altitude_target, t_ns=50_000_000))
    _, late = c.step(odom(z=p.altitude_target, t_ns=150_000_000))
    out = {
        "before": early["active_action"] == cf_sil.SAFE_NEUTRAL
                  and early["pitch_target_rad"] == 0.0,
        "after": late["active_action"] == cf_sil.FORWARD_SLOW
                 and late["pitch_target_rad"] > 0.0,
    }
    # 乱序派发按生效时间排序，而不是按调用顺序。
    q = cf_sil.CfSilController(p)
    q.dispatch(cf_sil.FORWARD_SLOW, 2_000_000_000)
    q.dispatch(cf_sil.SAFE_NEUTRAL, 1_000_000_000)
    _, a = q.step(odom(z=p.altitude_target, t_ns=1_050_000_000))
    _, b = q.step(odom(z=p.altitude_target, t_ns=2_150_000_000))
    out["sorted"] = (a["active_action"] == cf_sil.SAFE_NEUTRAL
                     and b["active_action"] == cf_sil.FORWARD_SLOW)
    return out


def action_bounds():
    p = params(latency_s=0.0)
    c = cf_sil.CfSilController(p)
    ok = 0
    for bad in (4, -1, 3.0, "1", None, True):
        try:
            c.dispatch(bad, 0)
        except ValueError:
            ok += 1
    return {"rejects_all": ok == 6, "queue_empty": c.pending == []}


def desaturation():
    p = params(latency_s=0.0)
    c = cf_sil.CfSilController(p)
    c.dispatch(cf_sil.YAW_LEFT, 0)
    # 实测角速率远低于设定点 -> 力矩远超可用油门范围，必须缩回 [0,1]。
    motors, telem = c.step(odom(z=p.altitude_target, t_ns=50_000_000,
                                w=(0.0, 0.0, -100.0)))
    u = telem["motor_norm"]
    return {
        "saturated": telem["saturated"],
        "bounded": all(0.0 <= x <= 1.0 for x in u)
                   and all(0.0 <= w <= p.max_omega for w in motors),
    }


def attitude_and_odom():
    roll, pitch, yaw = 0.1, -0.2, 0.5
    parsed = cf_sil.parse_odometry(odom(z=2.0, e=(roll, pitch, yaw)))
    r, p, y = cf_sil.quat_to_euler(parsed["orientation"])
    ok = abs(r - roll) < 1e-9 and abs(p - pitch) < 1e-9 and abs(y - yaw) < 1e-9
    try:
        cf_sil.parse_odometry({"pose": {"position": {}}})
        reject = False
    except ValueError:
        reject = True
    return {"euler_roundtrip": ok, "bad_odom_rejected": reject}


def flight_logs():
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.csv")
        with open(good, "w", encoding="utf-8") as f:
            f.write("t,qw,qx,qy,qz,wx,wy,wz,thrust,x,y,z\n")
            for i in range(3):
                f.write(f"{i*0.1},1,0,0,0,0,0,0,0.5,0,0,{2.0+i*0.1}\n")
        s = cf_sil.load_flight_log(good)
        out = {
            "csv_rows": s["rows"] == 3 and s["invalid_rows"] == 0,
            "csv_duration": abs(s["duration_s"] - 0.2) < 1e-9,
            "csv_groups": s["has_attitude"] and s["has_rates"]
                          and s["has_thrust"] and s["has_pos"]
                          and s["missing_groups"] == [],
        }
        bad = os.path.join(d, "bad.csv")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("qw,qx,qy,qz\n1,0,0,0\n")
        try:
            cf_sil.load_flight_log(bad)
            out["csv_missing_t"] = False
        except ValueError:
            out["csv_missing_t"] = True

    # .ulg 在 pyulog 缺失时必须明确报错，而不是静默跳过。
    old = sys.modules.get("pyulog", "missing")
    sys.modules["pyulog"] = None
    try:
        try:
            cf_sil.load_flight_log("/tmp/does-not-exist.ulg")
            out["ulg_needs_pyulog"] = False
        except RuntimeError as e:
            out["ulg_needs_pyulog"] = "pyulog" in str(e)
    finally:
        if old == "missing":
            sys.modules.pop("pyulog", None)
        else:
            sys.modules["pyulog"] = old
    try:
        cf_sil.load_flight_log("/tmp/log.bin")
        out["bad_ext"] = False
    except ValueError:
        out["bad_ext"] = True
    return out


def main():
    for group in (firmware_defaults(), hover(), saturation(), yaw_differential(),
                  forward_tilt(), latency_queue(), action_bounds(), desaturation(),
                  attitude_and_odom(), flight_logs()):
        for name, ok in group.items():
            check(name, ok)
    print(f"test_cf_sil: {passed} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
