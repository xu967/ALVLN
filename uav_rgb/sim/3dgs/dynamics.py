#!/usr/bin/env python3
"""3DGS pilot 动力学入口：直接复用 sim/tools/cf_blackbox.py（参数单源）。

不在这里定义第二套模型或默认参数。参数唯一来源是
sim/dynamics/cf_blackbox_v1.json；`CfBlackboxParams.profile()/manifest()`
直接写进 3DGS collector 的 manifest。固定 z=2 m、四动作、延迟/阻力/偏航一阶
都由 cf_blackbox 提供。
"""
import os
import sys

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from cf_blackbox import (  # noqa: E402,F401  统一从参数单源导入
    CfBlackbox, CfBlackboxParams, DEFAULT_PARAMS_PATH,
    SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, VALID_ACTIONS,
)


def make_blackbox(scene, params_path=None, params=None):
    """按场景 spawn 建飞控状态；params 传入时复用同一参数对象（manifest 一致）。"""
    params = params or CfBlackboxParams.from_json(params_path)
    spawn = scene["spawn"]
    return CfBlackbox(params, x=spawn["x"], y=spawn["y"],
                      yaw_deg=spawn.get("yaw_deg", 0.0))


def _selftest():
    p = CfBlackboxParams.from_json()
    assert p.calibrated is False and p.z_target_m == 2.0
    d = CfBlackbox(p, 0, 0, 0)
    for _ in range(20):
        s = d.step(FORWARD_SLOW, 0.5)
    assert s["x"] > 0.5 and abs(s["y"]) < 1e-9 and s["z"] == 2.0, s
    d2 = CfBlackbox(p, 0, 0, 0)
    for _ in range(20):
        s = d2.step(YAW_LEFT, 0.5)
    assert s["yaw_deg"] > 5 and s["z"] == 2.0, s
    print("dynamics selftest PASS", d.profile(), s["yaw_deg"])


if __name__ == "__main__":
    _selftest()
