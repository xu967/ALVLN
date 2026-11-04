"""3DGS 在线渲染 pilot：场景/渲染/黑盒动力学/碰撞/采集/评测。

与 Gazebo 管线完全隔离：本包只在 `$DATA_ROOT/uav_rgb/3dgs-venv` 下运行真实渲染，
其余模块（dynamics/collision/scene 的非渲染部分）纯 Python/numpy。接口说明见
docs/experiments/2026-11-05-3dgs-pilot.md。
"""
