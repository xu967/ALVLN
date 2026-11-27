"""按原教师和相机几何，在分岔的动作前状态重放一个目标的局部路标。"""
import importlib
import math

import numpy as np


vln = importlib.import_module("sim.3dgs.vln")
planner_module = importlib.import_module("sim.3dgs.planner")
COARSE_ACTION = {0: planner_module.FORWARD_SLOW, 1: planner_module.YAW_LEFT, 7: planner_module.YAW_RIGHT}


def _project(aim, altitude, camera, viewmat, nav_to_world, near_plane_m):
    view = np.asarray(viewmat, dtype=float)
    world = np.asarray(nav_to_world, dtype=float)
    if any(m.shape != (4, 4) or not np.isfinite(m).all() for m in (view, world)):
        raise ValueError("Camera and navigation transforms must be finite 4x4 matrices")
    fx, fy, cx, cy = (camera[k] for k in ("fx", "fy", "cx", "cy"))
    width, height = camera["width"], camera["height"]
    if not all(map(math.isfinite, (fx, fy, cx, cy, width, height, near_plane_m))) or min(fx, fy, width, height, near_plane_m) <= 0:
        raise ValueError("Invalid camera calibration or render near plane")

    # A* 只定义水平瞄准点，投影时取飞机当前高度。遮挡仍需独立测量。
    world_point = world @ np.array([aim[0], aim[1], altitude, 1.0])
    x, y, z = (view @ world_point)[:3]
    if z <= near_plane_m:
        return dict(camera_z_m=float(z), image_uv_same_altitude_px=None,
                    inside_actual_camera_horizontal_fov=False, inside_actual_image=False)
    u, v = float(cx + fx * x / z), float(cy + fy * y / z)
    horizontal = 0 <= u < width
    return dict(camera_z_m=float(z), image_uv_same_altitude_px=[u, v],
                inside_actual_camera_horizontal_fov=horizontal, inside_actual_image=horizontal and 0 <= v < height)


def label_goal(planner, state, waypoint, radius, camera, viewmat, nav_to_world,
               teacher_action, yaw_tol_deg, *, near_plane_m=0.2):
    """使用导航系的米、弧度、米/秒，按原 0.5 秒制动教师重放。

    viewmat 由动作前实测四元数与相机安装位姿生成。near_plane_m 取自场景渲染配置。
    """
    xy = (state.x, state.y)
    goal = (waypoint["x"], waypoint["y"])
    if not all(map(math.isfinite, (*xy, state.z, state.yaw, state.vx, state.vy, *goal, radius, yaw_tol_deg))):
        raise ValueError("Non-finite pose, velocity, waypoint, radius or teacher yaw tolerance")
    if radius <= 0 or yaw_tol_deg <= 0:
        raise ValueError("Reach radius and teacher yaw tolerance must be positive")
    speed = math.hypot(state.vx, state.vy)
    runtime_bin, near = vln.subgoal_bearing_bin(planner, state.yaw, xy, goal, speed, radius)

    teacher = planner_module.OccAstarTeacher(planner, waypoint, stop_radius=radius, hold_s=0.5,
                                             clearance="braking", tol_deg=yaw_tol_deg)
    action, _ = teacher(None, None, state)
    if action != teacher_action or teacher.stop_requested != bool(near):
        raise ValueError(f"Fresh teacher replay mismatch at {xy} for {goal}: registered action {teacher_action}, "
                         f"replayed {action} ({teacher.last_branch}), near={near}, stop={teacher.stop_requested}")
    result = dict(runtime_bin=int(runtime_bin), runtime_near=int(near), distance_to_goal_m=math.dist(xy, goal),
                  local_bearing_bin=None, original_action=teacher_action, replayed_teacher_action=int(action),
                  teacher_branch=teacher.last_branch, teacher_runtime_bin=int(teacher.last_bearing_bin),
                  teacher_yaw_tolerance_deg=yaw_tol_deg)
    if near:
        return dict(result, planning_status="near_goal_excluded")
    if teacher.last_path is None:
        return dict(result, planning_status="no_astar_path_runtime_used_direct_fallback")

    contract = vln.SUBGOAL_BEARING_CONTRACT
    lookahead = contract["lookahead_base_m"] + contract["lookahead_speed_gain_s"] * speed
    aim = planner.aim_point(teacher.last_path, *xy, lookahead=lookahead, los_max=contract["los_max_m"],
                            hard_los_max=min(contract["hard_los_max_m"], lookahead))
    if not all(map(math.isfinite, aim)):
        raise ValueError(f"Non-finite A* aim {aim} at {xy} for {goal}")
    if vln.bearing_bin(state.yaw, xy, aim) != runtime_bin:
        raise ValueError(f"Replayed A* aim {aim} at {xy} does not reproduce runtime bearing bin {runtime_bin}")
    if math.dist(xy, aim) <= 1e-8:
        return dict(result, planning_status="degenerate_local_vector")

    yaw_to_aim = math.atan2(aim[1] - state.y, aim[0] - state.x)
    angle = (math.degrees(yaw_to_aim - state.yaw) + 180) % 360 - 180
    coarse_action = COARSE_ACTION.get(runtime_bin)
    return dict(result, planning_status="astar", local_bearing_bin=int(runtime_bin), path_points=len(teacher.last_path),
                lookahead_m=lookahead, aim_xy_m=[float(value) for value in aim], relative_yaw_ccw_deg=angle,
                coarse_action_if_representable=coarse_action,
                agrees_with_original_action=coarse_action == teacher_action if coarse_action is not None else None,
                **_project(aim, state.z, camera, viewmat, nav_to_world, near_plane_m))
