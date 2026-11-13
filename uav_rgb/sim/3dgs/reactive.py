#!/usr/bin/env python3
"""不查全局地图的目标条件局部控制器。

这是一个用于方向验证的基线：目标点仍由实验脚本提供，但在线控制只看
当前深度、当前位置和航向。它不调用 A*，也不访问占据栅格。
"""
import math

from .dynamics import FORWARD_SLOW, SAFE_NEUTRAL, YAW_LEFT, YAW_RIGHT


def _wrap_deg(angle):
    """把角度折到 [-180, 180)。"""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _sector_min(depth, x0, x1, y0, y1):
    """返回深度扇区的最小有效值；没有有效深度时返回 0。"""
    best = math.inf
    valid = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            value = float(depth[y, x])
            if math.isfinite(value) and value > 0.0:
                best = min(best, value)
                valid += 1
    return best if valid else 0.0


class ReactiveGoalPolicy:
    """目标方向 + 深度扇区避障的最小局部基线。"""

    name = "reactive_goal"
    planner_name = None
    stop_requested = False

    def __init__(self, waypoint, stop_radius=1.0, turn_deadband_deg=15.0,
                 caution_m=1.5, emergency_m=0.8, switch_margin_m=0.25):
        if not isinstance(waypoint, dict) or not all(k in waypoint for k in ("x", "y")):
            raise ValueError("reactive_goal 需要 waypoint.x 和 waypoint.y")
        self.goal_x = float(waypoint["x"])
        self.goal_y = float(waypoint["y"])
        self.stop_radius = float(stop_radius)
        self.turn_deadband_deg = float(turn_deadband_deg)
        self.caution_m = float(caution_m)
        self.emergency_m = float(emergency_m)
        self.switch_margin_m = float(switch_margin_m)
        self.reset()

    def reset(self):
        self.stop_requested = False
        self.last_turn = None
        self.avoid_turn = None
        self.clear_streak = 0

    @staticmethod
    def _turn(err_deg):
        return YAW_LEFT if err_deg > 0.0 else YAW_RIGHT

    def _choose_clear_side(self, left, right):
        if left <= 0.0 and right <= 0.0:
            return None
        if self.last_turn == YAW_LEFT and left + self.switch_margin_m >= right:
            return YAW_LEFT
        if self.last_turn == YAW_RIGHT and right + self.switch_margin_m >= left:
            return YAW_RIGHT
        return YAW_LEFT if left >= right else YAW_RIGHT

    def __call__(self, rgb, depth, state):
        del rgb
        if getattr(depth, "ndim", 0) != 2 or min(depth.shape) < 5:
            raise ValueError(f"reactive_goal 需要二维深度图，得到 {getattr(depth, 'shape', None)}")
        height, width = depth.shape
        y0, y1 = int(height * 0.4), int(height * 0.6)
        x1, x2 = width // 3, width * 2 // 3
        left = _sector_min(depth, 0, x1, y0, y1)
        center = _sector_min(depth, x1, x2, y0, y1)
        right = _sector_min(depth, x2, width, y0, y1)

        dx = self.goal_x - float(state.x)
        dy = self.goal_y - float(state.y)
        distance = math.hypot(dx, dy)
        if distance <= self.stop_radius:
            self.stop_requested = True
            return SAFE_NEUTRAL, None
        self.stop_requested = False
        goal_bearing = math.degrees(math.atan2(dy, dx))
        error = _wrap_deg(goal_bearing - math.degrees(float(state.yaw)))

        if center <= 0.0:
            return SAFE_NEUTRAL, None
        if self.avoid_turn is not None:
            self.clear_streak = self.clear_streak + 1 if center >= self.caution_m else 0
            if self.clear_streak < 3:
                self.last_turn = self.avoid_turn
                return self.avoid_turn, None
            self.avoid_turn = None
        if center < self.emergency_m:
            if self.avoid_turn is None:
                self.avoid_turn = self._choose_clear_side(left, right)
            if self.avoid_turn is None:
                return SAFE_NEUTRAL, None
            self.last_turn = self.avoid_turn
            return self.avoid_turn, None
        if abs(error) > self.turn_deadband_deg:
            action = self._turn(error)
            self.last_turn = action
            return action, None
        if center < self.caution_m:
            action = self.avoid_turn or self._choose_clear_side(left, right)
            if action is not None and max(left, right) > center:
                self.avoid_turn = action
                self.last_turn = action
                return action, None
        return FORWARD_SLOW, None
