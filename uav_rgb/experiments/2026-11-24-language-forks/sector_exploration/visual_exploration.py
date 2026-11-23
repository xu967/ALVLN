"""局部探索。只读取 RGB 网络预测的三个扇区深度和已执行的动作。"""
import math
from collections import Counter


class VisualExploration:
    def __init__(self):
        self.x = self.y = self.yaw = 0.
        self.visits = Counter()
        self.last_cell = None
        self.turn = 2
        self.turn_windows_left = 0

    @staticmethod
    def cell(x, y):
        return round(x / 0.5), round(y / 0.5)

    def observe(self, previous_action, sectors, following_target):
        if previous_action == 1:
            self.x += 0.15 * math.cos(self.yaw)
            self.y += 0.15 * math.sin(self.yaw)
        if previous_action in (2, 3):
            self.yaw += math.radians(17 if previous_action == 2 else -17)
        here = self.cell(self.x, self.y)
        if here != self.last_cell:
            self.visits[here] += 1
            self.last_cell = here
        if following_target:
            self.turn_windows_left = 0
            return None
        if sectors is None or any(not math.isfinite(d) or d <= 0. for d in sectors):
            return self.turn
        if self.turn_windows_left:
            self.turn_windows_left -= 1
            return self.turn
        # 三个候选方向都从同一相机预测取得。回访惩罚按半米栅格计算。
        scores = []
        for offset, depth, action in zip((20, 0, -20), sectors, (2, 1, 3)):
            heading = self.yaw + math.radians(offset)
            visited = sum(self.visits[self.cell(self.x + distance * math.cos(heading),
                                                self.y + distance * math.sin(heading))]
                          for distance in (0.5, 1.0, 1.5))
            score = min(depth, 3.0) - 0.45 * visited
            if action == 1:
                score += 0.35
                if depth < 1.25:
                    score = -math.inf
            scores.append(score)
        action = (2, 1, 3)[max(range(3), key=scores.__getitem__)]
        if action in (2, 3):
            self.turn = action
            self.turn_windows_left = 1
        return action
