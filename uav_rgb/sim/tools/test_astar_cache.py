#!/usr/bin/env python3
"""teacher.astar_path 的栅格可走性缓存回归。

背景（2026-11-10 长课程第二轮）：评测每步都从当前位姿重跑一次 A*（教师臂是动作本身，学生臂是影子教师标签），
K=3 长课程的 A* 平均要 50–250 ms，占每步观测到派发延迟的大头，动作生效时长比 K=1 短约 11%。
cProfile：A* 里 85% 的时间在 _point_blocked（每个格子对每根柱子判一次）。柱子在一集内不变，所以可走性可以跨调用缓存。

覆盖：缓存有效（第二次调用几乎不再算格子）、路径与旧实现逐位相同（K=1 / K=3 / 带 radius 的圆柱 / 起点落在余量内 / 终点被堵）、
起点强制可走不污染缓存、柱子或余量变了不串缓存、缓存大小有上限。
"""
import heapq
import math
import os
import random
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene_course as sc  # noqa: E402
import scene_gen as sg  # noqa: E402
import teacher as T  # noqa: E402

SEEDS = [1, 113, 122, 153, 172, 176, 180, 230]


def astar_reference(pillars, start, goal, clearance=0.5, step=0.1):
    """改缓存之前的 astar_path 原样拷贝（每次调用新建 free 字典）；只作逐位对照的参照物。"""
    x0, y0, nx, ny = T._grid_spec(pillars, step)

    def idx(px, py):
        return int(round((px - x0) / step)), int(round((py - y0) / step))

    sx, sy = idx(*start)
    gx, gy = idx(*goal)
    if not (0 <= sx < nx and 0 <= sy < ny and 0 <= gx < nx and 0 <= gy < ny):
        return None
    free = {}

    def is_free(i, j):
        k = (i, j)
        if k not in free:
            free[k] = not T._point_blocked(x0 + i * step, y0 + j * step, pillars, clearance)
        return free[k]

    if not is_free(gx, gy):
        return None
    free[(sx, sy)] = True
    open_heap = [(0.0, 0, sx, sy)]
    gscore = {(sx, sy): 0.0}
    parent = {}
    while open_heap:
        _f, _g, i, j = heapq.heappop(open_heap)
        if (i, j) == (gx, gy):
            path = []
            cur = (i, j)
            while cur in parent:
                path.append((x0 + cur[0] * step, y0 + cur[1] * step))
                cur = parent[cur]
            path.append((x0 + sx * step, y0 + sy * step))
            return path[::-1]
        gi = gscore[(i, j)]
        for di, dj, w in T._MOVES:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not is_free(ni, nj):
                continue
            if di and dj and not (is_free(i + di, j) and is_free(i, j + dj)):
                continue
            ng = gi + w * step
            if ng < gscore.get((ni, nj), math.inf):
                gscore[(ni, nj)] = ng
                parent[(ni, nj)] = (i, j)
                h = math.hypot((ni - gx) * step, (nj - gy) * step)
                heapq.heappush(open_heap, (ng + h, ng, ni, nj))
    return None


def goal_of(pillars):
    return (max(p["x"] for p in pillars) + 2.0, 0.0)


def clear_cache():
    # 红灯阶段（缓存还没写）这个属性不存在：用 getattr 让用例以断言失败而不是 AttributeError 报红
    getattr(T, "_FREE_CACHE", {}).clear()


class AstarCache(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_second_call_reuses_cells(self):
        pillars, _ = sc.sample_course(1, 3)
        goal = goal_of(pillars)
        with mock.patch.object(T, "_point_blocked", wraps=T._point_blocked) as spy:
            T.astar_path(pillars, (-0.5, 0.0), goal, clearance=0.4)
            first = spy.call_count
            spy.reset_mock()
            T.astar_path(pillars, (2.0, 0.3), goal, clearance=0.4)
            second = spy.call_count
        self.assertGreater(first, 1000, "第一次调用应当要算大量格子")
        self.assertLess(second, 0.2 * first, f"第二次调用还在重算格子：{second} / {first}")

    def test_paths_identical_to_reference(self):
        rng = random.Random(20261110)
        n = 0
        for seed in SEEDS:
            for pillars in (sg.sample_pillars(seed)[0], sc.sample_course(seed, 3)[0]):
                goal = goal_of(pillars)
                for clearance in (0.4, 0.5):
                    for _ in range(6):
                        start = (rng.uniform(-1.0, goal[0] - 1.0), rng.uniform(-2.0, 2.0))
                        got = T.astar_path(pillars, start, goal, clearance=clearance)
                        want = astar_reference(pillars, start, goal, clearance=clearance)
                        self.assertEqual(got, want, f"seed={seed} start={start} clearance={clearance}")
                        n += 1
        self.assertGreater(n, 100)

    def test_start_inside_margin_not_cached_as_free(self):
        # 起点格被强制当自由格（旧实现写进本次调用的 free）；这个强制不能留在缓存里污染下一次调用
        pillars, _ = sc.sample_course(1, 2)
        goal = goal_of(pillars)
        p = pillars[0]
        near = (p["x"] - 0.5 * p["size"][0] - 0.1, p["y"])   # 离柱表面 0.1 m，在 0.4 m 余量内
        a = T.astar_path(pillars, near, goal, clearance=0.4)
        self.assertEqual(a, astar_reference(pillars, near, goal, clearance=0.4))
        b = T.astar_path(pillars, (p["x"] - 1.5, p["y"] + 1.0), goal, clearance=0.4)
        self.assertEqual(b, astar_reference(pillars, (p["x"] - 1.5, p["y"] + 1.0), goal, clearance=0.4))
        c = T.astar_path(pillars, near, goal, clearance=0.4)   # 再从同一个起点来一次
        self.assertEqual(c, a)

    def test_goal_blocked_and_out_of_grid(self):
        pillars, _ = sc.sample_course(1, 2)
        blocked_goal = (pillars[0]["x"], pillars[0]["y"])
        for goal in (blocked_goal, (99.0, 0.0)):
            self.assertEqual(T.astar_path(pillars, (-0.5, 0.0), goal, clearance=0.4),
                             astar_reference(pillars, (-0.5, 0.0), goal, clearance=0.4))

    def test_round_pillars_and_other_scenes_do_not_share_cache(self):
        rng = random.Random(7)
        trees = [{"name": f"t{i}", "x": 1.0 + 0.8 * i, "y": rng.uniform(-1.5, 1.5), "radius": 0.15, "size": [0.3, 0.3]} for i in range(8)]
        base, _ = sc.sample_course(113, 2)
        for scene in (trees, base, trees, base):   # 交替调用：不同场景不能串缓存
            goal = goal_of(scene)
            for clearance in (0.2, 0.4):
                got = T.astar_path(scene, (-0.5, 0.2), goal, clearance=clearance)
                self.assertEqual(got, astar_reference(scene, (-0.5, 0.2), goal, clearance=clearance))

    def test_moving_one_pillar_changes_the_result(self):
        pillars, _ = sc.sample_course(1, 2)
        goal = goal_of(pillars)
        T.astar_path(pillars, (-0.5, 0.0), goal, clearance=0.4)
        moved = [dict(p) for p in pillars]
        moved[3]["y"] = round(moved[3]["y"] + 0.4, 3)
        self.assertEqual(T.astar_path(moved, (-0.5, 0.0), goal, clearance=0.4),
                         astar_reference(moved, (-0.5, 0.0), goal, clearance=0.4))

    def test_cache_is_bounded(self):
        for seed in range(1, 40):
            pillars = sg.sample_pillars(seed)[0]
            T.astar_path(pillars, (-0.5, 0.0), goal_of(pillars), clearance=0.4)
        cache = getattr(T, "_FREE_CACHE", None)
        self.assertIsNotNone(cache, "teacher._FREE_CACHE 不存在")
        self.assertLessEqual(len(cache), T._FREE_CACHE_MAX_SCENES)


if __name__ == "__main__":
    unittest.main()
