#!/usr/bin/env python3
"""规则层上限的 3DGS 侧 oracle 策略（任务卡 K1）：各列可通行距离 + 规则 v1 + 路线方位，零训练。

可判定性分解里「图像决定得了的量」是各列到最近障碍的水平距离，「图像决定不了的量」是航向。这里两样都用特权信息给：
  列距离（20 列 × 3° = 60° 视场，上限 4.0 m，来源二选一，两个都算、只用主来源决策，另一个落日志做同位姿对照）：
    occ    在碰撞判定用的那个 Occupancy 上按飞行高度沿各列射线步进（每列 5 条射线含两条边缘，与 rule_policy.column_distances 同一套），
           首个碰撞判定为真的距离；步长 <= 半个体素，找到首个命中区间后二分到 ~0.4 mm。量到的是机体包络膨胀后的栅格边界，
           约等于「表面距离 − 0.1~0.2 m」（体素 0.1 / 0.2 m 的场景；膨胀 1 格），不是表面本身——规则的 r_safe 已含机体半径，这里再算一次，
           所以 r_safe 只在开发路线上选（见实验入口）。
    depth  渲染深度图（z 深度，米，0 = 无效）反投影，用动力学对象的 camera_viewmat（= 渲染用的同一个机体四元数 ⊗ camera_mount）换到 nav 系，
           只留高度在「机体中心 ± (包络半高 + 0.10 m)」内的点，按相对机头的方位角分到 20 列，每列取离相机最近的水平距离。
           深度是 z 深度不是射线距离：见 experiments/2026-11-10-rule-oracle-3dgs/depth_geometry_check.py 在真渲染器上对拍的结果。
  航向计数 n：Gazebo 里是已执行 YAW_LEFT 数减 YAW_RIGHT 数（正 = 机头偏左于目标方向）；3DGS 里换成路线方位误差的整数步数：
    err = guard.bearing_error_deg(...)（正 = 目标在左），n = round(−err / yaw_step_deg)。目标在左 -> n < 0 -> 规则的「转回」是 YAW_LEFT。
  规则：rule_policy.rule_v1_step，与 Gazebo 同一个函数、同一份 RULE_V1，只在构造时覆盖 r_safe / d_look / yaw_step_deg 这类数值；
    永不输出 SAFE_NEUTRAL；记忆按集清零（run_episode 每集开头调 reset()）。

接口：就是一个 policy (rgb, depth, state) -> (action, infer_ms)，state = 动力学对象（x / y / z / yaw 弧度，depth 来源还要 camera_viewmat）。
故意不带 stop_radius / prev / trace：没有 goal-stop（到点判定仍由评测端做，到点后继续飞），也不触发 driver 里护栏统计的分支；
逐步日志在 self.rows（driver 落成 rule.jsonl）。名字带 Oracle：occ 来源用了碰撞真值，航向用了真值位姿与占据 A* 路径，都是特权信息。

shortcut: depth 列只取包络高度带内的点，看不见带外的门楣 / 桌面 / 悬空物，何时升级：K1 的失败集里出现「撞到带外几何」就加多层列距离；
shortcut: n 用整数四舍五入，|err| < yaw_step/2 记 0（死区约 ±16°），何时升级：死区内的残余误差累积成明显偏航时改成带滞回的计数。
"""
import math
import os
import sys

import numpy as np

from .guard import bearing_error_deg
from .planner import FORWARD_SLOW, SAFE_NEUTRAL, YAW_LEFT, YAW_RIGHT  # noqa: F401  SAFE_NEUTRAL 供测试 / 调用方引用

_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import rule_policy as rp  # noqa: E402  sim/tools：规则 v1 与 Gazebo 侧同一份，这里只调用，不复制、不改

COL_DEG, N_RAYS = rp.COL_DEG, rp.N_RAYS
MAX_RANGE_M = 4.0           # = column_distances / obstacle_points 的默认 max_range，也是 RULE_V1["sum_clip"]；改它三处一起改
CAM_FORWARD_M = 0.045       # = e1_mini.json camera_mount.xyz[0] = rule_policy.obstacle_points 的 cam_forward（规则里写死，不能单改）
BAND_MARGIN_M = 0.10        # depth 来源的高度带：包络半高之外再加 0.10 m（任务卡）
REFINE_ITERS = 7            # 二分次数：半体素 0.05 m / 2^7 ≈ 0.4 mm
SOURCES = ("occ", "depth")
POLICY_VERSION = "uavrgb.rule_oracle.v1"
POLICY_VERSION_V2 = "uavrgb.rule_oracle.v2"      # 任务卡 K3：rule_version = "v2" 时的版本串；v1 的 POLICY_VERSION 不变
RULE_VERSIONS = ("v1", "v2")


def _hit(occ, px, py, pz):
    """与 Occupancy.collides / sweep 同一判定（地面 | 膨胀栅格）的向量化版本：px/py/pz 同形，返回同形 bool。"""
    pts = np.stack([px.reshape(-1), py.reshape(-1), pz.reshape(-1)], axis=1)
    return (occ._ground_hit(pts[:, 2]) | occ._grid_hit(pts)).reshape(px.shape)


def occ_columns(occ, x, y, z, yaw_deg, hfov_deg=60.0, col_deg=COL_DEG, n_rays=N_RAYS, max_range=MAX_RANGE_M,
                cam_forward=CAM_FORWARD_M, refine=True):
    """从左到右各列到最近占据的水平距离（m，沿射线量，上限 max_range），形状 [n_cols]。

    (x, y, z) 机体中心；射线从相机出发（机体前 cam_forward、同高度 z、朝向与机头一致）。
    第 j 列中心方位 (n/2 − 0.5 − j)·col_deg（正 = 左），扇区内 n_rays 条射线均匀取、含两条边缘，取最小。
    采样间距 = max_range / ceil(max_range / (voxel/2)) <= 半个体素；refine=True 时对首个命中的区间二分，起点就在占据里记 0。
    """
    n_cols = round(hfov_deg / col_deg)
    j = np.arange(n_cols)[:, None]
    k = np.arange(n_rays)[None, :]
    phi = (n_cols / 2 - 0.5 - j) * col_deg + (k / (n_rays - 1) - 0.5) * col_deg          # [C, R] 相对机头的方位（度）
    th = np.radians(yaw_deg + phi).reshape(-1)                                            # [C·R] nav 系射线角
    ux, uy = np.cos(th), np.sin(th)
    yaw = math.radians(yaw_deg)
    ox, oy = x + cam_forward * math.cos(yaw), y + cam_forward * math.sin(yaw)
    ts = np.linspace(0.0, max_range, int(math.ceil(max_range / (0.5 * occ.voxel))) + 1)
    hit = _hit(occ, ox + ts[None, :] * ux[:, None], oy + ts[None, :] * uy[:, None], np.full((len(th), len(ts)), float(z)))
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    d = np.where(any_hit, ts[first], max_range)
    if refine:
        idx = np.nonzero(any_hit & (first > 0))[0]
        if len(idx):
            lo, hi = ts[first[idx] - 1], ts[first[idx]]                                   # lo 不撞，hi 撞
            for _ in range(REFINE_ITERS):
                mid = 0.5 * (lo + hi)
                h = _hit(occ, ox + mid * ux[idx], oy + mid * uy[idx], np.full(len(idx), float(z)))
                hi, lo = np.where(h, mid, hi), np.where(h, lo, mid)
            d[idx] = hi
    return d.reshape(n_cols, n_rays).min(axis=1)


def depth_columns(depth, view_nav, body_z, yaw_deg, half_height_m, hfov_deg=60.0, col_deg=COL_DEG,
                  max_range=MAX_RANGE_M, band_margin_m=BAND_MARGIN_M, depth_is="z"):
    """渲染深度图 -> 各列最近水平距离（m，量自相机，上限 max_range），形状 [n_cols]；列定义同 occ_columns。

    depth [H, W]（米，0 / 非有限 = 无效像素，不产生点）；view_nav = nav -> camera 4x4（state.camera_viewmat(None)：
    机体四元数 ⊗ camera_mount，roll / pitch 都在里面，所以高度按完整姿态换到水平系再切带）；
    body_z 机体中心 nav 高度；yaw_deg 机头方位（度）；half_height_m 机体包络半高（占据代理同值）。
    像素中心 (u+0.5, v+0.5)、方形像素、主点在图像中心（scene.intrinsics）；depth_is="z"：像素值是相机系 z（gsplat expected depth）；
    "ray"：像素值是沿射线的距离——只给 depth_geometry_check 做反事实对照，策略不用。
    """
    if depth_is not in ("z", "ray"):
        raise ValueError(f"depth_is 必须是 z 或 ray，得到 {depth_is!r}")
    n_cols = round(hfov_deg / col_deg)
    out = np.full(n_cols, float(max_range))
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    fx = 0.5 * w / math.tan(math.radians(hfov_deg) / 2.0)
    vv, uu = np.nonzero(np.isfinite(depth) & (depth > 0.0))
    if len(vv) == 0:
        return out
    a, b = (uu + 0.5 - w / 2.0) / fx, (vv + 0.5 - h / 2.0) / fx                          # 相机系 x/z（右）、y/z（下）
    zc = depth[vv, uu]
    if depth_is == "ray":
        zc = zc / np.sqrt(1.0 + a * a + b * b)
    pc = np.stack([a * zc, b * zc, zc], axis=1)
    r_cn, t_c = np.asarray(view_nav, dtype=np.float64)[:3, :3], np.asarray(view_nav, dtype=np.float64)[:3, 3]
    p = (pc - t_c) @ r_cn                                                                 # R^T (p_c − t)：相机系 -> nav
    cam = -(r_cn.T @ t_c)                                                                 # 相机中心（nav）
    keep = np.abs(p[:, 2] - body_z) <= half_height_m + band_margin_m
    dx, dy = p[keep, 0] - cam[0], p[keep, 1] - cam[1]
    r = np.hypot(dx, dy)
    az = (np.degrees(np.arctan2(dy, dx)) - yaw_deg + 180.0) % 360.0 - 180.0              # 相对机头，正 = 左
    col = np.floor((hfov_deg / 2.0 - az) / col_deg).astype(int)                           # 第 0 列 = 最左 [+30°, +27°]
    ok = (col >= 0) & (col < n_cols)
    col, r = col[ok], r[ok]
    if len(col):
        order = np.argsort(r)
        uc, first = np.unique(col[order], return_index=True)                              # 每列第一个 = 最近
        out[uc] = np.minimum(out[uc], r[order][first])
    return out


class RuleOraclePolicy:
    """规则 v1 + 路线（或 waypoint）方位 + 完美列距离。每集新建一个，或让 run_episode 在集首调 reset()。

    occ        碰撞判定同一个 Occupancy（occ 来源的列距离、depth 来源的包络半高都从它来）；
    target_fn  (x, y) -> (tx, ty)：路线臂传 guard.OracleRouteGuide.target，终点臂传返回 waypoint 的函数；
    source     "occ" | "depth"：决策用哪一路列距离；
    params     覆盖 rule_policy.RULE_V1（rule_version="v2" 时是 RULE_V2）的数值，未知键抛 ValueError；v2 的 levels / order 等在构造时校验；
    shadow     True = 另一路列距离也算并落日志（同位姿一致性分析用）；depth 为 None 时跳过 depth。
    rule_version  "v1"（缺省，与 K1 逐位相同）| "v2"（任务卡 K3：rule_policy.rule_v2_step，两侧候选 + 分级通道 + 禁忌航向）。
               v2 的 n 仍是 round(−err / yaw_step_deg)，另把浮点值 e = −err / yaw_step_deg 交给规则做候选排序；
               上一步返回的动作（prev_action）由本类记，供「只许静止起步」的级判断。
    rows       逐步日志：step / err_deg / n / target / action / entry / mem_n / clear / src / d_occ / d_depth；
               v2 在 n 之后多 e，在 clear 之后多 level / m / tabu_n / ok（各级 × m = −n_cap … n_cap 的位图），其余键与次序同 v1。
    """
    name = "rule_oracle"            # manifest 的 teacher 字段；executed_by 会按「非 forward / student」记成 teacher，来源看 planner 字段
    planner_name = "rule_v1"
    version = POLICY_VERSION

    def __init__(self, occ, target_fn, source="occ", params=None, hfov_deg=60.0, shadow=True, rule_version="v1"):
        if source not in SOURCES:
            raise ValueError(f"source 必须是 {SOURCES}，得到 {source!r}")
        if rule_version not in RULE_VERSIONS:
            raise ValueError(f"rule_version 必须是 {RULE_VERSIONS}，得到 {rule_version!r}")
        table = rp.RULE_V1 if rule_version == "v1" else rp.RULE_V2
        unknown = sorted(set(params or {}) - set(table))
        if unknown:
            raise ValueError(f"未知规则参数 {unknown}；{rule_version} 可用 {sorted(table)}")
        self.occ, self.target_fn, self.source = occ, target_fn, source
        self.params = {**table, **(params or {})}
        self.rule_version = rule_version
        if rule_version == "v2":
            rp.check_v2_params(self.params)                              # 构造时就报错：levels 不逐级收紧、order 不认识等
            self.planner_name, self.version = "rule_v2", POLICY_VERSION_V2   # 实例属性盖掉类属性；v1 实例不设，保持 K1 的值
        self.hfov_deg, self.shadow = float(hfov_deg), bool(shadow)
        self.reset()

    def reset(self):
        """集边界：记忆、步号、上次转向、日志、条目计数全部清零（规则的记忆按集清零）。"""
        self.memory, self.step_idx, self.last_turn, self.rows = [], 0, None, []
        self.tabu, self.last_action = [], None                           # 仅 v2：禁忌表、上一步返回的动作
        self.entry_counts = {e: 0 for e in (rp.V1_ENTRIES if self.rule_version == "v1" else rp.V2_ENTRIES)}

    def __call__(self, rgb, depth, state):
        x, y, z = float(state.x), float(state.y), float(getattr(state, "z", 2.0))
        yaw_deg = math.degrees(float(state.yaw))
        if self.source == "depth" and depth is None:
            raise ValueError("source=depth 需要深度图（run_episode 会传）")
        d_occ = d_depth = None
        if self.source == "occ" or self.shadow:
            d_occ = occ_columns(self.occ, x, y, z, yaw_deg, self.hfov_deg)
        if depth is not None and (self.source == "depth" or self.shadow):
            d_depth = depth_columns(depth, state.camera_viewmat(None), z, yaw_deg, self.occ.body_half_height, self.hfov_deg)
        d = d_occ if self.source == "occ" else d_depth
        tx, ty = self.target_fn(x, y)
        err = bearing_error_deg(x, y, yaw_deg, (tx, ty))
        n = int(round(-err / self.params["yaw_step_deg"]))          # 目标在左 err > 0 -> n < 0 -> 规则朝左转回（YAW_LEFT）
        if self.rule_version == "v1":
            action, entry, self.memory, info = rp.rule_v1_step(self.memory, self.step_idx, x, y, yaw_deg, [float(v) for v in d], n,
                                                               self.last_turn, self.hfov_deg, self.params)
            extra = {}
        else:
            e = -err / self.params["yaw_step_deg"]                  # n 的浮点值（同一个符号约定）：规则用它给两侧候选排序
            action, entry, self.memory, self.tabu, info = rp.rule_v2_step(
                self.memory, self.tabu, self.step_idx, x, y, yaw_deg, [float(v) for v in d], n, self.last_turn, e=e,
                prev_action=self.last_action, hfov_deg=self.hfov_deg, params=self.params)
            extra = {"level": info["level"], "m": info["m"], "tabu_n": info["tabu_n"], "ok": info["ok"]}
        if action in (YAW_LEFT, YAW_RIGHT):
            self.last_turn = action
        self.last_action = int(action)
        self.entry_counts[entry] += 1
        row = {"step": self.step_idx, "err_deg": round(err, 2), "n": n}
        if self.rule_version == "v2":                                # v2 多出来的键：e 在 n 之后，level / m / tabu_n / ok 在 clear 之后
            row["e"] = round(e, 4)
        row.update({"target": [round(tx, 3), round(ty, 3)], "action": int(action), "entry": entry, "mem_n": info["mem_n"],
                    "clear": [int(c) for c in info["clear"]]})
        row.update(extra)
        row.update({"src": self.source,
                    "d_occ": None if d_occ is None else [round(float(v), 3) for v in d_occ],
                    "d_depth": None if d_depth is None else [round(float(v), 3) for v in d_depth]})
        self.rows.append(row)
        self.step_idx += 1
        return int(action), None
