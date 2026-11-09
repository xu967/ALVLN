#!/usr/bin/env python3
"""E1 MINI 机体外观的唯一一份几何：部件表 PARTS 与 sim/worlds/uav_rgb_e1.sdf 里 e1_mini/base_link 的
19 个 visual 逐项一致（test_e1_body.py 解析 SDF 对照）。不从这里生成 SDF：模板字节变了，sha 就变，采集数据
会被判成另一模板（VIZ 报告待确认 3）。

两域共用：
  draw_shaded   3DGS FOLLOW 格里的机体——部件表按位姿投影、画家算法 + 朗伯着色（Gazebo CHASE 由 Gazebo 自己渲染同一套 visual）
  draw_top_icon 两域 TOP 格里的机体图标——按地图比例的四桨盘 + 机臂 + 保护罩 + 机头红线，设最小像素尺寸
坐标：机体系 x 前、y 左、z 上（SDF 同）；cv2 作参数传入，便于测试注入假 cv2。
"""
import math

import numpy as np

_PCB = (0.05, 0.25, 0.10)       # 深绿 PCB（中心板 + 四臂）
_MOTOR = (0.25, 0.25, 0.25)
_PROP = (0.75, 0.75, 0.75)
_GUARD = (0.9, 0.9, 0.9)
_BATTERY = (0.10, 0.15, 0.45)
_CAMERA = (0.9, 0.2, 0.1)
_D = 0.7853981634               # 45°，SDF 里的字面量

# (SDF visual 名去掉 _visual, 形状, pose(x y z roll pitch yaw), 尺寸, 颜色 rgb 0..1, alpha)
#   box 尺寸 (sx, sy, sz)；cylinder 尺寸 (radius, length)，轴向 z。alpha = 1 - SDF <transparency>。
PARTS = (
    ("pcb_center", "box", (0, 0, 0, 0, 0, 0), (0.036, 0.036, 0.0016), _PCB, 1.0),
    ("arm_fl", "box", (0.021, 0.021, 0, 0, 0, _D), (0.068, 0.012, 0.0016), _PCB, 1.0),
    ("arm_fr", "box", (0.021, -0.021, 0, 0, 0, -_D), (0.068, 0.012, 0.0016), _PCB, 1.0),
    ("arm_rl", "box", (-0.021, 0.021, 0, 0, 0, -_D), (0.068, 0.012, 0.0016), _PCB, 1.0),
    ("arm_rr", "box", (-0.021, -0.021, 0, 0, 0, _D), (0.068, 0.012, 0.0016), _PCB, 1.0),
    ("motor_fl", "cylinder", (0.042, 0.042, 0.0078, 0, 0, 0), (0.00425, 0.014), _MOTOR, 1.0),
    ("motor_fr", "cylinder", (0.042, -0.042, 0.0078, 0, 0, 0), (0.00425, 0.014), _MOTOR, 1.0),
    ("motor_rl", "cylinder", (-0.042, 0.042, 0.0078, 0, 0, 0), (0.00425, 0.014), _MOTOR, 1.0),
    ("motor_rr", "cylinder", (-0.042, -0.042, 0.0078, 0, 0, 0), (0.00425, 0.014), _MOTOR, 1.0),
    ("prop_fl", "cylinder", (0.042, 0.042, 0.015, 0, 0, 0), (0.0325, 0.0015), _PROP, 0.5),
    ("prop_fr", "cylinder", (0.042, -0.042, 0.015, 0, 0, 0), (0.0325, 0.0015), _PROP, 0.5),
    ("prop_rl", "cylinder", (-0.042, 0.042, 0.015, 0, 0, 0), (0.0325, 0.0015), _PROP, 0.5),
    ("prop_rr", "cylinder", (-0.042, -0.042, 0.015, 0, 0, 0), (0.0325, 0.0015), _PROP, 0.5),
    ("guard_front", "box", (0.084, 0, 0.015, 0, 0, 0), (0.004, 0.168, 0.004), _GUARD, 1.0),
    ("guard_rear", "box", (-0.084, 0, 0.015, 0, 0, 0), (0.004, 0.168, 0.004), _GUARD, 1.0),
    ("guard_left", "box", (0, 0.084, 0.015, 0, 0, 0), (0.168, 0.004, 0.004), _GUARD, 1.0),
    ("guard_right", "box", (0, -0.084, 0.015, 0, 0, 0), (0.168, 0.004, 0.004), _GUARD, 1.0),
    ("battery", "box", (-0.012, 0, -0.0045, 0, 0, 0), (0.032, 0.017, 0.007), _BATTERY, 1.0),
    ("camera", "box", (0.045, 0, 0.015, 0, 0, 0), (0.009, 0.009, 0.009), _CAMERA, 1.0),
)
GUARD_HALF_M = 0.084            # 保护罩外沿半宽：整机宽 0.168 m（+0.004 罩厚）
ROTOR_Z_M = 0.015               # 桨盘 / 保护罩所在高度，机头线画在这一层
WIDTH_M = 2 * GUARD_HALF_M + 0.004
NOSE_BGR = (0, 0, 255)


def quat_to_rotmat(w, x, y, z):
    """单位四元数 (w,x,y,z) -> 3x3 旋转矩阵。"""
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)


def _rot_rpy(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]], np.float64)


def faces(parts=PARTS, n_sides=12):
    """部件表 -> 机体系面片 [(顶点 Nx3, bgr 0..255, alpha)]。盒 6 个四边形；圆柱 上下两个 n 边形 + n 个侧面。
    颜色在这里就转成 cv2 的 BGR 整数，画的时候只做光照乘法。"""
    out = []
    for _name, shape, pose, size, rgb, alpha in parts:
        r, t = _rot_rpy(*pose[3:]), np.asarray(pose[:3], np.float64)
        bgr = (int(round(rgb[2] * 255)), int(round(rgb[1] * 255)), int(round(rgb[0] * 255)))
        if shape == "box":
            hx, hy, hz = (0.5 * v for v in size)
            c = {(sx, sy, sz): (sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
            quads = (((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)),
                     ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)),
                     ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)))
            polys = [np.array([c[k] for k in q], np.float64) for q in quads]
        else:
            rad, length = size
            ang = np.linspace(0, 2 * math.pi, n_sides, endpoint=False)
            ring = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
            top = np.column_stack([ring, np.full(n_sides, 0.5 * length)])
            bot = np.column_stack([ring, np.full(n_sides, -0.5 * length)])
            polys = [top, bot[::-1]] + [np.array([bot[i], bot[(i + 1) % n_sides], top[(i + 1) % n_sides], top[i]])
                                        for i in range(n_sides)]
        out.extend((p @ r.T + t, bgr, alpha) for p in polys)
    return out


_FACES = None


def _fill(cv2, img, pts, color, alpha):
    """fillPoly；alpha<1 只在多边形包围盒内混合（整幅 addWeighted 太慢：桨盘 4×14 面 × 1280×720）。"""
    if alpha >= 1.0:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        return
    h, w = img.shape[:2]
    x0, y0 = max(0, int(pts[:, 0].min()) - 1), max(0, int(pts[:, 1].min()) - 1)
    x1, y1 = min(w, int(pts[:, 0].max()) + 2), min(h, int(pts[:, 1].max()) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.fillPoly(overlay, [pts - np.array([x0, y0], np.int32)], color, cv2.LINE_AA)
    roi[:] = cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0.0)


def draw_shaded(cv2, img, body_to_cam, k, light=(-0.4, -0.6, -0.7), ambient=0.35, parts=None, nose=True):
    """把机体画进相机图像：body_to_cam 是机体系->相机系 4x4（OpenCV 相机：x 右、y 下、z 前），k 是 3x3 内参。
    画家算法：面片按相机系平均深度从远到近 fillPoly；朗伯着色 ambient + (1-ambient)·max(0, n·l)，l 是相机系里
    指向光源的固定方向（默认左上、相机后方来光：顶面和朝相机的面亮，与 3DGS 画面的光照无关——机体本来就不在
    重建里）。任一顶点在相机后方（z<=0.05）的面不画。返回画到的像素包围盒 (x0, y0, x1, y1)，什么都没画返回 None。"""
    global _FACES
    if parts is None:
        if _FACES is None:
            _FACES = faces()
        fs = _FACES
    else:
        fs = faces(parts)
    rot, trans = body_to_cam[:3, :3], body_to_cam[:3, 3]
    lvec = np.asarray(light, np.float64)
    lvec /= np.linalg.norm(lvec)
    drawn = []
    for pts_b, bgr, alpha in fs:
        pc = pts_b @ rot.T + trans
        if (pc[:, 2] <= 0.05).any():
            continue
        n = np.cross(pc[1] - pc[0], pc[2] - pc[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ pc.mean(axis=0) > 0:         # 法线朝向相机一侧（面片两面都可能被看到）
            n = -n
        shade = ambient + (1.0 - ambient) * max(0.0, float(n @ lvec))
        uv = np.stack([k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2], k[1, 1] * pc[:, 1] / pc[:, 2] + k[1, 2]], axis=1)
        drawn.append((float(pc[:, 2].mean()), np.round(uv).astype(np.int32),
                      tuple(int(min(255, c * shade)) for c in bgr), alpha))
    if not drawn:
        return None
    drawn.sort(key=lambda d: -d[0])
    for _depth, pts, color, alpha in drawn:
        _fill(cv2, img, pts, color, alpha)
    allpts = np.concatenate([d[1] for d in drawn])
    if nose:
        # 机头红线：机体中心 -> 前保护罩中点，画在桨盘高度；线宽随机体像素尺寸走
        a = rot @ np.array([0.0, 0.0, ROTOR_Z_M]) + trans
        b = rot @ np.array([GUARD_HALF_M, 0.0, ROTOR_Z_M]) + trans
        if a[2] > 0.05 and b[2] > 0.05:
            pa = (int(round(k[0, 0] * a[0] / a[2] + k[0, 2])), int(round(k[1, 1] * a[1] / a[2] + k[1, 2])))
            pb = (int(round(k[0, 0] * b[0] / b[2] + k[0, 2])), int(round(k[1, 1] * b[1] / b[2] + k[1, 2])))
            width_px = allpts[:, 0].max() - allpts[:, 0].min()
            cv2.line(img, pa, pb, NOSE_BGR, max(1, int(round(width_px / 40))), cv2.LINE_AA)
    return (int(allpts[:, 0].min()), int(allpts[:, 1].min()), int(allpts[:, 0].max()), int(allpts[:, 1].max()))


def draw_top_icon(cv2, img, center, yaw, px_per_m, min_px=18, parts=PARTS):
    """顶视图标：四桨盘（按部件表位置/半径）+ 机臂 + 保护罩方框 + 机头红线，按地图比例画；整机像素宽小于
    min_px 时整体放大到 min_px（远景地图上仍看得出是四旋翼和朝向）。yaw 为 None 时不画机头线。
    图像约定与两域 TOP 一致：图右 = 世界 +x、图上 = 世界 +y。返回实际整机像素宽。"""
    scale = max(float(px_per_m), min_px / WIDTH_M)
    cx, cy = float(center[0]), float(center[1])
    ca, sa = (math.cos(yaw), math.sin(yaw)) if yaw is not None else (1.0, 0.0)

    def to_px(bx, by):
        wx, wy = bx * ca - by * sa, bx * sa + by * ca
        return int(round(cx + scale * wx)), int(round(cy - scale * wy))

    h = GUARD_HALF_M
    guard = np.array([to_px(h, h), to_px(-h, h), to_px(-h, -h), to_px(h, -h)], np.int32)
    cv2.polylines(img, [guard], True, (0, 0, 0), 3, cv2.LINE_AA)          # 黑描边让浅色地图上也看得见
    cv2.polylines(img, [guard], True, (235, 235, 235), 1, cv2.LINE_AA)
    rotors = [(pose[0], pose[1], size[0]) for name, shape, pose, size, _rgb, _a in parts if name.startswith("prop_")]
    for bx, by, rad in rotors:
        cv2.line(img, to_px(0, 0), to_px(bx, by), (40, 90, 40), 2, cv2.LINE_AA)
    for bx, by, rad in rotors:
        r_px = max(2, int(round(scale * rad)))
        cv2.circle(img, to_px(bx, by), r_px, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.circle(img, to_px(bx, by), max(1, r_px - 1), (200, 200, 200), -1, cv2.LINE_AA)
    if yaw is not None:
        cv2.line(img, to_px(0, 0), to_px(h, 0), NOSE_BGR, 2, cv2.LINE_AA)
    return int(round(scale * WIDTH_M))
