#!/usr/bin/env python3
"""e1_body 自检：部件表与 sim/worlds/uav_rgb_e1.sdf 的 19 个 visual 逐项一致；着色投影的尺寸 / 朝向 / 遮挡顺序；
顶视图标的最小尺寸与朝向。需要 numpy；绘制检查需要 cv2（.venv-tools / 3dgs-venv）。
用法：python3 sim/tools/test_e1_body.py
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_body as eb

SDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worlds", "uav_rgb_e1.sdf")


def check_parts_match_sdf():
    """SDF 是真相（Gazebo 画的就是它）；表里每一项的 pose / 几何 / 颜色 / 透明度都要对得上，一项不多一项不少。"""
    root = ET.parse(SDF).getroot()
    link = root.find(".//model[@name='e1_mini']/link[@name='base_link']")
    assert link is not None, "SDF 里没有 e1_mini/base_link"
    visuals = {v.get("name"): v for v in link.findall("visual")}
    names = {name + "_visual" for name, *_ in eb.PARTS}
    assert names == set(visuals), ("部件表与 SDF visual 名不一致", names ^ set(visuals))
    assert len(eb.PARTS) == 19
    for name, shape, pose, size, rgb, alpha in eb.PARTS:
        v = visuals[name + "_visual"]
        got_pose = tuple(float(t) for t in v.findtext("pose").split())
        assert len(got_pose) == 6 and all(abs(a - b) < 1e-12 for a, b in zip(got_pose, pose)), (name, got_pose, pose)
        geo = v.find("geometry")
        if shape == "box":
            got_size = tuple(float(t) for t in geo.findtext("box/size").split())
        else:
            got_size = (float(geo.findtext("cylinder/radius")), float(geo.findtext("cylinder/length")))
        assert geo.find(shape) is not None and all(abs(a - b) < 1e-12 for a, b in zip(got_size, size)), (name, got_size, size)
        diffuse = tuple(float(t) for t in v.findtext("material/diffuse").split())[:3]
        assert all(abs(a - b) < 1e-12 for a, b in zip(diffuse, rgb)), (name, diffuse, rgb)
        transparency = float(v.findtext("transparency") or 0.0)
        assert abs((1.0 - transparency) - alpha) < 1e-12, (name, transparency, alpha)
    # 几个派生常量也来自表：保护罩半宽、桨盘高度、整机宽
    guard = dict((n, p) for n, _s, p, *_ in eb.PARTS if n.startswith("guard_"))
    assert abs(guard["guard_front"][0] - eb.GUARD_HALF_M) < 1e-12 and abs(guard["guard_front"][2] - eb.ROTOR_Z_M) < 1e-12
    assert abs(eb.WIDTH_M - (2 * 0.084 + 0.004)) < 1e-12
    print("  ok  部件表 19 项与 uav_rgb_e1.sdf 的 visual 逐项一致（pose / 几何 / diffuse / transparency）")


def check_faces():
    fs = eb.faces()
    n_box = sum(1 for _n, s, *_ in eb.PARTS if s == "box")
    n_cyl = len(eb.PARTS) - n_box
    assert len(fs) == n_box * 6 + n_cyl * (2 + 12), len(fs)
    allpts = np.concatenate([p for p, _c, _a in fs])
    assert abs(allpts[:, 0].max() - 0.086) < 1e-9 and abs(allpts[:, 1].min() + 0.086) < 1e-9, "外沿应是保护罩 ±0.086"
    # 桨盘半透明、其余不透明；颜色已是 BGR 整数（相机块 rgb (0.9,0.2,0.1) -> bgr (26,51,230)）
    alphas = {a for _p, _c, a in fs}
    assert alphas == {0.5, 1.0}, alphas
    assert (26, 51, 230) in {c for _p, c, _a in fs}
    print("  ok  面片数、外沿尺寸、透明度、BGR 颜色")


def _cam_behind(back, up, hfov_deg, w, h):
    """相机在机体正后方 back、上方 up，看向机体：返回 (body_to_cam 4x4, K)。相机系 x 右、y 下、z 前。"""
    pitch = math.atan2(up, back)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # 机体系 (x 前, y 左, z 上) -> 相机系：相机前 = 机体 +x 方向俯视；先平移到相机位置再换轴
    r_world_to_cam = np.array([[0, -1, 0], [-sp, 0, -cp], [cp, 0, -sp]], np.float64)
    t = -r_world_to_cam @ np.array([-back, 0.0, up])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r_world_to_cam, t
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    return m, k


def check_draw_shaded():
    try:
        import cv2
    except ImportError:
        print("  skip 无 cv2：跳过着色投影检查")
        return
    w, h = 640, 360
    m, k = _cam_behind(1.1, 0.3, 60.0, w, h)
    img = np.zeros((h, w, 3), np.uint8)
    box = eb.draw_shaded(cv2, img, m, k)
    assert box is not None
    width = box[2] - box[0]
    # 最宽处是离相机最近的后保护罩（斜距 hypot(1.1-0.084, 0.3-0.015)），不是机体中心
    expect = (w / 2) / math.tan(math.radians(30)) * eb.WIDTH_M / math.hypot(1.1 - eb.GUARD_HALF_M, 0.3 - eb.ROTOR_Z_M)
    assert abs(width - expect) <= 4 and width >= 80, ("机体像素宽应 ≈ f·0.172/后罩斜距", width, expect)
    assert abs((box[0] + box[2]) / 2 - w / 2) <= 6, "机体应在画面水平中央"
    # 从正后方看：机头红线在上方（远端），机体中心 -> 前保护罩在图像上方向
    red = (img[:, :, 2] > 200) & (img[:, :, 1] < 60) & (img[:, :, 0] < 60)
    ys, xs = np.nonzero(red)
    assert len(ys) > 20 and ys.mean() < (box[1] + box[3]) / 2, "机头红线应画在机体远端（图像上方）"
    # 着色：有亮有暗（不是单色剪影）；桨盘半透明处能看到底色混合
    gray = img.max(axis=2)
    assert gray[gray > 0].std() > 20, "着色后应有明暗"
    # 机体在相机后方：什么都不画
    blank = np.zeros((h, w, 3), np.uint8)
    m2 = m.copy()
    m2[2, 3] -= 5.0
    assert eb.draw_shaded(cv2, blank, m2, k) is None and not blank.any()
    # 画家算法：近的盖远的。相机在机体正上方俯视：桨盘（z 0.015）在 PCB（z 0）之上，桨盘中心像素应是浅灰混合色而不是 PCB 深绿
    top = np.eye(4)
    top[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], np.float64)   # 相机 z 朝下
    top[:3, 3] = [0, 0, 1.0]
    img2 = np.full((h, w, 3), 128, np.uint8)
    eb.draw_shaded(cv2, img2, top, k)
    u, v = int(k[0, 0] * -0.042 / 1.0 + k[0, 2]), int(k[1, 1] * -0.042 / 1.0 + k[1, 2])
    pix = img2[v, u].astype(int)
    assert pix.min() > 80, ("桨盘中心应是浅色（半透明桨 + 底色），不是 PCB 深绿", pix.tolist())
    print(f"  ok  着色投影：1.1 m 后方 60° HFOV 640 宽 → 机体 {width} px（≥80）、居中、机头红线、明暗、后方不画、遮挡顺序")


def check_top_icon():
    try:
        import cv2
    except ImportError:
        print("  skip 无 cv2：跳过顶视图标检查")
        return
    img = np.full((360, 640, 3), 150, np.uint8)
    # 比例很小（0.172 m 只有 5 px）时放大到最小 18 px
    assert eb.draw_top_icon(cv2, img, (320, 180), 0.0, px_per_m=30.0, min_px=18) == 18
    assert eb.draw_top_icon(cv2, img, (320, 180), 0.0, px_per_m=500.0, min_px=18) == round(500 * eb.WIDTH_M)
    # 朝向：yaw=0 机头红线指向图右；yaw=90° 指向图上（图右 = +x、图上 = +y）
    for yaw, dx, dy in ((0.0, 1, 0), (math.pi / 2, 0, -1), (math.pi, -1, 0)):
        im = np.full((360, 640, 3), 150, np.uint8)
        eb.draw_top_icon(cv2, im, (320, 180), yaw, px_per_m=300.0)
        red = (im[:, :, 2] > 200) & (im[:, :, 1] < 80) & (im[:, :, 0] < 80)
        ys, xs = np.nonzero(red)
        assert len(xs) > 10
        vx, vy = xs.mean() - 320, ys.mean() - 180
        assert (vx * dx + vy * dy) > 0.8 * math.hypot(vx, vy), ("机头方向不对", yaw, vx, vy)
    im = np.full((360, 640, 3), 150, np.uint8)
    eb.draw_top_icon(cv2, im, (320, 180), None, px_per_m=300.0)
    assert not ((im[:, :, 2] > 200) & (im[:, :, 1] < 80)).any(), "yaw=None 不画机头线"
    print("  ok  顶视图标：最小 18 px、按比例放大、机头方向与两域 TOP 约定一致、yaw=None 不画机头")


def main():
    check_parts_match_sdf()
    check_faces()
    check_draw_shaded()
    check_top_icon()
    print("test_e1_body: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
