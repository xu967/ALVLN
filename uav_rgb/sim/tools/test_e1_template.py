#!/usr/bin/env python3
"""E1 MINI 世界模板单源检查：模板里的质量/惯量/碰撞体/相机安装位姿必须等于 e1_mini.json。

参数单源是 sim/dynamics/e1_mini.json（SITL 库 agent 写出）；它还不存在时退回硬件草案
$DATA_ROOT/uav_rgb/tmp/research-hw/e1_mini_params_draft.json（同名键）并打印警告。
另外检查：scene_gen 的两个注释标记、1 ms 步长、插件/话题约定、--vehicle/--camera-rate/
--no-aux-cameras 的生成结果、X3 默认路径逐字节不变。
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARAMS = os.path.join(REPO, "sim", "dynamics", "e1_mini.json")
DRAFT = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"),
                     "uav_rgb", "tmp", "research-hw", "e1_mini_params_draft.json")
TEMPLATE = sg.VEHICLES["e1_mini"]["template"]


def _num(text, tag):
    m = re.search(rf"<{tag}>([^<]+)</{tag}>", text)
    assert m, f"模板缺 <{tag}>"
    return float(m.group(1))


def _block(text, start_pat, end_pat):
    s = text.index(start_pat)
    return text[s:text.index(end_pat, s)]


def main():
    path = PARAMS if os.path.isfile(PARAMS) else DRAFT
    if path == DRAFT:
        print(f"警告: {sg._rel(PARAMS)} 不存在，按硬件草案 {DRAFT} 比对", file=sys.stderr)
    with open(path, encoding="utf-8") as f:
        prm = json.load(f)
    val = lambda k: prm[k]["value"]  # noqa: E731
    tpl = sg._read_template(TEMPLATE)

    # 1) scene_gen 标记、物理步长、载具识别
    for mark in (sg.START_MARK, sg.END_MARK, sg.LANDMARK_MARK, sg.LIGHT_MARK):
        assert tpl.count(mark) == 1, f"标记 {mark!r} 出现 {tpl.count(mark)} 次"
    assert sg.vehicle_of_template(tpl) == "e1_mini"
    assert _num(_block(tpl, "<physics", "</physics>"), "max_step_size") == 0.001

    # 2) 质量 / 惯量 / 碰撞体 / 相机位姿 == JSON
    model = _block(tpl, '<model name="e1_mini">', "</model>")
    assert abs(_num(model, "mass") - val("mass_flight")) < 1e-9, "mass != mass_flight"
    ixx, iyy, izz = val("inertia_diag")
    assert abs(_num(model, "ixx") - ixx) < 1e-12 and abs(_num(model, "iyy") - iyy) < 1e-12 \
        and abs(_num(model, "izz") - izz) < 1e-12, "inertia != inertia_diag"
    env = val("collision_envelope")
    coll = _block(model, '<collision name="base_link_collision">', "</collision>")
    assert abs(_num(coll, "radius") - env["radius_from_center"]) < 1e-9, "碰撞半径"
    assert abs(_num(coll, "length") - (env["z_max"] - env["z_min"])) < 1e-9, "碰撞高度"
    cz = float(re.search(r"<pose>\S+ \S+ (\S+)", coll).group(1))
    assert abs(cz - (env["z_max"] + env["z_min"]) / 2) < 1e-9, "碰撞体中心 z"
    cam = val("camera_mount")
    for sensor in ("camera", "depth"):
        blk = _block(model, f'<sensor name="{sensor}"', "</sensor>")
        pose = [float(v) for v in re.search(r"<pose>([^<]+)</pose>", blk).group(1).split()]
        assert pose[:3] == list(cam["xyz"]), (sensor, pose, cam)
        assert abs(pose[4] - (-cam["pitch_deg"] * 3.141592653589793 / 180.0)) < 1e-9, (sensor, pose)
        # 内参本轮不改：沿用 X3 的 320x240 / 60°
        assert "<width>320</width>" in blk and "<height>240</height>" in blk, sensor
        assert "1.0471975512" in blk, sensor
    # 3) 插件与话题约定（UI/demo 依赖）
    for s in ('filename="CfSitlSystem"', "<params>sim/dynamics/e1_mini.json</params>",
              "/e1_mini/action", "/e1_mini/cmd", "/e1_mini/state", "/uav/camera", "/uav/depth",
              "/uav/contact", "/uav/chase_camera", "/uav/top_camera",
              "<odom_publish_frequency>200</odom_publish_frequency>", '<model name="chase_cam">'):
        assert s in tpl, f"模板缺 {s}"
    assert "MulticopterMotorModel" not in tpl and "MulticopterVelocityControl" not in tpl
    assert "rotor_0" not in tpl

    # 4) scene_gen 生成：确定性、spawn 写进 e1_mini pose、相机帧率、辅助相机删除
    w1, m1 = sg.build_scene(tpl, 3, 3, "random", appearance="v1")
    w2, _ = sg.build_scene(tpl, 3, 3, "random", appearance="v1")
    assert w1 == w2 and m1["vehicle"] == "e1_mini" and m1["dynamics_profile"] == "cf_sitl_v1"
    pose = re.search(r'<model name="e1_mini">\s*<pose>([^<]*)</pose>', w1).group(1).split()
    assert abs(float(pose[0]) - m1["spawn"]["x"]) < 1e-6 and abs(float(pose[1]) - m1["spawn"]["y"]) < 1e-6
    assert sg.camera_rate_of(w1) == 10.0, sg.camera_rate_of(w1)
    w5 = sg.set_camera_rate(w1, 5)
    assert sg.camera_rate_of(w5) == 5.0 and w5.count("<update_rate>5</update_rate>") == 2
    assert "<update_rate>15</update_rate>" in w5 and "<update_rate>100</update_rate>" in w5  # chase/contact 不动
    stripped = sg.strip_aux_cameras(w1)
    assert '<model name="chase_cam">' not in stripped and '<model name="studio_top_camera">' not in stripped
    assert "<topic>/uav/chase_camera</topic>" not in stripped and "<topic>/uav/top_camera</topic>" not in stripped
    assert "/uav/camera" in stripped and "/uav/depth" in stripped and "/uav/contact" in stripped
    assert "<!-- aux-camera:" not in stripped and "<!-- /aux-camera -->" not in stripped
    # 几何与 X3 完全一致：同 seed 的 pillars 相同（只换载具）
    _, mx3 = sg.build_scene(sg._read_template(sg.DEFAULT_TEMPLATE), 3, 3, "random", appearance="v1")
    assert mx3["pillars"] == m1["pillars"] and mx3["spawn"] == m1["spawn"]
    assert mx3["vehicle"] == "x3" and mx3["dynamics_profile"] == "x3_v1"
    # X3 模板没有 aux 标记：--no-aux-cameras 必须报错而不是静默
    try:
        sg.strip_aux_cameras(sg._read_template(sg.DEFAULT_TEMPLATE))
    except ValueError:
        pass
    else:
        raise AssertionError("X3 模板 strip_aux_cameras 应报错")

    # 5) templates/ 归档副本（内容寻址）与 live 模板一致
    sha = hashlib.sha256(tpl.encode("utf-8")).hexdigest()
    archived = os.path.join(REPO, "sim", "worlds", "templates", f"uav_rgb_e1.{sha[:8]}.sdf")
    assert os.path.isfile(archived), f"缺归档模板 {sg._rel(archived)}（内容 sha256 {sha[:8]}）"
    with open(archived, encoding="utf-8") as f:
        assert f.read() == tpl, "归档模板与 live 模板不一致"
    print(f"test_e1_template: PASS（参数源 {os.path.basename(path)}；模板 sha256 {sha[:8]}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
