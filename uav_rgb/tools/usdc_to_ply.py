#!/usr/bin/env python3
"""将 SAGE-3D 二进制 PXR-USDC 碰撞网格提取为世界系三角网格 PLY（供 sim.3dgs.collision.from_mesh 使用）。

运行方式:
  /root/rivermind-data/uav_rgb/envs/usd-venv/bin/python tools/usdc_to_ply.py \
      --usd /root/rivermind-data/uav_rgb/pubdata/sage3d/collision/840139_collision.usd \
      --out /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_840139/collision_mesh.ply \
      --z-min -0.3 --z-max 1.8 --open-doors
"""
import argparse
import os
import sys
import numpy as np

# ========================= 顶部集中配置区 =========================
DEFAULT_OPEN_DOORS = True   # CAD 资产中 /Door_* 为闭合门板，而 NuRec 实景与 VLN 轨迹穿门通行，默认开启门洞
# =================================================================


def write_mesh_ply(path, vertices, faces):
    v = np.asarray(vertices, dtype="<f4")
    f = np.asarray(faces, dtype="<i4")
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"顶点/面形状非法: v={v.shape}, f={f.shape}")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    face_dtype = np.dtype([("n", "<u1"), ("idx", "<i4", (3,))])
    packed_faces = np.empty(len(f), dtype=face_dtype)
    packed_faces["n"] = 3
    packed_faces["idx"] = f
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as out:
        out.write(header)
        out.write(v.tobytes())
        out.write(packed_faces.tobytes())
    os.replace(tmp, path)


def fan_triangulate(counts, indices):
    cnts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)
    if len(cnts) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if (cnts == 3).all():
        return idx.reshape(-1, 3)
    tris, pos = [], 0
    for c in cnts:
        for k in range(1, int(c) - 1):
            tris.append((idx[pos], idx[pos + k], idx[pos + k + 1]))
        pos += int(c)
    return np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)


def extract_usdc_mesh(usd_path, open_doors=DEFAULT_OPEN_DOORS, z_min=None, z_max=None):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"无法打开 USD 文件: {usd_path}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    v_list, f_list, offset, skipped_doors = [], [], 0, 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = str(prim.GetPath())
        if open_doors and ("/Door_" in prim_path or prim_path.endswith("/Door")):
            skipped_doors += 1
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_vt = mesh.GetPointsAttr().Get()
        cnts_vt = mesh.GetFaceVertexCountsAttr().Get()
        idx_vt = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts_vt or not cnts_vt:
            continue
        pts = np.array(pts_vt, dtype=np.float64)
        faces_local = fan_triangulate(cnts_vt, idx_vt)
        if len(faces_local) == 0:
            continue
        mat = np.array(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
        wpts = (pts @ mat[:3, :3] + mat[3, :3]) * mpu
        if z_min is not None or z_max is not None:
            tz = wpts[faces_local, 2]
            keep = np.ones(len(faces_local), dtype=bool)
            if z_min is not None:
                keep &= tz.max(axis=1) >= float(z_min)
            if z_max is not None:
                keep &= tz.min(axis=1) <= float(z_max)
            faces_local = faces_local[keep]
            if len(faces_local) == 0:
                continue
        used_v, remap = np.unique(faces_local, return_inverse=True)
        v_list.append(wpts[used_v])
        f_list.append(remap.reshape(-1, 3) + offset)
        offset += len(used_v)
    if not v_list:
        raise ValueError(f"USD 未提取到有效网格面: {usd_path}")
    return np.concatenate(v_list, axis=0), np.concatenate(f_list, axis=0), skipped_doors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usd", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--open-doors", dest="open_doors", action="store_true", default=DEFAULT_OPEN_DOORS)
    p.add_argument("--keep-closed-doors", dest="open_doors", action="store_false")
    args = p.parse_args(argv)
    verts, faces, n_doors = extract_usdc_mesh(args.usd, open_doors=args.open_doors, z_min=args.z_min, z_max=args.z_max)
    write_mesh_ply(args.out, verts, faces)
    print(f"wrote {args.out}: verts={len(verts)} faces={len(faces)} doors_opened={n_doors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
