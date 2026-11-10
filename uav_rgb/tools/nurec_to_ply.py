#!/usr/bin/env python3
"""NVIDIA NuRec（SAGE-3D InteriorGS 的 .usdz 里的 <id>.usdz.nurec：gzip + msgpack，数组为 fp16 字节串）-> INRIA 3DGS binary PLY。

解释器：/opt/conda/bin/python3。msgpack 只在它里面，3dgs-venv 与 .venv-tools 都没有；缺 msgpack 时本脚本报错并提示，
不往现有 venv 装包。其余部分只用 numpy / 标准库。
  /opt/conda/bin/python3 tools/nurec_to_ply.py <in.usdz> <out.ply>        # 打印一行 JSON：高斯数、PLY 字节数

输出 PLY 与 sim/3dgs/scene.read_gaussian_ply 的 PLY_PROPS 同序（x y z nx ny nz f_dc_0-2 f_rest_0-44 opacity scale_0-2 rot_0-3，
全 float32）。opacity 写 NuRec 的 densities 原值（sigmoid 之前）、scale 写原值（取 exp 之前），读端自己激活；NuRec 与 INRIA 同口径，
bench 报告 2.3 在 841150 上用 gsplat 渲染验证过颜色。
"""
import gzip
import json
import os
import sys
import zipfile

import numpy as np

STATE_PREFIX = ".gaussians_nodes.gaussians."
NEEDED = ("positions", "rotations", "scales", "densities", "features_albedo", "features_specular")
DTYPES = {2: np.float16, 4: np.float32}
N_REST = 45                                    # 球谐 3 阶：15 个系数 × 3 通道
PROPS = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(N_REST)]
         + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def require_msgpack():
    try:
        import msgpack
    except ImportError:
        raise SystemExit("缺 msgpack：本脚本要用 /opt/conda/bin/python3 跑（msgpack 只在它里面；"
                         "3dgs-venv 与 .venv-tools 里没有，也不要往现有 venv 装包）") from None
    return msgpack


def decode_nurec(raw):
    """.nurec 字节（gzip + msgpack）-> {名字: float32 数组}，只取 NEEDED 六项；fp16 / fp32 都认，形状取 `<名字>.shape`。"""
    msgpack = require_msgpack()
    state = msgpack.unpackb(gzip.decompress(raw), raw=False, strict_map_key=False)["nre_data"]["state_dict"]
    out = {}
    for name in NEEDED:
        if STATE_PREFIX + name not in state or STATE_PREFIX + name + ".shape" not in state:
            raise ValueError(f"NuRec state_dict 缺 {STATE_PREFIX}{name}（或它的 .shape）")
        buf, shape = state[STATE_PREFIX + name], [int(v) for v in state[STATE_PREFIX + name + ".shape"]]
        n = int(np.prod(shape))
        item = len(buf) // n if n else 0
        if item not in DTYPES or item * n != len(buf):
            raise ValueError(f"{name}: 字节数 {len(buf)} 与形状 {shape} 对不上（每元素 {len(buf) / max(n, 1):g} 字节，只认 fp16 / fp32）")
        out[name] = np.frombuffer(buf, dtype=DTYPES[item]).reshape(shape).astype(np.float32)
    return out


def to_inria_table(a):
    """六项数组 -> (属性名列表, float32 [N, 62] 表)。

    f_rest 必须转置：NuRec 的 features_specular 是 [N, 15 个系数, 3 个通道]，INRIA 的 f_rest_{c*15+k} 是通道在前
    （第一版没转置，墙面渲染出粉绿色斑块，bench 报告 2.3）。
    """
    pos, rot, sca, den = a["positions"], a["rotations"], a["scales"], a["densities"]
    dc, rest = a["features_albedo"], a["features_specular"]
    n = len(pos)
    if rest.size != n * N_REST:
        raise ValueError(f"features_specular 元素数 {rest.size} != N×45（N={n}）")
    tab = np.zeros((n, len(PROPS)), dtype=np.float32)
    tab[:, 0:3] = pos
    tab[:, 6:9] = dc.reshape(n, 3)
    tab[:, 9:9 + N_REST] = rest.reshape(n, 15, 3).transpose(0, 2, 1).reshape(n, N_REST)
    tab[:, 54] = den.reshape(n)
    tab[:, 55:58] = sca.reshape(n, 3)
    tab[:, 58:62] = rot.reshape(n, 4)
    return list(PROPS), tab


def write_inria_ply(path, names, table):
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(table)}"]
    header += [f"property float {k}" for k in names] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(table, dtype="<f4").tobytes())


def convert(usdz_path, ply_path):
    """usdz（zip，内含 *.nurec）-> PLY；返回 {"n_gaussians", "ply_bytes", "props"}。"""
    with zipfile.ZipFile(usdz_path) as z:
        names = [n for n in z.namelist() if n.endswith(".nurec")]
        if not names:
            raise ValueError(f"{usdz_path} 里没有 .nurec 文件（zip 内容：{z.namelist()[:8]}）")
        raw = z.read(names[0])
    names, table = to_inria_table(decode_nurec(raw))
    write_inria_ply(ply_path, names, table)
    return {"n_gaussians": int(len(table)), "ply_bytes": os.path.getsize(ply_path), "props": len(names)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(convert(argv[0], argv[1])))


if __name__ == "__main__":
    main()
