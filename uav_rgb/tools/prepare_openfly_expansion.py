#!/usr/bin/env python3
"""OpenFly 6 场景 100 条因果扩量拉取、4/5 原语核验与 3D 空间簇无泄漏划分。

运行方式:
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python tools/prepare_openfly_expansion.py
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from urllib.parse import urljoin, urlsplit

import numpy as np
from PIL import Image
import requests

# ========================= 顶部集中配置区 =========================
CANDIDATE_JSON = Path(os.environ.get(
    "OPENFLY_CANDIDATE_JSON",
    "/root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/data_expansion_candidate.json",
))  # 预筛选的 6 场景 100 条官方 train 候选清单
OUTPUT_DIR = Path(os.environ.get(
    "OPENFLY_EXPANSION_OUT",
    "/root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/six_scene_causal_bc_v1",
))  # 因果组首图像、manifest.json 与 spatial_audit.json 输出目录
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")  # 国内可达镜像域名
SOURCE_REPO = "IPEC-COMMUNITY/OpenFly"  # 上游数据集仓库名
SOURCE_REVISION = "a12316d56a4e35a32ad626fb725ed7089937a1c4"  # 冻结提交哈希，防上游变更漂移
HF_TOKEN_FILE = Path(os.environ.get(
    "HF_TOKEN_FILE", "/root/.cache/huggingface/token",
))  # HuggingFace 只读访问令牌路径
CORRIDOR_DISTANCE_M = float(os.environ.get("OPENFLY_CORRIDOR_M", "30.0"))  # 同场景 3D 轨迹点 <30m 视为共享走廊聚入同一簇
SPLIT_SEED = int(os.environ.get("OPENFLY_SPLIT_SEED", "20261116"))  # 空间簇确定性划分种子（约 80/20）
WORKERS = int(os.environ.get("OPENFLY_WORKERS", "6"))  # 路线级并发下载线程数；兼顾镜像限流与吞吐
RANGE_CHUNKS = int(os.environ.get("OPENFLY_RANGE_CHUNKS", "4"))  # 单个 Parquet 分段 Range 并发数
MAX_ATTEMPTS = 5  # 单次 HTTP 请求最大重试次数
# =================================================================

PRIMITIVE_MAP = {
    ("go straight", 3): 1,
    ("turn left", 30): 2,
    ("turn right", 30): 3,
    ("go up", 3): 4,
    ("go down", 3): 5,
    ("stop", 0): 0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_session() -> requests.Session:
    client = requests.Session()
    client.trust_env = False
    client.headers["Accept-Encoding"] = "identity"
    return client


def resolve_redirect(client: requests.Session, url: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}", "Range": "bytes=0-0"} if token else {"Range": "bytes=0-0"}
    resp = client.get(url, headers=headers, allow_redirects=False, timeout=(15, 60))
    try:
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            require(bool(loc), f"重定向缺少 Location: {url}")
            return urljoin(url, loc)
        resp.raise_for_status()
        return url
    finally:
        resp.close()


def http_get_bytes(repo_path: str, token: str, byte_range=None) -> bytes:
    url = f"{HF_ENDPOINT}/datasets/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{repo_path}"
    for attempt in range(MAX_ATTEMPTS):
        try:
            with make_session() as client:
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                if byte_range is not None:
                    headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
                resp = client.get(url, headers=headers, allow_redirects=False, timeout=(15, 90))
                if resp.status_code in (301, 302, 303, 307, 308):
                    target = urljoin(url, resp.headers["Location"])
                    resp.close()
                    if urlsplit(target).netloc != urlsplit(url).netloc:
                        headers.pop("Authorization", None)
                    resp = client.get(target, headers=headers, timeout=(15, 120))
                try:
                    resp.raise_for_status()
                    return resp.content
                finally:
                    resp.close()
        except Exception:
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(min(3 * (attempt + 1), 15))
    raise RuntimeError("unreachable")


def fetch_parquet_parallel(repo_path: str, token: str, range_chunks: int = RANGE_CHUNKS) -> bytes:
    url = f"{HF_ENDPOINT}/datasets/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{repo_path}"
    for attempt in range(MAX_ATTEMPTS):
        try:
            with make_session() as client:
                cdn_url = resolve_redirect(client, url, token)
                same_host = urlsplit(cdn_url).netloc == urlsplit(url).netloc
                base_headers = {"Authorization": f"Bearer {token}"} if (token and same_host) else {}
                probe = client.get(cdn_url, headers={**base_headers, "Range": "bytes=0-0"}, timeout=(15, 30))
                try:
                    if probe.status_code != 206 or "Content-Range" not in probe.headers:
                        probe.close()
                        full = client.get(cdn_url, headers=base_headers, timeout=(15, 180))
                        full.raise_for_status()
                        return full.content
                    total = int(probe.headers["Content-Range"].split("/")[-1])
                finally:
                    probe.close()
            step = max(1 << 20, (total + range_chunks - 1) // range_chunks)
            ranges = [(i, min(i + step - 1, total - 1)) for i in range(0, total, step)]

            def pull_chunk(rng):
                for sub in range(MAX_ATTEMPTS):
                    try:
                        with make_session() as sub_client:
                            r = sub_client.get(
                                cdn_url,
                                headers={**base_headers, "Range": f"bytes={rng[0]}-{rng[1]}"},
                                timeout=(15, 120),
                            )
                            try:
                                require(r.status_code == 206, f"Range 状态异常: {r.status_code}")
                                data = r.content
                                require(len(data) == rng[1] - rng[0] + 1, "Range 字节长度不匹配")
                                return data
                            finally:
                                r.close()
                    except Exception:
                        if sub + 1 == MAX_ATTEMPTS:
                            raise
                        time.sleep(2 * (sub + 1))

            with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
                data = b"".join(pool.map(pull_chunk, ranges))
            require(len(data) == total, "Parquet 拼接长度不匹配")
            return data
        except Exception:
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(min(4 * (attempt + 1), 20))
    raise RuntimeError("unreachable")


def snappy_decompress(data: bytes) -> bytes:
    mv = memoryview(data)
    pos, limit = 0, len(mv)
    expected_len, shift = 0, 0
    while True:
        b = mv[pos]
        pos += 1
        expected_len |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    out = bytearray()
    while pos < limit:
        tag = mv[pos]
        pos += 1
        kind = tag & 0x03
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(mv[pos : pos + extra], "little")
                pos += extra
            length += 1
            out.extend(mv[pos : pos + length])
            pos += length
        else:
            if kind == 1:
                length = ((tag >> 2) & 0x07) + 4
                offset = ((tag & 0xE0) << 3) | mv[pos]
                pos += 1
            elif kind == 2:
                length = (tag >> 2) + 1
                offset = int.from_bytes(mv[pos : pos + 2], "little")
                pos += 2
            else:
                length = (tag >> 2) + 1
                offset = int.from_bytes(mv[pos : pos + 4], "little")
                pos += 4
            require(0 < offset <= len(out), "Snappy copy offset 越界")
            if offset >= length:
                start = len(out) - offset
                out.extend(out[start : start + length])
            else:
                for _ in range(length):
                    out.append(out[-offset])
    require(len(out) == expected_len, f"Snappy 解压长度不匹配: {len(out)} != {expected_len}")
    return bytes(out)


class _CompactReader:
    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = memoryview(buf)
        self.pos = pos

    def read_u8(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def read_varint(self) -> int:
        res, shift = 0, 0
        while True:
            b = self.buf[self.pos]
            self.pos += 1
            res |= (b & 0x7F) << shift
            if not (b & 0x80):
                return res
            shift += 7

    def read_zigzag(self) -> int:
        n = self.read_varint()
        return (n >> 1) ^ -(n & 1)

    def read_bytes(self) -> bytes:
        ln = self.read_varint()
        out = bytes(self.buf[self.pos : self.pos + ln])
        self.pos += ln
        return out

    def read_struct(self) -> dict:
        out, last_id = {}, 0
        while True:
            b = self.read_u8()
            if b == 0:
                return out
            delta, kind = b >> 4, b & 0x0F
            fid = (last_id + delta) if delta else self.read_zigzag()
            last_id = fid
            out[fid] = self._read_val(kind)

    def _read_val(self, kind: int):
        if kind in (1, 2):
            return kind == 1
        if kind == 3:
            return self.read_u8()
        if kind in (4, 5, 6):
            return self.read_zigzag()
        if kind == 7:
            val = struct.unpack_from("<d", self.buf, self.pos)[0]
            self.pos += 8
            return val
        if kind == 8:
            return self.read_bytes()
        if kind in (9, 10):
            hdr = self.read_u8()
            sz, elem_t = hdr >> 4, hdr & 0x0F
            if sz == 15:
                sz = self.read_varint()
            return [self._read_val(elem_t) for _ in range(sz)]
        if kind == 11:
            sz = self.read_varint()
            if not sz:
                return {}
            kv = self.read_u8()
            kt, vt = kv >> 4, kv & 0x0F
            return {self._read_val(kt): self._read_val(vt) for _ in range(sz)}
        if kind == 12:
            return self.read_struct()
        raise ValueError(f"未知 Thrift Compact 类型: {kind}")


def _decode_rle_bitpacked(buf: memoryview, bit_width: int, count: int) -> list:
    if bit_width == 0:
        return [0] * count
    pos, out = 0, []
    byte_width = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    while len(out) < count and pos < len(buf):
        hdr, shift = 0, 0
        while True:
            b = buf[pos]
            pos += 1
            hdr |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        if hdr & 1:
            groups = hdr >> 1
            n_vals = groups * 8
            n_bytes = groups * bit_width
            chunk = int.from_bytes(buf[pos : pos + n_bytes], "little")
            pos += n_bytes
            for i in range(n_vals):
                out.append((chunk >> (i * bit_width)) & mask)
        else:
            run_len = hdr >> 1
            val = int.from_bytes(buf[pos : pos + byte_width], "little")
            pos += byte_width
            out.extend([val] * run_len)
    require(len(out) >= count, "RLE/Bit-packed 解码数量不足")
    return out[:count]


def _unpack_plain_byte_arrays(raw: bytes, expected_count: int) -> list:
    out, pos, limit = [], 0, len(raw)
    for _ in range(expected_count):
        require(pos + 4 <= limit, "PLAIN BYTE_ARRAY 长度头越界")
        ln = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        require(pos + ln <= limit, "PLAIN BYTE_ARRAY 内容越界")
        out.append(raw[pos : pos + ln])
        pos += ln
    require(pos == limit, "PLAIN BYTE_ARRAY 尾部残留字节")
    return out


def extract_parquet_frames(parquet_bytes: bytes):
    """从上游 OpenFly traj/*.parquet 提取逐帧 (image_id, png_bytes) 列表。"""
    require(
        len(parquet_bytes) >= 12 and parquet_bytes[:4] == b"PAR1" and parquet_bytes[-4:] == b"PAR1",
        "非法 Parquet 魔数",
    )
    footer_len = struct.unpack("<I", parquet_bytes[-8:-4])[0]
    meta = _CompactReader(parquet_bytes, len(parquet_bytes) - 8 - footer_len).read_struct()
    # 计算叶子列最大 definition level（可选列/嵌套可选 struct 均需跳过 4-byte 长度前缀的 def level 块）
    schema_nodes = meta[2]
    max_def_levels = {}

    def walk(idx: int, path: tuple, def_lvl: int) -> int:
        node = schema_nodes[idx]
        name = node[4].decode()
        cur_path = path + ((name,) if idx > 0 else ())
        if idx > 0 and node.get(3, 0) == 1:  # OPTIONAL
            def_lvl += 1
        num_children = node.get(5, 0)
        if not num_children:
            max_def_levels[cur_path] = def_lvl
            return idx + 1
        nxt = idx + 1
        for _ in range(num_children):
            nxt = walk(nxt, cur_path, def_lvl)
        return nxt

    walk(0, (), 0)

    def read_byte_array_column(col_path: tuple) -> list:
        values = []
        max_def = max_def_levels[col_path]
        for rg in meta.get(4, []):
            col = next(c for c in rg[1] if tuple(x.decode() for x in c[3][3]) == col_path)
            cmeta = col[3]
            codec, total_vals = cmeta[4], cmeta[5]
            pos = cmeta.get(11, cmeta[9])
            dictionary = None
            col_vals = []
            while len(col_vals) < total_vals:
                rdr = _CompactReader(parquet_bytes, pos)
                ph = rdr.read_struct()
                ptype, ulen, clen = ph[1], ph[2], ph[3]
                payload = parquet_bytes[rdr.pos : rdr.pos + clen]
                pos = rdr.pos + clen
                if codec == 0:
                    raw = payload
                elif codec == 1:
                    raw = snappy_decompress(payload)
                else:
                    raise ValueError(f"不支持的 Parquet 压缩编码: {codec}")
                require(len(raw) == ulen, "Page 未压缩字节长度不匹配")
                if ptype == 2:  # DICTIONARY_PAGE
                    dictionary = _unpack_plain_byte_arrays(raw, ph[7][1])
                elif ptype == 0:  # DATA_PAGE v1
                    dph = ph[5]
                    n_val, enc = dph[1], dph[2]
                    offset = 0
                    if max_def > 0:
                        def_byte_len = struct.unpack_from("<I", raw, 0)[0]
                        def_bw = (max_def).bit_length()
                        defs = _decode_rle_bitpacked(memoryview(raw)[4 : 4 + def_byte_len], def_bw, n_val)
                        require(all(d == max_def for d in defs), f"列 {col_path} 存在空值")
                        offset = 4 + def_byte_len
                    if enc == 0:  # PLAIN
                        col_vals.extend(_unpack_plain_byte_arrays(raw[offset:], n_val))
                    elif enc in (2, 8):  # PLAIN_DICTIONARY / RLE_DICTIONARY
                        require(dictionary is not None, "缺少字典页")
                        bw = raw[offset]
                        idxs = _decode_rle_bitpacked(memoryview(raw)[offset + 1 :], bw, n_val)
                        col_vals.extend(dictionary[i] for i in idxs)
                    else:
                        raise ValueError(f"不支持的 DataPage 编码: {enc}")
                else:
                    raise ValueError(f"不支持的 Page 类型: {ptype}")
            values.extend(col_vals)
        require(len(values) == meta[3], f"列 {col_path} 行数不匹配")
        return values

    image_ids = [b.decode("utf-8") for b in read_byte_array_column(("image_id",))]
    png_blobs = read_byte_array_column(("image", "bytes"))
    return list(zip(image_ids, png_blobs))


def parse_and_verify_raw_pose(pose_path: Path, candidate: dict):
    """解析 pose.jsonl 并对齐原标注动作、组末位姿和每步 3D+yaw 运动学转移。"""
    decoder, text, raw = json.JSONDecoder(), pose_path.read_text(), []
    while text.strip():
        item, end = decoder.raw_decode(text.lstrip())
        text = text.lstrip()[end:]
        require(isinstance(item, dict), "raw pose 记录必须是对象")
        if "action" in item:
            raw.append(item["action"])
        else:
            require(set(item) == {"aim_landmark"}, "未知 raw pose 记录")

    def after(pose, token):
        x, y, z, yaw = pose
        if token in (1, 8, 9):
            step = {1: 3, 8: 6, 9: 9}[token]
            x, y = x + step * math.cos(yaw), y + step * math.sin(yaw)
        elif token in (2, 3):
            yaw += math.radians(30 if token == 2 else -30)
        elif token in (4, 5):
            z += 3 if token == 4 else -3
        return [x, y, z, yaw]

    def pose_matches(a, b):
        return (
            math.dist(a[:3], b[:3]) <= 1e-6
            and abs((a[3] - b[3] + math.pi) % (2 * math.pi) - math.pi) <= 1e-6
        )

    poses, tokens, raw_pairs = [], [], []
    require(bool(raw), "raw pose 为空")
    for i, act in enumerate(raw):
        require(act.get("imageid") == i, "raw imageid 必须连续从零递增")
        kind, value = act.get("type"), act.get("value")
        require((kind, value) in PRIMITIVE_MAP, f"未知原语: {(kind, value)}")
        poses.append(list(act["pos"][:3]) + [float(act["yaw"])])
        tokens.append(PRIMITIVE_MAP[(kind, value)])
        raw_pairs.append((kind, value))
    require(tokens[-1] == 0 and 0 not in tokens[:-1], "raw 仅允许唯一末尾 STOP")
    for i in range(len(raw) - 1):
        require(pose_matches(after(poses[i], tokens[i]), poses[i + 1]), f"相邻步转移不一致: {candidate['route_id']}#{i}")

    bounds, idx = [], 0
    while idx < len(raw):
        end = idx + 1
        if tokens[idx] == 1:
            while end < len(raw) and end < idx + 3 and tokens[end] == 1:
                end += 1
        bounds.append((idx, end - 1))
        idx = end
    require(len(bounds) == len(candidate["actions"]), "压缩组数与标注动作数不一致")
    groups = []
    for gi, (first, last) in enumerate(bounds):
        tok = {1: 1, 2: 8, 3: 9}[last - first + 1] if tokens[first] == 1 else tokens[first]
        require(tok == candidate["actions"][gi], "重建宏动作与原标注不一致")
        require(candidate["source_index_list"][gi].rsplit("_", 1)[-1] == str(last), "组末帧号不匹配")
        src_pose = list(candidate["source_pos"][gi][:3]) + [float(candidate["source_yaw"][gi])]
        require(pose_matches(src_pose, poses[last]), "组末位姿与原标注不一致")
        if gi + 1 < len(bounds):
            require(pose_matches(after(poses[first], tok), poses[bounds[gi + 1][0]]), "组首宏动作转移不一致")
        groups.append({"first_imageid": first, "last_imageid": last, "start_pose": poses[first]})
    return poses, bounds, groups, raw_pairs


def verify_png_bytes(data: bytes) -> tuple:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        rgb = img.convert("RGB")
        require(rgb.width > 0 and rgb.height > 0, "PNG 尺寸非法")
        return rgb.width, rgb.height


def cluster_scene_routes(route_ids: list, raw_xyz_list: list, corridor_distance_m: float = CORRIDOR_DISTANCE_M):
    """按同场景 3D 原始路径点最小欧氏距离 < corridor_distance_m 计算连通空间簇与成对距离矩阵。"""
    m = len(route_ids)
    require(m == len(raw_xyz_list) and m >= 2, "单场景至少需要两条路线才能划分空间簇")
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dist_mat = np.full((m, m), np.inf, dtype=np.float64)
    for a in range(m):
        pa = np.asarray(raw_xyz_list[a], dtype=np.float64)[:, :3]
        dist_mat[a, a] = 0.0
        for b in range(a + 1, m):
            pb = np.asarray(raw_xyz_list[b], dtype=np.float64)[:, :3]
            d = float(np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min())
            dist_mat[a, b] = dist_mat[b, a] = d
            if d < corridor_distance_m:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    comp_map = {}
    for idx in range(m):
        comp_map.setdefault(find(idx), []).append(idx)
    clusters = []
    for members in comp_map.values():
        anchor = min(route_ids[i] for i in members)
        clusters.append({"anchor_route": anchor, "member_indices": sorted(members),
                         "routes": [route_ids[i] for i in sorted(members)]})
    clusters.sort(key=lambda c: c["anchor_route"])
    return clusters, dist_mat


def assign_spatial_splits(
    scene_to_routes: dict,
    scene_order: list,
    corridor_distance_m: float = CORRIDOR_DISTANCE_M,
    seed: int = SPLIT_SEED,
):
    """将各场景连通空间簇按固定种子整簇划入 train/validation（约 80/20），严禁同簇跨划分。"""
    splits = {}
    scene_reports = {}
    for rank, scene in enumerate(scene_order):
        entries = scene_to_routes[scene]
        route_ids = [e["route_id"] for e in entries]
        xyz_list = [e["raw_xyz"] for e in entries]
        clusters, dist_mat = cluster_scene_routes(route_ids, xyz_list, corridor_distance_m)
        m = len(route_ids)
        target_val = max(2, round(m * 0.2))
        if len(scene_order) == 6 and m in (16, 17):
            # 6 场景共 100 条时精确取 4+4+3+3+3+3 = 20 条验证集（80/20）
            target_val = 4 if rank < 2 else 3

        ranked = sorted(
            clusters,
            key=lambda c: hashlib.sha256(f"{seed}:val_cluster:{c['anchor_route']}".encode()).hexdigest(),
        )
        val_indices = set()
        val_clusters = []
        for cluster in ranked:
            members = cluster["member_indices"]
            if len(val_indices) + len(members) <= target_val and (m - len(val_indices) - len(members)) >= 2:
                val_indices.update(members)
                val_clusters.append(cluster["anchor_route"])
            if len(val_indices) == target_val:
                break
        require(len(val_indices) >= 2 and (m - len(val_indices)) >= 2, f"场景 {scene} 空间簇不足以形成独立划分")
        train_indices = [i for i in range(m) if i not in val_indices]
        cross_pairs = [
            (float(dist_mat[i, j]), route_ids[i], route_ids[j])
            for i in train_indices
            for j in val_indices
        ]
        min_cross_dist, nearest_train, nearest_val = min(cross_pairs, key=lambda x: x[0])
        require(
            min_cross_dist >= corridor_distance_m,
            f"场景 {scene} 跨 split 最近距离 {min_cross_dist:.3f}m 低于隔离阈值 {corridor_distance_m}m",
        )
        for idx, rid in enumerate(route_ids):
            splits[rid] = "validation" if idx in val_indices else "train"
        scene_reports[scene] = {
            "total_routes": m,
            "spatial_clusters": len(clusters),
            "cluster_sizes": sorted([len(c["member_indices"]) for c in clusters], reverse=True),
            "train_routes": len(train_indices),
            "validation_routes": len(val_indices),
            "min_cross_split_3d_distance_m": min_cross_dist,
            "nearest_cross_split_pair": {"train_route": nearest_train, "validation_route": nearest_val},
            "validation_cluster_anchors": val_clusters,
        }
    return splits, scene_reports


def materialize_route(item_and_token):
    cand, parsed, out_dir, token = item_and_token
    poses, bounds, groups, _ = parsed
    route = cand["route_id"]
    route_img_dir = out_dir / "Image" / route
    route_img_dir.mkdir(parents=True, exist_ok=True)

    # 先查 HF Image/{route} 目录树是否存在独立发布的散装 PNG
    tree_url = f"{HF_ENDPOINT}/api/datasets/{SOURCE_REPO}/tree/{SOURCE_REVISION}/Image/{route}?expand=true"
    loose_pngs = {}
    for attempt in range(MAX_ATTEMPTS):
        try:
            with make_session() as client:
                r = client.get(tree_url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=(15, 45))
                r.raise_for_status()
                for entry in r.json():
                    p = entry.get("path", "")
                    if p.endswith(".png"):
                        stem = Path(p).stem
                        frame_no = int(stem.rsplit("_", 1)[-1])
                        loose_pngs[frame_no] = (stem, p)
                break
        except Exception:
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(2 * (attempt + 1))

    net_bytes = 0
    disk_bytes = 0
    index_list, image_paths, image_hashes = [], [], []
    if loose_pngs:
        source_mode = "loose_image_dir"
        for gi, (first, last) in enumerate(bounds):
            require(first in loose_pngs and last in loose_pngs, f"散装 PNG 缺少帧 {first}/{last}: {route}")
            first_stem, repo_png = loose_pngs[first]
            require(loose_pngs[last][0] == cand["source_index_list"][gi], "散装组末帧名与标注不一致")
            if first == last:
                require(first_stem == cand["source_index_list"][gi], "单步组帧名不一致")
            dst = route_img_dir / f"{first_stem}.png"
            if not dst.is_file():
                blob = http_get_bytes(repo_png, token)
                net_bytes += len(blob)
                verify_png_bytes(blob)
                part = dst.with_suffix(".png.part")
                part.write_bytes(blob)
                os.replace(part, dst)
            else:
                verify_png_bytes(dst.read_bytes())
            disk_bytes += dst.stat().st_size
            index_list.append(first_stem)
            image_paths.append(str(dst.resolve()))
            image_hashes.append(sha256_file(dst))
    else:
        source_mode = "traj_parquet_embedded"
        need_Download = True
        existing_by_num = {}
        for existing in route_img_dir.glob("*.png"):
            try:
                existing_by_num[int(existing.stem.rsplit("_", 1)[-1])] = existing
            except ValueError:
                pass
        if all(first in existing_by_num for first, _ in bounds):
            need_Download = False
            for gi, (first, last) in enumerate(bounds):
                dst = existing_by_num[first]
                if first == last:
                    require(dst.stem == cand["source_index_list"][gi], "已缓存单步组帧名不匹配")
                verify_png_bytes(dst.read_bytes())
                disk_bytes += dst.stat().st_size
                index_list.append(dst.stem)
                image_paths.append(str(dst.resolve()))
                image_hashes.append(sha256_file(dst))
        if need_Download:
            parquet_blob = fetch_parquet_parallel(f"traj/{route}.parquet", token)
            net_bytes += len(parquet_blob)
            frames = extract_parquet_frames(parquet_blob)
            require(len(frames) == len(poses), f"Parquet 帧数与 pose.jsonl 不一致: {route}")
            for i, (stem, _) in enumerate(frames):
                require(int(stem.rsplit("_", 1)[-1]) == i, f"Parquet image_id 序号不连续: {route}#{i}")
            for gi, (first, last) in enumerate(bounds):
                require(frames[last][0] == cand["source_index_list"][gi], "Parquet 组末帧名与标注不一致")
                first_stem, png_blob = frames[first]
                if first == last:
                    require(first_stem == cand["source_index_list"][gi], "Parquet 单步组帧名不一致")
                verify_png_bytes(png_blob)
                dst = route_img_dir / f"{first_stem}.png"
                part = dst.with_suffix(".png.part")
                part.write_bytes(png_blob)
                os.replace(part, dst)
                disk_bytes += dst.stat().st_size
                index_list.append(first_stem)
                image_paths.append(str(dst.resolve()))
                image_hashes.append(sha256_bytes(png_blob))

    pose_path = (out_dir / cand["raw_pose_repo_path"]).resolve()
    ep_record = {
        "route_id": route,
        "instruction": cand["instruction"],
        "actions": cand["actions"],
        "index_list": index_list,
        "source_index_list": cand["source_index_list"],
        "images": image_paths,
        "image_sha256": image_hashes,
        "raw_pose_path": str(pose_path),
        "raw_pose_sha256": sha256_file(pose_path),
        "groups": groups,
    }
    return ep_record, source_mode, net_bytes, disk_bytes


def run():
    t_start = time.time()
    cand_doc = json.loads(CANDIDATE_JSON.read_text())
    require(cand_doc["source_repo"] == SOURCE_REPO and cand_doc["source_revision"] == SOURCE_REVISION,
            "候选清单仓库或 revision 不匹配")
    train_source = Path(cand_doc["source_train"]["path"]).resolve()
    require(sha256_file(train_source) == cand_doc["source_train"]["sha256"], "官方 train.json SHA256 不匹配")

    # 校验与冻结 env18_causal_bc_v2 (16) 及 seen20 (20) 完全无混入
    frozen_dev = json.loads(Path(cand_doc["frozen_sources"]["train_development_manifest"]).read_text())
    frozen_seen = json.loads(Path(cand_doc["frozen_sources"]["seen_tasks"]).read_text())
    frozen_routes = {e["route_id"] for e in frozen_dev["episodes"]} | {t["route_id"] for t in frozen_seen["tasks"]}
    candidates = cand_doc["candidates"]
    require(len(candidates) == 100, "候选条数必须为 100")
    require(not any(c["route_id"] in frozen_routes for c in candidates), "候选混入了冻结集路线")

    token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.is_file() else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 拉取并校验全部 100 条 pose.jsonl
    def pull_pose(c):
        dst = OUTPUT_DIR / c["raw_pose_repo_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        fetched = 0
        if not dst.is_file():
            blob = http_get_bytes(c["raw_pose_repo_path"], token)
            fetched = len(blob)
            part = dst.with_suffix(".jsonl.part")
            part.write_bytes(blob)
            os.replace(part, dst)
        return dst, fetched

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pose_results = list(pool.map(pull_pose, candidates))

    pose_net_bytes = sum(x[1] for x in pose_results)
    pose_disk_bytes = sum(x[0].stat().st_size for x in pose_results)

    parsed_list = []
    vertical_audit = []
    observed_primitives = Counter()
    scene_to_routes = {}
    for cand, (pose_path, _) in zip(candidates, pose_results):
        poses, bounds, groups, raw_pairs = parse_and_verify_raw_pose(pose_path, cand)
        observed_primitives.update(f"{k}:{v}" for k, v in raw_pairs)
        if 4 in cand["actions"] or 5 in cand["actions"]:
            vert_steps = [
                {"imageid": i, "type": k, "value": v, "dz_m": round(poses[i + 1][2] - poses[i][2], 6)}
                for i, (k, v) in enumerate(raw_pairs[:-1])
                if k in ("go up", "go down")
            ]
            vertical_audit.append({"route_id": cand["route_id"], "actions": cand["actions"], "vertical_steps": vert_steps})
        parsed_list.append((poses, bounds, groups, raw_pairs))
        scene_to_routes.setdefault(cand["scene"], []).append({"route_id": cand["route_id"], "raw_xyz": poses})

    # 2. 3D 连通走廊空间簇隔离划分
    splits, scene_reports = assign_spatial_splits(
        scene_to_routes, cand_doc["scene_order"], CORRIDOR_DISTANCE_M, SPLIT_SEED
    )

    # 3. 拉取并逐张解码验证全部组首 PNG
    tasks = [(cand, parsed, OUTPUT_DIR, token) for cand, parsed in zip(candidates, parsed_list)]
    episodes = []
    png_net_bytes = 0
    png_disk_bytes = 0
    source_modes = Counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for idx, (ep_rec, mode, n_net, n_disk) in enumerate(pool.map(materialize_route, tasks), 1):
            ep_rec["split"] = splits[ep_rec["route_id"]]
            episodes.append(ep_rec)
            source_modes[mode] += 1
            png_net_bytes += n_net
            png_disk_bytes += n_disk
            if idx % 10 == 0 or idx == len(tasks):
                print(
                    f"[{idx}/{len(tasks)}] verified routes; png_disk={png_disk_bytes / (1 << 20):.2f} MiB, "
                    f"net_fetched={(pose_net_bytes + png_net_bytes) / (1 << 20):.2f} MiB",
                    flush=True,
                )

    manifest = {
        "schema": "uavrgb.openfly_subset.causal.v2",
        "annotation_source": str(train_source),
        "annotation_sha256": cand_doc["source_train"]["sha256"],
        "source_split": "train",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "supervision": "causal_group_start_macro_bc",
        "selection": (
            f"6-scene 100-route expansion from data_expansion_candidate.json (seed={SPLIT_SEED}); "
            f"3D raw-trajectory connected-corridor clustering (<{CORRIDOR_DISTANCE_M}m) with 80/20 cluster split; "
            f"seen20 and env18_causal_bc_v2 frozen and excluded"
        ),
        "episodes": episodes,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # 4. 调用 train_openfly_subset.load_manifest 做端到端契约复核
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_openfly_subset as trainer_mod
    trainer_mod.load_manifest(manifest_path)

    min_global_cross = min(r["min_cross_split_3d_distance_m"] for r in scene_reports.values())
    audit = {
        "schema": "uavrgb.openfly_spatial_audit.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t_start, 3),
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "endpoint_used": HF_ENDPOINT,
        "corridor_threshold_m": CORRIDOR_DISTANCE_M,
        "split_seed": SPLIT_SEED,
        "route_counts": {
            "total": len(episodes),
            "train": sum(e["split"] == "train" for e in episodes),
            "validation": sum(e["split"] == "validation" for e in episodes),
        },
        "frame_and_byte_summary": {
            "pose_jsonl_files": len(candidates),
            "pose_jsonl_disk_bytes": pose_disk_bytes,
            "group_start_png_files": sum(len(e["images"]) for e in episodes),
            "group_start_png_disk_bytes": png_disk_bytes,
            "total_subset_disk_bytes": pose_disk_bytes + png_disk_bytes,
            "network_bytes_fetched_this_run": pose_net_bytes + png_net_bytes,
            "route_materialization_modes": dict(source_modes),
        },
        "primitive_verification": {
            "observed_raw_type_value_counts": dict(sorted(observed_primitives.items())),
            "vertical_routes_count": len(vertical_audit),
            "vertical_routes_verified": vertical_audit,
        },
        "frozen_isolation": {
            "official_seen20_excluded": True,
            "official_seen20_sha256": sha256_file(Path(cand_doc["frozen_sources"]["seen_tasks"])),
            "frozen_env18_dev16_excluded": True,
            "exact_route_overlap_with_frozen": 0,
        },
        "global_min_cross_split_same_scene_3d_distance_m": min_global_cross,
        "per_scene_spatial_clusters": scene_reports,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    audit_path = OUTPUT_DIR / "spatial_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    (OUTPUT_DIR / "DONE").write_text(f"completed rc=0 manifest_sha256={audit['manifest_sha256']}\n")
    print(json.dumps({
        "manifest_sha256": audit["manifest_sha256"],
        "group_start_png_files": audit["frame_and_byte_summary"]["group_start_png_files"],
        "group_start_png_disk_bytes": png_disk_bytes,
        "global_min_cross_split_same_scene_3d_distance_m": min_global_cross,
    }, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    run()
