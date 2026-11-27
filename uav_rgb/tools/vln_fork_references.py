"""从登记原句恢复人工核对的终点参照。区间按 Unicode 字符计数。"""
import hashlib
import json
from pathlib import Path
import re


def read_reference_spans(path):
    """读取带版本的字符区间表，以原句与官方目标的联合摘要为键。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema"] != "uavrgb.vln_fork_reference_spans.v1":
        raise ValueError("不支持的终点参照表版本")
    by_digest = {}
    for row in data["rows"]:
        digest, status, spans = row["sha256"], row["status"], row["spans"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in by_digest:
            raise ValueError(f"终点参照摘要无效或重复 {digest}")
        if status not in ("explicit", "none", "ambiguous") or not isinstance(spans, list):
            raise ValueError(f"终点参照状态或区间无效 {digest}")
        if (status == "explicit" and not spans) or (status == "none" and spans):
            raise ValueError(f"终点参照状态与区间不符 {digest}")
        if status == "ambiguous" and not row.get("note"):
            raise ValueError(f"歧义参照缺少判定说明 {digest}")
        by_digest[digest] = row
    return by_digest


def lookup_reference(spans, text, goal):
    """联合摘要采用紧凑 JSON 数组 [原句, 官方目标] 的 UTF-8 字节，不做文本规范化。"""
    pair = json.dumps([text, goal], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(pair.encode("utf-8")).hexdigest()
    if digest not in spans:
        raise ValueError(f"原句与官方目标未在终点参照表中登记 {digest}")
    row = spans[digest]
    phrases = []
    previous_end = 0
    for start, end in row["spans"]:
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(text):
            raise ValueError(f"终点参照字符区间无效 {digest} [{start}, {end})")
        phrases.append(text[start:end])
        previous_end = end
    result = {"status": row["status"], "phrases": phrases, "instance_ids": None}
    if "note" in row:
        result["note"] = row["note"]
    return result
