#!/usr/bin/env python3
"""把自然语言指令落到一个外部提供的地标候选，不读取飞行器真值位姿。

这是 G1 的可审计起点：它只负责 ``text -> candidate name``，不假装拥有
物体坐标，也不直接发飞控动作。候选表必须由地图/感知模块提供，避免把
route waypoint 偷渡成“语言理解”。
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re


_STOPWORDS = {
    "a", "an", "and", "at", "by", "directly", "fly", "go", "located", "near",
    "of", "on", "over", "the", "then", "to", "toward", "towards", "under", "with",
    "above", "below", "left", "right", "inside", "into", "through", "past",
}

_RELATIONS = re.compile(
    r"\b(?:above|below|left|right|near|next\s+to|across\s+from|in\s+front\s+of|"
    r"behind|beside|between|within|around|opposite|adjacent\s+to)\b",
    re.IGNORECASE,
)


def _tokens(value):
    words = re.findall(r"[a-z0-9]+", str(value).lower().replace("_", " ").replace("-", " "))
    return tuple(word for word in words if word not in _STOPWORDS and not word.isdigit())


def _goal_text(instruction):
    """取最后一个明显的目的地短语，降低起点/路径描述对目标解析的干扰。"""
    text = str(instruction).lower().replace("_", " ").replace("-", " ")
    matches = list(re.finditer(r"\b(?:to|toward|towards|into|at)\b", text))
    return text[matches[-1].end():] if matches else text


def _goal_tokens(instruction):
    return set(_tokens(_goal_text(instruction)))


@dataclass(frozen=True)
class Grounding:
    target: str | None
    score: float
    margin: float
    accepted: bool
    reason: str


def load_catalog(path, scene_name=None, scene_path=None):
    """读取外部地标目录，返回 ``name -> {x, y, z?}``。

    目录必须来自独立地图或感知模块；这里不从 ``scene.json`` 猜坐标，避免
    把评测航线的 waypoint 伪装成语言理解结果。
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError("地标目录顶层必须是对象")
    if doc.get("schema") != "uavrgb.vln_landmark_catalog.v1":
        raise ValueError("地标目录 schema 必须是 uavrgb.vln_landmark_catalog.v1")
    scene = doc.get("scene")
    if isinstance(scene, bool) or not isinstance(scene, str) or not scene.strip():
        raise ValueError("地标目录 scene 必须是非空字符串")
    if scene_name is not None and scene != scene_name:
        raise ValueError(f"地标目录 scene={scene!r} 与当前场景 {scene_name!r} 不一致")
    # 仅接受可重算哈希的本地来源文件；哈希证明文件身份，不证明人工标注独立于测试答案。
    source = doc.get("source")
    source_hash = doc.get("source_sha256")
    if isinstance(source, dict):
        source_ref = (source.get("path") or source.get("uri") or source.get("id")
                      or source.get("name"))
        nested_hash = source.get("sha256")
        if source_hash is None:
            source_hash = nested_hash
        elif (nested_hash is not None and
              str(source_hash).lower() != str(nested_hash).lower()):
            raise ValueError("地标目录 source.sha256 与 source_sha256 不一致")
    else:
        source_ref = source
    if isinstance(source_ref, bool) or not isinstance(source_ref, str) or not source_ref.strip():
        raise ValueError("地标目录 source 必须指向可读取的来源文件")
    if (not isinstance(source_hash, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", source_hash) is None):
        raise ValueError("地标目录 source_sha256 必须是 64 位十六进制字符串")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", source_ref):
        raise ValueError("地标目录 source 必须是本地文件路径，不能使用未校验的 URI")
    source_path = source_ref
    if not os.path.isabs(source_path):
        source_path = os.path.join(os.path.dirname(os.path.abspath(path)), source_path)
    if not os.path.isfile(source_path):
        raise ValueError(f"地标目录来源文件不存在: {source_path}")
    digest = hashlib.sha256()
    with open(source_path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != source_hash.lower():
        raise ValueError("地标目录 source_sha256 与来源文件内容不一致")
    if scene_path is not None:
        if os.path.samefile(source_path, scene_path):
            raise ValueError("地标目录 source 不能指向当前 scene.json")
    if doc.get("frame_id") != "scene_nav" or doc.get("units") != "m":
        raise ValueError("地标目录坐标必须声明 frame_id=scene_nav、units=m")
    rows = doc.get("landmarks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("地标目录 landmarks 必须是非空数组")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("地标目录每项都需要 name")
        raw_name = row.get("name")
        if isinstance(raw_name, bool) or not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("地标目录每项都需要非空字符串 name")
        name = raw_name.strip()
        if name in out:
            raise ValueError(f"地标目录重复 name: {name}")
        point = {}
        for axis in ("x", "y", "z"):
            if axis not in row:
                if axis in ("x", "y"):
                    raise ValueError(f"地标 {name} 坐标无效: 缺少 {axis}")
                continue
            value = row[axis]
            if isinstance(value, bool):
                raise ValueError(f"地标 {name} 坐标不能是 bool")
            try:
                point[axis] = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"地标 {name} 坐标无效") from e
        if not all(math.isfinite(value) for value in point.values()):
            raise ValueError(f"地标 {name} 坐标必须有限")
        out[name] = point
    return out


class LexicalGrounder:
    """可解释的候选选择器；分数不是概率，不能替代视觉 grounder。"""

    def __init__(self, min_score=0.5, min_margin=0.15):
        if not (0.0 < float(min_score) <= 1.0):
            raise ValueError("min_score 必须在 (0, 1] 内")
        if float(min_margin) < 0.0:
            raise ValueError("min_margin 不能为负数")
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)

    def ground(self, instruction, candidates):
        """返回一个候选名；候选为空、无重叠或不够明确时拒绝。"""
        text = str(instruction).lower().replace("’", "'")
        # shortcut: 只支持英文单目标词法检索；复杂语义交给后续关系 grounder。
        reason = None
        if any(char.isalpha() and not char.isascii() for char in text):
            reason = "unsupported_language"
        elif re.search(r"\b(?:not|never|no|except|avoid|without)\b|n['’]t\b", text):
            reason = "unsupported_negation"
        elif re.search(r"\b(?:ascend|descend|climb|land|takeoff)\b|\b(?:go|fly|move)\s+(?:up|down)\b|\btake off\b", text):
            reason = "unsupported_vertical_command"
        elif re.search(r"\b(?:then|afterwards|both|either|or)\b|;", text):
            reason = "unsupported_multiple_goals"
        elif re.search(r"\band\b", _goal_text(text)):
            reason = "unsupported_multiple_goals"
        elif _RELATIONS.search(_goal_text(text)):
            reason = "unsupported_relation"
        if reason:
            return Grounding(None, 0.0, 0.0, False, reason)
        names = [str(name) for name in candidates if str(name).strip()]
        query = set(_tokens(instruction))
        goal = _goal_tokens(instruction)
        if not names or not query:
            return Grounding(None, 0.0, 0.0, False, "empty_query_or_candidates")
        scored = []
        normalized = str(instruction).lower().replace("_", " ").replace("-", " ")
        for name in dict.fromkeys(names):
            words = set(_tokens(name))
            if not words:
                score = 0.0
            else:
                overlap = len(query & words) / len(words)
                goal_overlap = len(goal & words) / len(words) if goal else 0.0
                if goal:
                    overlap = 0.25 * overlap + 0.75 * goal_overlap
                # 完整短语命中优先于零散词命中，但仍保持可解释的上限。
                phrase = " ".join(_tokens(name))
                target_text = normalized
                if goal:
                    target_text = normalized.rsplit(" to ", 1)[-1]
                score = 1.0 if phrase and phrase in target_text else overlap
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_name = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        accepted = best_score >= self.min_score and margin >= self.min_margin
        if accepted:
            reason = "accepted"
        elif best_score < self.min_score:
            reason = "below_min_score"
        else:
            reason = "ambiguous"
        return Grounding(best_name if accepted else None, best_score, margin, accepted, reason)


__all__ = ["Grounding", "LexicalGrounder", "load_catalog"]
