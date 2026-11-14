#!/usr/bin/env python3
"""把自然语言指令落到一个外部提供的地标候选，不读取飞行器真值位姿。

这是 G1 的可审计起点：它只负责 ``text -> candidate name``，不假装拥有
物体坐标，也不直接发飞控动作。候选表必须由地图/感知模块提供，避免把
route waypoint 偷渡成“语言理解”。
"""
from dataclasses import dataclass
import re


_STOPWORDS = {
    "a", "an", "and", "at", "by", "directly", "fly", "go", "located", "near",
    "of", "on", "over", "the", "then", "to", "toward", "towards", "under", "with",
    "above", "below", "left", "right", "inside", "into", "through", "past",
}


def _tokens(value):
    words = re.findall(r"[a-z0-9]+", str(value).lower().replace("_", " ").replace("-", " "))
    return tuple(word for word in words if word not in _STOPWORDS and not word.isdigit())


def _goal_tokens(instruction):
    """取最后一个明显的目的地短语，降低 ``from`` 起点的干扰。"""
    text = str(instruction).lower().replace("_", " ").replace("-", " ")
    matches = list(re.finditer(r"\b(?:to|toward|towards|into|at)\b", text))
    return set(_tokens(text[matches[-1].end():])) if matches else set()


@dataclass(frozen=True)
class Grounding:
    target: str | None
    score: float
    margin: float
    accepted: bool
    reason: str


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


__all__ = ["Grounding", "LexicalGrounder"]
