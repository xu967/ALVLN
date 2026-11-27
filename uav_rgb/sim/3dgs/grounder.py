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
_MOTION = re.compile(
    r"\b(?:go|fly|move|head|proceed|continue|navigate|travel)\b",
    re.IGNORECASE,
)
_TERMINAL_ACTION = re.compile(
    r"(?:^|\b(?:and(?:\s+then)?|then)\s+|[,.;!?]\s*)(?:please\s+)?"
    r"(?:stop|hover|wait|pause|hold(?:\s+position)?|stay|remain)(?:\s+(?:here|there|in\s+place))?$"
)
_CJK_TERMINAL_ACTION = re.compile(
    r"(?:^|[，。！？；、,.;!?]\s*|并且|并|然后|随后|之后|后|再)\s*(?:原地)?"
    r"(?:保持悬停|悬停|停下|停止|停住|停留|等待|停)(?:在[这那]里|在原地)?$"
)
_PUNCTUATION = " \t\r\n,.;!?，。！？；、"


def _navigation_text(instruction):
    """Remove a terminal action only when it names no destination of its own."""
    text = str(instruction or "").lower().replace("’", "'").strip(_PUNCTUATION)
    text = _TERMINAL_ACTION.sub("", text)
    return _CJK_TERMINAL_ACTION.sub("", text).rstrip(_PUNCTUATION)


def _tokens(value):
    words = re.findall(r"[a-z0-9]+", str(value).lower().replace("_", " ").replace("-", " "))
    return tuple(word for word in words if word not in _STOPWORDS and not word.isdigit())


def _goal_text(instruction):
    """优先明确运动目标，保留目标之后的约束，不把起点介词当目的地。"""
    text = instruction.replace("_", " ").replace("-", " ")
    motion = _MOTION.search(text)
    if motion:
        text = text[motion.end():]
    match = re.search(r"\b(?:to|toward|towards|into|at)\b", text)
    if match:
        for relation in _RELATIONS.finditer(text):
            if relation.start() <= match.start() < relation.end():
                return text[relation.start():]
    return text[match.end():] if match else text


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


_NEGATION_RE = re.compile(r"\b(?:not|never|no|except|avoid|without)\b|n['’]t\b")
_VERTICAL_RE = re.compile(r"\b(?:ascend|descend|climb|land|takeoff)\b|\b(?:go|fly|move)\s+(?:up|down)\b|\btake off\b")
_MULTI_SEQ_RE = re.compile(r"\b(?:then|afterwards|both|either|or)\b|;")
_AND_RE = re.compile(r"\band\b")

_BASE_GUARDS = (
    (lambda t: any(char.isalpha() and not char.isascii() for char in t), "unsupported_language"),
    (_NEGATION_RE.search, "unsupported_negation"),
    (_VERTICAL_RE.search, "unsupported_vertical_command"),
    (_MULTI_SEQ_RE.search, "unsupported_multiple_goals"),
)
_LEXICAL_EXTRA_GUARDS = (
    (lambda t: len(_MOTION.findall(t)) > 1, "unsupported_multiple_goals"),
    (lambda t: _AND_RE.search(_goal_text(t)), "unsupported_multiple_goals"),
    (lambda t: _RELATIONS.search(_goal_text(t)), "unsupported_relation"),
)


def _first_guard_rejection(text, guards):
    return next((reason for pred, reason in guards if pred(text)), None)


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
        text = _navigation_text(instruction)
        # shortcut: 只支持英文单目标词法检索；复杂语义交给后续关系 grounder。
        reason = _first_guard_rejection(text, _BASE_GUARDS + _LEXICAL_EXTRA_GUARDS)
        if reason:
            return Grounding(None, 0.0, 0.0, False, reason)
        names = [str(name) for name in candidates if str(name).strip()]
        query = set(_tokens(text))
        if not names or not query:
            return Grounding(None, 0.0, 0.0, False, "empty_query_or_candidates")
        goal_tokens = _tokens(_goal_text(text))
        goal = set(goal_tokens)
        target_text = " ".join(goal_tokens)
        scored = []
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
                score = 1.0 if phrase and f" {phrase} " in f" {target_text} " else overlap
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


_MODIFIER_SPLIT = re.compile(
    r"\b(?:located|situated|positioned|placed|standing|hanging|mounted|resting|sitting|"
    r"directly|right\s+next\s+to|next\s+to|in\s+front\s+of|across\s+from|adjacent\s+to|"
    r"to\s+the\s+(?:left|right)\s+of|on\s+the\s+(?:left|right)\s+side\s+of|"
    r"that\s+(?:is|has|contains)|which\s+(?:is|has|contains)|containing|with|"
    r"above|below|under|beneath|over|near|beside|behind|between|around|opposite|by|in|on|"
    r"after|along|inside|within)\b",
    re.IGNORECASE,
)
_TARGET_RELATION = re.compile(r"\b(?:near|next\s+to|beside)\b", re.IGNORECASE)
_TARGET_ACTION = re.compile(
    r"\b(?:polish(?:ing)?|adjust(?:ing)?|place|placing|showcase|showcasing|"
    r"display|displaying|hang|hanging)\s+(?:the|a|an)\s+",
    re.IGNORECASE,
)


_MAIN_TARGET = re.compile(
    r"(?:look\s+for|find|locate|polish|adjust|"
    r"(?:go|fly|move|head|navigate|travel|proceed)\s+(?:to|towards?|into))\s+"
)
_PLACEMENT_SITE = re.compile(
    r"(?:place|put|hang|showcase|display)\s+(?:it|them|(?:the|a|an)\s+[a-z-]+(?:\s+[a-z-]+)?)"
    r"\s+(?:(?:right\s+)?next\s+to|beside|near|close\s+to|into|onto|"
    r"(?:on|in)(?=\s+(?:the|a|an)\s+[a-z-]+(?:\s+[a-z-]+)?$))\s+"
)
_PRONOUN_PLACEMENT = re.compile(r"(?:place|put)\s+(?:it|them)\s+(?:in|on|into|onto)\s+(?=(?:the|a|an)\s)")
_ENDPOINT_MODIFIER = re.compile(
    r"\b(?:that|which|where|at|nearest(?=\s+(?:the|a|an)\b)|closest\s+to|close\s+to|"
    r"[a-z]+(?:ed|ing)\s+(?:for|to)|stored(?=\s+(?:[a-z]+ly|at|in|on)\b))\b"
)
_NEXT_ACTION = re.compile(
    r"\band\s+(?:next,?\s+)?(?=(?:bring|carry|continue|deliver|find|fly|go|hang|head|hover|leave|"
    r"locate|look|move|navigate|place|position|proceed|put|set|showcase|stop|take|transfer|transport|travel)\b)"
)
_NAVIGATION_VERB = re.compile(
    r"\b(?:(?:go|fly|move|head|navigate|travel|proceed|walk)(?:\s+from\b[^,;.]*?)?|"
    r"stop|hover|finish|end|arrive)\s+$"
)


def _english_destination(command):
    command = command.removeprefix("please ")
    final_action = _NEXT_ACTION.split(command)[-1].strip()
    # “take the pillow and place it in the cabinet near the fridge” 的终点由最后的放置动作确定。
    placement = _PRONOUN_PLACEMENT.match(final_action)
    if placement or _MAIN_TARGET.match(command):
        command = final_action
    marker = placement or _MAIN_TARGET.match(command) or _PLACEMENT_SITE.match(command)
    if marker:
        raw_goal = command[marker.end():].removeprefix("that ")
        head = _MODIFIER_SPLIT.split(raw_goal, maxsplit=1)[0]
        head = _ENDPOINT_MODIFIER.split(head, maxsplit=1)[0].strip()
        if re.search(r"\s(?:a|an|the)\s", head):
            return ""
        return head
    if re.match(r"(?:place|put|hang|showcase|display)\s+", command):
        return ""
    # 保留间接指令的既有目的地优先级。
    directions = list(re.finditer(r"(?<!next )\b(?:to|toward|towards|into|at)\b"
                                  r"(?!\s+the\s+(?:left|right)\s+of)", command))
    # 带起点的运动指令中，终点名词后面的 at 引入地点修饰语。
    if len(directions) >= 2 and directions[-1].group() == "at":
        start, location = directions[-2:]
        verb = command[:start.start()]
        goal = command[start.end():location.start()].strip()
        if (_NAVIGATION_VERB.search(verb) and re.match(r"(?:a|an|the)\s+", goal)
                and not re.search(r"[,;.]|\b(?:and|then)\b", goal)):
            return _MODIFIER_SPLIT.split(goal, maxsplit=1)[0]
    relations = list(_TARGET_RELATION.finditer(command))
    actions = list(_TARGET_ACTION.finditer(command))
    for matches in (directions, relations, actions):
        if matches:
            return _MODIFIER_SPLIT.split(command[matches[-1].end():], maxsplit=1)[0]
    return ""


def _cjk_instruction_target_matches(instruction, visible_target):
    if re.search(r"不要|不能|不许|不准|不得|禁止|(?:别|勿|不)(?:再)?(?:飞|去|到|前往|靠近|悬停|停|向|在)"
                 r"|(?:降落|起飞|上升|下降)(?:$|[，。；]|并|后|到|在)", instruction):
        return False
    destinations = list(re.finditer(r"前往|到|去|停在", instruction))
    label = str(visible_target or "").strip(_PUNCTUATION)
    if not destinations or not label or label in ("它", "东西", "物品", "目标", "none", "unknown"):
        return False
    target_phrase = instruction[destinations[-1].end():].strip(_PUNCTUATION)
    target_phrase = re.sub(r"(?:附近|旁边|周围|周边|区域|里面|内部)$", "", target_phrase)
    return bool(target_phrase) and target_phrase.endswith(label)


def instruction_target_matches(instruction, visible_target):
    """Accept only a visible label matching the final destination phrase."""
    text = _navigation_text(instruction)
    if _NEGATION_RE.search(text) or _VERTICAL_RE.search(text):
        return False
    if re.search(r"[\u4e00-\u9fff]", text):
        return _cjk_instruction_target_matches(text, visible_target)
    clauses = [part.strip() for part in re.split(r"[.;]|\bthen\b", text) if part.strip()]
    target_tokens = _tokens(_english_destination(clauses[-1])) if clauses else ()
    visible_tokens = _tokens(visible_target)
    if (not target_tokens or not visible_tokens
            or visible_tokens[0] in ("it", "thing", "object", "target", "word", "none", "unknown")
            or len(visible_tokens) > len(target_tokens)):
        return False
    return target_tokens[-len(visible_tokens):] == visible_tokens


class RelationPhraseGrounder(LexicalGrounder):
    """主目标短语提取器：把修饰从句（如 'located directly above the wall art'）拆成主目标与参照物。

    保留对非英文、否定词、升降动作和多阶段顺序指令的 fail-closed 拒绝；当目录中同类候选唯一或主短语
    能唯一定位目标时直接命中，同名多实例（如 chair_1 vs chair_2）无法单靠文本消歧时按 ambiguous 拒绝。
    """

    def ground(self, instruction, candidates):
        text = _navigation_text(instruction)
        reason = _first_guard_rejection(text, _BASE_GUARDS)
        if reason:
            return Grounding(None, 0.0, 0.0, False, reason)

        # 多子句或 "from X to Y" 句式：优先取最后一个目标介词短语（to / toward / into），避免把起点当目标
        clauses = [c.strip() for c in re.split(r"[.;]", text) if c.strip()]
        goal_clause = clauses[-1] if clauses else text
        dest_matches = list(re.finditer(r"\b(?:to|toward|towards|into)\b(?!\s+the\s+(?:left|right)\s+of\b)", goal_clause))
        raw_goal = goal_clause[dest_matches[-1].end():] if dest_matches else _goal_text(goal_clause)
        head_phrase = (_english_destination(goal_clause)
                       or _MODIFIER_SPLIT.split(raw_goal, maxsplit=1)[0].strip() or raw_goal)
        head_tokens = set(_tokens(head_phrase))
        full_tokens = set(_tokens(raw_goal))
        names = [str(name) for name in candidates if str(name).strip()]
        if not names or not (head_tokens or full_tokens):
            return Grounding(None, 0.0, 0.0, False, "empty_query_or_candidates")

        head_str = " ".join(_tokens(head_phrase))
        scored = []
        for name in dict.fromkeys(names):
            words = set(_tokens(name))
            if not words:
                scored.append((0.0, name))
                continue
            phrase = " ".join(_tokens(name))
            if phrase and f" {phrase} " in f" {head_str} ":
                score = 1.0
            else:
                h_cov = len(head_tokens & words) / len(words) if head_tokens else 0.0
                f_cov = len(full_tokens & words) / len(words) if full_tokens else 0.0
                # 主目标短语占 0.85 权重，避免把参照物（wall art）排在主目标（track light）前面
                score = 0.85 * h_cov + 0.15 * f_cov
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_name = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        accepted = best_score >= self.min_score and margin >= self.min_margin
        reason = "accepted" if accepted else ("below_min_score" if best_score < self.min_score else "ambiguous")
        return Grounding(best_name if accepted else None, best_score, margin, accepted, reason)


__all__ = [
    "BEARING_SEARCH", "Grounding", "LexicalGrounder", "RelationPhraseGrounder",
    "VisualGrounding", "VisualLanguageGrounder", "load_catalog", "parse_vlm_output",
]

BEARING_SEARCH = 8  # 当前视野未见目标，驱动机载策略原地旋转搜索

_SECTOR_TO_BIN = {
    "c": 0, "center": 0, "front": 0, "ahead": 0, "middle": 0, "0": 0,
    "l": 1, "left": 1, "fl": 1, "front_left": 1, "front-left": 1, "1": 1,
    "far_left": 2, "2": 2,
    "bl": 3, "back_left": 3, "3": 3,
    "b": 4, "back": 4, "behind": 4, "4": 4,
    "br": 5, "back_right": 5, "5": 5,
    "far_right": 6, "6": 6,
    "r": 7, "right": 7, "fr": 7, "front_right": 7, "front-right": 7, "7": 7,
    "n": 8, "none": 8, "search": 8, "unseen": 8, "not_visible": 8, "occluded": 8, "8": 8,
}
_BIN_SHORT = {0: "C", 1: "L", 2: "L", 3: "L", 4: "B", 5: "R", 6: "R", 7: "R", 8: "N"}
_GOAL_MODES = {
    "f": "final", "final": "final", "goal": "final", "target": "final",
    "s": "subgoal", "subgoal": "subgoal", "waypoint": "subgoal", "door": "subgoal", "exit": "subgoal",
    "n": "search", "none": "search", "search": "search",
}
VALID_GOAL_MODES = frozenset(("final", "subgoal", "search"))


@dataclass(frozen=True)
class VisualGrounding:
    bearing_bin: int
    near_goal: int
    conf: float
    reason: str
    latency_ms: float = 0.0
    goal_mode: str = "final"

    def __post_init__(self):
        if isinstance(self.bearing_bin, bool) or not (0 <= int(self.bearing_bin) <= 8):
            raise ValueError(f"bearing_bin 必须在 0..8 内，得到 {self.bearing_bin!r}")
        if isinstance(self.near_goal, bool) or int(self.near_goal) not in (0, 1):
            raise ValueError(f"near_goal 必须是 0 或 1，得到 {self.near_goal!r}")
        if not math.isfinite(float(self.conf)) or not (0.0 <= float(self.conf) <= 1.0):
            raise ValueError(f"conf 必须在 [0, 1] 内，得到 {self.conf!r}")
        if self.goal_mode not in VALID_GOAL_MODES:
            raise ValueError(f"goal_mode 必须在 {sorted(VALID_GOAL_MODES)} 内，得到 {self.goal_mode!r}")


def parse_vlm_output(raw_text, default_conf=0.8):
    """解析 VLM 输出（支持 G3 阶段数组 ["L","S",0,"door"]、旧三元组 ["C",1,"tv"] 及 JSON 对象）。"""
    txt = str(raw_text or "").strip()
    if "```" in txt:
        txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.IGNORECASE | re.MULTILINE).strip()
    fallback = (BEARING_SEARCH, 0, 0.0, "parse_fallback", "search")
    conf = default_conf
    mode_raw = "f"
    try:
        parsed = json.loads(txt)
    except ValueError:
        return fallback
    if isinstance(parsed, list) and len(parsed) in (3, 4):
        sector_raw = str(parsed[0]).strip()
        if len(parsed) == 4:
            _, mode_raw, near_raw, reason = parsed
        else:
            _, near_raw, reason = parsed
    elif isinstance(parsed, dict):
        sector_raw = (str(parsed["bearing_bin"])
                      if isinstance(parsed.get("bearing_bin"), int)
                      else str(parsed.get("sector", parsed.get("s", "N"))).strip())
        mode_raw = str(parsed.get("mode", parsed.get("goal_mode", "f"))).strip().lower()
        near_raw = parsed.get("near_goal", parsed.get("near", parsed.get("g", 0)))
        if "conf" in parsed or "c" in parsed:
            conf = parsed.get("conf", parsed.get("c", conf))
        reason = parsed.get("reason", parsed.get("r", parsed.get("why", "ok")))
    else:
        return fallback
    if not isinstance(reason, str) or not reason.strip():
        return fallback

    try:
        conf = float(conf)
    except (TypeError, ValueError, OverflowError):
        return fallback
    if not math.isfinite(conf) or not 0 <= conf <= 10:
        return fallback
    conf = conf / 10 if conf > 1 else conf
    mode = _GOAL_MODES.get(str(mode_raw).strip().lower())
    if mode is None or type(near_raw) not in (int, float) or near_raw not in (0, 1):
        return fallback
    bb = _SECTOR_TO_BIN.get(sector_raw.lower(), BEARING_SEARCH)
    if bb == BEARING_SEARCH or mode == "search":
        return BEARING_SEARCH, 0, conf, str(reason).strip()[:64], "search"
    return bb, int(near_raw) if mode == "final" else 0, conf, str(reason).strip()[:64], mode


class VisualLanguageGrounder:
    """零特权层次化视觉语言 Grounder：仅根据机载 RGB 与自然语言指令输出相对方位、阶段模式与近目标标志。"""

    DEFAULT_PYTHON = "/root/rivermind-data/uav_rgb/envs/vlm-venv/bin/python"
    DEFAULT_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_grounder_worker.py")

    def __init__(self, model_path=None, python_bin=None, worker_cmd=None, timeout_s=180.0):
        import subprocess
        self.timeout_s = float(timeout_s)
        self._recent_notes = []
        self.last_waypoint_probe = None
        if worker_cmd is not None:
            cmd = list(worker_cmd)
        else:
            py = python_bin or os.environ.get("UAV_RGB_VLM_PYTHON", self.DEFAULT_PYTHON)
            if not os.path.isfile(py):
                raise FileNotFoundError(f"找不到 VLM Python 环境: {py}")
            cmd = [py, self.DEFAULT_WORKER]
        env = os.environ.copy()
        if model_path:
            env["UAV_RGB_VLM_MODEL"] = str(model_path)
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        ready_line = self._readline(self.timeout_s, "startup")
        doc = json.loads(ready_line)
        if doc.get("status") != "ready":
            self.close()
            raise RuntimeError(f"VLM worker 未就绪: {ready_line}")

    def reset_episode(self):
        self._recent_notes.clear()
        self.last_waypoint_probe = None

    def note_action(self, action_name: str):
        if not self._recent_notes:
            return
        last = self._recent_notes[-1]
        if "->" not in last:
            self._recent_notes[-1] = f"{last}->{action_name}"

    def _readline(self, timeout_s, stage):
        import select
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("VLM worker 已关闭")
        rlist, _, _ = select.select([self._proc.stdout], [], [], timeout_s)
        if not rlist:
            self.close()
            raise TimeoutError(f"VLM worker {stage} 超时 ({timeout_s}s)")
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            self.close()
            raise RuntimeError(f"VLM worker {stage} 提前退出: {err.strip()}")
        return line

    @staticmethod
    def _encode_frames_b64(rgb_history):
        import base64
        import io
        import numpy as np
        from PIL import Image

        if rgb_history is None:
            return []
        if isinstance(rgb_history, np.ndarray) and rgb_history.ndim == 3:
            rgb_history = [rgb_history]
        encoded = []
        for frame in list(rgb_history)[-4:]:
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"RGB 帧必须是 HxWx3 uint8，得到 {arr.shape}")
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=85)
            encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return encoded

    def ground_step(self, rgb_history, instruction: str) -> VisualGrounding:
        """严格零特权接口：仅接收机载 RGB 数组（单帧 HWC 或帧序列）与语言指令。"""
        import time

        self.last_waypoint_probe = None
        text = str(instruction or "").strip()
        if not text:
            return VisualGrounding(BEARING_SEARCH, 0, 0.0, "empty_instruction", 0.0, "search")
        t0 = time.perf_counter()
        frames_b64 = self._encode_frames_b64(rgb_history)
        if not frames_b64:
            return VisualGrounding(BEARING_SEARCH, 0, 0.0, "empty_rgb_history", 0.0, "search")

        mem = "; ".join(self._recent_notes[-3:]) if self._recent_notes else "none"
        req = json.dumps({"frames_b64": frames_b64, "instruction": text, "memory": mem}, ensure_ascii=False)
        self._proc.stdin.write(req + "\n")
        self._proc.stdin.flush()
        resp = json.loads(self._readline(min(self.timeout_s, 30.0), "ground_step"))
        self.last_waypoint_probe = resp.get("waypoint_probe")
        bb, ng, conf, reason, goal_mode = parse_vlm_output(resp.get("raw", ""), default_conf=resp.get("conf", 0.5))
        tag = "F" if goal_mode == "final" else ("S" if goal_mode == "subgoal" else "N")
        self._recent_notes.append(f"{_BIN_SHORT.get(bb, 'N')}({tag}:{reason[:18]})")
        if len(self._recent_notes) > 6:
            self._recent_notes = self._recent_notes[-6:]
        ms = (time.perf_counter() - t0) * 1000.0
        return VisualGrounding(bb, ng, conf, reason, round(ms, 2), goal_mode)

    def close(self):
        proc, self._proc = getattr(self, "_proc", None), None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write('{"cmd":"close"}\n')
                proc.stdin.flush()
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()
        for s in (proc.stdout, proc.stderr):
            if s:
                s.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
