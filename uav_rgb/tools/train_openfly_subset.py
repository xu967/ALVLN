#!/usr/bin/env python3
"""真实 OpenFly 子集离线 BC：适配 AirVLN seq2seq/CMA（支持 v1 基线契约与 v2 语言/STOP 增强契约）。

运行方式:
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python tools/train_openfly_subset.py \
      --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/six_scene_causal_bc_v1/manifest.json \
      --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN \
      --out /root/rivermind-data/uav_rgb/exp-openfly-v2-20261116/training/v2_cma_rgb_language \
      --policy-type cma --training-mode rgb_language --epochs 15 --seed 20261116 \
      --imagenet-norm --use-bert --inflection-weight auto --stop-weight 3.0 \
      --progress-monitor --use-prev-action --prev-action-dropout 0.2
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

# ========================= 顶部集中配置区 =========================
SCHEMA = "uavrgb.openfly_subset.v1"
CAUSAL_SCHEMA = "uavrgb.openfly_subset.causal.v2"
CONTRACT_V1 = "uavrgb.openfly_external_bc.v1"
CONTRACT_V2 = "uavrgb.openfly_external_bc.v2"
INPUTS = ["instruction", "rgb_features"]
ACTION_SCHEMA = "openfly_0_9_annotation_indexed_offline_bc"
CAUSAL_ACTION_SCHEMA = "openfly_0_9_causal_group_start_macro_bc"
TRAINING_MODES = ("rgb_language", "rgb_only")
POLICY_TYPES = ("seq2seq", "cma")
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", "/root/rivermind-data/uav_rgb/hf-cache"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # TorchVision ResNet50 预训练通道均值
IMAGENET_STD = (0.229, 0.224, 0.225)   # TorchVision ResNet50 预训练通道标准差
# =================================================================


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_training_mode(args, mode):
    require(mode in TRAINING_MODES, "training_mode 必须是 rgb_language/rgb_only")
    # 保留相同架构和初始化，只使用官方开关阻断语言输入及其梯度。
    args.ablate_instruction = mode == "rgb_only"


def rgb_feature_shape(policy_type):
    require(policy_type in POLICY_TYPES, "policy_type 必须是 seq2seq/cma")
    return (2048, 4, 4) if policy_type == "cma" else (2048, 1, 1)


def preprocess_rgb_uint8(arrays_u8, device, imagenet_norm=False):
    """uint8 [B, H, W, 3] -> float32 [B, 3, H, W]；v2 开启 ImageNet mean/std 归一化。"""
    import torch
    x = torch.as_tensor(arrays_u8, device=device).permute(0, 3, 1, 2).float() / 255.0
    if imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


def load_bert_tokenizer(hf_home=DEFAULT_HF_HOME):
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)


def encode_instruction_text(text, tokenizer, use_bert=False, max_len=300):
    require(isinstance(text, str) and text.strip(), "指令为空")
    if not use_bert:
        words = tokenizer.split_sentence(text)
        require(0 < len(words) <= max_len, "指令为空或超长；禁止静默截断")
        tokens = tokenizer.encode_sentence(text)
        vocab_map = getattr(tokenizer, "word_to_index", None)
        oov = sum(w not in vocab_map for w in words) if vocab_map is not None else 0
        return tokens, len(words), oov
    raw_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    require(2 <= len(raw_ids) <= max_len, f"BERT token 长度 {len(raw_ids)} 超出 [2, {max_len}]；禁止静默截断")
    unk_id = tokenizer.unk_token_id
    oov = sum(1 for tid in raw_ids if tid == unk_id)
    padded = raw_ids + [int(tokenizer.pad_token_id)] * (max_len - len(raw_ids))
    return padded, len(raw_ids), oov


def build_trainer(observation_space, action_space, policy_type="seq2seq",
                  use_prev_action=False, prev_action_dropout=0.0,
                  progress_monitor=False, use_bert=False):
    """训练、重载和在线评估共用；v1 禁用前一动作，v2 可选启用因果 prev_action + 训练期整步置零 dropout。"""
    import torch
    from src.common.param import args
    from Model.il_trainer import VLNCETrainer
    require(0.0 <= float(prev_action_dropout) < 1.0, "prev_action_dropout 必须在 [0, 1) 内")
    rgb_feature_shape(policy_type)
    args.policy_type = policy_type
    args.SEQ2SEQ_use_prev_action = bool(use_prev_action)
    args.ablate_depth = True
    args.PROGRESS_MONITOR_use = bool(progress_monitor)
    args.tokenizer_use_bert = bool(use_bert)
    if use_bert:
        os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(DEFAULT_HF_HOME) / "hub"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    trainer = VLNCETrainer(False, observation_space, action_space)
    if not use_prev_action:
        if policy_type == "cma":
            # 官方 CMA 不读取 SEQ2SEQ 开关；将整个嵌入清零，连动作索引也不能泄漏。
            trainer.policy.net.prev_action_embedding.register_forward_hook(
                lambda module, inputs, output: torch.zeros_like(output))
    elif prev_action_dropout > 0.0 and hasattr(trainer.policy.net, "prev_action_embedding"):
        drop_p = float(prev_action_dropout)

        def _drop_prev_hook(module, inputs, output):
            if not module.training:
                return output
            keep = (torch.rand((output.shape[0],) + (1,) * (output.ndim - 1),
                               device=output.device) >= drop_p).to(output.dtype)
            return output * keep

        trainer.policy.net.prev_action_embedding.register_forward_hook(_drop_prev_hook)
    return trainer


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}\n".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def manifest_action_schema(manifest):
    require(manifest.get("schema") in (SCHEMA, CAUSAL_SCHEMA), "manifest schema 不匹配")
    return CAUSAL_ACTION_SCHEMA if manifest["schema"] == CAUSAL_SCHEMA else ACTION_SCHEMA


def validate_causal_episode(ep, original):
    raw_path = Path(ep["raw_pose_path"])
    route_parts = Path(ep["route_id"]).parts
    require(raw_path.is_absolute() and raw_path.name == "pose.jsonl"
            and tuple(raw_path.parent.parts[-len(route_parts):]) == route_parts, "raw pose 路线不匹配")
    require(sha256(raw_path) == ep["raw_pose_sha256"], "raw pose SHA256 不匹配")
    # 上游 pose.jsonl 是连续的多行 JSON 对象，不能逐行 json.loads。
    decoder, text, raw = json.JSONDecoder(), raw_path.read_text(), []
    while text.strip():
        item, end = decoder.raw_decode(text.lstrip())
        text = text.lstrip()[end:]
        require(isinstance(item, dict), "raw pose 记录必须是对象")
        if "action" in item:
            raw.append(item["action"])
        else:
            require(set(item) == {"aim_landmark"}, "未知 raw pose 记录")

    def pose_matches(a, b):
        return (isinstance(a, list) and len(a) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in a)
                and math.dist(a[:3], b[:3]) <= 1e-6
                and abs((a[3] - b[3] + math.pi) % (2 * math.pi) - math.pi) <= 1e-6)

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

    # 已逐组验证的原语：水平前进 3m、左右转向 30°、升降 ±3m（Z 向上为正）与末尾 STOP。
    primitive = {("go straight", 3): 1, ("turn left", 30): 2,
                 ("turn right", 30): 3, ("go up", 3): 4,
                 ("go down", 3): 5, ("stop", 0): 0}
    poses, tokens = [], []
    require(raw, "raw pose 无动作")
    for i, action in enumerate(raw):
        require(isinstance(action, dict) and type(action.get("imageid")) is int
                and action["imageid"] == i, "raw imageid 必须连续且从零开始")
        pos, yaw = action.get("pos"), action.get("yaw")
        require(isinstance(pos, list) and len(pos) in (3, 4)
                and all(type(v) in (int, float) and math.isfinite(v) for v in pos)
                and type(yaw) in (int, float) and math.isfinite(yaw), "raw pose 必须有限")
        kind, value = action.get("type"), action.get("value")
        require(isinstance(kind, str) and type(value) in (int, float)
                and (kind, value) in primitive, "未验证的 raw 动作类型或幅度")
        poses.append(pos[:3] + [yaw])
        tokens.append(primitive[(kind, value)])
    require(tokens[-1] == 0 and 0 not in tokens[:-1], "raw 只允许唯一末尾 STOP")
    for i in range(len(raw) - 1):
        require(pose_matches(after(poses[i], tokens[i]), poses[i + 1]), "raw 相邻动作转移不匹配")
    bounds, i = [], 0
    while i < len(raw):
        end = i + 1
        if tokens[i] == 1:
            while end < len(raw) and end < i + 3 and tokens[end] == 1:
                end += 1
        bounds.append((i, end - 1))
        i = end
    groups = ep.get("groups")
    require(isinstance(groups, list) and len(groups) == len(bounds) == len(ep["actions"]),
            "压缩组数量不匹配")
    for i, ((first, last), group) in enumerate(zip(bounds, groups)):
        require(isinstance(group, dict) and type(group.get("first_imageid")) is int
                and type(group.get("last_imageid")) is int
                and (group["first_imageid"], group["last_imageid"]) == (first, last), "压缩组边界不匹配")
        token = {1: 1, 2: 8, 3: 9}[last - first + 1] if tokens[first] == 1 else tokens[first]
        require(ep["actions"][i] == token, "原宏动作与 raw 分组不匹配")
        require(ep["source_index_list"][i].rsplit("_", 1)[-1] == str(last)
                and ep["index_list"][i].rsplit("_", 1)[-1] == str(first), "组首/组末图像索引不匹配")
        require(first != last or ep["index_list"][i] == ep["source_index_list"][i], "单帧组不得替换原图索引")
        require(pose_matches(original["pos"][i][:3] + [original["yaw"][i]], poses[last]),
                "原标注位姿与组末 raw 不匹配")
        require(pose_matches(group.get("start_pose"), poses[first]), "组首 start_pose 不匹配")
        if i + 1 < len(bounds):
            require(pose_matches(after(poses[first], token), poses[bounds[i + 1][0]]), "组首宏动作转移不匹配")
    return [poses[first] for first, _ in bounds]


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    manifest_action_schema(manifest)
    causal = manifest["schema"] == CAUSAL_SCHEMA
    if causal:
        require(manifest.get("supervision") == "causal_group_start_macro_bc", "causal supervision 不匹配")
    source = Path(manifest["annotation_source"])
    require(source.is_absolute() and source.name == "train.json", "只接受官方 train.json 内部划分")
    require(sha256(source) == manifest["annotation_sha256"], "Annotation SHA256 不匹配")
    episodes = manifest.get("episodes", [])
    require(isinstance(episodes, list) and episodes, "episodes 不能为空")
    routes, signatures, counts = set(), set(), {"train": 0, "validation": 0}
    originals = {}
    wanted = {ep["route_id"] for ep in episodes}
    for row in json.loads(source.read_text()):
        if row.get("image_path") in wanted:
            originals.setdefault(row["image_path"], []).append(row)
    for ep in episodes:
        route, split = ep["route_id"], ep["split"]
        require(isinstance(route, str) and route and route not in routes, "route_id 缺失或重复")
        require(split in counts, "split 只能是 train/validation")
        require(isinstance(ep["instruction"], str) and ep["instruction"].strip(), "缺少语言")
        actions = ep["actions"]
        require(isinstance(actions, list) and len(actions) >= 2, "必须是完整多步轨迹")
        require(all(type(a) is int and 0 <= a <= 9 for a in actions), "动作必须是原始整数 0..9")
        require(actions[-1] == 0 and 0 not in actions[:-1], "只允许唯一末尾 STOP")
        for key in ("index_list", "images", "image_sha256"):
            require(isinstance(ep[key], list) and len(ep[key]) == len(actions), f"{key} 与动作不等长")
        require(all(isinstance(i, str) and i for i in ep["index_list"]), "index_list 必须是原始字符串")
        require(len(set(ep["index_list"])) == len(actions), "重复帧编号")
        source_indices = ep.get("source_index_list") if causal else ep["index_list"]
        require(isinstance(source_indices, list) and len(source_indices) == len(actions)
                and all(isinstance(i, str) and i for i in source_indices), "source_index_list 不匹配")
        matched = [r for r in originals.get(route, []) if r["gpt_instruction"] == ep["instruction"]
                   and r["action"] == actions and r["index_list"] == source_indices]
        require(len(matched) == 1, "路线/语言/动作/帧序列无法唯一匹配官方 Annotation")
        positions, yaws = matched[0].get("pos"), matched[0].get("yaw")
        require(isinstance(positions, list) and isinstance(yaws, list)
                and len(positions) == len(yaws) == len(actions), "pos/yaw 必须与动作等长")
        require(all(isinstance(p, list) and len(p) in (3, 4)
                    and all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in positions)
                and all(type(y) in (int, float) and math.isfinite(y) for y in yaws),
                "pos/yaw 必须是有限数字，位置必须为 xyz 或 xyz+yaw")
        if causal:
            start_poses = validate_causal_episode(ep, matched[0])
            positions = [p[:3] for p in start_poses]
            yaws = [p[3] for p in start_poses]
        # 物理路线按实际使用的 xyz/yaw 比较，不受 JSON 整数/小数或冗余第四维影响。
        signature = tuple((tuple(p[:3]), y) for p, y in zip(positions, yaws))
        require(signature not in signatures, "重复物理路线，禁止改写指令后跨划分")
        signatures.add(signature)
        for image, index, digest in zip(ep["images"], ep["index_list"], ep["image_sha256"]):
            image = Path(image)
            require(image.is_absolute() and image.stem == index
                    and tuple(image.parent.parts[-len(Path(route).parts):]) == Path(route).parts,
                    "图片必须绝对路径且完整路线/文件名匹配 index_list")
            require(sha256(image) == digest, f"图片 SHA256 不匹配: {image}")
        routes.add(route)
        counts[split] += 1
    require(all(n >= 2 for n in counts.values()), "每个划分至少两条路线，才能做独立指令置换")
    return manifest


def resolve_inflection_coef(spec, train_episodes):
    if spec is None or str(spec) == "1" or str(spec) == "1.0":
        return 1.0
    if str(spec) == "auto":
        def modal_family(a):
            return 1 if a in (1, 8, 9) else a
        inf_steps, total_steps = 0, 0
        for ep in train_episodes:
            acts = ep["actions"]
            for t, a in enumerate(acts):
                total_steps += 1
                if t == 0 or modal_family(a) != modal_family(acts[t - 1]):
                    inf_steps += 1
        require(inf_steps > 0, "训练集无拐点样本")
        return max(1.5, float(total_steps - inf_steps) / float(inf_steps))
    val = float(spec)
    require(math.isfinite(val) and val > 0.0, "inflection_weight 必须为有限正数或 auto")
    return val


def episode_tensors(ep, features, tokens, device, use_prev_action=False,
                    inflection_coef=1.0, stop_weight=1.0):
    import torch
    require(features.shape[0] == len(ep["actions"]), "特征与动作不等长")
    target = torch.tensor(ep["actions"], dtype=torch.long, device=device).view(-1, 1)
    prev = torch.zeros_like(target)
    if use_prev_action and len(target) > 1:
        prev[1:, 0] = target[:-1, 0]
    masks = torch.ones_like(target, dtype=torch.uint8)
    masks[0] = 0
    weights = torch.ones_like(target, dtype=torch.float32)
    if inflection_coef != 1.0:
        for t in range(len(ep["actions"])):
            if t == 0 or ep["actions"][t] != ep["actions"][t - 1]:
                weights[t, 0] = float(inflection_coef)
    if stop_weight != 1.0:
        for t, a in enumerate(ep["actions"]):
            if a == 0:
                weights[t, 0] *= float(stop_weight)
    text = torch.as_tensor(tokens, dtype=torch.long, device=device).view(1, -1).repeat(len(target), 1)
    return {"instruction": text, "rgb_features": features.to(device)}, prev, masks, target, weights


def logits_for(trainer, batch):
    import torch
    obs, prev, masks, _, _ = batch
    require(sorted(obs) == INPUTS, "模型输入越过白名单")
    net = trainer.policy.net
    hidden = torch.zeros(1, net.num_recurrent_layers, net.state_encoder.hidden_size, device=trainer.device)
    return trainer.policy.build_distribution(obs, hidden, prev, masks).logits


def evaluate(trainer, batches, zero_language=False, permute=False):
    import torch
    require(len(batches) >= 2 or not permute, "置换语言至少需要两条路线")
    trainer.policy.eval()
    hook = trainer.policy.net.instruction_encoder.register_forward_hook(
        lambda module, inputs, output: torch.zeros_like(output)) if zero_language else None
    confusion = torch.zeros(10, 10, dtype=torch.long)
    total_loss, samples, two_cycle_osc = 0.0, 0, 0
    try:
        with torch.no_grad():
            for i, batch in enumerate(batches):
                if permute:
                    obs = dict(batch[0])
                    donor = batches[(i + 1) % len(batches)][0]["instruction"][0]
                    obs["instruction"] = donor.unsqueeze(0).repeat(len(batch[3]), 1)
                    batch = (obs, *batch[1:])
                logits, target = logits_for(trainer, batch), batch[3].view(-1)
                preds = logits.argmax(1)
                for t in range(1, len(preds)):
                    pair = (int(preds[t - 1].item()), int(preds[t].item()))
                    if pair in ((2, 3), (3, 2)):
                        two_cycle_osc += 1
                total_loss += float(torch.nn.functional.cross_entropy(logits, target, reduction="sum"))
                samples += len(target)
                confusion += torch.bincount((target * 10 + preds).cpu(), minlength=100).reshape(10, 10)
    finally:
        if hook is not None:
            hook.remove()
    require(samples > 0 and math.isfinite(total_loss), "评估没有有效样本或 loss 非有限")
    pred_stop = int(confusion[:, 0].sum())
    gt_stop = int(confusion[0].sum())
    tp_stop = int(confusion[0, 0])
    stop_prec = float(tp_stop) / pred_stop if pred_stop > 0 else 0.0
    stop_rec = float(tp_stop) / gt_stop if gt_stop > 0 else 0.0
    f1s = []
    for c in range(10):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum()) - tp
        fn = int(confusion[c, :].sum()) - tp
        if tp + fp + fn > 0:
            f1s.append(2.0 * tp / (2.0 * tp + fp + fn))
    out = {"samples": samples, "cross_entropy": total_loss / samples,
           "accuracy": float(confusion.diag().sum()) / samples,
           "macro_f1_present": sum(f1s) / len(f1s) if f1s else 0.0,
           "stop_precision": stop_prec,
           "stop_recall": stop_rec,
           "stop_f1": (2.0 * stop_prec * stop_rec / (stop_prec + stop_rec)) if (stop_prec + stop_rec) > 0 else 0.0,
           "two_cycle_turn_oscillations": two_cycle_osc,
           "confusion": confusion.tolist()}
    return out


def _is_v2_requested(cli):
    if getattr(cli, "contract_version", "auto") == "v2":
        return True
    if getattr(cli, "contract_version", "auto") == "v1":
        return False
    return any([
        bool(getattr(cli, "imagenet_norm", False)),
        bool(getattr(cli, "use_bert", False)),
        bool(getattr(cli, "use_prev_action", False)),
        bool(getattr(cli, "progress_monitor", False)),
        str(getattr(cli, "inflection_weight", "1.0")) not in ("1", "1.0"),
        float(getattr(cli, "stop_weight", 1.0)) != 1.0,
    ])


def run(cli):
    require(cli.epochs > 0, "epochs 必须为正")
    policy_type = getattr(cli, "policy_type", "seq2seq")
    feature_shape = rgb_feature_shape(policy_type)
    manifest = load_manifest(cli.manifest)
    require(not cli.out.exists(), "输出目录已存在，拒绝覆盖")
    use_v2 = _is_v2_requested(cli)
    imagenet_norm = bool(getattr(cli, "imagenet_norm", False))
    use_bert = bool(getattr(cli, "use_bert", False))
    use_prev_action = bool(getattr(cli, "use_prev_action", False))
    prev_dropout = float(getattr(cli, "prev_action_dropout", 0.0)) if use_prev_action else 0.0
    progress_monitor = bool(getattr(cli, "progress_monitor", False))
    stop_weight = float(getattr(cli, "stop_weight", 1.0))
    require(math.isfinite(stop_weight) and stop_weight > 0.0, "stop_weight 必须为有限正数")
    train_eps = [ep for ep in manifest["episodes"] if ep["split"] == "train"]
    inflection_coef = resolve_inflection_coef(getattr(cli, "inflection_weight", "1.0"), train_eps)
    if not use_v2:
        require(not any([imagenet_norm, use_bert, use_prev_action, progress_monitor])
                and inflection_coef == 1.0 and stop_weight == 1.0, "v1 契约不允许开启 v2 特性开关")

    import numpy as np
    import torch
    from PIL import Image
    from gym import spaces
    require(torch.cuda.is_available(), "此实验要求真实 CUDA；不会静默转 CPU")
    root = cli.airvln_root.resolve()
    require((Path(torch.hub.get_dir()) / "checkpoints/resnet50-0676ba61.pth").is_file(), "缺少已缓存 ResNet50，拒绝隐式下载")
    vocab_path = root.parent / "DATA/data/aerialvln/train_vocab.txt"
    vocab = vocab_path.read_text().splitlines()
    require(len(vocab) == len(set(vocab)) and vocab[:2] == ["<PAD>", "<UNK>"], "词表契约错误")
    cli.out.mkdir(parents=True)
    sys.path.insert(0, str(root))
    saved_argv = sys.argv
    sys.argv = [str(__file__), "--project_prefix", str(root.parent), "--policy_type", policy_type]
    try:
        from src.common.param import args
    finally:
        sys.argv = saved_argv
    args.DistributedDataParallel = False
    args.logger_file_name = str(cli.out / "airvln.log")
    args.ablate_depth = True
    args.PROGRESS_MONITOR_use = progress_monitor
    mode = getattr(cli, "training_mode", "rgb_language")
    configure_training_mode(args, mode)
    args.ablate_rgb = False
    args.tokenizer_use_bert = use_bert
    args.SEQ2SEQ_use_prev_action = use_prev_action
    args.vocab_size = len(vocab)
    from src.vlnce_src.util import Tokenizer
    from Model.aux_losses import AuxLosses
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = load_bert_tokenizer() if use_bert else Tokenizer(vocab, encoding_length=300)
    observation_space = spaces.Dict({"rgb": spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
                                     "depth": spaces.Box(0, 1, (256, 256, 1), dtype=np.float32)})
    trainer = build_trainer(observation_space, spaces.Discrete(10), policy_type,
                            use_prev_action=use_prev_action, prev_action_dropout=prev_dropout,
                            progress_monitor=progress_monitor, use_bert=use_bert)
    trainer.policy.eval()
    initial_state_sha256 = state_sha256(trainer.policy.state_dict())
    batches, oov, word_count = {"train": [], "validation": []}, 0, 0
    for ep in manifest["episodes"]:
        tokens, n_tok, n_oov = encode_instruction_text(ep["instruction"], tokenizer, use_bert=use_bert)
        oov += n_oov
        word_count += n_tok
        features = []
        hook = trainer.policy.net.rgb_encoder.layer_extract.register_forward_hook(
            lambda module, inputs, output: features.append(output.detach().cpu().clone()))
        try:
            with torch.no_grad():
                for start in range(0, len(ep["images"]), 16):
                    arrays = []
                    for path in ep["images"][start:start + 16]:
                        with Image.open(path) as image:
                            arrays.append(np.asarray(image.convert("RGB").resize((224, 224), Image.BILINEAR)).copy())
                    images = preprocess_rgb_uint8(np.stack(arrays), trainer.device, imagenet_norm=imagenet_norm)
                    trainer.policy.net.rgb_encoder.cnn(images)
        finally:
            hook.remove()
        features = torch.cat(features)
        require(tuple(features.shape[1:]) == feature_shape, "RGB 特征形状错误")
        is_tr = ep["split"] == "train"
        batches[ep["split"]].append(episode_tensors(
            ep, features, tokens, trainer.device,
            use_prev_action=use_prev_action,
            inflection_coef=inflection_coef if is_tr else 1.0,
            stop_weight=stop_weight if is_tr else 1.0,
        ))
    code_paths = [Path(__file__).resolve(), *sorted((root / "Model").rglob("*.py")),
                  root / "src/common/param.py", root / "src/vlnce_src/util.py"]
    prev_impl = ("causal_prev_action_with_step_dropout" if use_prev_action
                 else ("zero_embedding_output" if policy_type == "cma" else "disabled_by_config"))
    vis_desc = ("RGB; PIL bilinear 224x224; float32 /255; ImageNet mean/std normalization; frozen ResNet50 eval"
                if imagenet_norm
                else "RGB; PIL bilinear 224x224; float32 /255; no ImageNet normalization; frozen ResNet50 eval")
    contract = {"schema": CONTRACT_V2 if use_v2 else CONTRACT_V1,
                "action_schema": manifest_action_schema(manifest),
                "manifest_schema": manifest["schema"],
                "supervision": manifest.get("supervision", "annotation_indexed_offline_bc"),
                "training_mode": mode, "initial_state_sha256": initial_state_sha256,
                "policy_type": policy_type, "rgb_feature_shape": list(feature_shape),
                "previous_action_implementation": prev_impl,
                "baseline_variant": (
                    f"AirVLN {policy_type} v2: imagenet_norm={imagenet_norm}, bert={use_bert}, "
                    f"prev_action={use_prev_action}(drop={prev_dropout}), inflect={inflection_coef:.3f}, "
                    f"stop_w={stop_weight:.2f}, prog_mon={progress_monitor}"
                    if use_v2 else f"AirVLN {policy_type}: RGB, no depth, no previous action; adapted 10-action BC"),
                "parameter_count": sum(p.numel() for p in trainer.policy.parameters()),
                "parameter_counts": {"total": sum(p.numel() for p in trainer.policy.parameters()),
                                     "requires_grad": sum(p.numel() for p in trainer.policy.parameters() if p.requires_grad),
                                     "received_gradient": None},
                "parameter_count_note": "total includes frozen/unused branches; requires_grad is nominal; received_gradient counts parameters with Adam state after training",
                "language_branch": "disabled throughout training and evaluation" if mode == "rgb_only" else "enabled",
                "input_whitelist": INPUTS, "manifest_sha256": sha256(cli.manifest),
                "annotation_sha256": manifest["annotation_sha256"], "vocab_sha256": sha256(vocab_path),
                "code_sha256": {str(p): sha256(p) for p in code_paths}, "seed": cli.seed,
                "epochs": cli.epochs, "precision": "float32", "previous_action_enabled": use_prev_action,
                "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(trainer.device),
                "visual_preprocessing": vis_desc,
                "language": {"tokens": word_count, "oov": oov, "max_input": 300,
                             **({"encoder": "frozen_bert_base_uncased" if use_bert else "scratch_vocab_embedding"} if use_v2 else {})},
                **({"v2_config": {"imagenet_norm": imagenet_norm, "use_bert": use_bert,
                                  "use_prev_action": use_prev_action, "prev_action_dropout": prev_dropout,
                                  "inflection_weight_coef": inflection_coef, "stop_weight": stop_weight,
                                  "progress_monitor": progress_monitor}} if use_v2 else {}),
                "limits": ["causal group-start macro BC; raw pose transitions verified; physical flight unverified"
                           if manifest["schema"] == CAUSAL_SCHEMA
                           else "annotation-indexed offline BC; compressed token dynamics unverified",
                           "internal route validation from official train; no unseen-scene or closed-loop claim"]}
    (cli.out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    AuxLosses.deactivate()
    before = evaluate(trainer, batches["train"])
    lang_param = ("net.instruction_encoder.encoder_rnn.weight_ih_l0"
                  if use_bert else "net.instruction_encoder.embedding_layer.weight")
    watched = [lang_param, "action_distribution.linear.weight"]
    initial = {k: trainer.policy.state_dict()[k].detach().clone() for k in watched}
    if not progress_monitor:
        AuxLosses.activate()
    trainer.optimizer.zero_grad()
    updates = 0
    with (cli.out / "metrics.jsonl").open("x") as log:
        for epoch in range(cli.epochs):
            trainer.policy.train()
            trainer.policy.net.rgb_encoder.cnn.eval()
            if use_bert:
                trainer.policy.net.instruction_encoder.embedding_layer.eval()
            order = list(range(len(batches["train"])))
            random.shuffle(order)
            losses = []
            for i in order:
                batch_i = batches["train"][i]
                if progress_monitor:
                    obs_i = dict(batch_i[0])
                    t_len = len(batch_i[3])
                    obs_i["progress"] = torch.linspace(0.0, 1.0, steps=t_len, dtype=torch.float32, device=trainer.device)
                    AuxLosses.activate()
                    try:
                        loss, _, _ = trainer._update_agent(obs_i, *batch_i[1:])
                    finally:
                        AuxLosses.deactivate()
                else:
                    loss, _, _ = trainer._update_agent(*batch_i)
                require(math.isfinite(loss), "训练 loss 非有限")
                losses.append(loss)
                updates += 1
            row = {"epoch": epoch + 1, "updates": updates, "mean_train_update_loss": sum(losses) / len(losses),
                   "train_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                   "train": evaluate(trainer, batches["train"]), "validation": evaluate(trainer, batches["validation"])}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    AuxLosses.deactivate()
    changes = {k: float((trainer.policy.state_dict()[k] - initial[k]).abs().max()) for k in watched}
    contract["parameter_counts"]["received_gradient"] = sum(p.numel() for p, state in trainer.optimizer.state.items() if state)
    (cli.out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    require(all(math.isfinite(v) for v in changes.values()) and changes[watched[1]] > 0, "动作头没有真实更新或参数非有限")
    require(changes[watched[0]] == 0 if mode == "rgb_only" else changes[watched[0]] > 0,
            "语言参数变化不符合训练模态")
    require(all(bool(torch.isfinite(p).all()) for p in trainer.policy.parameters()), "训练参数出现非有限值")
    diagnostics = {split: {"correct": evaluate(trainer, data),
                          "zero_language_encoder_output": evaluate(trainer, data, zero_language=True),
                          "cyclic_instruction_swap": evaluate(trainer, data, permute=True)} for split, data in batches.items()}
    if mode == "rgb_only":
        require(all(d["correct"] == d["zero_language_encoder_output"] == d["cyclic_instruction_swap"]
                    for d in diagnostics.values()), "纯视觉训练模型仍受语言输入影响")
    trainer.policy.eval()
    with torch.no_grad():
        expected = logits_for(trainer, batches["validation"][0]).cpu()
    checkpoint = cli.out / "model.pt"
    torch.save({"contract": contract, "state_dict": trainer.policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(), "updates": updates}, checkpoint)
    reloaded = build_trainer(observation_space, spaces.Discrete(10), policy_type,
                             use_prev_action=use_prev_action, prev_action_dropout=prev_dropout,
                             progress_monitor=progress_monitor, use_bert=use_bert)
    saved = torch.load(checkpoint, map_location=reloaded.device)
    require(saved["contract"].get("training_mode", "rgb_language") == mode, "checkpoint 训练模态不匹配")
    require(saved["contract"].get("policy_type", "seq2seq") == policy_type, "checkpoint 策略类型不匹配")
    reloaded.policy.load_state_dict(saved["state_dict"], strict=True)
    reloaded.policy.eval()
    with torch.no_grad():
        actual = logits_for(reloaded, batches["validation"][0]).cpu()
    difference = float((expected - actual).abs().max())
    require(torch.allclose(expected, actual, rtol=1e-6, atol=1e-7), "checkpoint 重载 logits 不一致")
    result = {"updates": updates, "before_train": before, "diagnostics": diagnostics,
              "weight_changes": changes, "reload_max_abs_logit_diff": difference,
              "checkpoint_sha256": sha256(checkpoint), "contract": contract}
    (cli.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (cli.out / "DONE").write_text(f"{contract['supervision']} completed\n")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--airvln-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--policy-type", choices=POLICY_TYPES, default="seq2seq",
                   help="seq2seq 池化视觉特征 2048x1x1；cma 保留空间视觉特征 2048x4x4 与跨模态注意力")
    p.add_argument("--training-mode", choices=TRAINING_MODES, default="rgb_language",
                   help="rgb_only 从头单独训练并始终禁用语言；不是推理时消融")
    p.add_argument("--contract-version", choices=("auto", "v1", "v2"), default="auto")
    p.add_argument("--imagenet-norm", action="store_true", help="开启 ResNet50 ImageNet mean/std 归一化")
    p.add_argument("--use-bert", action="store_true", help="启用冻结 bert-base-uncased InstructionBertEncoder")
    p.add_argument("--inflection-weight", default="1.0", help="拐点步权重系数（数字或 auto）")
    p.add_argument("--stop-weight", type=float, default=1.0, help="末尾 STOP 步额外乘性权重")
    p.add_argument("--progress-monitor", action="store_true", help="训练期启用归一化进度辅助损失")
    p.add_argument("--use-prev-action", action="store_true", help="启用因果上一动作输入")
    p.add_argument("--prev-action-dropout", type=float, default=0.2, help="训练期 prev_action 整步置零概率")
    return p


def main():
    print(json.dumps(run(parser().parse_args())), flush=True)


if __name__ == "__main__":
    main()
