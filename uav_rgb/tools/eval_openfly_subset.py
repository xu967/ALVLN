#!/usr/bin/env python3
"""OpenFly 原生动作的真实渲染反馈诊断；位姿步进，不是物理飞行或正式基准。"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
import sys
import textwrap
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_openfly_subset import (
    CAUSAL_SCHEMA, CONTRACT_V1, CONTRACT_V2, INPUTS,
    encode_instruction_text, load_bert_tokenizer, load_manifest,
    manifest_action_schema, preprocess_rgb_uint8, require, sha256,
)

LEGACY_PROTOCOL = "openfly_eval_pose_step_v1"
PROTOCOL = "openfly_eval_pose_step_v2"
KINEMATIC_PROTOCOL = "openfly_eval_kinematic_step_v3"
METRIC_PROTOCOL = "uavrgb.openfly_local_metrics.v2"
SUCCESS_RADIUS_M = 20.0
INITIAL_RGB_RETRIES = 5
PROTOCOL_ERRORS = ("pose_application_error", "stale_rgb_timestamp", "black_rgb")
ALLOWED_EVAL_ENVS = (
    "env_airsim_16", "env_airsim_18", "env_airsim_23",
    "env_airsim_26", "env_airsim_gz", "env_airsim_sh",
)
ACTIONS = ("STOP", "FORWARD_3M", "LEFT_30DEG", "RIGHT_30DEG", "UP_3M",
           "DOWN_3M", "LEFT_3M", "RIGHT_3M", "FORWARD_6M", "FORWARD_9M")


# 10 个离散动作 -> 机体系增量 (forward_m, left_m, up_m, dyaw_deg)
_ACTION_DELTAS = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (3.0, 0.0, 0.0, 0.0),
    2: (0.0, 0.0, 0.0, 30.0),
    3: (0.0, 0.0, 0.0, -30.0),
    4: (0.0, 0.0, 3.0, 0.0),
    5: (0.0, 0.0, -3.0, 0.0),
    6: (0.0, 3.0, 0.0, 0.0),
    7: (0.0, -3.0, 0.0, 0.0),
    8: (6.0, 0.0, 0.0, 0.0),
    9: (9.0, 0.0, 0.0, 0.0),
}


def next_pose(pose, action):
    """对应官方快照 train/eval.py::getPoseAfterMakeAction，yaw 为弧度。"""
    require(type(action) is int and action in _ACTION_DELTAS, "非法动作，不能静默转换为 STOP")
    require(len(pose) == 4 and all(math.isfinite(v) for v in pose), "位姿非有限或维度错误")
    x, y, z, yaw = pose
    fwd, left, up, dyaw_deg = _ACTION_DELTAS[action]
    c, s = math.cos(yaw), math.sin(yaw)
    nx = x + fwd * c - left * s
    ny = y + fwd * s + left * c
    return [nx, ny, z + up, (yaw + math.radians(dyaw_deg) + math.pi) % (2 * math.pi) - math.pi]


def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def yaw_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def recurrent_step(policy, observation, hidden, previous, mask):
    require(sorted(observation) == INPUTS, "模型输入超出 RGB/语言白名单")
    features, hidden = policy.net(observation, hidden, previous, mask)
    logits = policy.action_distribution(features).logits
    require(tuple(logits.shape) == (1, 10), "动作输出不是 [1,10]")
    require(bool(logits.isfinite().all()) and bool(hidden.isfinite().all()), "非有限模型输出或历史状态")
    return logits, hidden


def checkpoint_training_mode(contract):
    mode = contract.get("training_mode", "rgb_language")
    require(mode in ("rgb_language", "rgb_only"), "checkpoint training_mode 不支持")
    return mode


class BlindForward:
    """只从训练标签固定前进动作和 STOP 时刻，不读取图像、指令或评估终点。"""
    checkpoint_sha256, contract, input_whitelist = None, None, []

    def __init__(self, manifest):
        train = [ep for ep in manifest["episodes"] if ep["split"] == "train"]
        counts = Counter(a for ep in train for a in ep["actions"] if a in (1, 8, 9))
        require(train and counts, "盲基线必须有 train 路线与前进标签")
        lengths = [len(ep["actions"]) - 1 for ep in train]
        self.action = min(counts, key=lambda a: (-counts[a], a))
        self.stop_after = math.ceil(median(lengths))
        self.details = {"policy": "blind_forward", "training_mode": None, "active_inputs": [],
                        "blind_baseline": {"selection_split": "train",
                            "forward_action": self.action, "stop_after_forward_actions": self.stop_after,
                            "action_rule": "most frequent train forward token among 1/8/9; ties choose smaller token",
                            "stop_rule": "ceil(median(train route action counts excluding final STOP))",
                            "train_forward_counts": {str(a): counts[a] for a in (1, 8, 9)},
                            "train_route_lengths": [dict(route_id=ep["route_id"], nonstop_actions=n)
                                                    for ep, n in zip(train, lengths)]}}

    def reset(self, instruction):
        self.steps = 0

    def step(self, rgb):
        action = self.action if self.steps < self.stop_after else 0
        self.steps += 1
        return action, None


class Policy:
    input_whitelist = INPUTS
    imagenet_norm = False
    use_bert = False
    use_prev_action = False
    prev_dropout = 0.0
    progress_monitor = False

    def __init__(self, cli, manifest):
        import numpy as np
        import torch
        from gym import spaces
        from train_openfly_subset import build_trainer, rgb_feature_shape
        self.torch = torch
        root = cli.airvln_root.resolve()
        saved = torch.load(cli.checkpoint, map_location="cpu")
        contract = saved["contract"]
        training_mode = checkpoint_training_mode(contract)
        policy_type = contract.get("policy_type", "seq2seq")
        self.feature_shape = rgb_feature_shape(policy_type)
        require(tuple(contract.get("rgb_feature_shape", (2048, 1, 1))) == self.feature_shape,
                "checkpoint RGB 特征形状不匹配")
        self.use_prev_action = bool(contract.get("previous_action_enabled", False))
        if policy_type == "cma" and not self.use_prev_action:
            require(contract.get("previous_action_implementation") == "zero_embedding_output",
                    "CMA checkpoint 未禁用前一动作输入")
        schema = contract.get("schema")
        require(schema in (CONTRACT_V1, CONTRACT_V2)
                and contract["action_schema"] == manifest_action_schema(manifest), "checkpoint 协议错误")
        v2_cfg = contract.get("v2_config") or {}
        self.imagenet_norm = bool(v2_cfg.get("imagenet_norm", False))
        self.use_bert = bool(v2_cfg.get("use_bert", False))
        self.prev_dropout = float(v2_cfg.get("prev_action_dropout", 0.0))
        self.progress_monitor = bool(v2_cfg.get("progress_monitor", False))
        if schema == CONTRACT_V1:
            require(contract["previous_action_enabled"] is False, "checkpoint 输入/精度不兼容")
        require(contract["input_whitelist"] == INPUTS and contract["precision"] == "float32",
                "checkpoint 输入/精度不兼容")
        require(contract["manifest_sha256"] == sha256(cli.manifest)
                and contract["annotation_sha256"] == manifest["annotation_sha256"], "checkpoint 数据清单不匹配")
        vocab_path = root.parent / "DATA/data/aerialvln/train_vocab.txt"
        require(contract["vocab_sha256"] == sha256(vocab_path), "词表 SHA256 不匹配")
        for path, digest in contract["code_sha256"].items():
            original = Path(path)
            local = root / str(original).split("/AirVLN/", 1)[1] if "/AirVLN/" in path else original
            require(sha256(local) == digest, f"训练源码变化: {local}")
        require(torch.cuda.is_available(), "此诊断要求 CUDA，不静默降级")
        sys.path.insert(0, str(root))
        old_argv = sys.argv
        sys.argv = [__file__, "--project_prefix", str(root.parent), "--policy_type", policy_type]
        try:
            from src.common.param import args
        finally:
            sys.argv = old_argv
        args.DistributedDataParallel = False
        args.logger_file_name = str(cli.out / "airvln.log")
        args.ablate_depth = True
        args.PROGRESS_MONITOR_use = self.progress_monitor
        args.ablate_instruction = training_mode == "rgb_only" or cli.language == "zero"
        args.ablate_rgb = False
        args.tokenizer_use_bert = self.use_bert
        args.SEQ2SEQ_use_prev_action = self.use_prev_action
        vocab = vocab_path.read_text().splitlines()
        args.vocab_size = len(vocab)
        from src.vlnce_src.util import Tokenizer
        from Model.aux_losses import AuxLosses
        AuxLosses.deactivate()
        observation_space = spaces.Dict({"rgb": spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
                                         "depth": spaces.Box(0, 1, (256, 256, 1), dtype=np.float32)})
        trainer = build_trainer(observation_space, spaces.Discrete(10), policy_type,
                                use_prev_action=self.use_prev_action, prev_action_dropout=self.prev_dropout,
                                progress_monitor=self.progress_monitor, use_bert=self.use_bert)
        trainer.policy.load_state_dict(saved["state_dict"], strict=True)
        self.policy, self.device = trainer.policy.eval(), trainer.device
        self.tokenizer = load_bert_tokenizer() if self.use_bert else Tokenizer(vocab, encoding_length=300)
        self.contract = contract
        self.checkpoint_sha256 = sha256(cli.checkpoint)
        self.details = {"policy": "student", "policy_type": policy_type,
                        "training_mode": training_mode, "blind_baseline": None,
                        "active_inputs": ["rgb_features"] if args.ablate_instruction else INPUTS}

    def reset(self, instruction):
        torch = self.torch
        tokens = [0] * 300
        if instruction:
            tokens, _, _ = encode_instruction_text(instruction, self.tokenizer, use_bert=self.use_bert)
        self.tokens = torch.as_tensor(tokens, device=self.device).view(1, -1)
        net = self.policy.net
        self.hidden = torch.zeros(1, net.num_recurrent_layers, net.state_encoder.hidden_size, device=self.device)
        self.previous = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

    def step(self, rgb):
        import numpy as np
        from PIL import Image
        torch, captured = self.torch, []
        image = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)).copy()
        tensor = preprocess_rgb_uint8(image[None, ...], self.device, imagenet_norm=self.imagenet_norm)
        hook = self.policy.net.rgb_encoder.layer_extract.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            with torch.no_grad():
                self.policy.net.rgb_encoder.cnn(tensor)
                require(len(captured) == 1 and tuple(captured[0].shape[1:]) == self.feature_shape,
                        "在线 RGB 特征形状不匹配")
                observation = {"instruction": self.tokens, "rgb_features": captured[0]}
                logits, self.hidden = recurrent_step(self.policy, observation, self.hidden, self.previous, self.mask)
                action = int(logits.argmax(1).item())
                self.previous.fill_(action)
                self.mask.fill_(True)
                return action, logits[0].cpu().tolist()
        finally:
            hook.remove()


class Simulator:
    def __init__(self, cli):
        import airsim
        self.api, self.vehicle, self.camera = airsim, cli.vehicle, cli.camera
        self.protocol = getattr(cli, "execution_protocol", KINEMATIC_PROTOCOL)
        require(self.protocol in (LEGACY_PROTOCOL, PROTOCOL, KINEMATIC_PROTOCOL), "未知执行协议")
        self.client = airsim.MultirotorClient(port=cli.port, timeout_value=10)
        require(self.client.ping(), "AirSim RPC 无响应")

    def begin(self, pose):
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.vehicle)
        self.client.armDisarm(True, vehicle_name=self.vehicle)
        self.client.simPause(True)
        self.advance(pose)

    def advance(self, pose):
        x, y, z, yaw = pose
        target = self.api.Pose(self.api.Vector3r(x, -y, -z), self.api.to_quaternion(0, 0, -yaw))
        if self.protocol == KINEMATIC_PROTOCOL:
            # 位姿步进不继承上一窗下落速度；这是运动学诊断，不是物理飞行控制。
            state = self.api.KinematicsState()
            state.position, state.orientation = target.position, target.orientation
            state.linear_velocity = self.api.Vector3r(0, 0, 0)
            state.angular_velocity = self.api.Vector3r(0, 0, 0)
            state.linear_acceleration = self.api.Vector3r(0, 0, 0)
            state.angular_acceleration = self.api.Vector3r(0, 0, 0)
            self.client.simSetKinematics(state, True, vehicle_name=self.vehicle)
        else:
            self.client.simSetVehiclePose(target, True, vehicle_name=self.vehicle)
        phases = [(self.client.simContinueForTime, 0.02)]
        if self.protocol != LEGACY_PROTOCOL:
            # physics 更新与 pawn 渲染分属两阶段；固定追加帧，不按反馈重发位姿。
            phases.append((self.client.simContinueForFrames, 2))
        for advance, duration in phases:
            self._advance_tick(advance, duration)
        time.sleep(0.1)

    def _advance_tick(self, advance, duration):
        advance(duration)
        deadline = time.monotonic() + 10
        while not self.client.simIsPause():
            require(time.monotonic() < deadline, "仿真 tick 超时")
            time.sleep(0.01)

    def refresh_initial_rgb(self):
        """相机曝光需要新的仿真帧。保留当前位姿，不重发动作。"""
        self._advance_tick(self.client.simContinueForFrames, 2)

    def observe(self):
        import numpy as np
        from PIL import Image
        response = self.client.simGetImages([self.api.ImageRequest(self.camera, self.api.ImageType.Scene, False, True)],
                                            vehicle_name=self.vehicle)[0]
        require(response.width > 0 and response.height > 0 and response.image_data_uint8, "空 RGB 响应")
        with Image.open(io.BytesIO(bytes(response.image_data_uint8))) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle)
        _, _, yaw = self.api.to_eularian_angles(pose.orientation)
        position = [pose.position.x_val, -pose.position.y_val, -pose.position.z_val, -yaw]
        require(all(math.isfinite(v) for v in position), "仿真返回非有限位姿")
        collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle)
        return rgb, position, bool(collision.has_collided), int(response.time_stamp)

    def close(self):
        self.client.simPause(True)
        self.client.cancelLastTask(vehicle_name=self.vehicle)


def open_video(path, size):
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim/tools"))
    from view_layout import VideoSink, find_ffmpeg
    require(find_ffmpeg() is not None, "没有 ffmpeg，不能静默生成不可播放的 MP4")
    sink = VideoSink(cv2, path, fps=2, size=size, crf=20)
    require(sink.isOpened(), "H264 视频编码器未打开")
    return sink


class Recorder:
    def __init__(self, directory, original, supplied, record_video):
        from PIL import ImageFont
        self.directory, self.enabled, self.sink, self.count = directory, record_video, None, 0
        self.video_error = None
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        self.lines = ["Original instruction:", *textwrap.wrap(original, width=105)]
        if supplied != original:
            self.lines += ["Policy instruction:", *textwrap.wrap(supplied or "[empty]", width=105)]
        self.height = 706 + 26 * len(self.lines)
        self.height += self.height % 2

    def write(self, rgb, row):
        import numpy as np
        from PIL import Image, ImageDraw
        image = Image.fromarray(rgb)
        if self.count == 0:
            image.save(self.directory / "first_rgb.png")
        image.save(self.directory / "last_rgb.png")
        canvas = Image.new("RGB", (1280, self.height), (20, 23, 29))
        image.thumbnail((1248, 560))
        canvas.paste(image, ((1280 - image.width) // 2, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 8), "DECISION FRAMES | pose stepping, NOT real-time physical flight | 2 frames/s", font=self.font)
        action = row["action"]
        label = ACTIONS[action] if action is not None else "NO ACTION"
        status = f"frame={row['frame']}  predicted={label}  goal distance={row['goal_distance_m']:.2f}m [evaluation only]"
        draw.text((16, 604), status, font=self.font)
        draw.text((16, 634), f"policy={row['policy']}  training={row['training_mode'] or 'not_trained'}  language={row['language']}", font=self.font)
        status = f"end={row.get('termination') or 'running'}"
        if row.get("blind_baseline"):
            baseline = row["blind_baseline"]
            status += f" | train-only rule: {ACTIONS[baseline['forward_action']]} x {baseline['stop_after_forward_actions']} then STOP"
        draw.text((16, 660), status, font=self.font)
        for i, line in enumerate(self.lines):
            draw.text((16, 696 + i * 26), line, font=self.font)
        if self.count == 0:
            canvas.save(self.directory / "preview_first.png")
        canvas.save(self.directory / "preview_last.png")
        if self.enabled:
            try:
                if self.sink is None:
                    self.sink = open_video(self.directory / "decisions.mp4", canvas.size)
                self.sink.write(np.asarray(canvas)[:, :, ::-1].copy())
            except Exception as exc:
                self.video_error = f"{type(exc).__name__}: {exc}"
                self.enabled = False
                self.close()
        self.count += 1

    def close(self):
        if self.sink is not None:
            try:
                self.sink.release()
            except Exception as exc:
                self.video_error = f"{self.video_error or ''}; finalize: {type(exc).__name__}: {exc}"
            finally:
                self.sink = None


def rollout(simulator, policy, recorder, trace, start, goal, instruction, cli):
    began, pose, travelled, steps, frame = time.monotonic(), list(start), 0.0, 0, 0
    reason, last_distance, error, video_error, previous_timestamp = None, None, None, None, None
    closest_distance, initial_rgb_retries = None, 0
    try:
        require(len(pose) == 4 and all(math.isfinite(v) for v in pose), "起始位姿非有限或维度错误")
        require(len(goal) == 3 and all(math.isfinite(v) for v in goal), "评估终点非有限或维度错误")
        policy.reset(instruction)
        simulator.begin(pose)
        while True:
            rgb, actual, collided, timestamp = simulator.observe()
            require(len(actual) == 4 and all(math.isfinite(v) for v in actual), "反馈位姿非有限或维度错误")
            last_distance = distance(actual, goal)
            pose_error = distance(actual, pose)
            yaw_error_deg = math.degrees(yaw_error(actual[3], pose[3]))
            action, logits, candidate = None, None, None
            if collided:
                reason = "collision_reported"
            elif pose_error > 0.2 or yaw_error_deg > 1.0:
                reason = "pose_application_error"
            elif not rgb.any():
                protocol = getattr(simulator, "protocol", KINEMATIC_PROTOCOL)
                retry_limit = 0 if protocol == LEGACY_PROTOCOL else INITIAL_RGB_RETRIES
                if frame == 0 and initial_rgb_retries < retry_limit and time.monotonic() - began < cli.timeout:
                    simulator.refresh_initial_rgb()
                    if type(timestamp) is int and timestamp > 0:
                        previous_timestamp = max(previous_timestamp or timestamp, timestamp)
                    initial_rgb_retries += 1
                    continue
                reason = "black_rgb"
            elif type(timestamp) is not int or timestamp <= 0 or (
                    previous_timestamp is not None and timestamp <= previous_timestamp):
                reason = "stale_rgb_timestamp"

            observation_valid = reason is None
            if observation_valid:
                closest_distance = last_distance if closest_distance is None else min(closest_distance, last_distance)
            if observation_valid and time.monotonic() - began >= cli.timeout:
                reason = "time_budget"
            if reason is None and steps >= cli.max_steps:
                reason = "step_budget"
            if reason is None:
                action, logits = policy.step(rgb)
                candidate = next_pose(pose, action)
                if time.monotonic() - began >= cli.timeout:
                    reason = "time_budget"
                elif action == 0:
                    reason = "fixed_stop" if policy.details["policy"] == "blind_forward" else "predicted_stop"
                elif travelled + distance(candidate, pose) > cli.max_distance:
                    reason = "distance_budget"
            row = {**policy.details, "frame": frame, "timestamp": timestamp if type(timestamp) is int else None,
                   "protocol": getattr(simulator, "protocol", KINEMATIC_PROTOCOL),
                   "pose": actual, "requested_pose": pose,
                   "goal_distance_m": last_distance, "pose_error_m": pose_error,
                   "yaw_error_deg": yaw_error_deg, "observation_valid": observation_valid,
                   "action": action, "action_log_probs": logits, "collision_reported": collided,
                   "language": cli.language, "termination": reason,
                   "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                   "rgb_shape": list(rgb.shape), "action_requested": reason is None}
            try:
                if recorder is not None:
                    recorder.write(rgb, row)
            except Exception as exc:
                video_error = f"{type(exc).__name__}: {exc}"
            # 编码也计入墙钟预算；已过期的候选动作不能再发给模拟器。
            if reason is None and time.monotonic() - began >= cli.timeout:
                reason = row["termination"] = "time_budget"
                row["action_requested"] = False
            row["video_error"] = video_error or getattr(recorder, "video_error", None)
            trace.write(json.dumps(row) + "\n")
            trace.flush()
            frame += 1
            previous_timestamp = timestamp
            if reason is not None:
                break
            simulator.advance(candidate)
            travelled += distance(candidate, pose)
            pose, steps = candidate, steps + 1
    except Exception as exc:
        reason, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if recorder is not None:
                recorder.close()
        except Exception as exc:
            video_error = f"{video_error or ''}; finalize: {type(exc).__name__}: {exc}"
    return {**policy.details, "termination": reason, "error": error, "executed_actions": steps, "frames": frame,
            "video_error": video_error or getattr(recorder, "video_error", None),
            "commanded_distance_m": travelled, "final_goal_distance_m": last_distance,
            "min_goal_distance_m": closest_distance, "initial_rgb_retries": initial_rgb_retries,
            "within_20m_diagnostic": last_distance < SUCCESS_RADIUS_M if last_distance is not None else None,
            "ever_within_20m_diagnostic": closest_distance is not None and closest_distance < SUCCESS_RADIUS_M,
            "stopped_within_20m_diagnostic": reason in ("predicted_stop", "fixed_stop") and last_distance < SUCCESS_RADIUS_M,
            "elapsed_s": time.monotonic() - began}


def sequence_parity(policy, episode):
    """仅用于门禁：同一离线图像序列，逐帧记忆输出必须对齐整段前向。"""
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from train_openfly_subset import episode_tensors, logits_for
    torch, online, online_actions, arrays, features = policy.torch, [], [], [], []
    policy.reset(episode["instruction"])
    for path in episode["images"]:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        act, lgs = policy.step(rgb)
        online_actions.append(act)
        online.append(lgs)
        arrays.append(np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)).copy())
    hook = policy.policy.net.rgb_encoder.layer_extract.register_forward_hook(
        lambda module, inputs, output: features.append(output))
    try:
        with torch.no_grad():
            tensor = preprocess_rgb_uint8(
                np.stack(arrays), policy.device, imagenet_norm=getattr(policy, "imagenet_norm", False))
            policy.policy.net.rgb_encoder.cnn(tensor)
            batch = episode_tensors(episode, features[0], policy.tokens[0], policy.device)
            if getattr(policy, "use_prev_action", False) and len(online_actions) > 1:
                batch[1][1:, 0] = torch.tensor(online_actions[:-1], dtype=torch.long, device=policy.device)
            offline = logits_for(SimpleNamespace(policy=policy.policy, device=policy.device), batch).cpu()
    finally:
        hook.remove()
    online = torch.tensor(online)
    difference = float((online - offline).abs().max())
    require(torch.allclose(online, offline, atol=1e-5, rtol=1e-5), "逐帧与整段前向不一致")
    return {"frames": len(arrays), "max_abs_logit_diff": difference,
            "argmax_equal": bool(torch.equal(online.argmax(1), offline.argmax(1))),
            "scope": "offline sequence gate only; rollout observations remain real-time RGB"}


def load_evaluation_tasks(path, manifest_path, manifest):
    """独立读取官方 seen 任务；训练清单、策略输入和评估答案保持各自契约。"""
    data = json.loads(path.read_text())
    require(data.get("schema") == "uavrgb.openfly_official_seen_tasks.v1", "评测任务 schema 不支持")
    require(data["training_manifest_sha256"] == sha256(manifest_path), "评测任务绑定的训练清单不匹配")
    source = Path(data["annotation_source"])
    require(source.is_absolute() and source.name == "seen.json", "评测任务必须来自官方 seen.json")
    require(sha256(source) == data["annotation_sha256"], "评测标注 SHA256 不匹配")
    require(data["source_repo"] == manifest["source_repo"]
            and data["source_revision"] == manifest["source_revision"], "评测与训练来源版本不匹配")
    rows = json.loads(source.read_text())
    training_rows = json.loads(Path(manifest["annotation_source"]).read_text())
    environment = data["selection"]["env"]
    require(environment in ALLOWED_EVAL_ENVS, f"评测任务场景 {environment!r} 不在已验证 AirSim 场景集合内")
    training_rows = [r for r in training_rows if r["image_path"].split("/")[0] == environment]
    excluded_routes = {r["image_path"] for r in training_rows}
    excluded_routes.update(ep["route_id"] for ep in manifest["episodes"])
    def geometry(row):
        return tuple((tuple(p[:3]), yaw) for p, yaw in zip(row["pos"], row["yaw"]))
    excluded_geometry = {geometry(r) for r in training_rows}
    tasks = data.get("tasks", [])
    require(isinstance(tasks, list) and tasks, "评测任务为空")
    routes, signatures = set(), set()
    for task in tasks:
        index = task["source_row_index"]
        require(type(index) is int and 0 <= index < len(rows), "评测源行索引非法")
        row = rows[index]
        digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(digest == task["source_record_sha256"], "评测源行 SHA256 不匹配")
        route = row["image_path"]
        require(route.startswith(environment + "/astar_data/medium_short/")
                and route not in excluded_routes and route not in routes, "评测路线重复、跨场景或与训练重叠")
        actions, positions, yaws = row["action"], row["pos"], row["yaw"]
        require(isinstance(actions, list) and len(actions) >= 2
                and all(type(a) is int and 0 <= a <= 9 for a in actions)
                and actions[-1] == 0 and 0 not in actions[:-1], "评测源动作必须完整且仅末尾 STOP")
        require(len(positions) == len(yaws) == len(row["index_list"]) == len(actions)
                and all(isinstance(p, list) and len(p) in (3, 4)
                        and all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in positions)
                and all(type(y) in (int, float) and math.isfinite(y) for y in yaws), "评测源位姿非法")
        require(isinstance(row["gpt_instruction"], str) and row["gpt_instruction"].strip(), "评测指令为空")
        require(task["route_id"] == route and task["instruction"] == row["gpt_instruction"]
                and task["start_pose"] == positions[0][:3] + [yaws[0]]
                and task["goal_position"] == positions[-1][:3], "评测任务内容偏离源标注")
        signature = geometry(row)
        require(signature not in excluded_geometry and signature not in signatures, "评测物理路线重复或与训练重叠")
        routes.add(route)
        signatures.add(signature)
    return data


def run(cli):
    require(all(math.isfinite(v) and v > 0 for v in
                (cli.limit, cli.max_steps, cli.max_distance, cli.timeout)), "预算必须为有限正数")
    if cli.policy == "blind_forward":
        require(cli.checkpoint is None and cli.airvln_root is None, "盲基线不使用 checkpoint 或 AirVLN 模型")
        require(cli.language == "correct", "盲基线不消费语言，不适用 swap/zero")
    else:
        require(cli.checkpoint is not None and cli.airvln_root is not None, "student 需要 --checkpoint 和 --airvln-root")
    require(not cli.out.exists(), "输出已存在，拒绝覆盖")
    manifest = load_manifest(cli.manifest)
    evaluation = load_evaluation_tasks(cli.evaluation_tasks, cli.manifest, manifest) if cli.evaluation_tasks else None
    episodes = evaluation["tasks"] if evaluation else [ep for ep in manifest["episodes"] if ep["split"] == "validation"]
    if getattr(cli, "scene", None):
        episodes = [ep for ep in episodes if ep["route_id"].split("/", 1)[0] == cli.scene]
    instructions = [ep["instruction"] for ep in episodes]
    require(episodes and (cli.language != "swap" or len(set(instructions)) >= 2), "无有效验证路线或无法置换")
    wanted = {ep["route_id"] for ep in episodes}
    source = {}
    if not evaluation and manifest["schema"] != CAUSAL_SCHEMA:
        source = {r["image_path"]: r for r in json.loads(Path(manifest["annotation_source"]).read_text())
                  if r["image_path"] in wanted}
    require(all("medium_short" in ep["route_id"] for ep in episodes), "此小样本诊断只验证水平相机 medium_short")
    cli.out.mkdir(parents=True)
    policy = BlindForward(manifest) if cli.policy == "blind_forward" else Policy(cli, manifest)
    protocol = cli.execution_protocol
    contract = {**policy.details, "protocol": protocol, "checkpoint_sha256": policy.checkpoint_sha256,
                "training_contract": policy.contract, "evaluator_sha256": sha256(__file__),
                "metric_protocol": METRIC_PROTOCOL,
                "metrics": {"radius_m": SUCCESS_RADIUS_M,
                            "estimated_local_osr": "closest valid observation over the episode",
                            "invalid_observations": list(PROTOCOL_ERRORS) + ["collision_reported"]},
                "manifest_sha256": sha256(cli.manifest), "annotation_sha256": manifest["annotation_sha256"],
                "manifest_schema": manifest["schema"], "supervision": manifest.get("supervision"),
                "action_schema": manifest_action_schema(manifest),
                "evaluation_tasks_sha256": sha256(cli.evaluation_tasks) if evaluation else None,
                "evaluation_annotation_sha256": evaluation["annotation_sha256"] if evaluation else manifest["annotation_sha256"],
                "evaluation_scope": "official_seen_subset" if evaluation else "internal_development",
                "evaluation_start": "published pos[0]/yaw[0]" if evaluation else "training manifest observation pose",
                "synchronization": {"pose_requests_per_action": 1, "sim_continue_for_time_argument_s": 0.02,
                    "render_frames": 0 if protocol == LEGACY_PROTOCOL else 2, "retries": 0,
                    "initial_black_rgb_retries": INITIAL_RGB_RETRIES,
                    "initial_black_rgb_retry_limit": 0 if protocol == LEGACY_PROTOCOL else INITIAL_RGB_RETRIES,
                    "initial_black_rgb_sim_continue_for_time_s_per_retry": 0,
                    "initial_black_rgb_render_frames_per_retry":
                        0 if protocol == LEGACY_PROTOCOL else 2,
                    "state_reset": ("atomic pose + zero linear/angular velocity and acceleration"
                                    if protocol == KINEMATIC_PROTOCOL else "pose only; velocity retained")},
                "execution_semantics": ("Kinematic reset per step; not physical flight or official OpenFly protocol"
                                        if protocol == KINEMATIC_PROTOCOL else "Pose teleport with residual physics"),
                "input_whitelist": policy.input_whitelist, "language": cli.language, "record_video": cli.record_video,
                "max_steps": cli.max_steps, "max_distance_m": cli.max_distance, "timeout_s": cli.timeout,
                "timeout_semantics": "wall clock checked before inference and motion dispatch; STOP must be predicted before deadline; final recording does not revoke timely termination",
                "camera": cli.camera, "vehicle": cli.vehicle, "port": cli.port,
                "pose_feedback_gate": {"xyz_tolerance_m": 0.2, "yaw_tolerance_deg": 1,
                                       "timestamp": "strictly increasing after actions; initial black RGB may be retried"},
                "renderer_capture": "AirSim compressed PNG decoded with PIL RGB; latest feedback only",
                "limits": [("Official seen env18 subset only; safety gates differ from upstream; no full benchmark or leaderboard claim"
                            if evaluation else "Internal held-out routes from official train; no unseen-scene or official SR claim"),
                           "Pose teleport ignores collision response; reported endpoint collision is not swept-path safety",
                           ("Official evaluation reset and goal use published Annotation endpoints, separate from causal training observations" if evaluation else
                            "Causal manifest uses verified raw group-start observations and poses with unchanged published macro labels; no official benchmark claim"
                            if manifest["schema"] == CAUSAL_SCHEMA else
                            "Published group-end observations are not causal macro starts; this frozen manifest is a protocol-limited diagnostic"),
                           "Video has one frame per observation, fixed 2fps; no interpolation or synthetic frames"]}
    (cli.out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    simulator, results, cleanup_error, run_error = None, [], None, None
    try:
        simulator = Simulator(cli)
        for i, ep in enumerate(episodes[:cli.limit]):
            directory = cli.out / f"episode_{i:03d}"
            directory.mkdir()
            if evaluation:
                start, goal = list(ep["start_pose"]), list(ep["goal_position"])
            elif manifest["schema"] == CAUSAL_SCHEMA:
                start = list(ep["groups"][0]["start_pose"])
                goal = list(ep["groups"][-1]["start_pose"][:3])
            else:
                original = source[ep["route_id"]]
                start = list(original["pos"][0][:3]) + [original["yaw"][0]]
                goal = list(original["pos"][-1][:3])
            supplied = "" if cli.language == "zero" else ep["instruction"]
            if cli.language == "swap":
                supplied = next(text for text in instructions[i + 1:] + instructions[:i] if text != supplied)
            recorder, recorder_error = None, None
            try:
                recorder = Recorder(directory, ep["instruction"], supplied, cli.record_video)
            except Exception as exc:
                recorder_error = f"{type(exc).__name__}: {exc}"
            with (directory / "trace.jsonl").open("x") as trace:
                result = rollout(simulator, policy, recorder, trace, start, goal, supplied, cli)
            result["video_error"] = recorder_error or result["video_error"]
            result.update(route_id=ep["route_id"], instruction=ep["instruction"], supplied_instruction=supplied,
                          start=start, goal_for_evaluation_only=goal, output=str(directory), protocol=protocol)
            (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print(json.dumps(result), flush=True)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if simulator is not None:
                simulator.close()
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
    summary = {**policy.details, "protocol": protocol, "checkpoint_sha256": policy.checkpoint_sha256, "episodes": results,
               "metric_protocol": METRIC_PROTOCOL, "n_requested": min(len(episodes), cli.limit),
               "errors": sum(r["error"] is not None for r in results),
               "protocol_errors": sum(r["termination"] in PROTOCOL_ERRORS for r in results),
               "video_errors": sum(r["video_error"] is not None for r in results),
               "cleanup_error": cleanup_error, "run_error": run_error,
               "note": "Closed-loop rendered feedback diagnosis; not physical flight or official benchmark"}
    (cli.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if summary["errors"] or summary["protocol_errors"] or summary["video_errors"] or cleanup_error or run_error else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--policy", choices=("student", "blind_forward"), default="student")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--airvln-root", type=Path)
    p.add_argument("--evaluation-tasks", type=Path, help="独立官方 seen 任务清单；默认使用内部开发集")
    p.add_argument("--scene", choices=ALLOWED_EVAL_ENVS, default=None, help="多场景验证时仅跑当前打开的 AirSim 场景")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-distance", type=float, default=300)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--port", type=int, default=41458)
    p.add_argument("--vehicle", default="drone_1")
    p.add_argument("--camera", default="front_custom")
    p.add_argument("--execution-protocol", choices=(LEGACY_PROTOCOL, PROTOCOL, KINEMATIC_PROTOCOL),
                   default=KINEMATIC_PROTOCOL)
    p.add_argument("--language", choices=("correct", "swap", "zero"), default="correct")
    video = p.add_mutually_exclusive_group()
    video.add_argument("--record-video", dest="record_video", action="store_true")
    video.add_argument("--no-record-video", dest="record_video", action="store_false")
    p.set_defaults(record_video=True)
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
