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
from train_openfly_subset import ACTION_SCHEMA, INPUTS, load_manifest, require, sha256

PROTOCOL = "openfly_eval_pose_step_v1"
ACTIONS = ("STOP", "FORWARD_3M", "LEFT_30DEG", "RIGHT_30DEG", "UP_3M",
           "DOWN_3M", "LEFT_3M", "RIGHT_3M", "FORWARD_6M", "FORWARD_9M")


def next_pose(pose, action):
    """对应官方快照 train/eval.py::getPoseAfterMakeAction，yaw 为弧度。"""
    require(type(action) is int and 0 <= action < 10, "非法动作，不能静默转换为 STOP")
    require(len(pose) == 4 and all(math.isfinite(v) for v in pose), "位姿非有限或维度错误")
    x, y, z, yaw = pose
    if action in (1, 8, 9):
        distance = {1: 3.0, 8: 6.0, 9: 9.0}[action]
        x, y = x + distance * math.cos(yaw), y + distance * math.sin(yaw)
    elif action in (2, 3):
        yaw += math.radians(30 if action == 2 else -30)
    elif action in (4, 5):
        z += 3 if action == 4 else -3
    elif action in (6, 7):
        direction = 1 if action == 6 else -1
        x, y = x - direction * 3 * math.sin(yaw), y + direction * 3 * math.cos(yaw)
    return [x, y, z, (yaw + math.pi) % (2 * math.pi) - math.pi]


def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def yaw_error(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def recurrent_step(policy, observation, hidden, previous, mask):
    require(sorted(observation) == INPUTS, "模型输入超出 RGB/语言白名单")
    features, hidden = policy.net(observation, hidden, previous, mask)
    logits = policy.action_distribution(features).logits
    require(tuple(logits.shape) == (1, 10), "动作输出不是 [1,10]")
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

    def __init__(self, cli, manifest):
        import numpy as np
        import torch
        from gym import spaces
        self.torch = torch
        root = cli.airvln_root.resolve()
        saved = torch.load(cli.checkpoint, map_location="cpu")
        contract = saved["contract"]
        training_mode = checkpoint_training_mode(contract)
        require(contract["schema"] == "uavrgb.openfly_external_bc.v1"
                and contract["action_schema"] == ACTION_SCHEMA, "checkpoint 协议错误")
        require(contract["input_whitelist"] == INPUTS and contract["previous_action_enabled"] is False
                and contract["precision"] == "float32", "checkpoint 输入/精度不兼容")
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
        sys.argv = [__file__, "--project_prefix", str(root.parent), "--policy_type", "seq2seq"]
        try:
            from src.common.param import args
        finally:
            sys.argv = old_argv
        args.DistributedDataParallel = False
        args.logger_file_name = str(cli.out / "airvln.log")
        args.ablate_depth, args.PROGRESS_MONITOR_use = True, False
        args.ablate_instruction = training_mode == "rgb_only" or cli.language == "zero"
        args.ablate_rgb, args.tokenizer_use_bert = False, False
        args.SEQ2SEQ_use_prev_action = False
        vocab = vocab_path.read_text().splitlines()
        args.vocab_size = len(vocab)
        from src.vlnce_src.util import Tokenizer
        from Model.il_trainer import VLNCETrainer
        observation_space = spaces.Dict({"rgb": spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
                                         "depth": spaces.Box(0, 1, (256, 256, 1), dtype=np.float32)})
        trainer = VLNCETrainer(False, observation_space, spaces.Discrete(10))
        trainer.policy.load_state_dict(saved["state_dict"], strict=True)
        self.policy, self.device = trainer.policy.eval(), trainer.device
        self.tokenizer = Tokenizer(vocab, encoding_length=300)
        self.contract = contract
        self.checkpoint_sha256 = sha256(cli.checkpoint)
        self.details = {"policy": "student", "training_mode": training_mode, "blind_baseline": None,
                        "active_inputs": ["rgb_features"] if args.ablate_instruction else INPUTS}

    def reset(self, instruction):
        torch = self.torch
        words = self.tokenizer.split_sentence(instruction)
        require(0 < len(words) <= 300, "指令空白或超长，禁止截断")
        self.tokens = torch.as_tensor(self.tokenizer.encode_sentence(instruction), device=self.device).view(1, -1)
        net = self.policy.net
        self.hidden = torch.zeros(1, net.num_recurrent_layers, net.state_encoder.hidden_size, device=self.device)
        self.previous = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)

    def step(self, rgb):
        import numpy as np
        from PIL import Image
        torch, captured = self.torch, []
        image = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)).copy()
        tensor = torch.as_tensor(image, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        hook = self.policy.net.rgb_encoder.layer_extract.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            with torch.no_grad():
                self.policy.net.rgb_encoder.cnn(tensor)
                observation = {"instruction": self.tokens, "rgb_features": captured[0]}
                logits, self.hidden = recurrent_step(self.policy, observation, self.hidden, self.previous, self.mask)
                require(bool(torch.isfinite(logits).all()), "非有限模型输出")
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
        self.client.simSetVehiclePose(target, True, vehicle_name=self.vehicle)
        self.client.simContinueForTime(0.02)
        deadline = time.monotonic() + 10
        while not self.client.simIsPause():
            require(time.monotonic() < deadline, "仿真 tick 超时")
            time.sleep(0.01)
        time.sleep(0.1)

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
            self.lines += ["Policy receives swapped instruction:", *textwrap.wrap(supplied, width=105)]
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
        status = f"frame={row['frame']}  action={label}  goal distance={row['goal_distance_m']:.2f}m [evaluation only]"
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
    try:
        policy.reset(instruction)
        simulator.begin(pose)
        while True:
            rgb, actual, collided, timestamp = simulator.observe()
            last_distance = distance(actual, goal)
            action, logits, candidate = None, None, None
            if collided:
                reason = "collision_reported"
            elif distance(actual, pose) > 0.2 or yaw_error(actual[3], pose[3]) > math.radians(1):
                reason = "pose_application_error"
            elif previous_timestamp is not None and timestamp <= previous_timestamp:
                reason = "stale_rgb_timestamp"
            elif time.monotonic() - began >= cli.timeout:
                reason = "time_budget"
            elif steps >= cli.max_steps:
                reason = "step_budget"
            else:
                action, logits = policy.step(rgb)
                candidate = next_pose(pose, action)
                if action == 0:
                    reason = "fixed_stop" if policy.details["policy"] == "blind_forward" else "predicted_stop"
                elif travelled + distance(candidate, pose) > cli.max_distance:
                    reason = "distance_budget"
            row = {**policy.details, "frame": frame, "timestamp": timestamp, "pose": actual, "requested_pose": pose,
                   "goal_distance_m": last_distance, "pose_error_m": distance(actual, pose),
                   "yaw_error_deg": math.degrees(yaw_error(actual[3], pose[3])),
                   "action": action, "action_log_probs": logits, "collision_reported": collided,
                   "language": cli.language, "termination": reason,
                   "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                   "rgb_shape": list(rgb.shape), "action_requested": reason is None}
            try:
                recorder.write(rgb, row)
            except Exception as exc:
                video_error = f"{type(exc).__name__}: {exc}"
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
            recorder.close()
        except Exception as exc:
            video_error = f"{video_error or ''}; finalize: {type(exc).__name__}: {exc}"
    return {**policy.details, "termination": reason, "error": error, "executed_actions": steps, "frames": frame,
            "video_error": video_error or getattr(recorder, "video_error", None),
            "commanded_distance_m": travelled, "final_goal_distance_m": last_distance,
            "within_20m_diagnostic": last_distance < 20 if last_distance is not None else None,
            "stopped_within_20m_diagnostic": reason in ("predicted_stop", "fixed_stop") and last_distance < 20,
            "elapsed_s": time.monotonic() - began}


def sequence_parity(policy, episode):
    """仅用于门禁：同一离线图像序列，逐帧记忆输出必须对齐整段前向。"""
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from train_openfly_subset import episode_tensors, logits_for
    torch, online, arrays, features = policy.torch, [], [], []
    policy.reset(episode["instruction"])
    for path in episode["images"]:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        online.append(policy.step(rgb)[1])
        arrays.append(np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)).copy())
    hook = policy.policy.net.rgb_encoder.layer_extract.register_forward_hook(
        lambda module, inputs, output: features.append(output))
    try:
        with torch.no_grad():
            tensor = torch.as_tensor(np.stack(arrays), device=policy.device).permute(0, 3, 1, 2).float() / 255.0
            policy.policy.net.rgb_encoder.cnn(tensor)
            batch = episode_tensors(episode, features[0], policy.tokens[0], policy.device)
            offline = logits_for(SimpleNamespace(policy=policy.policy, device=policy.device), batch).cpu()
    finally:
        hook.remove()
    online = torch.tensor(online)
    difference = float((online - offline).abs().max())
    require(torch.allclose(online, offline, atol=1e-5, rtol=1e-5), "逐帧与整段前向不一致")
    return {"frames": len(arrays), "max_abs_logit_diff": difference,
            "argmax_equal": bool(torch.equal(online.argmax(1), offline.argmax(1))),
            "scope": "offline sequence gate only; rollout observations remain real-time RGB"}


def run(cli):
    require(cli.limit > 0 and cli.max_steps > 0 and cli.max_distance > 0 and cli.timeout > 0, "预算必须为正")
    if cli.policy == "blind_forward":
        require(cli.checkpoint is None and cli.airvln_root is None, "盲基线不使用 checkpoint 或 AirVLN 模型")
        require(cli.language == "correct", "盲基线不消费语言，不适用 swap/zero")
    else:
        require(cli.checkpoint is not None and cli.airvln_root is not None, "student 需要 --checkpoint 和 --airvln-root")
    require(not cli.out.exists(), "输出已存在，拒绝覆盖")
    manifest = load_manifest(cli.manifest)
    episodes = [ep for ep in manifest["episodes"] if ep["split"] == "validation"]
    require(episodes and (cli.language != "swap" or len(episodes) >= 2), "无有效验证路线或无法置换")
    wanted = {ep["route_id"] for ep in episodes}
    source = {r["image_path"]: r for r in json.loads(Path(manifest["annotation_source"]).read_text())
              if r["image_path"] in wanted}
    require(all("medium_short" in ep["route_id"] for ep in episodes), "此小样本诊断只验证水平相机 medium_short")
    cli.out.mkdir(parents=True)
    policy = BlindForward(manifest) if cli.policy == "blind_forward" else Policy(cli, manifest)
    contract = {**policy.details, "protocol": PROTOCOL, "checkpoint_sha256": policy.checkpoint_sha256,
                "training_contract": policy.contract, "evaluator_sha256": sha256(__file__),
                "manifest_sha256": sha256(cli.manifest), "annotation_sha256": manifest["annotation_sha256"],
                "action_schema": ACTION_SCHEMA,
                "input_whitelist": policy.input_whitelist, "language": cli.language, "record_video": cli.record_video,
                "max_steps": cli.max_steps, "max_distance_m": cli.max_distance, "timeout_s": cli.timeout,
                "camera": cli.camera, "vehicle": cli.vehicle, "port": cli.port,
                "pose_feedback_gate": {"xyz_tolerance_m": 0.2, "yaw_tolerance_deg": 1,
                                       "timestamp": "strictly increasing after each executed action"},
                "renderer_capture": "AirSim compressed PNG decoded with PIL RGB; latest feedback only",
                "limits": ["Internal held-out routes from official train; no unseen-scene or official SR claim",
                           "Pose teleport ignores collision response; reported endpoint collision is not swept-path safety",
                           "This 16-route subset matches greedy grouping of up to 3 raw forward actions into 1/8/9, retaining each group's final raw pre-action frame. This observed 150-frame pattern does not verify official compression intent; native macro execution from retained poses is not raw-action replay. Manifest and labels remain frozen for protocol-limited comparison.",
                           "Video has one frame per observation, fixed 2fps; no interpolation or synthetic frames"]}
    (cli.out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    simulator, results, cleanup_error = Simulator(cli), [], None
    try:
        for i, ep in enumerate(episodes[:cli.limit]):
            directory = cli.out / f"episode_{i:03d}"
            directory.mkdir()
            original = source[ep["route_id"]]
            start = list(original["pos"][0][:3]) + [original["yaw"][0]]
            goal = list(original["pos"][-1][:3])
            supplied = episodes[(i + 1) % len(episodes)]["instruction"] if cli.language == "swap" else ep["instruction"]
            recorder = Recorder(directory, ep["instruction"], supplied, cli.record_video)
            with (directory / "trace.jsonl").open("x") as trace:
                result = rollout(simulator, policy, recorder, trace, start, goal, supplied, cli)
            result.update(route_id=ep["route_id"], instruction=ep["instruction"], supplied_instruction=supplied,
                          start=start, goal_for_evaluation_only=goal, output=str(directory), protocol=PROTOCOL)
            (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        try:
            simulator.close()
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
    summary = {**policy.details, "protocol": PROTOCOL, "checkpoint_sha256": policy.checkpoint_sha256, "episodes": results,
               "errors": sum(r["error"] is not None for r in results),
               "protocol_errors": sum(r["termination"] in ("pose_application_error", "stale_rgb_timestamp")
                                      for r in results),
               "video_errors": sum(r["video_error"] is not None for r in results), "cleanup_error": cleanup_error,
               "note": "Closed-loop rendered feedback diagnosis; not physical flight or official benchmark"}
    (cli.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if summary["errors"] or summary["protocol_errors"] or summary["video_errors"] or cleanup_error else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--policy", choices=("student", "blind_forward"), default="student")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--airvln-root", type=Path)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-distance", type=float, default=300)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--port", type=int, default=41458)
    p.add_argument("--vehicle", default="drone_1")
    p.add_argument("--camera", default="front_custom")
    p.add_argument("--language", choices=("correct", "swap", "zero"), default="correct")
    video = p.add_mutually_exclusive_group()
    video.add_argument("--record-video", dest="record_video", action="store_true")
    video.add_argument("--no-record-video", dest="record_video", action="store_false")
    p.set_defaults(record_video=True)
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
