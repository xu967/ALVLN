"""供 train.py 使用的冻结 CLIP 三扇区特征、局部路标方向训练及逐帧评分。"""
import hashlib
import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory


CLASSES = ("C", "L", "R")
BEARING_BINS = (0, 1, 7)
MODELS = ("region_text", "region_blank", "global_text")
FEATURES = ("left_text", "center_text", "right_text", "left_global", "center_global", "right_global")
MODEL_FILES = ("config.json", "merges.txt", "preprocessor_config.json", "pytorch_model.bin",
               "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_inputs(cfg):
    labels = Path(cfg["labels"])
    protocol = labels.parent / "region_registration.json"
    corpus = labels.parent / "registration.json"
    if sha(protocol) != cfg["protocol_sha256"] or sha(corpus) != cfg["corpus_registration_sha256"]:
        raise ValueError("Candidate protocol or original corpus registration SHA256 mismatch")
    registered = json.loads(corpus.read_text())
    snapshot = labels / "summary.json"
    raw = snapshot.read_bytes()
    summary = json.loads(raw)
    if summary.get("schema") != "uavrgb.vln.fork_bearings.v1":
        raise ValueError(f"Unexpected fork label schema in {snapshot}")
    archive = (labels.parent / cfg["original_manifests"]).resolve()
    originals = {Path(p).name: h for p, h in summary["sources_sha256"].items() if Path(p).parent.name == archive.name}
    manifests, frozen_sha = {}, {}
    for split in ("train", "development"):
        if set(registered["splits"][split]) != set(cfg[f"{split}_rooms"]):
            raise ValueError(f"Room split differs from the registration: {split}")
        expected = []
        for pool in (split, f"equal_{split}"):
            path = archive / f"{pool}.jsonl"
            source = path.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if digest != registered["registered_inputs"][path.name] or digest != originals.get(path.name):
                raise ValueError(f"Original manifest SHA256 mismatch: {path}")
            frozen_sha[path.name] = digest
            for number, line in enumerate(source.splitlines(), 1):
                row = json.loads(line)
                expected.append((pool, number, identity(row, row["instructions"])))
        derived = (labels / f"{split}.jsonl").read_bytes()
        manifests[split] = hashlib.sha256(derived).hexdigest()
        if not (manifests[split] == summary["output_sha256"][f"{split}.jsonl"] == cfg["label_sha256"][split]):
            raise ValueError(f"Label manifest SHA256 differs from the frozen export or corpus summary: {split}")
        lines = derived.splitlines()
        if len(lines) != len(expected) or len(lines) != registered["denominators"][f"{split}_states"]:
            raise ValueError(f"Incomplete registered coverage in {split}: {len(lines)} states, expected {len(expected)}")
        for i, (line, original) in enumerate(zip(lines, expected), 1):
            row = json.loads(line)
            if (row["source_pool"], row["source_line"], identity(row, [a["instruction"] for a in row["labels"]])) != original:
                raise ValueError(f"Derived sample identity differs from the original manifest: {split}, line {i}")
    checkpoint = Path(cfg["checkpoint"])
    files = {name: sha(checkpoint / name) for name in MODEL_FILES}
    signature = "".join(f"{name} {digest}\n" for name, digest in sorted(files.items()))
    if hashlib.sha256(signature.encode()).hexdigest() != cfg["model_files_digest"]:
        raise ValueError(f"CLIP checkpoint/tokenizer SHA256 differs from the registered revision: {checkpoint}")
    return dict(label_sha256=manifests, label_summary_sha256=hashlib.sha256(raw).hexdigest(),
                model_files_sha256=files, original_manifest_sha256=frozen_sha,
                protocol_sha256=cfg["protocol_sha256"], corpus_registration_sha256=cfg["corpus_registration_sha256"])


def identity(row, instructions):
    return tuple(row[k] for k in ("split", "scene", "pair", "spawn_group", "arm", "step", "rgb_sha256")) + tuple(instructions)


@dataclass
class Frame:
    scene: str
    pair: str
    spawn: str
    arm: int
    step: int
    shared: bool
    rgb: Path
    rgb_sha256: str
    instructions: tuple
    bins: tuple
    status: tuple
    inside: tuple

    @property
    def classes(self):
        return [BEARING_BINS.index(b) if s == "astar" and b in BEARING_BINS else -1
                for s, b in zip(self.status, self.bins)]


def read_frames(cfg, manifests):
    base = Path(cfg["labels"])
    frames = {}
    seen_rgb = set()
    for split, digest in manifests.items():
        source = base / f"{split}.jsonl"
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Label manifest changed after preflight: {source}")
        rows = []
        allowed = set(cfg[f"{split}_rooms"])
        for line, data in enumerate(raw.splitlines(), 1):
            row = json.loads(data)
            arms = row["labels"]
            if row["split"] != split or row["scene"] not in allowed or len(arms) != 2:
                raise ValueError(f"Unexpected split, scene or number of instructions at {source}:{line}")
            digest_rgb = row["rgb_sha256"]
            if digest_rgb in seen_rgb:
                raise ValueError(f"Duplicate RGB within/across splits: {digest_rgb}")
            seen_rgb.add(digest_rgb)
            texts = tuple(a["instruction"] for a in arms)
            if any(not isinstance(t, str) or not t.strip() for t in texts) or texts[0] == texts[1]:
                raise ValueError(f"Missing or identical fork instructions at {source}:{line}")
            for arm in arms:
                if arm["planning_status"] == "astar" and (type(arm["local_bearing_bin"]) is not int
                        or arm["local_bearing_bin"] not in range(8) or type(arm["inside_actual_image"]) is not bool):
                    raise ValueError(f"Invalid A* bearing or projection at {source}:{line}")
            if type(row["verified_shared_prefix"]) is not bool:
                raise ValueError(f"Invalid shared-prefix flag at {source}:{line}")
            rows.append(Frame(row["scene"], row["pair"], row["spawn_group"], row["arm"], row["step"],
                              row["verified_shared_prefix"], base / row["rgb"], digest_rgb, texts,
                              tuple(a.get("local_bearing_bin") for a in arms),
                              tuple(a["planning_status"] for a in arms), tuple(a.get("inside_actual_image") for a in arms)))
        if {row.scene for row in rows} != allowed:
            raise ValueError(f"Incomplete registered rooms in {source}")
        frames[split] = rows
    for attr in ("scene", "pair", "spawn"):
        if {getattr(r, attr) for r in frames["train"]} & {getattr(r, attr) for r in frames["development"]}:
            raise ValueError(f"Training and development overlap on {attr}")
    return frames


def load_rgb(frame):
    from PIL import Image
    with Image.open(frame.rgb) as image:
        rgb = image.convert("RGB")
    if hashlib.sha256(rgb.tobytes()).hexdigest() != frame.rgb_sha256:
        raise ValueError(f"Decoded RGB SHA256 mismatch: {frame.rgb}")
    return rgb


class FrozenClip:
    def __init__(self, cfg, device):
        import torch
        from transformers import AutoTokenizer, CLIPModel

        checkpoint = Path(cfg["checkpoint"])
        self.device = device
        self.cfg = cfg
        self.model = CLIPModel.from_pretrained(checkpoint, local_files_only=True).eval().requires_grad_(False)
        self.model.to(device=device, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.preprocess = json.loads((checkpoint / "preprocessor_config.json").read_text())
        self.text_cache = {}

    def text_vectors(self, texts):
        import torch
        import torch.nn.functional as F

        missing = sorted(set(texts) - self.text_cache.keys())
        batch_size = self.cfg["batch_size"]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            if tokens.input_ids.shape[1] > self.model.config.text_config.max_position_embeddings:
                raise ValueError("Full instruction exceeds CLIP's text window")
            with torch.inference_mode():
                text = self.model.text_model(**tokens.to(self.device)).pooler_output
                values = F.normalize(self.model.text_projection(text), dim=-1).cpu().numpy()
            self.text_cache.update(zip(batch, values))
        return self.text_cache

    def image_vectors(self, images):
        import numpy as np
        from PIL import Image, ImageOps
        import torch
        import torch.nn.functional as F

        size = self.cfg["image_size"]
        squares = [ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC,
                                color=tuple(self.cfg["pad_rgb"])) for im in images]
        pixels = np.stack([np.asarray(image, dtype=np.float32) for image in squares]) / 255.0
        pixels -= np.asarray(self.preprocess["image_mean"], dtype=np.float32)
        pixels /= np.asarray(self.preprocess["image_std"], dtype=np.float32)
        data = torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.inference_mode():
            vision = self.model.vision_model(pixel_values=data)
            patches = vision.last_hidden_state[:, 1:].reshape(len(images), 7, 7, -1)
            sectors = torch.stack([patches[:, :, a:b].mean(dim=(1, 2)) for a, b in self.cfg["patch_columns"]], dim=1)
            sectors = self.model.vision_model.post_layernorm(sectors)
            regional = F.normalize(self.model.visual_projection(sectors), dim=-1)
            global_cls = F.normalize(self.model.visual_projection(vision.pooler_output), dim=-1)
        return regional.cpu().numpy(), global_cls.cpu().numpy()


def extract_features(rows, encoder, batch_size):
    import numpy as np

    texts = encoder.text_vectors(["", *(t for row in rows for t in row.instructions)])
    result = {mode: [] for mode in MODELS}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        regions, globals_ = encoder.image_vectors([load_rgb(row) for row in batch])
        for row, region, global_cls in zip(batch, regions, globals_):
            region_global = region @ global_cls
            blank = np.r_[region @ texts[""], region_global]
            for text in row.instructions:
                word = texts[text]
                result["region_text"].append(np.r_[region @ word, region_global])
                result["region_blank"].append(blank)
                # 同一单位向量的余弦恒为 1；标准化会把浮点舍入残差放大成伪特征。
                result["global_text"].append([global_cls @ word] * 3 + [1.0] * 3)
    return {mode: np.asarray(v, dtype=np.float64) for mode, v in result.items()}


def fit_classifier(features, truth, cfg):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    keep = truth >= 0
    x, y = features[keep], truth[keep]
    counts = np.bincount(y, minlength=3)
    if min(counts) == 0:
        raise ValueError(f"All three classes need eligible training examples: {counts}")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - mean) / scale
    n, d = x.shape
    sample_weight = n / (3 * counts[y])
    def objective(flat):
        weights, bias = flat[:-3].reshape(d, 3), flat[-3:]
        logits = x @ weights + bias
        normalizer = logsumexp(logits, axis=1)
        loss = sample_weight @ (normalizer - logits[np.arange(n), y]) + 0.5 * np.sum(weights**2) / cfg["C"]
        error = np.exp(logits - normalizer[:, None])
        error[np.arange(n), y] -= 1.0
        error *= sample_weight[:, None]
        gradient = np.r_[(x.T @ error + weights / cfg["C"]).ravel(), error.sum(axis=0)]
        return loss / n, gradient / n

    fit = minimize(objective, np.zeros(3 * (d + 1)), jac=True, method="L-BFGS-B", tol=cfg["tol"],
                   options={"maxiter": cfg["max_iter"]})
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"Multinomial logistic fit failed: {fit.message}")
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=fit.x[:-3].reshape(d, 3).tolist(),
                bias=fit.x[-3:].tolist(), train_class_counts=counts.tolist(), iterations=int(fit.nit))


def predict(params, features):
    import numpy as np
    standard = (features - params["mean"]) / params["scale"]
    return np.argmax(standard @ params["weights"] + params["bias"], axis=1)


def fraction(num, den):
    return num / den if den else None


def label_metrics(y, pred):
    eligible = int((y >= 0).sum())
    correct = int((pred == y).sum())
    per_class = {}
    for c, name in enumerate(CLASSES):
        support = int((y == c).sum())
        hits = int(((y == c) & (pred == c)).sum())
        per_class[name] = dict(correct=hits, support=support, recall=fraction(hits, support))
    recalls = [per_class[c]["recall"] for c in CLASSES]
    return dict(total_requests=y.size, eligible=eligible, correct=correct, coverage=fraction(eligible, y.size),
                accuracy_all_requests=fraction(correct, y.size), accuracy_eligible=fraction(correct, eligible),
                per_class=per_class, macro_recall=sum(recalls) / 3 if None not in recalls else None)


def pair_metrics(rows, y, pred, mask):
    idx = mask.nonzero()[0]
    correct_idx = idx[(y[idx] == pred[idx]).all(axis=1)]
    return dict(states=len(idx), correct=len(correct_idx), both_correct_rate=fraction(len(correct_idx), len(idx)),
                predicted_different=int((pred[idx, 0] != pred[idx, 1]).sum()),
                route_pairs=len({rows[i].pair for i in idx}), spawn_groups=len({rows[i].spawn for i in idx}),
                correct_spawn_groups=sorted({rows[i].spawn for i in correct_idx}))


def score(rows, pred):
    import numpy as np
    y = np.array([row.classes for row in rows])
    eligible = (y >= 0).all(axis=1)
    different = eligible & (y[:, 0] != y[:, 1])
    inside = np.array([all(r.inside) for r in rows])
    shared = np.array([r.shared for r in rows])
    masks = {"all": np.ones(len(rows), dtype=bool), "both_expressible": eligible, "different": different,
             "both_inside": inside, "shared": shared, "both_inside_shared": inside & shared,
             "expressible_both_inside": eligible & inside, "expressible_shared": eligible & shared,
             "expressible_both_inside_shared": eligible & inside & shared,
             "different_both_inside": different & inside, "different_shared": different & shared,
             "different_both_inside_shared": different & inside & shared}
    by_group = {}
    for attr in ("scene", "pair", "spawn"):
        group = np.array([getattr(row, attr) for row in rows])
        by_group[attr] = {key: pair_metrics(rows, y, pred, different & (group == key)) for key in sorted(set(group))}
    return dict(single=label_metrics(y, pred),
                pairs={name: pair_metrics(rows, y, pred, mask) for name, mask in masks.items()},
                different_by_group=by_group)


def promotion(scores):
    full = scores["region_text"]
    comparisons = [scores[k] for k in ("region_blank", "global_text", "always_center")]
    primary = full["pairs"]["different"]
    more_pairs = all(primary["correct"] > s["pairs"]["different"]["correct"] for s in comparisons)
    macro = full["single"]["macro_recall"]
    macros = [s["single"]["macro_recall"] for s in comparisons]
    recall_pass = macro is not None and all(m is not None and macro >= m for m in macros)
    spawns = len(primary["correct_spawn_groups"])
    return dict(strictly_more_different_pairs=more_pairs, matches_best_macro_recall=recall_pass,
                correct_different_spawn_groups=spawns, passed=more_pairs and recall_pass and spawns >= 2)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def write_predictions(path, rows, predictions):
    with path.open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            arms = [dict(instruction=t, planning_status=row.status[j], local_bearing_bin=row.bins[j],
                         inside_actual_image=row.inside[j], truth=CLASSES[c] if c >= 0 else None,
                         predicted={name: CLASSES[p[i, j]] for name, p in predictions.items()})
                    for j, (t, c) in enumerate(zip(row.instructions, row.classes))]
            item = dict(scene=row.scene, pair=row.pair, spawn_group=row.spawn, arm=row.arm, step=row.step,
                        verified_shared_prefix=row.shared, rgb=str(row.rgb), rgb_sha256=row.rgb_sha256, labels=arms)
            print(json.dumps(item, ensure_ascii=False, allow_nan=False), file=stream)


def run(cfg, checked, run_dir):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = read_frames(cfg, checked["label_sha256"])
    code = Path(__file__).resolve()
    registration = dict(schema="uavrgb.fork_region.fit.v1", config=cfg, inputs=checked, classes=CLASSES,
                        bearing_bins=BEARING_BINS, features=FEATURES, device=device, precision="float32 CLIP / float64 fit",
                        versions={k: version(k) for k in ("torch", "transformers", "numpy", "scipy", "Pillow")},
                        code_sha256={f.name: sha(f) for f in (code, code.parent.parent / "train.py")},
                        rgb_sha256_decoded={split: [r.rgb_sha256 for r in rows] for split, rows in frames.items()})
    dst = run_dir / "fork_region"
    dst.mkdir(exist_ok=True)
    register = dst / "registration.json"
    registered = json_text(registration)
    overwrite = os.environ.get("UAVRGB_OVERWRITE", "0") == "1"
    if register.is_file() and register.read_text() != registered and not overwrite:
        raise ValueError(f"Registered inputs, source, model or environment changed; use a new run: {dst}")
    out = dst / "summary.json"
    if out.is_file() and not overwrite:
        ready = json.loads(out.read_text())
        if not register.is_file() or any(sha(dst / k) != h for k, h in ready["output_sha256"].items()):
            raise ValueError(f"Existing region outputs differ from their recorded hashes: {dst}")
        return ready

    with TemporaryDirectory(dir=dst) as working:
        tmp = Path(working)
        (tmp / register.name).write_text(registered)
        (tmp / register.name).replace(register)
        train, dev = frames["train"], frames["development"]
        encoder = FrozenClip(cfg, device)
        print(f"[fork_region] {device}, {len(train)} train frames, {len(dev)} development frames", flush=True)
        train_x = extract_features(train, encoder, cfg["batch_size"])
        train_y = np.array([label for row in train for label in row.classes])
        fits = {mode: fit_classifier(train_x[mode], train_y, cfg) for mode in MODELS}
        (tmp / "linear_models.json").write_text(json_text(dict(classes=CLASSES, features=FEATURES, fits=fits)))
        dev_x = extract_features(dev, encoder, cfg["batch_size"])
        predictions = {mode: predict(fits[mode], dev_x[mode]).reshape(-1, 2) for mode in MODELS}
        predictions["always_center"] = np.zeros((len(dev), 2), dtype=int)
        scores = {mode: score(dev, p) for mode, p in predictions.items()}
        write_predictions(tmp / "development_predictions.jsonl", dev, predictions)
        (tmp / "metrics.json").write_text(json_text(scores))
        hashes = {register.name: sha(register)}
        for name in ("linear_models.json", "development_predictions.jsonl", "metrics.json"):
            hashes[name] = sha(tmp / name)
            (tmp / name).replace(dst / name)
        result = dict(schema="uavrgb.fork_region.result.v1", path=str(dst), output_sha256=hashes,
                      train_requests=len(train_y), eligible_train_requests=int((train_y >= 0).sum()),
                      development_requests=2 * len(dev), promotion=promotion(scores))
        (tmp / out.name).write_text(json_text(result))
        (tmp / out.name).replace(out)
    return result
