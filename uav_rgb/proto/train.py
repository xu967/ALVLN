#!/usr/bin/env python3
"""PyTorch 原型线的训练配方复刻：照 src/trainer.cpp、trainer_dev.cpp、model_dev.cu、nn_cuda.cu、softmax_cuda.cu 逐项写，
目标是同一份数据、同一个种子下训出与 uavrgb_train 统计上不可分的模型（卡 E2E-P0 步骤三）。

逐项对应（括号里是主线出处）：
  初值     std::mt19937(seed) 的 Xavier 均匀抽样，权重按 c1 c2 c3 fc 的内存顺序、每个元素一次（model.cpp make_cnn）；cpp_rng 与 libstdc++ 逐位对拍
  打乱     另起 std::mt19937(seed)；order=train_idx（meta 里 episode 的顺序、去碰撞），每个 epoch 就地 std::shuffle，引擎持续不重置（trainer.cpp）
  批次     batch=128，末批 8205 % 128 = 13 条不丢，每 epoch 65 步，30 epoch 共 1950 步
  损失     类别权重 auto：loss = Σ w[label]·CE_i / Σ w[label]（CE 用 double 算，float 存），梯度 dlogits = w[label]·(p−onehot)/Σw（softmax_cuda.cu 的两个核）
  裁剪     全局范数：double 累加平方和，coef = min(1, max_norm / (float(norm)+1e-6))，coef<1 才缩放（model_dev.cu backward_step）
  优化器   Adam，β1=0.9f β2=0.999f eps=1e-8f lr=0.001f 常数（warmup 0、无余弦、无衰减、无标签平滑、无增强）；
           bc = float(1 − pow(double(β), t))，p −= lr·((m/bc1)/(sqrt(v/bc2)+eps))，逐元素 float32（nn_cuda.cu adam_step_kernel）
  评估     每个 epoch 末在 val 上按 batch=128 评估：无权 CE、acc、宏 F1、平衡准确率、每类召回、混淆矩阵；宏 F1 严格大于历史 best 才更新 best（trainer.cpp、trainer_eval.cpp）
  留档     每个 epoch 写 pt/epNN.pt（权重 + Adam 状态），last.pt、best.pt；训练结束后 export.py 写成 C++ 的 ckpt_epNN.bin / ckpt.bin / ckpt_best.bin，
           再用 ckpt_best.bin（不是内存里的权重）在 test 上评估一次（2026-11-03 审计的口径）
不复刻的项（配置里没用到）：软标签蒸馏、深度辅助头、文本分支、双帧、prev-action、增强、warmup/余弦、权重衰减>0、续训。

用法（仓库根）：
  CUDA_VISIBLE_DEVICES=<GPU> $D/teacher-venv/bin/python proto/train.py --seed 0 [--out <run 目录>] [--epochs 30] [--device cuda]
  --max-steps N 只给对拍调试：跑到第 N 步就停，不做 epoch 末评估和落盘。
产物（<run>/）：config.json、metrics.jsonl、results.csv（格式同 C++ run 目录）、pt/、ckpt_*.bin、test_eval.jsonl、DONE。
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io as cio  # noqa: E402
import cpp_rng  # noqa: E402
import data as dataio  # noqa: E402
import export as exporter  # noqa: E402
import metrics as M  # noqa: E402
import model as pm  # noqa: E402
import config as cfgio  # noqa: E402

# ───────────────────────── 配置区（每项一行，依据写在行尾）─────────────────────────
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data/uav_rgb")
DATA = DATA_ROOT + "/tmp/agent6-distill/trunc/ds/dataset.uavr"   # 任务卡验收 (a) 指定的数据集（sha256 534ca590…，e1-noise 三个参照种子也用它）
OUT_ROOT = DATA_ROOT + "/exp-proto-parity-20261112/step3-train"  # 任务卡：原型训练出的权重放 exp-proto-parity-20261112/ 下
EPOCHS = 30             # 参照 e1-noise/seed0/run/config.json: epochs=30
BATCH = 128             # 同上: batch=128
LR = 0.001              # 同上: lr=0.00100000005（float32(0.001)）；全程常数：warmup_steps=0、cosine_lr=false
BETA1 = 0.9             # 同上: beta1=0.899999976（float32(0.9)）
BETA2 = 0.999           # 同上: beta2=0.999000013
EPS = 1e-8              # 同上: eps=9.99999994e-09
WEIGHT_DECAY = 0.0      # 同上: weight_decay=0
MAX_GRAD_NORM = 5.0     # 同上: max_grad_norm=5（全局范数裁剪）
DROP_COLLISION = True   # 同上: drop_collision=true，三个 split 都删碰撞步
DISTILL_ALPHA = 0.5     # 只写进 113 B 契约；没有软标签时是占位。参照 ckpt 契约里就是 0.5（TrainConfig 默认）
DEVICE = "cuda"         # 训练在 GPU 上；测试用 cpu
# ──────────────────────────────────────────────────────────

F32 = np.float32


def num(x):
    """同 src/metrics_json.cpp 的 num()：%.9g。"""
    return float("%.9g" % float(x))


# ───────────────────────── 配方里的几个小核（float32，口径同 CUDA 核） ─────────────────────────
def softmax_parts(logits):
    """softmax_cuda.cu 的写法：z = x − 行最大，e = expf(z)，s 按类下标顺序逐个累加（float32）。"""
    z = logits - logits.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    s = e[:, 0]
    for k in range(1, e.shape[1]):
        s = s + e[:, k]
    return z, e, s


def ce_per_sample(z, s, y):
    """−double(z[label]) + log(double(s))，与 softmax_ce_loss_kernel 同一个表达式。"""
    zl = z.gather(1, y[:, None])[:, 0].double()
    return -zl + torch.log(s.double())


def plain_ce(logits, y):
    """无权 CE：double 求和再除 n，最后转 float32（softmax_ce_loss_cuda）。返回 python float（float32 值）。"""
    z, _, s = softmax_parts(logits)
    return float((ce_per_sample(z, s, y).sum() / logits.shape[0]).float())


def weighted_loss_and_grad(logits, y, w):
    """加权 CE 的 loss（float32 值）与 dlogits（float32 [N,C]）。

    loss = Σ w[label]·ce_i / Σ w[label]（double），dlogits = w[label]·(p − onehot) / float(Σ w[label])，p = e / s。
    Σ w[label] 是至多 128 个 float32 值的 double 和：最小量级 2^-25、最大 2^13，位宽 < 53，与求和顺序无关。
    """
    n, c = logits.shape
    z, e, s = softmax_parts(logits)
    ce = ce_per_sample(z, s, y)
    wy = w[y]
    wsum = wy.double().sum()
    loss = ((wy.double() * ce).sum() / wsum).float()
    p = e / s[:, None]
    onehot = F.one_hot(y, c).to(logits.dtype)
    g = (wy[:, None] * (p - onehot)) / wsum.float()
    return float(loss), g


def count_correct(logits, y):
    """行内第一个最大值的下标为预测（平局取小下标，同 count_correct 的 `>` 比较）。"""
    return int((logits.argmax(dim=1) == y).sum())


def clip_global_norm(flat_grad, max_norm):
    """model_dev.cu backward_step：double 累加平方和，coef = min(1, max_norm/(float(norm)+1e-6))，coef<1 才缩放。返回裁剪前的范数。"""
    g64 = flat_grad.double()
    sumsq = float((g64 * g64).sum())
    if not math.isfinite(sumsq):
        raise RuntimeError("梯度含非有限值")
    norm = math.sqrt(sumsq)
    raw = F32(max_norm) / (F32(norm) + F32(1e-6))
    coef = F32(1.0) if raw > F32(1.0) else raw
    if coef < F32(1.0):
        flat_grad.mul_(float(coef))
    return norm


class FlatAdam:
    """nn_cuda.cu adam_step_kernel 的逐元素 float32 复刻；所有参数共用一个步数 t（C++ 每段各计一次，值相同）。"""

    def __init__(self, flat, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, weight_decay=WEIGHT_DECAY):
        self.p = flat
        self.m = torch.zeros_like(flat)
        self.v = torch.zeros_like(flat)
        self.t = 0
        self.lr, self.b1, self.b2, self.eps, self.wd = F32(lr), F32(beta1), F32(beta2), F32(eps), F32(weight_decay)
        self.c1 = F32(1.0) - self.b1      # 1.0f - b1（float32）
        self.c2 = F32(1.0) - self.b2

    def bias_corrections(self, t):
        return (F32(1.0 - math.pow(float(self.b1), t)), F32(1.0 - math.pow(float(self.b2), t)))

    def step(self, g):
        self.t += 1
        bc1, bc2 = self.bias_corrections(self.t)
        f = float   # float32 值转 double 是精确的；torch 的标量乘除在 float32 张量上按 float32 取标量
        m, v = self.m, self.v
        m.mul_(f(self.b1)).add_(g * f(self.c1))               # mi = b1*m + (1-b1)*g
        v.mul_(f(self.b2)).add_((g * f(self.c2)) * g)         # vi = b2*v + ((1-b2)*g)*g
        upd = (m / f(bc1)) / (torch.sqrt(v / f(bc2)) + f(self.eps))
        if self.wd != 0:
            upd = upd + f(self.wd) * self.p
        self.p.sub_(upd * f(self.lr))                         # p -= lr * (...)

    def state_arrays(self, params):
        """（m 列表, v 列表）：按 params 的形状切开，numpy float32。"""
        return split_flat(self.m, params), split_flat(self.v, params)


def split_flat(flat, params):
    out, off = [], 0
    for p in params:
        n = p.numel()
        out.append(flat[off:off + n].detach().cpu().numpy().reshape(tuple(p.shape)).astype(np.float32, copy=True))
        off += n
    return out


def flatten_params(model):
    """把 cpp_params 的各张量搬进同一块连续存储（参数是它的视图），返回 (flat, params)。"""
    params = model.cpp_params()
    flat = torch.cat([p.detach().reshape(-1) for p in params]).contiguous()
    off = 0
    for p in params:
        n = p.numel()
        p.data = flat[off:off + n].view_as(p.data)
        off += n
    return flat, params


# ───────────────────────── 数据 ─────────────────────────
def load_images(ds):
    """整份图像区读进内存（uint8 张量，约 560 MB）；标签 int64。图像留在 CPU，按批搬进显存：显存只放激活，单进程不超过卡规定的 2 GB。"""
    imgs = torch.empty((ds.n, ds.c, ds.h, ds.w), dtype=torch.uint8)
    for a in range(0, ds.n, 1024):
        b = min(ds.n, a + 1024)
        imgs[a:b] = torch.from_numpy(np.array(ds.images[a:b], copy=True))   # memmap 是只读的，先拷成可写数组
    return imgs, torch.from_numpy(ds.labels.astype(np.int64))


@torch.no_grad()
def evaluate(model, images, labels, idx, batch=BATCH, classes=4):
    """trainer_eval.cpp evaluate：按 idx 的顺序、batch 一组，无权 CE（每批 float32 均值 × 批大小，double 累加）、混淆矩阵、宏 F1 等。
    images/labels 在 CPU，按批搬到模型所在的设备。"""
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        raise ValueError("evaluate: 空的下标集")
    model.eval()
    dev = next(model.parameters()).device
    total_ce, preds, truths = 0.0, [], []
    for s in range(0, len(idx), batch):
        part = torch.from_numpy(idx[s:s + batch])
        y = labels[part].to(dev)
        logits = model(images[part].to(dev))
        total_ce += plain_ce(logits, y) * len(part)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        truths.append(y.cpu().numpy())
    conf = M.confusion(np.concatenate(truths), np.concatenate(preds), classes)
    out = M.summarize(conf)
    out["ce"] = total_ce / len(idx)
    out["confusion_matrix"] = conf
    return out


# ───────────────────────── 日志 ─────────────────────────
class RunLog:
    """metrics.jsonl（每步 flush）、results.csv、stdout。事件名与字段同 C++ 的 tinyvln.metrics.v1。"""

    def __init__(self, run_dir):
        self.dir = run_dir
        self.metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        self.csv = open(os.path.join(run_dir, "results.csv"), "a", encoding="utf-8")
        if os.path.getsize(os.path.join(run_dir, "results.csv")) == 0:
            self.csv.write("epoch,train_ce,dev_ce,dev_accuracy\n")
            self.csv.flush()

    def event(self, kind, **fields):
        rec = {"schema": "tinyvln.metrics.v1", "event": kind}
        rec.update(fields)
        self.metrics.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
        self.metrics.flush()

    def status(self, message):
        self.event("status", message=message)

    def csv_row(self, epoch, train_ce, dev_ce, dev_acc):
        self.csv.write(f"{epoch},{train_ce:.9g},{dev_ce:.9g},{dev_acc:.9g}\n")
        self.csv.flush()

    def close(self):
        self.metrics.close()
        self.csv.close()


def git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ───────────────────────── 训练 ─────────────────────────
class Trainer:
    def __init__(self, ds, seed, device, epochs=EPOCHS, batch=BATCH, run_dir=None, images=None, labels=None, config=None):
        self.ds, self.seed, self.device = ds, int(seed), torch.device(device)
        self.epochs, self.batch = int(epochs), int(batch)
        self.run_dir = run_dir
        self.config = config or {}
        train_cfg = self.config.get("train", {})
        self.lr = F32(train_cfg.get("lr", LR))
        self.beta1 = F32(train_cfg.get("beta1", BETA1))
        self.beta2 = F32(train_cfg.get("beta2", BETA2))
        self.eps = F32(train_cfg.get("eps", EPS))
        self.weight_decay = F32(train_cfg.get("weight_decay", WEIGHT_DECAY))
        self.max_grad_norm = F32(train_cfg.get("max_grad_norm", MAX_GRAD_NORM))
        self.drop_collision = bool(train_cfg.get("drop_collision", DROP_COLLISION))
        self.distill_alpha = F32(train_cfg.get("distill_alpha", DISTILL_ALPHA))
        pm.configure_determinism()
        torch.manual_seed(self.seed)
        self.train_idx = ds.splits["train"].astype(np.int64)
        self.val_idx = ds.splits["val"].astype(np.int64)
        self.test_idx = ds.splits["test"].astype(np.int64)
        if len(self.val_idx) == 0:
            raise ValueError("数据集没有 val：C++ 会退回 train 做 dev，原型线不复刻，直接拒绝")
        self.class_weights = ds.auto_class_weights()     # float32 [C]，class_weights.cpp auto
        self.w = torch.from_numpy(self.class_weights).to(self.device)
        self.images, self.labels = (images, labels) if images is not None else load_images(ds)
        self.model = pm.ProtoCnn(in_h=ds.h, in_w=ds.w, in_c=ds.c, num_classes=dataio.NUM_CLASSES)
        self.model.init_xavier(cpp_rng.CppXavier(self.seed))
        self.model.to(self.device)
        self.flat, self.params = flatten_params(self.model)
        self.adam = FlatAdam(self.flat, lr=self.lr, beta1=self.beta1, beta2=self.beta2,
                             eps=self.eps, weight_decay=self.weight_decay)
        self.rng = cpp_rng.CppMt19937(self.seed)         # 打乱用的引擎，与初值那个独立（C++ 里是 make_cnn 里的局部变量）
        self.order = self.train_idx.copy()
        self.per_epoch = (len(self.order) + self.batch - 1) // self.batch
        self.total_steps = self.per_epoch * self.epochs
        scheme_sha = hashlib.sha256(ds.split_scheme.encode()).digest() if ds.split_scheme else None   # C++：scheme 为空则契约里记全 0
        self.contract = cio.make_contract(ds.sha256, self.drop_collision, self.lr, self.seed, self.batch,
                                          scheme_sha, None, self.distill_alpha)
        self.best_f1 = F32(-1.0)
        self.step = 0

    # ---- 一步 ----
    def train_step(self, idx):
        """idx：本批的全局样本下标（CPU 上的 int64 张量）。返回 (loss, acc, plain_ce, 裁剪前范数)。"""
        self.model.train()
        x, y = self.images[idx].to(self.device), self.labels[idx].to(self.device)
        logits = self.model(x)
        with torch.no_grad():
            lg = logits.detach()
            loss, dlogits = weighted_loss_and_grad(lg, y, self.w)
            if not math.isfinite(loss):
                raise RuntimeError(f"loss 非有限值（step {self.step + 1}）")
            acc = F32(count_correct(lg, y)) / F32(len(y))
            plain = plain_ce(lg, y)
        grads = torch.autograd.grad(logits, self.params, grad_outputs=dlogits)
        with torch.no_grad():
            g = torch.cat([t.reshape(-1) for t in grads])
            norm = clip_global_norm(g, self.max_grad_norm)
            self.adam.step(g)
        self.step += 1
        return loss, float(acc), plain, norm

    def epoch_batches(self):
        """打乱 order（就地、引擎持续），按 batch 切开，产出 CPU 上的下标张量。"""
        self.rng.shuffle(self.order)
        order = torch.from_numpy(self.order.copy())
        for s in range(0, len(self.order), self.batch):
            yield order[s:s + self.batch]

    # ---- 留档 ----
    def save_pt(self, name, epoch):
        m, v = self.adam.state_arrays(self.params)
        exporter.save_pt(os.path.join(self.run_dir, "pt", name), structure=self.model.structure(),
                         params=split_flat(self.flat, self.params), adam_m=m, adam_v=v, adam_t=self.adam.t, epoch=epoch,
                         class_weights=self.class_weights, best_macro_f1=float(self.best_f1), contract=self.contract)

    def config_record(self, args, log_extra):
        ds = self.ds
        rec = {
            "schema": "uavrgb.train_config.v1", "backend": "proto-pytorch", "data": ds.path, "epochs": self.epochs,
            "batch": self.batch, "lr": num(self.lr), "seed": self.seed, "text_dim": 0, "vision_only": False,
            "max_grad_norm": num(self.max_grad_norm), "drop_collision": self.drop_collision,
            "class_weights": [num(x) for x in self.class_weights], "class_weights_source": "auto",
            "soft_targets": "", "distill_alpha": num(self.distill_alpha), "dataset_sha256": ds.sha256, "split_scheme": ds.split_scheme,
            "window_from": ds.meta.get("window_from", ""), "hold_s": ds.meta.get("hold_s", ""),
            "beta1": num(self.beta1), "beta2": num(self.beta2), "eps": num(self.eps), "augment": "", "warmup_steps": 0, "cosine_lr": False,
            "weight_decay": num(self.weight_decay), "label_smoothing": 0, "two_frame": False, "prev_action": False, "depth_aux": False,
            "model_profile": self.config.get("model", {}).get("profile", "nano_proto_v6"),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(self.device),
            "proto_git": git_head(os.path.dirname(HERE)), "train_samples": int(len(self.train_idx)),
            "val_samples": int(len(self.val_idx)), "test_samples": int(len(self.test_idx)),
        }
        rec.update(log_extra)
        return rec

    # ---- 主流程 ----
    def run(self, args=None, max_steps=0):
        os.makedirs(os.path.join(self.run_dir, "pt"), exist_ok=True)
        self.config_hash = cfgio.write_resolved(self.run_dir, self.config)
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config_record(args, {}), f, ensure_ascii=False, separators=(",", ":"))
        log = RunLog(self.run_dir)
        log.status(f"start step 0, {self.ds.n} samples ({len(self.train_idx)} train / {len(self.val_idx)} dev), "
                   f"params={sum(p.numel() for p in self.params)}, backend=proto-pytorch")
        t0 = time.time()
        stopped = False
        try:
            for ep in range(1, self.epochs + 1):
                ce_sum, seen = 0.0, 0
                for idx in self.epoch_batches():
                    loss, acc, plain, _ = self.train_step(idx)
                    elapsed = time.time() - t0
                    eta = elapsed / self.step * (self.total_steps - self.step)
                    log.event("train_step", step=self.step, total_steps=self.total_steps, loss=num(loss),
                              action_accuracy=num(acc), lr=num(self.lr), elapsed_sec=num(elapsed), eta_sec=num(eta))
                    ce_sum += plain * len(idx)
                    seen += len(idx)
                    if max_steps and self.step >= max_steps:
                        stopped = True
                        break
                if stopped:
                    log.status(f"stopped by --max-steps={max_steps} (debug): no eval, no checkpoint")
                    break
                train_ce = ce_sum / seen
                dev = evaluate(self.model, self.images, self.labels, self.val_idx, self.batch)
                if not all(math.isfinite(dev[k]) for k in ("ce", "accuracy", "macro_f1", "balanced_accuracy")):
                    raise RuntimeError(f"dev 指标非有限值（epoch {ep}）")
                log.csv_row(ep, train_ce, dev["ce"], dev["accuracy"])
                log.event("validation", step=self.step, loss=num(F32(dev["ce"])), action_accuracy=num(F32(dev["accuracy"])),
                          split="val", macro_f1=num(F32(dev["macro_f1"])), balanced_accuracy=num(F32(dev["balanced_accuracy"])),
                          confusion=dev["confusion"])
                print(f"epoch {ep} dev   ce={dev['ce']:.4f} acc={dev['accuracy']:.4f} macro_f1={dev['macro_f1']:.4f} "
                      f"bal_acc={dev['balanced_accuracy']:.4f} recall=[" + " ".join(f"{r:.3f}" for r in dev["recall"]) + "]",
                      flush=True)
                log.event("class_weights", epoch=ep, step=self.step, class_weights=[num(x) for x in self.class_weights],
                          class_weights_source="auto")
                if dev["macro_f1"] > float(self.best_f1):          # 严格大于；比较在 double 里，best 以 float32 存
                    self.best_f1 = F32(dev["macro_f1"])
                    self.save_pt("best.pt", ep)
                    log.status(f"best checkpoint: epoch {ep} macro_f1={num(dev['macro_f1'])}")
                self.save_pt("last.pt", ep)                         # 在更新 best 之后写：last 里的 best_macro_f1 不落后
                self.save_pt(f"ep{ep:02d}.pt", ep)
            if stopped:
                log.close()
                return {"stopped": True, "steps": self.step}
            # 导出成 C++ 的文件名，再用交付的 ckpt_best.bin 在 test 上评估一次
            exported = exporter.export_run(self.run_dir)
            log.status(f"exported {len(exported)} checkpoints to C++ layout")
            best_bin = os.path.join(self.run_dir, "ckpt_best.bin")
            best_model = pm.model_from_ckpt(cio.read_checkpoint(best_bin), self.device)
            test = evaluate(best_model, self.images, self.labels, self.test_idx, self.batch)
            log.event("validation", step=self.step, loss=num(F32(test["ce"])), action_accuracy=num(F32(test["accuracy"])),
                      split="test", macro_f1=num(F32(test["macro_f1"])), balanced_accuracy=num(F32(test["balanced_accuracy"])),
                      confusion=test["confusion"])
            now = datetime.datetime.now(datetime.timezone.utc)
            with open(os.path.join(self.run_dir, "test_eval.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"schema": "uavrgb.test_eval.v1", "utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "origin": "proto_train_inline", "ckpt": best_bin, "data": self.ds.path,
                                    "dataset_sha256": self.ds.sha256, "drop_collision": self.drop_collision,
                                    "samples": int(len(self.test_idx)), "split": "test", "eval_count": 1,
                                    "macro_f1": test["macro_f1"], "balanced_accuracy": test["balanced_accuracy"],
                                    "recall": test["recall"], "accuracy": test["accuracy"], "confusion": test["confusion"]},
                                   separators=(",", ":")) + "\n")
            elapsed = time.time() - t0
            log.status(f"done: {self.epochs * len(self.train_idx)} samples in {elapsed:.3f} s, ckpt={self.run_dir}/ckpt.bin")
            print(f"test (ckpt_best) macro_f1={test['macro_f1']:.4f} bal_acc={test['balanced_accuracy']:.4f} "
                  "recall=[" + " ".join(f"{r:.3f}" for r in test["recall"]) + f"] elapsed={elapsed:.1f}s", flush=True)
            return {"stopped": False, "steps": self.step, "test": test, "elapsed": elapsed}
        finally:
            log.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=None, help="训练种子（初值与打乱共用，同 uavrgb_train --seed）")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None, help=f"run 目录，缺省 {OUT_ROOT}/seed<种子>/run；必须是空目录或不存在")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="调试：跑到第 N 步就停，不做评估和落盘")
    ap.add_argument("--config", default=None, help="YAML/JSON 配置；显式 CLI 参数覆盖配置")
    a = ap.parse_args(argv)
    if not a.config and a.seed is None:
        ap.error("旧 CLI 模式必须提供 --seed；配置模式可在 train.seed 中提供")
    raw = cfgio.load(a.config)
    # 旧 CLI 的默认值不能覆盖配置文件；只有显式传入的值才应覆盖。
    cli = {"seed": a.seed, "data": a.data,
           "out": a.out, "epochs": a.epochs,
           "batch": a.batch, "device": a.device}
    cfg = cfgio.resolve(raw, cli=cli)
    if cfg["model"]["profile"] != "nano_proto_v6":
        raise SystemExit("base_torch_v1 已注册但当前 C++ 对拍训练器不支持；请使用独立 Torch runner")
    train_cfg = cfg["train"]
    seed = int(train_cfg.get("seed", a.seed))
    data_path = train_cfg.get("data") or a.data or DATA
    device = train_cfg.get("device") or a.device or DEVICE
    out = train_cfg.get("out") or a.out or os.path.join(OUT_ROOT, f"seed{seed}", "run")
    os.makedirs(out, exist_ok=True)
    lock = open(os.path.join(out, ".run.lock"), "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"{out} 正被另一个训练进程使用", file=sys.stderr)
        return 2
    if [n for n in os.listdir(out) if n != ".run.lock"]:
        print(f"{out} 不是空目录，拒绝混写（旧结果改名挪走再训）", file=sys.stderr)
        return 2
    ds = dataio.Uavr(data_path, drop_collision=bool(train_cfg.get("drop_collision", DROP_COLLISION)))
    input_cfg = cfg["model"]["input"]
    if (int(input_cfg["height"]), int(input_cfg["width"])) != (ds.h, ds.w):
        raise SystemExit(f"配置输入 {input_cfg['height']}x{input_cfg['width']} 与 UAVR {ds.h}x{ds.w} 不一致")
    tr = Trainer(ds, seed, device, epochs=int(train_cfg.get("epochs", EPOCHS)),
                 batch=int(train_cfg.get("batch", BATCH)), run_dir=out, config=cfg)
    res = tr.run(a, max_steps=a.max_steps)
    if not res["stopped"]:
        with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as f:
            f.write(f"rc=0 {datetime.datetime.now():%F %T} seed={seed} steps={res['steps']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
