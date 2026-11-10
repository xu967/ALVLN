#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baseline_infer.py：外部基线适配器。说与 uavrgb_infer 同一套行协议，把公开模型放进我们的闭环评测。

不改评测代码：
    eval_closed_loop.py --policy student --ckpt <任意存在的文件> --infer-bin <sim/tools/baselines/launch_*.sh> ...
`--ckpt` 只被检查「文件存在」与记 sha256，内容不读（生成方式见 README.md）。

协议（sim/tools/student.py 的 StudentPolicy、src/infer_main.cpp、src/infer.cpp）：
    启动：  <bin> --ckpt <file> [--text id,id,...]
    握手：  stdout 先写一行 hello JSON：{"in_h","in_w","in_c","num_classes":4,"text_dim","prev_action_dim","aux_classes"}
    每步：  stdin 读 in_h*in_w*3*(in_c/3) 字节（HWC uint8 RGB，当前帧在前）+ prev_action_dim>0 时 1 字节；
            stdout 写一行 {"action":0..3,"probs":[4 个浮点],"infer_ms":浮点}（额外字段被 StudentPolicy 忽略，这里多带 raw）
    结束：  stdin EOF 时退出码 0（StudentPolicy.close() 对非 0 退出码抛错）
运行：BASELINE=forward python baseline_infer.py --ckpt /任意存在的文件   （实际用 launch_*.sh，环境变量写死在启动器里）

动作：0 SAFE_NEUTRAL，1 FORWARD_SLOW，2 YAW_LEFT（偏航角为正，逆时针），3 YAW_RIGHT。
"""
import importlib.util
import json
import math
import os
import sys
import time
import types

sys.dont_write_bytecode = True  # 根分区只剩 2.4 GB，不落 __pycache__

# ------------------------------ 配置区（环境变量可覆盖；启动器里写死各基线的取值） ------------------------------
BASELINE = os.environ.get("BASELINE", "")                       # forward | pulp_v3 | pulp_tiny | dronet；没有默认：漏设就报错，不静默跑成别的基线
BASE = os.environ.get("BASELINE_ROOT", "/root/rivermind-data/uav_rgb/baselines")  # 第三方代码与权重留在数据盘，仓库只放适配代码
DEVICE = os.environ.get("BASELINE_DEVICE", "cpu")               # 第一档网络 <1.5 MB、41 M MAC：CPU 毫秒级，不占 GPU 配额
THREADS = int(os.environ.get("BASELINE_THREADS", "1"))          # torch 线程数：评测时同机有 7 个 gz 软件渲染，1 个线程足够 1 Hz 决策（单步约 10–30 ms）
PREP = os.environ.get("BASELINE_PREP", "native")                # 灰度 200x200 取法：native(各自原文的裁法) | resize_crop | squash
WEIGHTS = os.environ.get("BASELINE_WEIGHTS", "")                # 同架构「在我们数据上重训」臂的 .pth；空 = 官方公开权重（零样本臂）
YAW_STEP_DEG = float(os.environ.get("YAW_STEP_DEG", "25"))      # 一个 YAW 动作的名义转角：公共任务书 §1 写 23–27°，取中值
HOLD_S = float(os.environ.get("HOLD_S", "1.0"))                 # 每个动作保持 1.0 s（公共任务书 §1）
STEER_SIGN = float(os.environ.get("STEER_SIGN", "1"))           # +1：steer>0 = 左转（PULP README：YawRate>0 逆时针即左）；各基线的实测符号写在各自启动器里，见 README「符号」
P_CRIT = float(os.environ.get("P_CRIT", "0.7"))                 # 碰撞概率刹停阈值：PULP config_dronet.h CRITICAL_PROB_COLL 0.7，DroNet deep_navigation.cpp critical_prob 0.7
YAW_SCALE_PULP = float(os.environ.get("YAW_SCALE_PULP", "90"))  # steer=1 对应的偏航角速度 °/s：PULP config_dronet.h L45 YAW_SCALING 90（另两组配置 120）
YAW_SCALE_DRONET = float(os.environ.get("YAW_SCALE_DRONET", "20"))  # DroNet Bebop outdoor.yaml L31 SpeedSettingsMaxRotationSpeedCurrent 20 °/s
FWD_STEP_M = float(os.environ.get("FWD_STEP_M", "0.25"))        # 一个 FORWARD_SLOW 动作的名义位移：公共任务书 §1 写约 0.25 m；只有连续速度映射用
BASELINE_LOG = os.environ.get("BASELINE_LOG", "")               # 非空：每步追加一行原始输出 JSONL（steer / p_coll），事后对账映射
# --------------------------------------------------------------------------------------------------------------

SAFE, FWD, LEFT, RIGHT = 0, 1, 2, 3


# ------------------------------ 输出 → 4 个动作的映射（纯函数，参数显式传入，测试直接调） ------------------------------
def map_steer_coll(steer, p_coll, yaw_scale_deg_s, sign=1.0, p_crit=0.7, yaw_step_deg=25.0, hold_s=1.0):
    """转向量 + 碰撞概率 → 离散动作。

    转：本步想转的角度 = sign*steer*yaw_scale*hold_s；不小于半个 YAW 步长就转。半步长是「转 0 步还是转 1 步」的最近邻分界。
    进：碰撞概率不超过 p_crit 才前进（原文控制律 v=(1-p)*vmax，p>0.7 时紧急刹停）；否则 SAFE。
    转与进冲突时先转：离散动作不能边转边进，朝向错了前进只会更近障碍。
    steer 或 p_coll 不是有限数（网络输出坏了）一律 SAFE：不动，不乱飞。
    """
    if not (math.isfinite(steer) and math.isfinite(p_coll)):
        return SAFE
    yaw_cmd = sign * steer * yaw_scale_deg_s * hold_s
    if abs(yaw_cmd) >= 0.5 * yaw_step_deg:
        return LEFT if yaw_cmd > 0 else RIGHT
    return FWD if p_coll <= p_crit else SAFE


def map_velocity(vx_mps, yaw_rate_deg_s, yaw_step_deg=25.0, fwd_step_m=0.25, hold_s=1.0):
    """连续速度（前向 m/s、偏航 °/s）→ 动作：把 hold_s 内的位移与转角量化到最近的原语（0.25 m / 25°）。

    shortcut: 目前没有基线调用它（DiffPhysDrone、NavRL 一类输出连续速度的基线还没接入）；接入时再核对符号约定与门槛。
    """
    dyaw, dx = yaw_rate_deg_s * hold_s, vx_mps * hold_s
    if abs(dyaw) >= 0.5 * yaw_step_deg:
        return LEFT if dyaw > 0 else RIGHT
    return FWD if dx >= 0.5 * fwd_step_m else SAFE


# ------------------------------ 各基线适配器 ------------------------------
def _import_torch():
    import torch
    if THREADS > 0:
        torch.set_num_threads(THREADS)
    return torch


class Adapter:
    in_h, in_w, in_c, prev_dim = 240, 320, 3, 0   # hello 声明原生 320x240：StudentPolicy 的 PIL resize 同尺寸即直通，各基线在 step() 里自己灰度/裁剪/缩放

    def step(self, frames, prev_action):          # frames: [当前帧(, 上一帧)]，每个 uint8 [H,W,3]；返回 (动作, raw 字典)
        raise NotImplementedError


class Forward(Adapter):
    """盲前进：协议自检用，确认包装本身没有改变闭环结果（应与 --policy forward 逐步一致）。"""

    def step(self, frames, prev_action):
        return FWD, {}


def _gray(frame, mode, bottom_aligned=False):
    """RGB uint8 [240,320,3] → 灰度 float32 [200,200]，0..1。"""
    from PIL import Image
    import numpy as np
    im = Image.fromarray(frame).convert("L")      # ITU-R 601 亮度，与 PIL/cv2 灰度一致
    w, h = im.size
    if mode == "native":                          # 原文流水线：QVGA 中心裁 200x200（DroNet 取底部居中）；视场比原相机窄，这是零样本的固有偏差
        x0, y0 = (w - 200) // 2, (h - 200 if bottom_aligned else (h - 200) // 2)
        im = im.crop((x0, y0, x0 + 200, y0 + 200))
    elif mode == "resize_crop":                   # 高缩到 200 再中心裁宽：保留完整竖直视场
        im = im.resize((round(w * 200 / h), 200), Image.Resampling.BILINEAR)
        x0 = (im.size[0] - 200) // 2
        im = im.crop((x0, 0, x0 + 200, 200))
    elif mode == "squash":                        # 整幅压成方形：保留完整水平视场，宽高比失真
        im = im.resize((200, 200), Image.Resampling.BILINEAR)
    else:
        raise ValueError(f"BASELINE_PREP={mode}")
    return np.asarray(im, dtype=np.float32) / 255.0


class PulpDronetV3(Adapter):
    """pulp-platform/pulp-dronet 的 tiny-pulp-dronet-v3：灰度 200x200 → (steer∈[-1,1], p_coll)。"""

    def __init__(self, variant):
        torch = self.torch = _import_torch()
        root = f"{BASE}/pulp-dronet/src/tiny-pulp-dronet-v3"
        # 第三方模型文件顶部 import nemo（量化库）；FP 前向里 PACT_IntegerAdd 等价于 a+b，用桩替代，不改第三方源码，也不装包。
        nemo, quant, pact = (types.ModuleType(n) for n in ("nemo", "nemo.quant", "nemo.quant.pact"))

        class PACT_IntegerAdd(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        pact.PACT_IntegerAdd, quant.pact, nemo.quant = PACT_IntegerAdd, pact, quant
        sys.modules.update({"nemo": nemo, "nemo.quant": quant, "nemo.quant.pact": pact})
        spec = importlib.util.spec_from_file_location("dronet_v3", f"{root}/model/dronet_v3.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if variant == "v3":      # README：ResBlock、depth_mult=1.0、bypass=True，320,928 参数
            net, path = mod.dronet(depth_mult=1.0, block_class=mod.ResBlock, bypass=True), "pulp-dronet-v3-resblock-1.0.pth"
        elif variant == "tiny":  # README：Depthwise、depth_mult=0.125、bypass=False，2,784 参数
            net, path = mod.dronet(depth_mult=0.125, block_class=mod.Depthwise_Separable, bypass=False), "tiny-pulp-dronet-v3-dw-pw-0.125.pth"
        else:
            raise ValueError(f"PULP variant={variant}")
        ck = torch.load(WEIGHTS or f"{root}/model/{path}", map_location="cpu")
        net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        self.net = net.eval().to(DEVICE)
        self.n_params = sum(p.numel() for p in net.parameters())

    def step(self, frames, prev_action):
        torch = self.torch
        x = torch.from_numpy(_gray(frames[0], PREP))[None, None].to(DEVICE)
        with torch.no_grad():
            steer, coll = self.net(x)
        s, p = float(steer[0]), float(coll[0])
        return map_steer_coll(s, p, YAW_SCALE_PULP, STEER_SIGN, P_CRIT, YAW_STEP_DEG, HOLD_S), {"steer": s, "p_coll": p}


class DroNet(Adapter):
    """uzh-rpg/rpg_public_dronet 的 ResNet-8：Keras 权重（h5 已转 npz）用 torch 算子重写前向；灰度 200x200 → (steer, p_coll)。

    shortcut: 本机没有 TF 1.x，无法与 Keras 原模型逐位对拍；独立复核（逐层 NumPy 解释器）见 experiments/2026-11-10-baselines-zeroshot/offline_check.py。
    """

    def __init__(self):
        import numpy as np
        torch = self.torch = _import_torch()
        z = np.load(f"{BASE}/dronet/dronet_best_weights.npz")
        self.w = {k.split("|")[0] + "/" + k.split("|")[-1].split("/")[-1].replace(":0", ""): torch.from_numpy(z[k]).to(DEVICE)
                  for k in z.files}
        self.eps = 1e-3   # model_struct.json 里 BN epsilon=0.001

    @staticmethod
    def _same(x, k, s):   # TF/Keras 'same'：多出的一格补在右/下，和 torch 的对称 padding 不同
        import torch.nn.functional as F

        def tot(n):
            return max((-(-n // s) - 1) * s + k - n, 0)
        ph, pw = tot(x.shape[-2]), tot(x.shape[-1])
        return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))

    def _conv(self, x, name, k, s):
        import torch.nn.functional as F
        w = self.w[f"{name}/kernel"].permute(3, 2, 0, 1).contiguous()    # Keras [kh,kw,in,out] → torch [out,in,kh,kw]
        return F.conv2d(self._same(x, k, s), w, self.w[f"{name}/bias"], stride=s)

    def _bn(self, x, name):
        g, b = self.w[f"{name}/gamma"], self.w[f"{name}/beta"]
        m, v = self.w[f"{name}/moving_mean"], self.w[f"{name}/moving_variance"]
        sh = (1, -1, 1, 1)
        return (x - m.view(sh)) / (v.view(sh) + self.eps).sqrt() * g.view(sh) + b.view(sh)

    def forward_layers(self, x):
        """x: float [1,1,200,200] → 字典：第一层卷积输出 conv2d_1、最终 steer、碰撞 logit（供独立复核逐层对账）。"""
        torch = self.torch
        relu = torch.relu
        out = {}
        with torch.no_grad():
            x = self._conv(x, "conv2d_1", 5, 2)
            out["conv2d_1"] = x
            x1 = torch.nn.functional.max_pool2d(x, 3, 2)
            for bn_a, c_a, bn_b, c_b, sc in (("batch_normalization_1", "conv2d_2", "batch_normalization_2", "conv2d_3", "conv2d_4"),
                                              ("batch_normalization_3", "conv2d_5", "batch_normalization_4", "conv2d_6", "conv2d_7"),
                                              ("batch_normalization_5", "conv2d_8", "batch_normalization_6", "conv2d_9", "conv2d_10")):
                x2 = self._conv(relu(self._bn(x1, bn_a)), c_a, 3, 2)
                x2 = self._conv(relu(self._bn(x2, bn_b)), c_b, 3, 1)
                x1 = self._conv(x1, sc, 1, 2) + x2
                out[sc] = x1                                             # 每个残差块的和（add_1/2/3）
            f = relu(x1.permute(0, 2, 3, 1).reshape(1, -1))              # Keras Flatten 是 NHWC 顺序
            out["steer"] = (f @ self.w["dense_1/kernel"] + self.w["dense_1/bias"]).item()
            out["coll_logit"] = (f @ self.w["dense_2/kernel"] + self.w["dense_2/bias"]).item()
        return out

    def step(self, frames, prev_action):
        torch = self.torch
        x = torch.from_numpy(_gray(frames[0], PREP, bottom_aligned=(PREP == "native")))[None, None].to(DEVICE)
        o = self.forward_layers(x)
        steer, coll = o["steer"], float(torch.sigmoid(torch.tensor(o["coll_logit"])))
        s = max(-1.0, min(1.0, steer))                                   # 原控制节点对 steer 截断到 [-1,1]
        return map_steer_coll(s, coll, YAW_SCALE_DRONET, STEER_SIGN, P_CRIT, YAW_STEP_DEG, HOLD_S), {"steer": steer, "p_coll": coll}


REGISTRY = {"forward": Forward, "pulp_v3": lambda: PulpDronetV3("v3"), "pulp_tiny": lambda: PulpDronetV3("tiny"), "dronet": DroNet}


# ------------------------------ 协议循环 ------------------------------
def main():
    args = sys.argv[1:]
    ckpt = None
    while args:
        k = args.pop(0)
        if k == "--ckpt" and args:
            ckpt = args.pop(0)
        elif k == "--text" and args:
            args.pop(0)            # 指令 token id：像素基线忽略
        else:
            sys.stderr.write("用法: baseline_infer.py --ckpt <file> [--text id,id,...]\n")
            return 2
    if not ckpt:
        sys.stderr.write("缺 --ckpt\n")
        return 2
    if BASELINE not in REGISTRY:
        sys.stderr.write(f"baseline_infer: 环境变量 BASELINE={BASELINE!r} 无效，可选 {sorted(REGISTRY)}\n")
        return 2
    # stdout 只许协议行：保留真 stdout 的副本，再把 fd 1 指向 stderr，第三方库的 print 不会污染协议。
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    try:
        ad = REGISTRY[BASELINE]()
    except Exception as e:  # noqa: BLE001 - 构造失败要让调用方从 stderr 看出原因
        sys.stderr.write(f"baseline_infer: 构造 {BASELINE} 失败: {type(e).__name__}: {e}\n")
        return 1
    hello = {"in_h": ad.in_h, "in_w": ad.in_w, "in_c": ad.in_c, "num_classes": 4, "text_dim": 0,
             "prev_action_dim": ad.prev_dim, "aux_classes": 0}
    out.write((json.dumps(hello) + "\n").encode())
    import numpy as np
    one = ad.in_h * ad.in_w * 3
    payload = one * (ad.in_c // 3) + (1 if ad.prev_dim else 0)
    log = open(BASELINE_LOG, "a") if BASELINE_LOG else None
    n = 0
    stdin = sys.stdin.buffer
    while True:
        buf = stdin.read(payload)      # BufferedReader 在管道上读满 payload 或遇 EOF
        if not buf:
            break                      # 干净 EOF
        if len(buf) != payload:
            sys.stderr.write("baseline_infer: 帧数据不足（截断）\n")
            return 1
        frames = [np.frombuffer(buf, np.uint8, one, one * i).reshape(ad.in_h, ad.in_w, 3) for i in range(ad.in_c // 3)]
        prev = buf[-1] if ad.prev_dim else 0
        t0 = time.perf_counter()
        a, raw = ad.step(frames, prev)
        ms = (time.perf_counter() - t0) * 1e3
        probs = [1.0 if i == a else 0.0 for i in range(4)]      # 基线没有四类概率，one-hot；原始输出在 raw
        out.write((json.dumps({"action": int(a), "probs": probs, "infer_ms": ms, "raw": raw}) + "\n").encode())
        if log:
            # pid：每集一个 collect 进程 → 一个本进程；同一臂的日志里靠 pid 切开各集，n 是本进程内的帧序号（0 = 预热帧）
            log.write(json.dumps({"pid": os.getpid(), "n": n, "action": int(a), "ms": ms, **raw}) + "\n")
            log.flush()
        n += 1
    sys.stderr.write(f"baseline_infer[{BASELINE}]: {n} 帧\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
