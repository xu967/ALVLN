#!/usr/bin/env python3
"""gsplat 渲染器：加载 3DGS .ply，按 nav 位姿渲染 RGB + 期望深度。

真实后端要求 `$DATA_ROOT/uav_rgb/3dgs-venv`（torch 2.7.1+cu128, gsplat 1.5.3）；
torch/gsplat 只在 GsplatRenderer 内部延迟导入，mock/单测不需要它们。
gsplat 的 depth 是 expected depth（sum(w z)/sum(w)，米），alpha 过低的像素置 0
表示无效（与 Gazebo depth 的 0=无回波一致）。

CLI：
  python -m sim.3dgs.renderer --scene .../scene.json --pose 0.5,0,2,0 \
      --width 320 --height 240 --fov 60 --out /tmp/rgb.png --depth-out /tmp/d.npy --bench 50
  python -m sim.3dgs.renderer --mock --pose 0,0,2,0 --out /tmp/mock.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import scene as scene_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(REPO_ROOT, "sim", "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _save_rgb(path, rgb):
    import save_frame
    h, w, nch = rgb.shape
    if nch != 3:
        raise ValueError(f"RGB 必须是 3 通道，实际 {nch}")
    with open(path, "wb") as f:
        f.write(save_frame._png(w, h, nch, np.ascontiguousarray(rgb).tobytes()))


class GsplatRenderer:
    """一次性把高斯放到 GPU；每帧只传 viewmat/K。"""

    backend = "gsplat"

    def __init__(self, scene, width=None, height=None, fov_deg=None,
                 device="cuda", sh_degree=0, background=(0.7, 0.75, 0.8)):
        try:
            import torch
            from gsplat import rasterization
        except Exception as e:  # noqa: BLE001 - 安装问题要原样报给用户
            raise RuntimeError(f"gsplat/torch 导入失败（请用 3dgs-venv）: {e}") from e
        self.torch = torch
        self.rasterization = rasterization
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；检查 3dgs-venv 与驱动")
        cam = scene.get("camera", {})
        self.width = int(width or cam.get("width", 320))
        self.height = int(height or cam.get("height", 240))
        self.fov_deg = float(fov_deg or cam.get("fov_deg", 60.0))
        self.sh_degree = int(sh_degree)
        self.nav_to_world = scene.get("_nav_to_world")
        # 不用 gsplat 的 backgrounds：1.5.3 在 packed=True 时要求 (channels,)，
        # 与高层 [C,3] 的约定冲突。改为拿 alpha 后自己合成背景。
        self._bg = torch.tensor(background, dtype=torch.float32,
                                device=self.device).view(1, 1, 3)

        g = scene_mod.read_gaussian_ply(scene["ply_path"])
        self.n_gaussians = int(len(g["means"]))
        t = torch.from_numpy
        self.means = t(g["means"]).to(self.device).contiguous()
        self.quats = t(g["quats"]).to(self.device).contiguous()
        self.scales = t(g["scales"]).to(self.device).contiguous()
        self.opacities = t(g["opacities"]).to(self.device).contiguous()
        # sh_degree=None -> colors 直接是 RGB [N,3]，避免为 pilot 读 45 维 SH。
        self.colors = t(g["colors"]).to(self.device).contiguous()
        self.scene = scene
        self.samples = 0
        self.total_ms = 0.0

    def render_pose(self, x, y, z, yaw_deg, pitch_deg=0.0, width=None, height=None):
        """nav 位姿 -> (rgb uint8 HxWx3, depth float32 HxW[米,0=无效], 耗时 ms)。"""
        width = int(width or self.width)
        height = int(height or self.height)
        view = scene_mod.viewmat_from_pose(x, y, z, yaw_deg, pitch_deg,
                                           self.nav_to_world)
        k = scene_mod.intrinsics(width, height, self.fov_deg)
        torch = self.torch
        vm = torch.from_numpy(view).float().to(self.device)[None]
        ks = torch.from_numpy(k).float().to(self.device)[None]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            colors, alphas, _meta = self.rasterization(
                self.means, self.quats, self.scales, self.opacities, self.colors,
                vm, ks, width, height,
                sh_degree=None, render_mode="RGB+ED",
                backgrounds=None, packed=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        out = colors[0].detach()
        alpha = alphas[0].detach().float()
        # 手动背景合成：accumulated RGB 已是预乘 alpha。
        rgb_t = out[..., :3] + self._bg * (1.0 - alpha)
        rgb = (rgb_t.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        depth = out[..., 3].float().cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        depth = np.where(alpha_np[..., 0] >= 0.05, depth, 0.0).astype(np.float32)
        self.samples += 1
        self.total_ms += ms
        return rgb, depth, ms

    def stats(self):
        return {"backend": self.backend, "n_gaussians": self.n_gaussians,
                "width": self.width, "height": self.height, "fov_deg": self.fov_deg,
                "sh_degree": self.sh_degree, "device": str(self.device),
                "samples": self.samples,
                "mean_ms": self.total_ms / self.samples if self.samples else None}


class MockRenderer:
    """显式 mock：固定渐变色 + 常量深度；只用于单测/离线冒烟，不产生真数据。"""

    backend = "mock"

    def __init__(self, scene=None, width=None, height=None, fov_deg=None,
                 depth_m=5.0):
        cam = (scene or {}).get("camera", {})
        self.width = int(width or cam.get("width", 320))
        self.height = int(height or cam.get("height", 240))
        self.fov_deg = float(fov_deg or cam.get("fov_deg", 60.0))
        self.depth_m = float(depth_m)
        self.samples = 0
        self.total_ms = 0.0

    def render_pose(self, x, y, z, yaw_deg, pitch_deg=0.0, width=None, height=None):
        w = int(width or self.width)
        h = int(height or self.height)
        t0 = time.perf_counter()
        # 位姿驱动一个可区分的渐变，避免所有帧完全一样。
        yy, xx = np.mgrid[0:h, 0:w]
        phase = (x * 7.0 + y * 3.0 + yaw_deg) % 360.0
        rgb = np.stack([
            (xx / max(w - 1, 1) * 255 * 0.5 + phase) % 256,
            (yy / max(h - 1, 1) * 255 * 0.5 + z * 10) % 256,
            np.full_like(xx, 96, dtype=np.float64)], axis=-1).astype(np.uint8)
        depth = np.full((h, w), self.depth_m, dtype=np.float32)
        ms = (time.perf_counter() - t0) * 1e3
        self.samples += 1
        self.total_ms += ms
        return rgb, depth, ms

    def stats(self):
        return {"backend": self.backend, "width": self.width, "height": self.height,
                "fov_deg": self.fov_deg, "samples": self.samples,
                "mean_ms": self.total_ms / self.samples if self.samples else None}


def make_renderer(scene, mock=False, width=None, height=None, fov_deg=None,
                  device="cuda"):
    """显式选择后端；真实后端不可用时直接报错，不静默回退 mock。"""
    if mock:
        return MockRenderer(scene, width, height, fov_deg)
    return GsplatRenderer(scene, width, height, fov_deg, device=device)


def parse_pose(text):
    """'x,y,z,yaw' 或 'x,y,z,yaw,pitch' -> tuple[float]。"""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) not in (4, 5):
        raise ValueError(f"--pose 需要 x,y,z,yaw[,pitch]，实际 {text!r}")
    vals = tuple(float(p) for p in parts)
    if not all(np.isfinite(vals)):
        raise ValueError(f"--pose 含非有限值: {text!r}")
    return vals


def bench(renderer, pose, n):
    """预热 2 帧后测 n 帧；返回单帧耗时分位数与 FPS（含 depth）。"""
    for _ in range(2):
        renderer.render_pose(*pose)
    times = []
    for _ in range(max(1, int(n))):
        _, _, ms = renderer.render_pose(*pose)
        times.append(ms)
    times.sort()
    k = len(times)
    return {"frames": k, "mean_ms": sum(times) / k,
            "p50_ms": times[k // 2], "p95_ms": times[min(k - 1, int(0.95 * k))],
            "fps_mean": 1000.0 / (sum(times) / k)}


def main():
    p = argparse.ArgumentParser(description="3DGS 单帧渲染（gsplat）")
    p.add_argument("--scene", default=None, help="scene.json")
    p.add_argument("--ply", default=None, help="直接给 .ply（无 scene.json 时）")
    p.add_argument("--pose", required=True, help="x,y,z,yaw[,pitch]（nav 世界系）")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fov", type=float, default=None, help="水平 FOV 度")
    p.add_argument("--out", required=True, help="RGB PNG")
    p.add_argument("--depth-out", default=None, help="深度 .npy（米，0=无效）")
    p.add_argument("--bench", type=int, default=0, help="额外计时帧数")
    p.add_argument("--mock", action="store_true", help="显式 mock 后端（仅冒烟）")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.scene:
        scene = scene_mod.load_scene(args.scene)
    elif args.ply:
        scene = {"ply_path": os.path.abspath(args.ply), "camera": {},
                 "_nav_to_world": np.eye(4)}
    else:
        p.error("--scene 或 --ply 必填")
    pose = parse_pose(args.pose)
    renderer = make_renderer(scene, args.mock, args.width, args.height, args.fov,
                             args.device)
    rgb, depth, ms = renderer.render_pose(*pose)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    _save_rgb(args.out, rgb)
    if args.depth_out:
        np.save(args.depth_out, depth)
    stats = {"first_frame_ms": ms, **renderer.stats()}
    if args.bench:
        stats["bench"] = bench(renderer, pose, args.bench)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
