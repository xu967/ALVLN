#!/usr/bin/env python3
"""OpenFly v2 六场景因果扩量 + 预训练 BERT + 拐点/STOP 加权 + 因果 prev_action 实验流水线。

运行方式（守护进程）:
  OUT=/root/rivermind-data/uav_rgb/exp-openfly-v2-20261116; mkdir -p "$OUT"
  setsid nohup /root/rivermind-data/uav_rgb/envs/airvln/bin/python \
      experiments/2026-11-16-openfly-v2/run.py > "$OUT/run.log" 2>&1 < /dev/null & echo $! > "$OUT/run.pid"
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

# ========================= 顶部集中配置区 =========================
REPO_ROOT = Path("/root/claude-code-backup-20261028/code/uav_rgb")
PYTHON_BIN = Path(os.environ.get("OPENFLY_PYTHON", "/root/rivermind-data/uav_rgb/envs/airvln/bin/python"))
AIRVLN_ROOT = Path("/root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN")
SUBSET_DIR = Path(os.environ.get(
    "OPENFLY_SUBSET_DIR",
    "/root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/six_scene_causal_bc_v1",
))  # 6 场景 100 条因果扩量目录
OUT_DIR = Path(os.environ.get(
    "OPENFLY_EXP_OUT",
    "/root/rivermind-data/uav_rgb/exp-openfly-v2-20261116",
))  # 实验产物与 DONE 标记输出目录
TORCH_HOME = "/root/rivermind-data/uav_rgb/torch-cache"  # ResNet50 预训练缓存目录（峰值显存 < 2 GiB）
HF_HOME = "/root/rivermind-data/uav_rgb/hf-cache"        # bert-base-uncased 离线缓存目录
TMP_DIR = "/root/rivermind-data/uav_rgb/tmp"             # 临时目录放数据盘，绝不写 /root/rivermind-fs
EPOCHS = int(os.environ.get("OPENFLY_EPOCHS", "15"))     # 80 train routes × 15 epochs = 1200 updates（预算可比）
SEED = int(os.environ.get("OPENFLY_SEED", "20261116"))   # 固定随机种子
STOP_WEIGHT = float(os.environ.get("OPENFLY_STOP_WEIGHT", "3.0"))       # 强化末步 STOP 召回与精度
PREV_DROPOUT = float(os.environ.get("OPENFLY_PREV_DROPOUT", "0.2"))     # 打破二周期原地摆头吸引子
# =================================================================


def run_cmd(cmd, env):
    print(f"[CMD] {' '.join(str(x) for x in cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"命令失败 rc={res.returncode}: {cmd}")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "TORCH_HOME": TORCH_HOME,
        "HF_HOME": HF_HOME,
        "TRANSFORMERS_CACHE": f"{HF_HOME}/hub",
        "TMPDIR": TMP_DIR,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    })

    manifest_path = SUBSET_DIR / "manifest.json"
    audit_path = SUBSET_DIR / "spatial_audit.json"
    if not (manifest_path.is_file() and audit_path.is_file()):
        print("[STEP 1] 拉取并校验 6 场景 100 条因果组首图与 3D 空间簇划分...", flush=True)
        run_cmd([str(PYTHON_BIN), "tools/prepare_openfly_expansion.py"], env)
    else:
        print(f"[STEP 1] 复用已校验清单 {manifest_path}", flush=True)

    train_root = OUT_DIR / "training"
    train_root.mkdir(parents=True, exist_ok=True)
    arms = [
        ("v1_cma_rgb_language_6scene", [
            "--policy-type", "cma", "--training-mode", "rgb_language",
            "--contract-version", "v1",
        ]),
        ("v2_cma_rgb_only", [
            "--policy-type", "cma", "--training-mode", "rgb_only",
            "--imagenet-norm", "--inflection-weight", "auto",
            "--stop-weight", str(STOP_WEIGHT), "--progress-monitor",
            "--use-prev-action", "--prev-action-dropout", str(PREV_DROPOUT),
        ]),
        ("v2_cma_rgb_language", [
            "--policy-type", "cma", "--training-mode", "rgb_language",
            "--imagenet-norm", "--use-bert", "--inflection-weight", "auto",
            "--stop-weight", str(STOP_WEIGHT), "--progress-monitor",
            "--use-prev-action", "--prev-action-dropout", str(PREV_DROPOUT),
        ]),
    ]

    arm_results = {}
    for arm_name, extra_flags in arms:
        arm_dir = train_root / arm_name
        done_file = arm_dir / "DONE"
        if not done_file.is_file():
            print(f"[STEP 2] 训练臂 {arm_name}...", flush=True)
            cmd = [
                str(PYTHON_BIN), "tools/train_openfly_subset.py",
                "--manifest", str(manifest_path),
                "--airvln-root", str(AIRVLN_ROOT),
                "--out", str(arm_dir),
                "--epochs", str(EPOCHS),
                "--seed", str(SEED),
                *extra_flags,
            ]
            run_cmd(cmd, env)
        arm_results[arm_name] = json.loads((arm_dir / "result.json").read_text())

    print("[STEP 3] 执行 20 条空间隔离验证集在线/离线逐帧 sequence_parity 校验...", flush=True)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    os.environ.update({"TORCH_HOME": TORCH_HOME, "HF_HOME": HF_HOME, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    import eval_openfly_subset as eval_mod
    import train_openfly_subset as train_mod
    manifest = train_mod.load_manifest(manifest_path)
    val_eps = [ep for ep in manifest["episodes"] if ep["split"] == "validation"]
    parity_summary = {}
    for arm_name, _ in arms:
        ckpt = train_root / arm_name / "model.pt"
        p_out = OUT_DIR / "parity" / arm_name
        p_out.mkdir(parents=True, exist_ok=True)
        cli = SimpleNamespace(
            checkpoint=ckpt, manifest=manifest_path, airvln_root=AIRVLN_ROOT,
            out=p_out, language="correct",
        )
        pol = eval_mod.Policy(cli, manifest)
        diffs = []
        for ep in val_eps:
            rep = eval_mod.sequence_parity(pol, ep)
            diffs.append(rep["max_abs_logit_diff"])
            if not rep["argmax_equal"]:
                raise RuntimeError(f"{arm_name} 在 {ep['route_id']} 上在线/离线 argmax 不一致")
        parity_summary[arm_name] = {
            "validation_episodes": len(val_eps),
            "max_abs_logit_diff": max(diffs),
            "all_argmax_equal": True,
        }

    spatial_audit = json.loads(audit_path.read_text())
    summary = {
        "schema": "uavrgb.openfly_v2_ablation_summary.v1",
        "elapsed_s": round(time.time() - t0, 2),
        "dataset": {
            "manifest": str(manifest_path),
            "manifest_sha256": spatial_audit["manifest_sha256"],
            "routes": spatial_audit["route_counts"],
            "group_start_png_files": spatial_audit["frame_and_byte_summary"]["group_start_png_files"],
            "total_subset_disk_bytes": spatial_audit["frame_and_byte_summary"]["total_subset_disk_bytes"],
            "min_cross_split_same_scene_3d_distance_m": spatial_audit["global_min_cross_split_same_scene_3d_distance_m"],
        },
        "arms": {
            name: {
                "schema": res["contract"]["schema"],
                "updates": res["updates"],
                "validation_correct": {
                    k: v for k, v in res["diagnostics"]["validation"]["correct"].items() if k != "confusion"
                },
                "validation_swap": {
                    k: v for k, v in res["diagnostics"]["validation"]["cyclic_instruction_swap"].items() if k != "confusion"
                },
                "validation_zero": {
                    k: v for k, v in res["diagnostics"]["validation"]["zero_language_encoder_output"].items() if k != "confusion"
                },
                "sequence_parity": parity_summary[name],
            }
            for name, res in arm_results.items()
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "DONE").write_text(f"rc=0 finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "DONE").write_text(f"rc=1 error={type(exc).__name__}: {exc}\n")
        raise
