#!/usr/bin/env python3
"""任务卡 S1（双帧 + 深度辅助 + 15维动态方位 + 第5维独立STOP头 + 1轮DAgger）与卡 G2（零特权VLM 1Hz + 机载10Hz学生闭环）正式实测。

运行方式（守护进程）:
  OUT=/root/rivermind-data/uav_rgb/exp-s1-g2-scenelong-20261116; mkdir -p "$OUT"
  setsid nohup /root/rivermind-data/uav_rgb/3dgs-venv/bin/python \
      experiments/2026-11-16-s1-g2-scenelong/run_s1_g2_v2.py >> "$OUT/run.log" 2>&1 < /dev/null & echo $! > "$OUT/run.pid"
"""
import glob
import json
import os
from pathlib import Path
import subprocess
import time

# ========================= 顶部集中配置区 =========================
REPO_ROOT = Path("/root/claude-code-backup-20261028/code/uav_rgb")
DATA_ROOT = Path("/root/rivermind-data/uav_rgb")
OUT_DIR = Path(os.environ.get("UAV_EXP_OUT", "/root/rivermind-data/uav_rgb/exp-s1-g2-scenelong-20261116"))
PY_3DGS = Path("/root/rivermind-data/uav_rgb/3dgs-venv/bin/python")
PY_TOOLS = REPO_ROOT / ".venv-tools/bin/python"
PY_VLM = Path("/root/rivermind-data/uav_rgb/envs/vlm-venv/bin/python")
TRAIN_BIN = REPO_ROOT / "build/check/uavrgb_train"
INFER_BIN = REPO_ROOT / "build/check/uavrgb_infer"
CF_SITL_PROC = DATA_ROOT / "tmp/build-cf-sitl/cf_sitl_proc"
GSPLAT_EXT_DIR = DATA_ROOT / "torch-cache/extensions-sm75-20261109"
CACHE_ROOT = DATA_ROOT / "cache"   # 严禁向 /root/rivermind-fs 写入任何字节

# 仅使用 3 个训练公寓的 train 路线训练（保留 840865 给零特权 VLM 未见场景评测，保留 841980 与 839992 ti_* 给留出评测）
TRAIN_SCENES = ("sage3d_840139", "sage3d_841012", "sage3d_841150")
VAL_SCENE = "sage3d_839992"
DEMO_LIMIT_PER_SCENE = int(os.environ.get("S1_DEMO_LIMIT", "25"))     # 3 × 25 = 75 集初始示范
DAGGER_LIMIT_PER_SCENE = int(os.environ.get("S1_DAGGER_LIMIT", "15")) # 3 × 15 = 45 集 DAgger 纠偏
VAL_LIMIT = int(os.environ.get("S1_VAL_LIMIT", "15"))                 # 仅采 train split 作选模 val
SEARCH_DROPOUT = "0.35"  # 将 35% 视野外左转帧标为 BEARING_UNKNOWN(8)，学会未知方位时单向旋转扫描且不误停
EPOCHS_R0 = 20           # 第一轮蒸馏轮数（无类权重偏置，避免把 SAFE/YAW 放大压制 FORWARD）
EPOCHS_DAGGER = 30       # DAgger 聚合训练轮数
# =================================================================


def make_env():
    env = os.environ.copy()
    env.update({
        "DATA_ROOT": "/root/rivermind-data",
        "UAV_RGB_BUILD_DIR": str(REPO_ROOT / "build/check"),
        "CF_SITL_PROC": str(CF_SITL_PROC),
        "TORCH_EXTENSIONS_DIR": str(GSPLAT_EXT_DIR),
        "UAV_RGB_VLM_PYTHON": str(PY_VLM),
        "UAV_RGB_CACHE_ROOT": str(CACHE_ROOT),
        "TMPDIR": str(DATA_ROOT / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "UAVRGB_CUDA_BLOCKING_SYNC": "1",
    })
    return env


def run_cmd(cmd, env, allow_rc1=False):
    print(f"[CMD] {' '.join(str(c) for c in cmd)}", flush=True)
    res = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT), env=env, check=False)
    if res.returncode != 0 and not (allow_rc1 and res.returncode == 1):
        raise RuntimeError(f"命令失败 rc={res.returncode}: {cmd}")


def build_unseen_vlm_counterfactual_scene(out_json: Path) -> Path:
    """在完全未参与训练的公寓 sage3d_840865 中提取两组原生同起点异目标真实指令对：
    - 对 1: acoustic_equipment_1 -> sofa_1 (idx 53) vs body_pillow_8 (idx 71)，出生净空 1.19m
    - 对 2: basin_cabinet_2 -> mirror_cabinet_1 (idx 17) vs window_8 (idx 67)，出生净空 1.20m
    每对共享完全相同的出生位姿 (x, y, z, yaw_deg)，在 swap 模式下 (0<->1, 2<->3) 严格互换目标指令。"""
    src = json.loads((DATA_ROOT / "3dgs-scenes/sage3d_840865/scene.json").read_text())
    src["ply"] = str((DATA_ROOT / "3dgs-scenes/sage3d_840865/scene.ply").resolve())
    src["collision"]["geometry"] = str((DATA_ROOT / "3dgs-scenes/sage3d_840865/collision_mesh.ply").resolve())
    routes = src["routes"]
    pair1_a = json.loads(json.dumps(routes[53]))
    pair1_b = json.loads(json.dumps(routes[71]))
    pair1_b["spawn"] = dict(pair1_a["spawn"])
    pair2_a = json.loads(json.dumps(routes[17]))
    pair2_b = json.loads(json.dumps(routes[67]))
    pair2_b["spawn"] = dict(pair2_a["spawn"])
    picked = [pair1_a, pair1_b, pair2_a, pair2_b]
    src["routes"] = picked
    src["spawn"] = picked[0]["spawn"]
    src["waypoint"] = picked[0]["waypoint"]
    src["max_steps"] = max(int(r["max_steps"]) for r in picked)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(src, indent=2, ensure_ascii=False) + "\n")
    return out_json


def main():
    t0 = time.time()
    env = make_env()
    v2_dir = OUT_DIR / "card_s1_v2"
    col_tr = v2_dir / "collect_train"
    col_val = v2_dir / "collect_val"
    ds_r0 = v2_dir / "ds_r0"
    run_r0 = v2_dir / "run_r0"
    col_dag = v2_dir / "collect_dagger"
    ds_dag = v2_dir / "ds_dagger"
    run_dag = v2_dir / "run_dagger"

    train_scene_flags = []
    for s in TRAIN_SCENES:
        train_scene_flags += ["--scene", DATA_ROOT / f"3dgs-scenes/{s}/scene.json"]

    if not (col_tr / "vln_eval.json").is_file():
        run_cmd([
            PY_3DGS, "-m", "sim.3dgs.run_vln",
            *train_scene_flags, "--split", "train",
            "--policy", "teacher", "--teacher-clearance", "braking",
            "--collect", "--save-depth", "--limit", str(DEMO_LIMIT_PER_SCENE),
            "--seed", "101", "--out", col_tr,
        ], env)

    if not (col_val / "vln_eval.json").is_file():
        run_cmd([
            PY_3DGS, "-m", "sim.3dgs.run_vln",
            "--scene", DATA_ROOT / f"3dgs-scenes/{VAL_SCENE}/scene.json", "--split", "train",
            "--policy", "teacher", "--teacher-clearance", "braking",
            "--collect", "--save-depth", "--limit", str(VAL_LIMIT),
            "--seed", "201", "--out", col_val,
        ], env)

    if not (ds_r0 / "dataset.uavr").is_file():
        eps = [d for d in sorted(glob.glob(str(col_tr / "*/*")) + glob.glob(str(col_val / "*/*")))
               if os.path.isfile(os.path.join(d, "episode_manifest.json"))]
        run_cmd([
            PY_TOOLS, "tools/preprocess.py",
            "--drop-collision", "--cond", "vln-nav", "--vln-search-dropout", SEARCH_DROPOUT,
            "--out", ds_r0, *eps,
        ], env)

    if not (run_r0 / "ckpt_best.bin").is_file():
        run_cmd([
            TRAIN_BIN, "--data", ds_r0 / "dataset.uavr", "--out", run_r0,
            "--epochs", str(EPOCHS_R0), "--batch", "32", "--lr", "1e-3", "--seed", "20261116",
            "--two-frame", "--vln-nav", "--stop-head", "--depth-aux", "--aux-lambda", "0.5",
            "--augment", "photo", "--cosine-lr", "--drop-collision", "--cuda-blocking-sync",
        ], env)

    if not (col_dag / "vln_eval.json").is_file():
        run_cmd([
            PY_3DGS, "-m", "sim.3dgs.run_vln",
            *train_scene_flags, "--split", "train",
            "--policy", "student", "--ckpt", run_r0 / "ckpt_best.bin", "--infer-bin", INFER_BIN,
            "--teacher-clearance", "braking", "--collect", "--save-depth",
            "--dagger-beta", "0.3", "--dagger-round", "1",
            "--limit", str(DAGGER_LIMIT_PER_SCENE), "--seed", "301", "--out", col_dag,
        ], env, allow_rc1=True)

    if not (ds_dag / "dataset.uavr").is_file():
        eps = [d for d in sorted(
            glob.glob(str(col_tr / "*/*")) + glob.glob(str(col_dag / "*/*")) + glob.glob(str(col_val / "*/*"))
        ) if os.path.isfile(os.path.join(d, "episode_manifest.json"))]
        run_cmd([
            PY_TOOLS, "tools/preprocess.py",
            "--drop-collision", "--cond", "vln-nav", "--vln-search-dropout", SEARCH_DROPOUT,
            "--out", ds_dag, *eps,
        ], env)

    ckpt_final = run_dag / "ckpt_best.bin"
    if not ckpt_final.is_file():
        run_cmd([
            TRAIN_BIN, "--data", ds_dag / "dataset.uavr", "--out", run_dag,
            "--epochs", str(EPOCHS_DAGGER), "--batch", "32", "--lr", "1e-3", "--seed", "20261116",
            "--two-frame", "--vln-nav", "--stop-head", "--depth-aux", "--aux-lambda", "0.5",
            "--augment", "photo", "--cosine-lr", "--drop-collision", "--cuda-blocking-sync",
        ], env)

    eval_scenes = [
        "--scene", DATA_ROOT / "3dgs-scenes/sage3d_839992/scene.json",
        "--scene", DATA_ROOT / "3dgs-scenes/sage3d_841980/scene.json",
        "--split", "test_Instruction_Unseen",
        "--split", "test_Scene_Unseen",
    ]
    eval_arms = {
        "teacher_v5": ["--policy", "teacher", "--teacher-clearance", "braking"],
        "student_g0_subgoal": ["--policy", "student", "--ckpt", ckpt_final, "--infer-bin", INFER_BIN, "--videos", "2"],
        "student_blind_unknown": ["--policy", "student", "--ckpt", ckpt_final, "--infer-bin", INFER_BIN, "--grounder", "blind_unknown"],
        "student_blind_rgb": ["--policy", "student", "--ckpt", ckpt_final, "--infer-bin", INFER_BIN, "--blind-rgb"],
    }
    s1_eval = {}
    for arm, flags in eval_arms.items():
        adir = v2_dir / f"eval_{arm}"
        if not (adir / "vln_eval.json").is_file():
            run_cmd([PY_3DGS, "-m", "sim.3dgs.run_vln", *eval_scenes, *flags, "--out", adir], env, allow_rc1=True)
        s1_eval[arm] = json.loads((adir / "vln_eval.json").read_text())["overall"]

    # 卡 G2：未见公寓 sage3d_840865 同起点双目标对 × 零特权 VLM(1Hz) + 机载 C++ 学生(10Hz) 四组反事实闭环
    g2_scene = build_unseen_vlm_counterfactual_scene(OUT_DIR / "card_g2_v2/sage3d_840865_cf4.json")
    g2_eval = {}
    for cf in ("correct", "swap", "zero"):
        cdir = OUT_DIR / f"card_g2_v2/vlm_student_{cf}"
        if not (cdir / "vln_eval.json").is_file():
            vid = ["--videos", "2"] if cf in ("correct", "swap") else []
            run_cmd([
                PY_3DGS, "-m", "sim.3dgs.run_vln",
                "--scene", g2_scene, "--limit", "4",
                "--policy", "student", "--ckpt", ckpt_final, "--infer-bin", INFER_BIN,
                "--grounder", "vlm", "--vlm-every", "2", "--counterfactual", cf, *vid, "--out", cdir,
            ], env, allow_rc1=True)
        rep = json.loads((cdir / "vln_eval.json").read_text())
        eps_rows = [json.loads(l) for l in (cdir / "episodes.jsonl").read_text().splitlines() if l.strip()]
        g2_eval[cf] = {
            "overall": rep["overall"],
            "vlm_latency_ms": rep.get("vlm_latency_ms"),
            "episodes": [
                {
                    "route": r["route_name"],
                    "goal_object": r.get("goal_object"),
                    "supplied": r["supplied_text"][:80],
                    "success": r["success"],
                    "oracle_success": r["oracle_success"],
                    "ne_m": round(r["ne_m"], 3),
                    "spl": round(r["spl"], 3),
                    "stopped": r["stopped"],
                    "collision": r["collision"],
                }
                for r in eps_rows
            ],
        }

    # 同时补跑一组 Wanderland 户外街道长航线 student_g0_subgoal 满电闭环实测！
    wl_eval_dir = v2_dir / "eval_wanderland_student"
    if not (wl_eval_dir / "vln_eval.json").is_file():
        run_cmd([
            PY_3DGS, "-m", "sim.3dgs.run_vln",
            "--scene", DATA_ROOT / "3dgs-scenes/wanderland_1MlUucug/scene.json",
            "--policy", "student", "--ckpt", ckpt_final, "--infer-bin", INFER_BIN,
            "--dynamics-params", DATA_ROOT / "exp-wanderland-10nvb-20261114/soc100/e1_mini_soc100.json",
            "--limit", "3", "--videos", "1", "--out", wl_eval_dir,
        ], env, allow_rc1=True)
    wl_eval = json.loads((wl_eval_dir / "vln_eval.json").read_text())["overall"]

    prev_summary = json.loads((OUT_DIR / "summary.json").read_text())
    prev_summary["card_s1_cpp_student_v2_dagger"] = {
        "ckpt_best": str(ckpt_final),
        "heldout_eval": s1_eval,
        "wanderland_outdoor_long_eval": wl_eval,
    }
    prev_summary["card_g2_zero_privilege_vlm_unseen_cf"] = g2_eval
    prev_summary["v2_followup_elapsed_s"] = round(time.time() - t0, 2)
    (OUT_DIR / "summary.json").write_text(json.dumps(prev_summary, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "DONE").write_text(f"rc=0 v2_finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print(json.dumps({"card_s1_v2": s1_eval, "wanderland_student": wl_eval, "card_g2_v2": g2_eval}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (OUT_DIR / "DONE").write_text(f"rc=1 v2_error={type(exc).__name__}: {exc}\n")
        raise
