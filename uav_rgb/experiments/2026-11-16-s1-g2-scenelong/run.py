#!/usr/bin/env python3
"""任务卡 SCENE-LONG + 卡 S1 + 卡 G2 联合真机动力学(cf_sitl)与 GPU 闭环实测流水线。

运行方式（守护进程）:
  OUT=/root/rivermind-data/uav_rgb/exp-s1-g2-scenelong-20261116; mkdir -p "$OUT"
  setsid nohup /root/rivermind-data/uav_rgb/3dgs-venv/bin/python \
      experiments/2026-11-16-s1-g2-scenelong/run.py > "$OUT/run.log" 2>&1 < /dev/null & echo $! > "$OUT/run.pid"
"""
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# ========================= 顶部集中配置区 =========================
REPO_ROOT = Path("/root/claude-code-backup-20261028/code/uav_rgb")
DATA_ROOT = Path("/root/rivermind-data/uav_rgb")
OUT_DIR = Path(os.environ.get(
    "UAV_EXP_OUT",
    "/root/rivermind-data/uav_rgb/exp-s1-g2-scenelong-20261116",
))  # 实验产物与 DONE 标记输出目录
PY_3DGS = Path("/root/rivermind-data/uav_rgb/3dgs-venv/bin/python")
PY_TOOLS = REPO_ROOT / ".venv-tools/bin/python"
PY_USD = Path("/root/rivermind-data/uav_rgb/envs/usd-venv/bin/python")
PY_VLM = Path("/root/rivermind-data/uav_rgb/envs/vlm-venv/bin/python")
TRAIN_BIN = REPO_ROOT / "build/check/uavrgb_train"
INFER_BIN = REPO_ROOT / "build/check/uavrgb_infer"
CF_SITL_PROC = DATA_ROOT / "tmp/build-cf-sitl/cf_sitl_proc"
GSPLAT_EXT_DIR = DATA_ROOT / "torch-cache/extensions-sm75-20261109"
CACHE_ROOT = DATA_ROOT / "cache"   # 缓存一律放数据盘，严禁写 /root/rivermind-fs

SAGE_IDS = ("839992", "840139", "840865", "841012", "841150", "841980")
TRAIN_EPOCHS = 30                  # C++ 双帧+15维动态方位+独立STOP头训练轮数（< 15s）
TRAIN_BATCH = 32                   # 显存占用 < 1 GiB
TRAIN_LR = "1e-3"                  # 配合 --cosine-lr 稳定收敛
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


def run_cmd(cmd, env, allow_rc1_eval=False):
    print(f"[CMD] {' '.join(str(c) for c in cmd)}", flush=True)
    res = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT), env=env, check=False)
    if res.returncode != 0 and not (allow_rc1_eval and res.returncode == 1):
        raise RuntimeError(f"命令失败 rc={res.returncode}: {cmd}")
    return res.returncode


def step1_scene_long(env):
    print("=== STEP 1: 任务卡 SCENE-LONG（SAGE 门洞开启碰撞网格 + 跨房间多段长航线 + Wanderland 原生长航线 + 质量探针）===", flush=True)
    sage_before = {}
    sage_after = {}
    for sid in SAGE_IDS:
        sc_dir = DATA_ROOT / f"3dgs-scenes/sage3d_{sid}"
        old_sum_p = sc_dir / "import_summary.json"
        if old_sum_p.is_file():
            old_s = json.loads(old_sum_p.read_text())
            # 若已有备份则读备份对应的原统计
            bak_p = sc_dir / "scene.gaussians_v1.json"
            if bak_p.is_file():
                bak_sc = json.loads(bak_p.read_text())
                sage_before[sid] = {"n_ok": len(bak_sc.get("routes", []))}
            else:
                sage_before[sid] = {"n_ok": old_s.get("n_ok")}
        align = json.loads((sc_dir / "align.json").read_text())
        gz = float(align["ground"]["ground_z"])
        usd_p = DATA_ROOT / f"pubdata/sage3d/collision/{sid}_collision.usd"
        mesh_p = sc_dir / "collision_mesh.ply"
        if not mesh_p.is_file():
            run_cmd([
                PY_USD, "tools/usdc_to_ply.py",
                "--usd", usd_p, "--out", mesh_p,
                "--z-min", f"{gz - 0.15:.3f}", "--z-max", f"{gz + 1.60:.3f}",
                "--open-doors",
            ], env)
        env_s = dict(env, IMPORT_STEPS="scene", IMPORT_OUT=str(OUT_DIR / f"sage_import/{sid}"))
        run_cmd([PY_3DGS, "tools/scene_import_sage3d.py", sid], env_s)
        new_s = json.loads((sc_dir / "import_summary.json").read_text())
        new_sc = json.loads((sc_dir / "scene.json").read_text())
        ge15 = sum(1 for r in new_sc["routes"] if float(r["sage"]["astar_path_m"]) >= 15.0)
        sage_after[sid] = {
            "collision_source": new_s["collision_source"],
            "n_candidates": new_s["n_candidates"],
            "n_ok": new_s["n_ok"],
            "n_ok_ge_15m": ge15,
            "max_astar_m": new_s["astar_path_m_ok"].get("max"),
            "n_multi_stage_30_55m": new_s["n_multi_stage_30_55m"],
        }

    wl_summary = {}
    for sname, sid in (
        ("wanderland_1MlUucug", "1MlUucugRAnrx86cpUuQt8fUtxFMddxju"),
        ("wanderland_10nvb", "10nvbQnL3fUaY8zb5Orh_t3mk7wnal26i"),
    ):
        w_out = OUT_DIR / f"wl_import/{sname}"
        run_cmd([
            PY_3DGS, "tools/scene_import_wanderland.py",
            "--scene-id", sid, "--scene-name", sname,
            "--out", w_out, "--steps", "scene", "--use-episodes-json",
        ], env)
        w_sc = json.loads((DATA_ROOT / f"3dgs-scenes/{sname}/scene.json").read_text())
        ge30 = sum(1 for r in w_sc["routes"] if float(r["vln"]["geodesic_distance_m"]) >= 30.0)
        ge40 = sum(1 for r in w_sc["routes"] if float(r["vln"]["geodesic_distance_m"]) >= 40.0)
        max_g = max(float(r["vln"]["geodesic_distance_m"]) for r in w_sc["routes"])
        wl_summary[sname] = {
            "n_ok": len(w_sc["routes"]),
            "n_ge_30m": ge30,
            "n_ge_40m": ge40,
            "max_geodesic_m": round(max_g, 2),
            "render": w_sc.get("render"),
        }

    probe_out = OUT_DIR / "probe_wl_r0"
    if not (probe_out / "routes_quality.json").is_file():
        subprocess.run([
            str(PY_3DGS), "tools/scene_probe_quality.py",
            "--scene", str(DATA_ROOT / "3dgs-scenes/wanderland_1MlUucug/scene.json"),
            "--route", "0", "--out", str(probe_out),
        ], cwd=str(REPO_ROOT), env=env, check=False)
    probe_q = json.loads((probe_out / "routes_quality.json").read_text())["routes"][0]
    return {
        "sage_before": sage_before,
        "sage_after": sage_after,
        "sage_totals": {
            "single_stage_ok": sum(x["n_ok"] for x in sage_after.values()),
            "single_stage_ge_15m": sum(x["n_ok_ge_15m"] for x in sage_after.values()),
            "multi_stage_30_55m": sum(x["n_multi_stage_30_55m"] for x in sage_after.values()),
        },
        "wanderland": wl_summary,
        "probe_wanderland_r0": {k: probe_q.get(k) for k in (
            "name", "path_m", "n_frames", "low_frac", "lower_alpha50_min",
            "lower_alpha_pass", "sweep_n_frames", "sweep_low_frac", "sweep_pass", "pass",
        )},
    }


def step2_card_s1_student(env):
    print("=== STEP 2: 任务卡 S1（采集 VLN 示范 + 打包 vln-nav + 训练 C++ 5维动作/STOP模型 + 闭环消融评测）===", flush=True)
    s1_dir = OUT_DIR / "card_s1"
    col_train = s1_dir / "collect_train"
    col_val = s1_dir / "collect_val"
    ds_dir = s1_dir / "ds"
    run_dir = s1_dir / "run_cpp"

    if not (col_train / "vln_eval.json").is_file():
        run_cmd([
            PY_3DGS, "-m", "sim.3dgs.run_vln",
            "--scene", DATA_ROOT / "3dgs-scenes/sage3d_840865/scene.json",
            "--scene", DATA_ROOT / "3dgs-scenes/sage3d_841012/scene.json",
            "--policy", "teacher", "--teacher-clearance", "braking",
            "--collect", "--limit", "24", "--seed", "100",
            "--out", col_train,
        ], env)

    if not (col_val / "vln_eval.json").is_file():
        run_cmd([
            PY_3DGS, "-m", "sim.3dgs.run_vln",
            "--scene", DATA_ROOT / "3dgs-scenes/sage3d_839992/scene.json",
            "--policy", "teacher", "--teacher-clearance", "braking",
            "--collect", "--limit", "10", "--seed", "200",
            "--out", col_val,
        ], env)

    if not (ds_dir / "dataset.uavr").is_file():
        ep_dirs = sorted(
            glob.glob(str(col_train / "*/*")) + glob.glob(str(col_val / "*/*"))
        )
        ep_dirs = [d for d in ep_dirs if os.path.isfile(os.path.join(d, "episode_manifest.json"))]
        run_cmd([
            PY_TOOLS, "tools/preprocess.py",
            "--drop-collision", "--cond", "vln-nav",
            "--out", ds_dir, *ep_dirs,
        ], env)

    ckpt_best = run_dir / "ckpt_best.bin"
    if not ckpt_best.is_file():
        run_cmd([
            TRAIN_BIN,
            "--data", ds_dir / "dataset.uavr",
            "--out", run_dir,
            "--epochs", str(TRAIN_EPOCHS),
            "--batch", str(TRAIN_BATCH),
            "--lr", TRAIN_LR,
            "--seed", "20261116",
            "--two-frame", "--vln-nav", "--stop-head",
            "--class-weights", "auto",
            "--augment", "flip",
            "--cosine-lr",
            "--drop-collision",
            "--cuda-blocking-sync",
        ], env)

    # 在留出场景 sage3d_839992 (test_Instruction_Unseen + test_Trajectory_Unseen) 与未见场景 sage3d_841980 上评测
    eval_scenes = [
        "--scene", DATA_ROOT / "3dgs-scenes/sage3d_839992/scene.json",
        "--scene", DATA_ROOT / "3dgs-scenes/sage3d_841980/scene.json",
        "--split", "test_Instruction_Unseen",
        "--split", "test_Scene_Unseen",
    ]
    eval_arms = {
        "teacher_v5": ["--policy", "teacher", "--teacher-clearance", "braking"],
        "student_g0_goal": ["--policy", "student", "--ckpt", ckpt_best, "--infer-bin", INFER_BIN, "--videos", "1"],
        "student_blind_unknown": ["--policy", "student", "--ckpt", ckpt_best, "--infer-bin", INFER_BIN, "--grounder", "blind_unknown"],
        "student_blind_rgb": ["--policy", "student", "--ckpt", ckpt_best, "--infer-bin", INFER_BIN, "--blind-rgb"],
    }
    eval_results = {}
    for arm, flags in eval_arms.items():
        adir = s1_dir / f"eval_{arm}"
        if not (adir / "vln_eval.json").is_file():
            run_cmd([PY_3DGS, "-m", "sim.3dgs.run_vln", *eval_scenes, *flags, "--out", adir], env, allow_rc1_eval=True)
        eval_results[arm] = json.loads((adir / "vln_eval.json").read_text())["overall"]

    return {
        "ckpt_best": str(ckpt_best),
        "dataset_samples": json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").is_file() else None,
        "heldout_eval": eval_results,
    }


def step3_card_g2_vlm(env, ckpt_best):
    print("=== STEP 3: 任务卡 G2（零特权 VLM Grounder 1Hz + 机载 10Hz C++ 学生 四组反事实闭环）===", flush=True)
    g2_dir = OUT_DIR / "card_g2"
    # 选取未见房间/留出指令集评测零特权闭环：correct vs swap vs zero
    target_scenes = [
        "--scene", DATA_ROOT / "3dgs-scenes/sage3d_840865/scene.json",
        "--limit", "4", "--seed", "7",
    ]
    cf_results = {}
    for cf in ("correct", "swap", "zero"):
        cdir = g2_dir / f"vlm_student_{cf}"
        if not (cdir / "vln_eval.json").is_file():
            vid = ["--videos", "1"] if cf in ("correct", "swap") else []
            run_cmd([
                PY_3DGS, "-m", "sim.3dgs.run_vln",
                *target_scenes,
                "--policy", "student", "--ckpt", ckpt_best, "--infer-bin", INFER_BIN,
                "--grounder", "vlm", "--vlm-every", "2",
                "--counterfactual", cf, *vid,
                "--out", cdir,
            ], env, allow_rc1_eval=True)
        rep = json.loads((cdir / "vln_eval.json").read_text())
        cf_results[cf] = {
            "overall": rep["overall"],
            "vlm_latency_ms": rep.get("vlm_latency_ms"),
        }
    return cf_results


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env()
    s1_long = step1_scene_long(env)
    s2_cpp = step2_card_s1_student(env)
    s3_g2 = step3_card_g2_vlm(env, s2_cpp["ckpt_best"])
    summary = {
        "schema": "uavrgb.s1_g2_scenelong_verification.v1",
        "elapsed_s": round(time.time() - t0, 2),
        "scene_long": s1_long,
        "card_s1_cpp_student": {
            "ckpt_best": s2_cpp["ckpt_best"],
            "heldout_eval": s2_cpp["heldout_eval"],
        },
        "card_g2_zero_privilege_vlm": s3_g2,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "DONE").write_text(f"rc=0 finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "DONE").write_text(f"rc=1 error={type(exc).__name__}: {exc}\n")
        raise
