#!/usr/bin/env python3
"""两数据集对比流水线：A=Gazebo 柱阵（E1 固件在环采集）、B=3DGS Callinstrasse 路线、AB=并集 → 同结构同超参各训一个学生
→ 两域测试集交叉闭环（teacher / forward / 旧 X3 学生 / S_A / S_B / S_AB）→ tools/stats_report 汇总成论文表。

矩阵：docs/experiments/2026-11-07-dataset-compare-plan.md §2；口径：docs/EVAL_STANDARD.md §5.3；
表模板：docs/experiments/PAPER_TABLES_TEMPLATE.md（表 4 跨场景泛化）。

运行（仓库根；守护进程，与 e1_pipeline 同一套约定；Gazebo 评测要核，E1 流水线结束后再起）：
  export UAV_RGB_PYTHON=/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python
  OUT=$DATA_ROOT/uav_rgb/compare-$(date +%Y%m%d-%H%M); mkdir -p "$OUT"
  setsid nohup "$UAV_RGB_PYTHON" tools/compare_datasets.py --out "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
  echo $! > "$OUT/run.pid"; ps -o pid,pgid,etime,cmd -p $(cat "$OUT/run.pid")
看进度：tail -f $OUT/run.log；每步状态 $OUT/status.json；结束落 $OUT/DONE（rc + 最后一步）；停：kill -- -<PGID>。
续跑 --from train / eval / summary（已有 ckpt / eval.json 的项自动跳过）。--dry-run 全链只打印命令、不写文件。
改参数只改 CFG，不改函数。

已知前提（2026-11-07 EVAL agent 补完时）：
  ① A 与 B 的口径不同（A hold 1.0 s / window_from observation；B pilot v3 hold 0.5 s / render_observation），
     preprocess 拒绝混包。pack 前先读 manifest 预检：hold_s 或 window_from 不唯一的数据集标 blocked、跳过它的学生组、
     summary 写明原因；B 放量改成 hold 1.0 + observation（并补 camera_profile）后 AB 才成立（DATA 报告待确认 6）。
     跨域评测时两域 hold 不同也会在 summary 里标出（学生面对的每步位移不同：0.25 m vs 0.12 m）。
  ② 3DGS 教师组：eval_3dgs 没有 teacher 策略，用 collect_3dgs --policy astar（占据栅格 A*，与 B 采集同一教师）
     + --stop-on-collision 采，再按 eval_3dgs.summarize_3dgs_episode / aggregate 写成同格式 eval.json。
  ③ Gazebo 固定 seed：tools/eval_seeds.EVAL_SEEDS_BASE_20（pick_test_seeds 挑的 20 个独立几何，v5 下都是 test，与新 E1
     学生和旧 X3 学生的训练几何都零重叠）。2026-11-09 前这里是另一张「v3/v5 双 salt 都是 test」的 15 个 seed 清单；
     双 salt 只保证旧 X3 学生按它自己的 split 没见过，不如直接排除它的训练几何，且两张表的 n 不同不便对照，故收成同一来源。
"""
import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
BUILD = os.environ.get("UAV_RGB_BUILD_DIR", os.path.join(REPO, "build", "check"))
PY = sys.executable
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)
import stats_report  # noqa: E402
import eval_seeds  # noqa: E402  - 冻结评测 seed 的唯一来源
import pipeline_common as pc  # noqa: E402  - 三条流水线共用：续跑判定等

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
CFG = dict(
    # 数据来源：每个集合 = 若干 episode 目录 glob（{a,b} 在 Python 里展开）。目录里要有 episode_manifest.json。
    sources=dict(
        A=[f"{DATA_ROOT}/uav_rgb/e1-pipeline-20261107-0417/collection/{{train,val,test}}/*/"],
        #   柱阵 base 300 集：tools/e1_pipeline.py 04:17 产出，E1 固件在环 / hold 1.0 / observation / 教师 v5（余量 0.40/0.30 + 脱困 v2）；
        #   audit 除 class_share、safe_collision（任务结构，见 DATA 报告）外全 PASS。
        #   f"{DATA_ROOT}/uav_rgb/tmp/agent-forest/teacher3/{{sparse,medium,dense}}/train/*/",
        #   森林三档 18 集：旧 0.20/0.15 方框教师、Gazebo 未验证（DATA 报告待确认 4），先不进 A；接入时 preprocess 自动加 --allow-mixed-geometry。
        B=[f"{DATA_ROOT}/uav_rgb/datasets/e1-3dgs-callinstrasse-pilot-v3-fov60/route_*/ep_*/"],
        #   3DGS Callinstrasse pilot v3（60° FOV、6 路线、22 集、hold 0.5 / render_observation / teacher_occ.v1）：只够链路验证。
        #   放量后换目录，例如 f"{DATA_ROOT}/uav_rgb/datasets/e1-3dgs-callinstrasse-v4-hold1/route_*/ep_*/"（名字由 3DGS agent 定）。
    ),
    datasets=dict(A=("A",), B=("B",), AB=("A", "B")),   # 三个训练集 = 来源集合的并；口径不一致的集合在 pack 预检被标 blocked
    collision_filter="drop_steps",   # drop_steps = --drop-collision（与 e1_pipeline 同）；exclude_episodes = --exclude-collision-episodes
                                     # （DATA 报告建议：连带丢掉碰撞集里余量内的 SAFE/左右反转标签，丢 16/245 集 = 6.5%）
    # 训练（与 tools/e1_pipeline.py CFG 相同，学生只换数据）
    epochs=30, batch=128, lr=1e-3, class_weights="auto", text_dim=0,
    train_seeds=(0,),        # 论文用 (0, 1, 2)：同一数据三个训练种子，stats_report 报 IQM + 分层 bootstrap；每个 ≈2 min（E1 流水线 30 轮 112 s）
    old_x3_ckpt=f"{DATA_ROOT}/uav_rgb/t6-full-1640/run/ckpt_best.bin",   # 旧 X3 学生（T6 best，X3/x3_velocity/hold 1.0 训的）：载具迁移差距
    # 评测域 1：Gazebo 柱阵、E1 机体、observation、hold 1.0
    gazebo=dict(
        dynamics="cf_sitl", vehicle="e1_mini", hold=1.0, window_from="observation",
        seeds=eval_seeds.csv(eval_seeds.EVAL_SEEDS_BASE_20),
        #   与 e1_pipeline 同一批 20 个独立几何（tools/eval_seeds.py 唯一来源；见文件头 ③）。
        steps=80, sim_timeout=120.0, episode_timeout=600.0,
        parallel=4,          # 同时起几个 gz（各 1 实例）：4 实例 6.8 核（cgroup 12 核）；用户 YOLO 占 2 核
        env={"UAVRGB_RENDER_BACKEND": "software", "DISPLAY": ":20", "XDG_RUNTIME_DIR": "/tmp/xdg",
             "UAVRGB_CF_SITL_PLUGIN_DIR": f"{DATA_ROOT}/uav_rgb/tmp/build-cf-sitl/gz"},
    ),
    # 评测域 2：3DGS Callinstrasse 路线（GPU 渲染，串行）
    gs=dict(
        python=f"{DATA_ROOT}/uav_rgb/3dgs-venv/bin/python",   # torch/gsplat 在这个 venv；跑前 unset TORCH_CUDA_ARCH_LIST（BRIEF §1）
        scene=f"{DATA_ROOT}/uav_rgb/3dgs-scenes/callinstrasse/scene.json",
        routes=(3,),         # 评测路线。pilot v3 六条路线都进了训练集，这里还是「域内路线」；留出路线（split=test）等 3DGS agent 在 scene.routes 标出后改
        episodes_per_route=10,
        steps=None,          # None = 路线自带 max_steps（planner.route_check 的步数预算）；给数字则覆盖
        hold=0.5,            # 与 B pilot v3 采集一致（学生按 0.5 s 窗口训的）；B 放量改 hold 1.0 后同步改。与 gazebo 1.0 不同 → summary 自动标注
        dynamics="cf_sitl", dynamics_params=f"{REPO}/sim/dynamics/e1_mini.json",
        spawn_jitter=0.0,    # 同一路线多集靠出生扰动区分；0 = 每集同出生点（只在 pilot 口径）
        env={"TORCH_CUDA_ARCH_LIST": ""},
    ),
    metrics=stats_report.DEFAULT_METRICS,   # 表列 SR/CR/false_forward/SPL；base 任务没有 waypoint，task_success 恒 0 不进表
    n_boot=2000,
)
# 评测组：(名字, policy, ckpt 来源)。None=无模型；"old_x3_ckpt"=CFG 里的路径；"A"/"B"/"AB"=本脚本训出的学生（每个训练种子一个 run）。
GROUPS = (
    ("teacher", "teacher", None),
    ("forward", "forward", None),
    ("student-x3-old", "student", "old_x3_ckpt"),
    ("S_A", "student", "A"),
    ("S_B", "student", "B"),
    ("S_AB", "student", "AB"),
)
EVAL_DOMAINS = ("gazebo", "3dgs")
STEPS = ("pack", "train", "eval", "summary")
# ──────────────────────────────────────────────────────────────────────────

DRY = False   # main 里与 pipeline_common.DRY 一起置位；helper 都在 tools/pipeline_common.py（A4）
log, run, sh, read_json, write_json, mark, need = pc.log, pc.run_cmd, pc.sh, pc.read_json, pc.write_json, pc.mark, pc.need


def expand_braces(pattern):
    """把 {a,b} 展开成多条 glob（subprocess 不经 shell，这里自己展开）。"""
    if "{" not in pattern:
        return [pattern]
    head, _, rest = pattern.partition("{")
    alts, _, tail = rest.partition("}")
    return [p for alt in alts.split(",") for p in expand_braces(head + alt + tail)]


def episode_dirs(source_names):
    eps = []
    for name in source_names:
        for pat in CFG["sources"][name]:
            for g in expand_braces(pat):
                eps += sorted(d for d in glob.glob(g) if os.path.isfile(os.path.join(d, "episode_manifest.json")))
    return eps


def dataset_protocol(eps):
    """读每集 manifest 的口径键：hold_s / window_from 必须各只有一个值，否则 preprocess 会拒绝混包（①）。"""
    keys = ("hold_s", "window_from", "geometry_version", "dynamics", "camera_profile")
    seen = {k: set() for k in keys}
    for d in eps:
        m = read_json(os.path.join(d, "episode_manifest.json"))
        for k in keys:
            seen[k].add(json.dumps(m.get(k), ensure_ascii=False))
    proto = {k: sorted(v) for k, v in seen.items()}
    proto["n_episodes"] = len(eps)
    proto["blocked_by"] = [k for k in ("hold_s", "window_from") if len(proto[k]) > 1]
    return proto


def ds_dir(out, ds_name):
    return os.path.join(out, "ds", ds_name)


_BLOCKED = {}   # 本进程 pack 预检的结果（dry-run 不落盘也要让后面的步骤知道）


def blocked_reason(out, ds_name):
    """pack 预检写的 BLOCKED.json（口径不一致）；没有则 None。"""
    if ds_name in _BLOCKED:
        return _BLOCKED[ds_name]
    path = os.path.join(ds_dir(out, ds_name), "BLOCKED.json")
    return read_json(path)["reason"] if os.path.isfile(path) else None


# ───────────────────────────── 四个步骤，按顺序 ─────────────────────────────
def pack(out):
    """每个训练集一个 .uavr。先预检口径（hold/window），不一致的标 blocked 并继续别的集；跨族（几何版本不同）显式 --allow-mixed-geometry。"""
    blocked = {}
    for ds_name, srcs in CFG["datasets"].items():
        ds = ds_dir(out, ds_name)
        eps = episode_dirs(srcs)
        if not eps:
            if DRY:
                log(f"[dry] {ds_name}: {srcs} 下现在没有 episode 目录，命令按占位符打印")
                proto = {"blocked_by": [], "geometry_version": ["?"]}
            else:
                raise RuntimeError(f"{ds_name}: {srcs} 下没有 episode 目录")
        else:
            proto = dataset_protocol(eps)
            log(f"{ds_name}: {proto['n_episodes']} 集；hold_s={proto['hold_s']} window_from={proto['window_from']} "
                f"geometry_version={proto['geometry_version']} camera_profile={proto['camera_profile']}")
        if proto["blocked_by"]:
            reason = (f"{ds_name} 的 {proto['blocked_by']} 不唯一（{[proto[k] for k in proto['blocked_by']]}），"
                      "preprocess 会拒绝混包；B 放量统一 hold 1.0 / window_from observation 后再打包")
            log(f"!! blocked: {reason}")
            blocked[ds_name] = _BLOCKED[ds_name] = reason
            if not DRY:
                os.makedirs(ds, exist_ok=True)
                write_json(os.path.join(ds, "BLOCKED.json"), {"reason": reason, "protocol": proto})
            continue
        if os.path.isfile(os.path.join(ds, "dataset.uavr")):
            log(f"跳过 pack {ds_name}：已有 dataset.uavr")
            continue
        cmd = [PY, f"{REPO}/tools/preprocess.py", "--out", ds]
        if len(proto["geometry_version"]) > 1:
            cmd.append("--allow-mixed-geometry")
        cmd.append({"drop_steps": "--drop-collision", "exclude_episodes": "--exclude-collision-episodes"}
                   [CFG["collision_filter"]])
        sh(cmd + (eps or [f"<{'+'.join(srcs)} 的 episode 目录>"]), log_path=None if DRY else os.path.join(out, f"pack-{ds_name}.log"))
    mark(out, "pack", "ok" if not blocked else "partial", blocked=blocked)
    return blocked


def train_runs(out, ds_name):
    """(run 标签, run 目录) 列表；一个训练种子一个 run。"""
    return [(f"s{seed}", os.path.join(out, "run", ds_name, f"s{seed}")) for seed in CFG["train_seeds"]]


def train(out):
    for ds_name in CFG["datasets"]:
        if blocked_reason(out, ds_name):
            log(f"跳过 train {ds_name}：{blocked_reason(out, ds_name)}")
            continue
        ds = os.path.join(ds_dir(out, ds_name), "dataset.uavr")
        need(ds, f"先跑 pack（{ds_name}）")
        for label, run_dir in train_runs(out, ds_name):
            if pc.train_finished(run_dir):   # 完成 = metrics.jsonl 末行 done:（ckpt 文件每轮都写，不算）
                log(f"跳过 train {ds_name}/{label}：metrics.jsonl 已记 done")
                continue
            if os.path.isdir(run_dir) and os.listdir(run_dir):
                raise RuntimeError(f"{run_dir} 非空且训练没跑完（metrics.jsonl 末行不是 done:）：uavrgb_train 拒绝复用，删掉后重跑")
            seed = int(label[1:])
            cmd = [f"{BUILD}/uavrgb_train", "--data", ds, "--out", run_dir,
                   "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", repr(float(CFG["lr"])),
                   "--seed", str(seed), "--class-weights", CFG["class_weights"]]
            if CFG["collision_filter"] == "drop_steps":
                cmd.append("--drop-collision")
            if CFG["text_dim"] > 0:
                cmd += ["--text-dim", str(CFG["text_dim"])]
            elif os.path.isfile(os.path.join(ds_dir(out, ds_name), "instructions.jsonl")):
                cmd.append("--vision-only")
            sh(cmd, log_path=None if DRY else os.path.join(out, f"train-{ds_name}-{label}.log"))
            need(f"{run_dir}/ckpt_best.bin", "训练结束但没有 best ckpt，看 run/metrics.jsonl")


def ckpts_of(out, key):
    """ckpt 来源 -> [(run 后缀, ckpt 路径)]。无模型 [("", None)]；blocked 数据集 []（该组整行缺）。"""
    if key is None:
        return [("", None)]
    if key == "old_x3_ckpt":
        return [("", CFG["old_x3_ckpt"])]
    if blocked_reason(out, key):
        return []
    return [(f"-{label}", os.path.join(run_dir, "ckpt_best.bin")) for label, run_dir in train_runs(out, key)]


def eval_gazebo_cmd(out, name, policy, ckpt):
    g = CFG["gazebo"]
    edir = os.path.join(out, "eval-gazebo", name)
    cmd = [PY, f"{REPO}/sim/tools/eval_closed_loop.py", "--policy", policy, "--task", "base",
           "--split", "test", "--seeds", g["seeds"], "--episodes", str(len(g["seeds"].split(","))),
           "--steps", str(g["steps"]), "--out", edir,
           "--sim-timeout", str(g["sim_timeout"]), "--episode-timeout", str(g["episode_timeout"]),
           "--dynamics", g["dynamics"], "--vehicle", g["vehicle"], "--hold", str(g["hold"]),
           "--window-from", g["window_from"]]
    if ckpt:
        cmd += ["--ckpt", ckpt, "--infer-bin", f"{BUILD}/uavrgb_infer"]
    return cmd, g["env"], edir


def eval_3dgs_cmd(out, name, policy, ckpt, route):
    """student/forward 走 eval_3dgs；teacher 走 collect_3dgs --policy astar（②），之后由 teacher_3dgs_eval_json 补 eval.json。"""
    g = CFG["gs"]
    edir = os.path.join(out, "eval-3dgs", f"{name}-route{route}")
    common = ["--scene", g["scene"], "--route", str(route), "--episodes", str(g["episodes_per_route"]),
              "--hold", str(g["hold"]), "--dynamics", g["dynamics"], "--dynamics-params", g["dynamics_params"],
              "--spawn-jitter", str(g["spawn_jitter"]), "--out", edir]
    if g["steps"] is not None:
        common += ["--steps", str(g["steps"])]
    if policy == "teacher":
        cmd = [g["python"], "-m", "sim.3dgs.collect_3dgs", "--policy", "astar", "--stop-on-collision", "--split", "test"] + common
    else:
        cmd = [g["python"], "-m", "sim.3dgs.eval_3dgs", "--policy", policy] + common
        if ckpt:
            cmd += ["--ckpt", ckpt, "--infer-bin", f"{BUILD}/uavrgb_infer"]
    return cmd, g["env"], edir


def teacher_3dgs_eval_json(edir, route):
    """把 collect_3dgs --policy astar 的采集目录按 eval_3dgs 的格式写成 eval.json（同一套 summarize/aggregate，含 SPL）。"""
    scene_mod = importlib.import_module("sim.3dgs.scene")
    collision = importlib.import_module("sim.3dgs.collision")
    planner = importlib.import_module("sim.3dgs.planner")
    ev = importlib.import_module("sim.3dgs.eval_3dgs")
    run_meta = read_json(os.path.join(edir, "run.json"))
    scene = scene_mod.apply_route(scene_mod.load_scene(CFG["gs"]["scene"]), route)
    occ_planner = planner.OccPlanner(collision.from_scene(scene))
    results, failures = [], list(run_meta.get("failures") or [])
    for ep in run_meta.get("episodes") or []:
        d = os.path.join(edir, ep["dir"])
        if ep.get("error"):
            failures.append({"episode": ep["episode"], "error": ep["error"], "dir": ep["dir"]})
            continue
        actions, oracle = ev._read_jsonl(os.path.join(d, "actions.jsonl")), ev._read_jsonl(os.path.join(d, "oracle.jsonl"))
        s = ev.summarize_3dgs_episode(actions, oracle, scene, shortest_path_m=ev.route_shortest_path_m(occ_planner, oracle, scene))
        s.update({"seed": read_json(os.path.join(d, "episode_manifest.json")).get("seed", ep["episode"]),
                  "episode": ep["episode"], "dir": ep["dir"], "policy": "teacher", "actions": {}})
        for a in actions:
            k = int(a.get("action_executed", -1))
            s["actions"][k] = s["actions"].get(k, 0) + 1
        results.append(s)
    n_req = len(run_meta.get("episodes") or []) + len(run_meta.get("failures") or [])
    agg = ev.aggregate(results, requested=n_req, attempted=n_req, failed_count=len(failures))
    report = {"schema": "uavrgb.eval.v1", "policy": "teacher", "teacher": "astar_occ (sim.3dgs.planner)",
              "source": "collect_3dgs --policy astar --stop-on-collision，由 tools/compare_datasets.teacher_3dgs_eval_json 汇总",
              "metric_version": ev.METRIC_VERSION, "success_criterion": ev.SUCCESS_CRITERION,
              "reach_radius_m": ev.reach_radius_of(scene), "spl_reference": ev.SPL_REFERENCE,
              "scene": {"name": scene.get("name"), "path": scene["path"], "route": scene.get("route"),
                        "waypoint": scene.get("waypoint"), "spawn": scene.get("spawn"), "split": "test"},
              "dynamics": run_meta.get("dynamics"), "dynamics_profile": run_meta.get("dynamics_profile"),
              "episodes_requested": n_req, "steps": run_meta.get("steps"), "hold_s": run_meta.get("hold_s"),
              "seeds": [r["seed"] for r in results], "per_episode": results, "failed": failures, **agg}
    write_json(os.path.join(edir, "eval.json"), report)
    return report


def eval_jobs(out, check_ckpt=True):
    """所有 (域, 组名, 后缀, 命令, env, 目录, route)。blocked 数据集的学生组不出现；summary 阶段不查 ckpt。"""
    jobs = []
    for name, policy, key in GROUPS:
        for suffix, ckpt in ckpts_of(out, key):
            if ckpt and check_ckpt:
                need(ckpt, f"{name}{suffix} 的 ckpt")
            if "gazebo" in EVAL_DOMAINS:
                cmd, env, edir = eval_gazebo_cmd(out, name + suffix, policy, ckpt)
                jobs.append(("gazebo", name, suffix, cmd, env, edir, None))
            if "3dgs" in EVAL_DOMAINS:
                for route in CFG["gs"]["routes"]:
                    cmd, env, edir = eval_3dgs_cmd(out, name + suffix, policy, ckpt, route)
                    jobs.append(("3dgs", name, suffix, cmd, env, edir, route))
    return jobs


def evaluate(out):
    """gazebo 域按 CFG.gazebo.parallel 并行（各 1 个 gz）；3dgs 域串行（共用一块 GPU）。已有 eval.json 的跳过。"""
    jobs = eval_jobs(out)
    pending = []
    for domain, name, suffix, cmd, env, edir, route in jobs:
        tag = f"{domain}/{name}{suffix}" + (f" route {route}" if route is not None else "")
        if pc.eval_finished(edir):   # 完成 = eval.json 在、failed_count==0、episodes_done>0、非 partial
            log(f"跳过 {tag}：eval.json 完整")
            continue
        if not DRY:
            moved = pc.retire_unfinished_eval(edir)   # 有 eval.json 但没完成（失败集/中断）：让出目录重跑
            if moved:
                log(f"{tag} 上次没跑完，旧目录挪到 {moved}，重跑")
        pending.append((domain, name, suffix, cmd, env, edir, route))
    gz = [j for j in pending if j[0] == "gazebo"]
    par = max(1, int(CFG["gazebo"]["parallel"]))
    for i in range(0, len(gz), par):
        procs = []
        for _d, name, suffix, cmd, env, edir, _r in gz[i:i + par]:
            log_path = os.path.join(out, f"eval-gazebo-{name}{suffix}.log")
            log("$ " + " ".join(cmd) + f"  > {log_path}")
            if not DRY:
                procs.append((name + suffix, edir, pc.popen(cmd, env=dict(os.environ, **env),   # 登记：SIGTERM 时连 gz 一起收
                                                            stdout=open(log_path, "ab"), stderr=subprocess.STDOUT)))
        for label, edir, proc in procs:
            rc = proc.wait()
            log(f"eval-gazebo/{label} rc={rc}" + ("（有失败/跳过集，见 eval.json 的 failed/skipped）" if rc else ""))
        for label, edir, _proc in procs:   # 等齐这一批再查：中途抛出会让 main 写 DONE 退出，其余组的 gz 留成孤儿（AUDIT Must 3）
            need(os.path.join(edir, "eval.json"), f"{label} 没写出 eval.json，看 {out}/eval-gazebo-{label}.log")
    for _d, name, suffix, cmd, env, edir, route in [j for j in pending if j[0] == "3dgs"]:
        log_path = os.path.join(out, f"eval-3dgs-{name}{suffix}-route{route}.log")
        rc = run(cmd, env, log_path=log_path)
        if rc != 0:
            log(f"eval-3dgs/{name}{suffix} route {route} rc={rc}（有失败集或采集错误，见 {log_path}）")
        if name == "teacher":
            if DRY:
                log(f"[dry] 之后由 teacher_3dgs_eval_json 把 {edir} 汇总成 eval.json")
            else:
                need(os.path.join(edir, "run.json"), f"教师采集没写 run.json，看 {log_path}")
                rep = teacher_3dgs_eval_json(edir, route)
                log(f"eval-3dgs/teacher route {route}: SR={rep['success_rate']} CR={rep['collision_rate']} "
                    f"spl={rep['spl']} n={rep['episodes_done']}")
        if not DRY:
            need(os.path.join(edir, "eval.json"), f"{name}{suffix} route {route} 没写出 eval.json，看 {log_path}")


def summary(out):
    """stats_report：行 = 组，列 = 域(×路线) × 指标，Wilson/bootstrap/IQM + 两两 McNemar（同 seed）+ Cochran's Q；写 summary.md/json。"""
    entries, missing = [], []
    for domain, name, suffix, _cmd, _env, edir, route in eval_jobs(out, check_ckpt=False):
        scene = "gazebo" if domain == "gazebo" else f"3dgs-r{route}"
        path = os.path.join(edir, "eval.json")
        if os.path.isfile(path):
            entries.append((name, scene, path))   # 同名多 run（训练种子）→ stats_report 报 IQM
        else:
            missing.append(f"{name}{suffix} @ {scene}")
    blocked = {ds: blocked_reason(out, ds) for ds in CFG["datasets"] if blocked_reason(out, ds)}
    notes = [f"- 训练集 {ds} 未打包：{reason}；组 {[g for g, _p, k in GROUPS if k == ds]} 整行缺。" for ds, reason in blocked.items()]
    if missing:
        notes.append(f"- 缺 eval.json：{missing}")
    notes.append(f"- 域口径：gazebo hold {CFG['gazebo']['hold']} s / {CFG['gazebo']['window_from']} / "
                 f"{len(CFG['gazebo']['seeds'].split(','))} 个独立几何 seed（请求数 = seed 数）；"
                 f"3dgs hold {CFG['gs']['hold']} s / 路线 {list(CFG['gs']['routes'])}（pilot v3 路线都在训练集里，不是留出路线）。")
    if abs(CFG["gazebo"]["hold"] - CFG["gs"]["hold"]) > 1e-9:
        notes.append(f"- **两域 hold 不同（gazebo {CFG['gazebo']['hold']} s vs 3dgs {CFG['gs']['hold']} s）**：学生每步位移不同，跨域列不可与域内列直接比，"
                     "报告里必须标注（dataset-compare-plan §5）。")
    notes.append("- E1 外径 0.20 m vs X3 0.71 m：旧 X3 学生那一行是载具迁移差距，不是同机体对照。")
    if not entries:
        log("summary: 还没有任何 eval.json" + ("（dry-run）" if DRY else ""))
        for n in notes:
            log(n)
        return
    rep = stats_report.build_report(entries, metrics=CFG["metrics"], pair_metrics=("SR",), n_boot=CFG["n_boot"],
                                    title="数据集对比（A=Gazebo 柱阵，B=3DGS 路线，AB=并集）")
    rep["blocked_datasets"] = blocked
    rep["notes"] = notes
    rep["cfg"] = CFG
    md = stats_report.to_markdown(rep) + "\n## 备注\n\n" + "\n".join(notes) + "\n"
    for line in md.splitlines():
        log(line)
    if not DRY:
        with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
            f.write(md)
        write_json(os.path.join(out, "summary.json"), rep)


RUN = dict(pack=pack, train=train, eval=evaluate, summary=summary)


def preflight(steps):
    """只查这次要跑的步骤用得到的东西：`--from summary` 只读 eval.json，不该因为没编译二进制而停。"""
    if "train" in steps or "eval" in steps:
        need(f"{BUILD}/uavrgb_train", "先 bash tools/check.sh 构建，或设 UAV_RGB_BUILD_DIR")
        need(f"{BUILD}/uavrgb_infer", "同上")
    if "eval" in steps:
        need(CFG["old_x3_ckpt"], "旧 X3 学生 ckpt")
        if "gazebo" in EVAL_DOMAINS:
            need(CFG["gazebo"]["env"]["UAVRGB_CF_SITL_PLUGIN_DIR"], "cf_sitl Gazebo 插件目录（sim/cf_sitl/gz/README）")
        if "3dgs" in EVAL_DOMAINS:
            need(CFG["gs"]["python"], "3DGS venv（torch/gsplat）")
            need(CFG["gs"]["scene"], "3DGS scene.json")
            if CFG["gs"]["dynamics"] == "cf_sitl":
                need(os.environ.get("CF_SITL_PROC", f"{DATA_ROOT}/uav_rgb/tmp/build-cf-sitl/cf_sitl_proc"),
                     "cf_sitl_proc（3DGS 侧固件在环子进程）")
    if CFG["collision_filter"] not in ("drop_steps", "exclude_episodes"):
        raise RuntimeError(f"CFG.collision_filter={CFG['collision_filter']!r} 不认识")
    for name, _p, key in GROUPS:
        if key not in (None, "old_x3_ckpt") and key not in CFG["datasets"]:
            raise RuntimeError(f"GROUPS 里 {name} 的 ckpt 来源 {key} 不在 CFG.datasets")
    for m in CFG["metrics"]:
        if m not in stats_report.METRICS:
            raise RuntimeError(f"CFG.metrics 里 {m} 不在 stats_report.METRICS")


def main():
    global DRY
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出根目录（数据盘）")
    p.add_argument("--from", dest="start", choices=STEPS, default="pack", help="从哪一步开始（续跑）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行、不写文件")
    a = p.parse_args()
    pc.DRY = DRY = a.dry_run
    out = os.path.abspath(a.out)
    if not DRY:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "config.json"),
                   {"cfg": CFG, "groups": GROUPS, "argv": sys.argv, "repo": REPO, "build": BUILD, "python": PY,
                    "git": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()})
        if os.path.exists(os.path.join(out, "DONE")):
            os.remove(os.path.join(out, "DONE"))
    pc.install_signal_handlers()   # SIGTERM/Ctrl-C → Interrupted：收子进程组、status 标 interrupted、DONE 写真实 rc
    rc, last = 0, None
    try:
        preflight(STEPS[STEPS.index(a.start):])
        for step in STEPS[STEPS.index(a.start):]:
            last = step
            log(f"==== {step} 开始")
            mark(out, step, "running")
            t0 = time.monotonic()
            RUN[step](out)
            if step != "pack":   # pack 自己按 blocked 情况标 ok/partial
                mark(out, step, "ok", seconds=round(time.monotonic() - t0, 1))
            log(f"==== {step} 完成 {time.monotonic() - t0:.0f}s")
    except pc.Interrupted as e:
        rc = e.rc
        n = pc.terminate_children()
        mark(out, last or "preflight", "interrupted", signal=e.name, children_terminated=n)
        log(f"!!!! {e}：已收掉 {n} 个子进程组，{last or 'preflight'} 标 interrupted，rc={rc}")
    except Exception as e:   # noqa: BLE001
        rc = 1
        mark(out, last or "preflight", "failed", error=f"{type(e).__name__}: {e}")
        log(f"!!!! {last or 'preflight'} 失败：{e}")
    finally:
        if not DRY:
            with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as fp:
                fp.write(f"rc={rc} {time.strftime('%F %T')} last_step={last}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
