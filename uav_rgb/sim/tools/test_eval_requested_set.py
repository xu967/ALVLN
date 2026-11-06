#!/usr/bin/env python3
"""eval_closed_loop 固定请求集（AUDIT #11）+ cf_sitl 的 false_forward 滑行窗口 / stable_termination。

不起 Gazebo：用假的 run_fn 喂 scan_episodes，再过 aggregate。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ec  # noqa: E402


def _summary(seed, ok=True, success=True):
    if not ok:
        return {"seed": seed, "collect_rc": 1, "steps": 0, "stderr_tail": ["boom"],
                "dedup_key": f"k{seed}", "success": False, "collision": False, "max_x": None}
    return {"seed": seed, "collect_rc": 0, "steps": 10, "dedup_key": f"k{seed}",
            "success": success, "collision": False, "max_x": 6.0, "infer_ms": [],
            "geometry_gid": f"g{seed}", "stable_termination": None}


def _old_scan(seed_list, episodes, run_fn):
    """修复前的循环（按 len(results) 停）：只用来复现 C 报告里的矛盾数字。"""
    results, seen, failures, skips, attempted = [], set(), [], [], []
    for seed in seed_list:
        if len(results) >= episodes:
            break
        summary, skip = run_fn(seed, seen)
        if skip:
            skips.append(seed)
            continue
        attempted.append(seed)
        if summary["collect_rc"] != 0 or summary["steps"] == 0:
            failures.append(seed)
            continue
        results.append(summary)
    return results, failures, skips, attempted


def test_requested_set_frozen():
    def run_fn(seed, seen):
        if seed == 1:
            return _summary(seed, ok=False), None
        if seed == 9:
            return None, "split=train"
        return _summary(seed), None

    seeds = [0, 1, 9, 2, 3, 4]
    quiet = lambda *a, **k: None  # noqa: E731
    # 修复前：3 集请求、seed 1 失败 → 补扫 seed 3，成功 3 集：SR 1.0 / execution 1.0 却 completion 0.75。
    res_old, fail_old, _, att_old = _old_scan(seeds, 3, run_fn)
    agg_old = ec.aggregate(res_old, requested=3, attempted=len(att_old), failed_count=len(fail_old))
    assert agg_old["requested_success_rate"] == 1.0 and agg_old["execution_rate"] == 1.0
    assert agg_old["completion_rate"] == 0.75 and agg_old["failed_count"] == 1
    # 修复后：请求集冻结为前 3 个尝试过的 seed {0,1,2}；失败占名额；skip 不占。
    res, fails, skips, scanned, attempted = ec.scan_episodes(seeds, 3, run_fn, log=quiet)
    assert attempted == [0, 1, 2], attempted
    assert scanned == [0, 1, 9, 2], scanned
    assert [f["seed"] for f in fails] == [1] and [s["seed"] for s in skips] == [9]
    assert [r["seed"] for r in res] == [0, 2]
    agg = ec.aggregate(res, requested=3, attempted=len(attempted),
                       failed_count=len(fails), skipped_count=len(skips))
    assert abs(agg["requested_success_rate"] - 2 / 3) < 1e-12
    assert abs(agg["execution_rate"] - 2 / 3) < 1e-12
    assert abs(agg["completion_rate"] - 2 / 3) < 1e-12
    assert agg["failed_count"] == 1 and agg["skipped_count"] == 1 and agg["episodes_done"] == 2
    # 异常也占名额
    def run_raise(seed, seen):
        if seed == 0:
            raise RuntimeError("sim died")
        return _summary(seed), None
    res, fails, _, _, attempted = ec.scan_episodes([0, 1, 2], 2, run_raise, log=quiet)
    assert attempted == [0, 1] and [r["seed"] for r in res] == [1]
    assert fails[0]["seed"] == 0 and "sim died" in fails[0]["error"]
    # --seeds 全成功：请求集 == seed 列表
    res, fails, _, _, attempted = ec.scan_episodes([5, 6], 2, lambda s, _: (_summary(s), None), log=quiet)
    assert attempted == [5, 6] and len(res) == 2 and not fails


def test_coast_false_forward():
    s = lambda k: int(k * 1e9)  # noqa: E731
    # 碰撞步本身是 SAFE，但 0.6 s 前下过 FORWARD → cf_sitl 口径 True，旧口径 False
    rows = [{"action_executed": 1, "collision": False, "action_start_ns": s(0.0)},
            {"action_executed": 0, "collision": False, "action_start_ns": s(1.0)},
            {"action_executed": 1, "collision": False, "action_start_ns": s(2.0)},
            {"action_executed": 0, "collision": True, "src": "contact", "action_start_ns": s(2.6)}]
    oracle = [{"pose_gt": {"x": i * 0.3}} for i in range(4)]
    old = ec.summarize_episode(rows, oracle, 6.0)
    new = ec.summarize_episode(rows, oracle, 6.0, coast_window_s=1.0)
    assert old["false_forward"] is False and old["metric_version"] == ec.METRIC_VERSION
    assert new["false_forward"] is True and new["metric_version"] == ec.METRIC_VERSION + ".coast1s"
    # FORWARD 在窗口外（1.6 s 前）→ False；stuck 不算碰撞
    rows2 = [{"action_executed": 1, "collision": False, "action_start_ns": s(0.0)},
             {"action_executed": 0, "collision": False, "action_start_ns": s(1.0)},
             {"action_executed": 0, "collision": True, "src": "contact", "action_start_ns": s(1.6)}]
    assert ec.summarize_episode(rows2, oracle[:3], 6.0, coast_window_s=1.0)["false_forward"] is False
    rows3 = [{"action_executed": 1, "collision": True, "src": "stuck", "action_start_ns": s(0.0)}]
    assert ec.summarize_episode(rows3, oracle[:1], 6.0, coast_window_s=1.0)["false_forward"] is False
    # 缺时间戳退回「碰撞步动作 == FORWARD」
    rows4 = [{"action_executed": 1, "collision": True, "src": "contact"}]
    assert ec.coast_false_forward(rows4, 1.0) is True
    assert ec.coast_false_forward([{"action_executed": 0, "collision": True, "src": "contact"}], 1.0) is False
    # 空集也带后缀
    assert ec.summarize_episode([], [], 6.0, coast_window_s=1.0)["metric_version"].endswith(".coast1s")


def test_stable_termination():
    rows = [{"action_executed": 0, "collision": False}]
    assert ec.summarize_episode(rows, [{"pose_gt": {"x": 1.0}}], 6.0)["stable_termination"] is None
    slow = ec.summarize_episode(rows, [{"pose_gt": {"x": 1.0}, "speed_mps": 0.02}], 6.0)
    fast = ec.summarize_episode(rows, [{"pose_gt": {"x": 1.0}, "speed_mps": 0.31}], 6.0)
    assert slow["stable_termination"] is True and fast["stable_termination"] is False
    done = [dict(_summary(0), stable_termination=True), dict(_summary(1), stable_termination=False),
            dict(_summary(2), stable_termination=None)]
    assert ec.aggregate(done)["stable_termination_rate"] == 0.5
    assert ec.aggregate([_summary(0)])["stable_termination_rate"] is None
    assert "cf_sitl" in ec.DYNAMICS_PROFILES and ec.COAST_WINDOW_S["cf_sitl"] == 1.0


if __name__ == "__main__":
    test_requested_set_frozen()
    test_coast_false_forward()
    test_stable_termination()
    print("test_eval_requested_set: PASS")
