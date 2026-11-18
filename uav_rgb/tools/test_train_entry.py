#!/usr/bin/env python3
"""唯一训练与评测入口 train.py 的配置守卫、拼写检查、防覆盖与预设模式单测（不碰 GPU）。"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "train.py"


class TrainEntryGuardTests(unittest.TestCase):
    def _run_entry(self, *args, extra_env=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("UAVRGB_")}
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, str(TRAIN_PY), *args],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_check_only_succeeds_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run_entry("--check", extra_env={"UAVRGB_OUT_ROOT": tmp, "UAVRGB_NAME": "unit_check"})
            self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
            self.assertIn("--check 预检全部通过", res.stdout)
            self.assertIn("总预计耗时", res.stdout)

    def test_unknown_uavrgb_env_is_rejected_with_suggestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run_entry(
                "--check",
                extra_env={"UAVRGB_OUT_ROOT": tmp, "UAVRGB_S1_BATCh": "16"},
            )
            self.assertNotEqual(res.returncode, 0)
            combined = res.stdout + res.stderr
            self.assertIn("UAVRGB_S1_BATCh", combined)
            self.assertIn("UAVRGB_S1_BATCH", combined)

    def test_completed_run_dir_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "done_run"
            run_dir.mkdir()
            (run_dir / "DONE").write_text("rc=0 finished_at=2026-11-17T00:00:00+0800\n")
            res = self._run_entry("--check", extra_env={"UAVRGB_OUT_ROOT": tmp, "UAVRGB_NAME": "done_run"})
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("拒绝启动：目标目录已完成", res.stdout + res.stderr)

            res_ok = self._run_entry(
                "--check",
                extra_env={"UAVRGB_OUT_ROOT": tmp, "UAVRGB_NAME": "done_run", "UAVRGB_OVERWRITE": "1"},
            )
            self.assertEqual(res_ok.returncode, 0, msg=res_ok.stderr + res_ok.stdout)

    def test_presets_set_expected_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for preset in ("vln_3dgs", "s1_direct", "s1_subgoal", "g3_vlm", "openfly_eval", "openfly_v2", "full_all"):
                res = self._run_entry(preset, "--check", extra_env={"UAVRGB_OUT_ROOT": tmp})
                self.assertEqual(res.returncode, 0, msg=f"{preset}: {res.stderr}\n{res.stdout}")
                self.assertIn(f"模式={preset}", res.stdout)

    def test_rivermind_fs_output_is_rejected(self):
        res = self._run_entry("--check", extra_env={"UAVRGB_OUT_ROOT": "/root/rivermind-fs/tmp"})
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("rivermind-fs", res.stdout + res.stderr)

    def test_video_gallery_parses_openfly_episode_fields(self):
        sys.path.insert(0, str(REPO_ROOT))
        import train
        with tempfile.TemporaryDirectory() as tmp:
            rdir = Path(tmp)
            ep_dir = rdir / "openfly_v2/closedloop/seen20_test/episode_000"
            ep_dir.mkdir(parents=True)
            (ep_dir / "decisions.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
            sdoc = {
                "episodes": [{
                    "output": str(ep_dir),
                    "route_id": "env_airsim_18/astar_data/r0",
                    "termination": "predicted_stop",
                    "final_goal_distance_m": 14.25,
                    "stopped_within_20m_diagnostic": True,
                    "within_20m_diagnostic": True,
                    "executed_actions": 6,
                    "supplied_instruction": "fly straight",
                }]
            }
            (ep_dir.parent / "summary.json").write_text(__import__("json").dumps(sdoc))
            res = train.build_video_gallery(rdir)
            self.assertEqual(res["total_videos"], 1)
            html_txt = Path(res["index_html"]).read_text(encoding="utf-8")
            self.assertIn("SUCCESS", html_txt)
            self.assertIn("NE=14.25m", html_txt)
            self.assertIn("steps=6", html_txt)

if __name__ == "__main__":
    unittest.main()
