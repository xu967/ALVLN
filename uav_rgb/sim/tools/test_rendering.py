#!/usr/bin/env python3
"""渲染路径与性能统计回归；不依赖显卡或显示服务器。"""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import collect
import collect_parallel as cp
import test_collect as fixtures


class RenderingTests(unittest.TestCase):
    def test_gpu_clears_software_overrides_and_display(self):
        with patch.dict(os.environ, {"DISPLAY": ":20", "LIBGL_ALWAYS_SOFTWARE": "1",
                                    "LP_NUM_THREADS": "1", "GALLIUM_DRIVER": "llvmpipe"},
                        clear=True):
            env = cp.gz_env("isolated")
        self.assertEqual(env["UAVRGB_RENDER_BACKEND"], "gpu")
        self.assertEqual(env["GZ_PARTITION"], "isolated")
        self.assertTrue(env["__EGL_VENDOR_LIBRARY_FILENAMES"].endswith("10_nvidia.json"))
        for key in ("DISPLAY", "LIBGL_ALWAYS_SOFTWARE", "LP_NUM_THREADS", "GALLIUM_DRIVER"):
            self.assertNotIn(key, env)

    def test_software_is_explicit_and_preserves_display(self):
        with patch.dict(os.environ, {"UAVRGB_RENDER_BACKEND": "software", "DISPLAY": ":42"},
                        clear=True):
            env = cp.gz_env("test")
        self.assertEqual(env["DISPLAY"], ":42")
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE", env)
        self.assertEqual(env["LP_NUM_THREADS"], "1")
        self.assertNotIn("__EGL_VENDOR_LIBRARY_FILENAMES", env)

    def test_invalid_backend_fails(self):
        with patch.dict(os.environ, {"UAVRGB_RENDER_BACKEND": "typo"}):
            with self.assertRaises(ValueError):
                cp.gz_env("test")

    def test_start_command_and_missing_driver(self):
        for mode in ("gpu", "software"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {"UAVRGB_RENDER_BACKEND": mode}), \
                 patch.object(cp.os.path, "isfile", return_value=True), \
                 patch.object(cp.subprocess, "Popen", return_value=Mock()) as popen, \
                 patch.object(cp, "_register"):
                cp._start_sim("world.sdf", "isolated", directory + "/sim.log")
                self.assertEqual("--headless-rendering" in popen.call_args.args[0], mode == "gpu")
                self.assertEqual(popen.call_args.kwargs["env"]["GZ_PARTITION"], "isolated")
        with patch.dict(os.environ, {"UAVRGB_RENDER_BACKEND": "gpu"}), \
             patch.object(cp.os.path, "isfile", return_value=False), \
             patch.object(cp.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "EGL"):
                cp._start_sim("world.sdf", "isolated", "/unused.log")
            popen.assert_not_called()

    def test_performance_uses_ratio_of_sums(self):
        rows = [dict(io=.1, window=wall, block=wall, tail=.01,
                     total=wall+.11, dispatch_sim_dt=1.) for wall in (1., 4.)]
        summary = collect.performance_summary(rows)
        self.assertEqual(summary["steps"], 2)
        self.assertAlmostEqual(summary["action_rtf"], .4)
        self.assertAlmostEqual(summary["mean_step_s"]["total"], 2.61)
        self.assertEqual(collect.performance_summary([]), {"steps": 0})

    def test_slow_inference_does_not_inflate_rtf_or_change_window_contract(self):
        for mode, delay in (("observation", .8), ("dispatch", .8), ("observation", 1.2)):
            with self.subTest(mode=mode, delay=delay):
                clock = [10.2]
                pose = fixtures.ClockPoseSub(fixtures.pose_msg(clock[0]))
                subs = {"rgb": fixtures.FakeSub(after=fixtures.image_msg(4, 6, 10)),
                        "depth": fixtures.FakeSub(frames=[(1, fixtures._fresh_depth(4, 6))]),
                        "pose": pose, "contact": fixtures.FakeSub()}

                def advance(seconds):
                    clock[0] += seconds
                    pose.current = fixtures.pose_msg(clock[0])

                class Student:
                    def act(self, msg, prev_action=None):   # 与 student.StudentPolicy.act 同签名（collect_one 传 prev_action）
                        advance(delay)
                        return 1, delay * 1000, [0., 1., 0., 0.]

                class Publisher(fixtures.FakePub):
                    def start(self, *args):
                        # 仅用来保证没有仿真 tick 的立即零速也有可测墙钟差。
                        self.t += 1e-9
                        return clock[0] + self.t

                with tempfile.TemporaryDirectory() as out, \
                     patch.object(collect, "write_rgb_png", return_value=(True, None)), \
                     patch.object(collect.time, "perf_counter", side_effect=lambda: clock[0]), \
                     patch.object(collect.time, "sleep", side_effect=advance):
                    row, error = collect.collect_one(out, 0, 1., "student", subs,
                                                     Publisher(), student=Student(),
                                                     window_from=mode)
                self.assertIsNone(error)
                summary = collect.performance_summary([row["timing"]])
                self.assertEqual(summary["rtf_basis"], "dispatch_pose_to_after_pose")
                if delay > 1.:
                    self.assertAlmostEqual(row["sim_dt"], delay)
                    self.assertIsNone(summary["action_rtf"])
                else:
                    self.assertAlmostEqual(row["sim_dt"], 1.)
                    self.assertAlmostEqual(summary["action_rtf"], 1., places=6)
                    self.assertAlmostEqual(summary["action_sim_s"],
                                           .2 if mode == "observation" else 1.)
                self.assertEqual(row["action_start_ns"],
                                 10_200_000_000 if mode == "observation" else 11_000_000_000)


if __name__ == "__main__":
    unittest.main()
