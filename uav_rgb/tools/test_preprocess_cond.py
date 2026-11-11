#!/usr/bin/env python3
"""preprocess --cond {prev-action,heading-code} 的回归（任务卡 W1-9；纯 CPU，临时目录，不读真实数据）。

同一批 episode 两种 --cond 打包：图像与标签逐字节相同，meta 只差 created_utc 与 cond，侧车只有 prev_action 列不同；
heading-code 的列 = 按集从头累计「已执行」动作得到的航向码（含被 --drop-collision 丢掉的碰撞步；DAgger 的执行动作 ≠ 教师标签）；
默认值（prev-action）的列与 meta 不变；waypoint 任务拒绝 heading-code（H 未定义）。

运行（仓库根）：.venv-tools/bin/python tools/test_preprocess_cond.py
"""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "tools"))
import nav_cond  # noqa: E402

PY = sys.executable
AUX_HEADER_SIZE = 128
F, L, R = 1, 2, 3

# 每集：(执行动作, 教师标签, 碰撞步集合)。标签故意与执行动作不同（DAgger 形态）：码按执行动作算，标签列照教师。
EPISODES = {
    "ep1": ([F, L, L, F, R, L], [3, 1, 0, 2, 2, 1], {2}),   # 第 2 步是转向且碰撞：丢样本不能丢计数
    "ep2": ([L, L, L, R], [2, 2, 2, 3], set()),
    "ep3": ([R, R, R, F, L], [1, 1, 0, 0, 3], set()),
}
# 手算：每步执行前的 n → 码（n≤1 → 0；n≥2 → 2；n≤−2 → 3）
CODES = {"ep1": [0, 0, 0, 2, 2, 0], "ep2": [0, 0, 2, 2], "ep3": [0, 0, 3, 3, 3]}
# 手算：上一步执行的动作（首帧 0）
PREV = {"ep1": [0, F, L, L, F, R], "ep2": [0, L, L, L], "ep3": [0, R, R, R, F]}


def read_uavr(path):
    """-> (图像+标签区的原始字节, meta dict)。"""
    raw = Path(path).read_bytes()
    meta_off = struct.unpack_from("<Q", raw, 36)[0]
    meta_len = struct.unpack_from("<I", raw, 44)[0]
    return raw[48:meta_off], json.loads(raw[meta_off:meta_off + meta_len])


def read_aux(path):
    """-> (prev_index, prev_action)；这批夹具没有深度，载荷 = u32[N] + u8[N]。"""
    raw = Path(path).read_bytes()
    n = struct.unpack_from("<Q", raw, 8)[0]
    body = raw[AUX_HEADER_SIZE:]
    assert len(body) == 5 * n, "夹具不该带深度分带"
    return list(struct.unpack("<%dI" % n, body[:4 * n])), list(body[4 * n:])


class PreprocessCondTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"), "--out", str(self.tmp / "ep")],
                       check=True, capture_output=True)
        self.eps = {n: self.tmp / "ep" / n for n in EPISODES}
        for name, (executed, expert, collided) in EPISODES.items():
            rows = [{"step": i, "timestamp_ns": 1_000_000_000 + 100_000_000 * i, "action_expert": expert[i],
                     "action_executed": executed[i], "collision": i in collided} for i in range(len(executed))]
            (self.eps[name] / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def pack(self, name, *flags, eps=None, check=True):
        out = self.tmp / name
        r = subprocess.run([PY, str(REPO / "tools/preprocess.py"), "--out", str(out), *flags,
                            *map(str, eps if eps is not None else self.eps.values())],
                           capture_output=True, text=True)
        if check:
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return out, r

    def per_episode(self, out, column):
        """把侧车某一列按 meta 的 sample_start/sample_count 切回每集；键 = 夹具集名（ep1…）。"""
        _, meta = read_uavr(out / "dataset.uavr")
        by_id = {1: "ep1", 2: "ep2", 3: "ep3"}
        return {by_id[e["episode_id"]]: column[e["sample_start"]:e["sample_start"] + e["sample_count"]]
                for e in meta["episodes"]}

    @staticmethod
    def kept(values, name):
        """去掉被 --drop-collision 丢掉的步。"""
        collided = EPISODES[name][2]
        return [v for i, v in enumerate(values) if i not in collided]

    def test_heading_code_column_counts_executed_actions_including_dropped_steps(self):
        out, _ = self.pack("hc", "--drop-collision", "--cond", "heading-code")
        _, column = read_aux(out / "dataset.aux")
        got = self.per_episode(out, column)
        for name in EPISODES:
            self.assertEqual(got[name], self.kept(CODES[name], name), name)
        # ep1 第 3 步的码是 2：丢掉的第 2 步是转向，只数留下来的样本会得到 0
        self.assertEqual(got["ep1"], [0, 0, 2, 2, 0])

    def test_codes_equal_nav_cond_on_the_executed_chain(self):
        # 夹具的手算表与 nav_cond 是同一件事：两边对不上说明测试本身写错了
        for name, (executed, _expert, _c) in EPISODES.items():
            self.assertEqual(nav_cond.heading_codes(executed), CODES[name], name)

    def test_default_column_is_still_the_previous_executed_action(self):
        out, _ = self.pack("default", "--drop-collision")
        _, column = read_aux(out / "dataset.aux")
        got = self.per_episode(out, column)
        for name in EPISODES:
            self.assertEqual(got[name], self.kept(PREV[name], name), name)

    def test_explicit_prev_action_equals_omitted_flag(self):
        a, _ = self.pack("omitted", "--drop-collision")
        b, _ = self.pack("explicit", "--drop-collision", "--cond", "prev-action")
        self.assertEqual(read_aux(a / "dataset.aux"), read_aux(b / "dataset.aux"))
        (ia, ma), (ib, mb) = read_uavr(a / "dataset.uavr"), read_uavr(b / "dataset.uavr")
        ma.pop("created_utc")
        mb.pop("created_utc")
        self.assertEqual((ia, ma), (ib, mb))

    def test_two_conds_differ_only_in_the_condition_column(self):
        a, _ = self.pack("pa", "--drop-collision", "--cond", "prev-action")
        b, _ = self.pack("hc", "--drop-collision", "--cond", "heading-code")
        (img_a, meta_a), (img_b, meta_b) = read_uavr(a / "dataset.uavr"), read_uavr(b / "dataset.uavr")
        self.assertEqual(img_a, img_b, "图像与标签区逐字节相同")
        for m in (meta_a, meta_b):
            m.pop("created_utc")
        self.assertEqual(meta_b.pop("cond"), "heading_code.v1")
        self.assertNotIn("cond", meta_a)
        self.assertEqual(meta_a, meta_b, "meta 只差 cond（与时间戳）")
        (idx_a, act_a), (idx_b, act_b) = read_aux(a / "dataset.aux"), read_aux(b / "dataset.aux")
        self.assertEqual(idx_a, idx_b, "prev_index 不变")
        self.assertNotEqual(act_a, act_b, "条件列不同")
        self.assertEqual(len(act_a), len(act_b))

    def test_labels_stay_the_teacher_labels_not_the_executed_actions(self):
        out, _ = self.pack("hc", "--cond", "heading-code")   # 不丢碰撞步
        raw = (out / "dataset.uavr").read_bytes()
        n = struct.unpack_from("<I", raw, 8)[0]
        meta_off = struct.unpack_from("<Q", raw, 36)[0]
        labels = list(raw[meta_off - n:meta_off])
        _, meta = read_uavr(out / "dataset.uavr")
        by_id = {1: "ep1", 2: "ep2", 3: "ep3"}
        for e in meta["episodes"]:
            name = by_id[e["episode_id"]]
            self.assertEqual(labels[e["sample_start"]:e["sample_start"] + e["sample_count"]], EPISODES[name][1], name)

    def test_without_drop_collision_every_step_keeps_its_code(self):
        out, _ = self.pack("hc", "--cond", "heading-code")
        _, column = read_aux(out / "dataset.aux")
        self.assertEqual(self.per_episode(out, column), CODES)

    def test_waypoint_episode_is_refused_for_heading_code_only(self):
        man_path = self.eps["ep1"] / "episode_manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["waypoint"] = {"x": 3.0, "y": 0.5}
        man_path.write_text(json.dumps(man), encoding="utf-8")
        _, r = self.pack("hc", "--cond", "heading-code", check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("waypoint", r.stderr)
        self.assertIn("heading-code", r.stderr)
        self.pack("default", "--drop-collision")   # 默认路径不受影响

    def test_unknown_cond_is_rejected_by_argparse(self):
        _, r = self.pack("bad", "--cond", "bogus", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--cond", r.stderr)


if __name__ == "__main__":
    unittest.main()
