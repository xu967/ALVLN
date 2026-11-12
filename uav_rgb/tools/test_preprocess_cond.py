#!/usr/bin/env python3
"""preprocess --cond {prev-action,heading-code} 的回归（任务卡 W1-9；纯 CPU，临时目录，不读真实数据）。

同一批 episode 两种 --cond 打包：图像与标签逐字节相同，meta 只差 created_utc 与 cond，侧车只有 prev_action 列不同；
heading-code 的列 = 按集从头累计「已执行」动作得到的航向码（含被 --drop-collision 丢掉的碰撞步；DAgger 的执行动作 ≠ 教师标签）；
默认值（prev-action）的列与 meta 不变；waypoint 任务拒绝 heading-code（H 未定义）。

运行（仓库根）：.venv-tools/bin/python tools/test_preprocess_cond.py

T2a（审核 P-10）：三集夹具（合成 episode + 每集的 actions.jsonl）在 setUpModule 造一次（TMPDIR 下，模块结束清掉），
用例各拷一份再改，每个用例结束核对夹具内容哈希没变。打包默认进程内调 preprocess.main()；两个用例走真子进程，锁命令行契约：
test_unknown_cond_is_rejected_by_argparse 锁 argparse 报错的退出码 2 与 stderr 里的 --cond，
test_waypoint_episode_is_refused_for_heading_code_only 的第一次打包锁 die() 的退出码非 0 与 stderr 里的 waypoint / heading-code。
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "tools"))
sys.path.insert(0, str(REPO / "tools"))
import nav_cond  # noqa: E402
from preprocess_testlib import SharedFixture, make_synthetic, run_preprocess, run_preprocess_cli  # noqa: E402

AUX_HEADER_SIZE = 128
FIXTURE = None  # setUpModule 里造；用例只读，要改先 copy_to
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


def build_fixture(root):
    make_synthetic(root)
    for name, (executed, expert, collided) in EPISODES.items():
        rows = [{"step": i, "timestamp_ns": 1_000_000_000 + 100_000_000 * i, "action_expert": expert[i],
                 "action_executed": executed[i], "collision": i in collided} for i in range(len(executed))]
        (root / name / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def setUpModule():
    global FIXTURE
    FIXTURE = SharedFixture("cond", build_fixture)
    unittest.addModuleCleanup(FIXTURE.cleanup)


def tearDownModule():
    FIXTURE.verify()


class PreprocessCondTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        FIXTURE.copy_to(self.tmp / "ep")
        self.eps = {n: self.tmp / "ep" / n for n in EPISODES}

    def tearDown(self):
        FIXTURE.verify()

    def pack(self, name, *flags, eps=None, check=True, cli=False):
        out = self.tmp / name
        run = run_preprocess_cli if cli else run_preprocess
        r = run("--out", out, *flags, *(eps if eps is not None else self.eps.values()))
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
        _, r = self.pack("hc", "--cond", "heading-code", check=False, cli=True)  # 真子进程：die() 的退出码与 stderr 关键字
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("waypoint", r.stderr)
        self.assertIn("heading-code", r.stderr)
        self.pack("default", "--drop-collision")   # 默认路径不受影响

    def test_unknown_cond_is_rejected_by_argparse(self):
        _, r = self.pack("bad", "--cond", "bogus", check=False, cli=True)  # 真子进程：argparse 报错的退出码 2
        self.assertEqual(r.returncode, 2)
        self.assertIn("--cond", r.stderr)

    def test_invalid_action_executed_on_any_step_is_rejected_with_file_and_line(self):
        # I2-6b / TEST 2e：无论跳号前的步（step=0,2 里的 step=0）还是集末步（step=5），
        # 每行自身的 action_executed 都必须直接校验为 0..3 的整数，报错指明文件与行号。
        ep1 = self.eps["ep1"]
        act_path = ep1 / "actions.jsonl"
        man_path = ep1 / "episode_manifest.json"
        orig_man = json.loads(man_path.read_text(encoding="utf-8"))
        orig_act = act_path.read_text(encoding="utf-8")

        # 1) 跳号步 step = 0, 2 且 step=0 的 action_executed 非法
        man = dict(orig_man, num_steps=2, instruction=None)
        man_path.write_text(json.dumps(man), encoding="utf-8")
        for bad in ("BAD_STR", 99, -1, True, None, 1.5):
            act_path.write_text(
                json.dumps({"step": 0, "action_expert": 1, "action_executed": bad, "collision": False}) + "\n" +
                json.dumps({"step": 2, "action_expert": 1, "action_executed": 2, "collision": False}) + "\n",
                encoding="utf-8",
            )
            for flags in (("--cond", "heading-code"), ("--cond", "prev-action")):
                _, r = self.pack("bad_gap", *flags, eps=[ep1], check=False)
                self.assertNotEqual(r.returncode, 0, (bad, flags))
                self.assertIn(str(act_path), r.stderr)
                self.assertIn("第 1 行", r.stderr)
                self.assertIn("action_executed", r.stderr)

        # 2) 连续步集末行（第 6 行，step=5）的 action_executed 非法：旧代码因无人查 step=5 作为上一步而漏检
        man_path.write_text(json.dumps(orig_man), encoding="utf-8")
        rows = [json.loads(ln) for ln in orig_act.splitlines() if ln.strip()]
        for bad in ("BAD_LAST", 99, True):
            bad_rows = [dict(x) for x in rows]
            bad_rows[-1]["action_executed"] = bad
            act_path.write_text("".join(json.dumps(x) + "\n" for x in bad_rows), encoding="utf-8")
            for flags in (("--cond", "prev-action"), ("--cond", "heading-code")):
                _, r = self.pack("bad_last", *flags, check=False)
                self.assertNotEqual(r.returncode, 0, (bad, flags))
                self.assertIn(str(act_path), r.stderr)
                self.assertIn("第 6 行", r.stderr)

    def test_step_gaps_in_source_actions_jsonl_reset_both_conditions_uniformly(self):
        # I2-6b：输入 actions.jsonl 自身步号有缺口（缺 step=2，只存 step=0,1,3,4,5）时，
        # 两种条件统一在缺口处断链重置为 0，随后在新连续段内重新传递/累计。
        from PIL import Image
        import numpy as np

        ep1 = self.eps["ep1"]
        man_path = ep1 / "episode_manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["num_steps"] = 5
        man_path.write_text(json.dumps(man), encoding="utf-8")
        # 补一张 rgb_00005.png（原 ep1 有 0..5 共 6 张图，已齐）
        assert (ep1 / "rgb_00005.png").is_file()
        steps_exec = [(0, L), (1, L), (3, L), (4, L), (5, F)]
        rows = [{"step": st, "timestamp_ns": 1_000_000_000 + 100_000_000 * st,
                 "action_expert": F, "action_executed": ex, "collision": False}
                for st, ex in steps_exec]
        (ep1 / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

        out_pa, _ = self.pack("gap_pa", "--cond", "prev-action")
        out_hc, _ = self.pack("gap_hc", "--cond", "heading-code")
        idx_pa, col_pa = read_aux(out_pa / "dataset.aux")
        idx_hc, col_hc = read_aux(out_hc / "dataset.aux")
        self.assertEqual(idx_pa[:5], [0, 0, 2, 2, 3])
        self.assertEqual(idx_hc[:5], [0, 0, 2, 2, 3])
        # 段 1 (step 0,1): prev=[0, L], hc=[0, 0]；缺口后段 2 (step 3,4,5): prev=[0, L, L], hc=[0, 0, 2]
        self.assertEqual(self.per_episode(out_pa, col_pa)["ep1"], [0, L, 0, L, L])
        self.assertEqual(self.per_episode(out_hc, col_hc)["ep1"], [0, 0, 0, 0, 2])


if __name__ == "__main__":
    unittest.main()
