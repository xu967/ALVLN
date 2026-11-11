#!/usr/bin/env python3
"""tools/make_params_variant.py 的测试（任务卡 W1-10）。不起 Gazebo，不需要 GPU。

覆盖：
  点路径：`sim.battery_soc_init` 自动穿过 {"value": ...} 这一层，写 `sim.value.battery_soc_init` 也行；找不到路径、类型不符、
          非 JSON 的新值都报错（打错键名不能静默造出一个没人读的新键）。
  指纹：  与 cf_sitl_exec.params_values_sha256 是同一个函数的结果；满电副本 = 卡里写的 06e5f017，原文件 = 1bc8fedf；
          没有 --set 的副本指纹与原文件相同（重新序列化不改 value）；只改 unit / basis 这类注释，指纹不变。
  副本：  与原文件逐项比较只差指定的路径；原文件字节不动；cf_sitl_exec.CfSitlParams.from_json 能读；
          拒绝覆盖原文件、拒绝覆盖已有输出（除非 --force）。
运行（仓库根）：.venv-tools/bin/python tools/test_make_params_variant.py
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import cf_sitl_exec  # noqa: E402
import make_params_variant as mpv  # noqa: E402

SRC = os.path.join(REPO, "sim", "dynamics", "e1_mini.json")
SCRIPT = os.path.join(HERE, "make_params_variant.py")


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def leaf_diffs(a, b, prefix=""):
    """两个 JSON 对象逐叶比较：[(路径, a 的值, b 的值)]。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{prefix}{k}", a.get(k, "<缺>"), b.get(k, "<缺>")))
            else:
                out += leaf_diffs(a[k], b[k], f"{prefix}{k}.")
        return out
    return [] if a == b else [(prefix[:-1], a, b)]


class SetPath(unittest.TestCase):
    def test_shorthand_steps_through_value(self):
        obj = load(SRC)
        new, changes = mpv.apply_sets(obj, ["sim.battery_soc_init=1.0"])
        self.assertEqual(new["sim"]["value"]["battery_soc_init"], 1.0)
        self.assertEqual(changes, [("sim.value.battery_soc_init", 0.5, 1.0)])
        self.assertEqual(obj["sim"]["value"]["battery_soc_init"], 0.5, "输入对象不能被改")

    def test_explicit_value_path_is_the_same(self):
        obj = load(SRC)
        a, _ = mpv.apply_sets(obj, ["sim.battery_soc_init=1.0"])
        b, _ = mpv.apply_sets(obj, ["sim.value.battery_soc_init=1.0"])
        self.assertEqual(a, b)

    def test_scalar_entry_sets_its_value(self):
        obj = load(SRC)
        new, changes = mpv.apply_sets(obj, ["battery_capacity=1.0"])
        self.assertEqual(new["battery_capacity"]["value"], 1.0)
        self.assertEqual(changes[0][0], "battery_capacity.value")
        self.assertEqual(new["battery_capacity"]["unit"], obj["battery_capacity"]["unit"], "注释项不动")

    def test_nested_value_key_is_reached_through_value(self):
        obj = load(SRC)
        new, changes = mpv.apply_sets(obj, ["drag_linear.xy=0.05"])
        self.assertEqual(new["drag_linear"]["value"], {"xy": 0.05, "z": 0.02})
        self.assertEqual(changes, [("drag_linear.value.xy", 0.04, 0.05)])

    def test_list_value_keeps_its_length(self):
        obj = load(SRC)
        new, _ = mpv.apply_sets(obj, ["cg_offset=[0.0, 0.0, 0.0]"])
        self.assertEqual(new["cg_offset"]["value"], [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            mpv.apply_sets(obj, ["cg_offset=[0.0, 0.0]"])
        with self.assertRaises(ValueError):
            mpv.apply_sets(obj, ['cg_offset=[0.0, 0.0, "x"]'])

    def test_int_for_float_is_stored_as_float(self):
        """json 里 1 与 1.0 的指纹不同；原值是浮点时按浮点存，免得同一个数得到两个指纹。"""
        obj = load(SRC)
        a, _ = mpv.apply_sets(obj, ["sim.battery_soc_init=1"])
        b, _ = mpv.apply_sets(obj, ["sim.battery_soc_init=1.0"])
        self.assertIsInstance(a["sim"]["value"]["battery_soc_init"], float)
        self.assertEqual(mpv.values_sha256(a), mpv.values_sha256(b))
        self.assertEqual(mpv.values_sha256(a)[:8], "06e5f017")
        c, _ = mpv.apply_sets(obj, ["battery_c_rating=30"])
        self.assertIsInstance(c["battery_c_rating"]["value"], int, "原值是整数就保持整数")

    def test_object_target_rejected(self):
        for bad in ("drag_linear=1", "sim=1", 'sim={"battery_soc_init": 1.0}'):
            with self.assertRaises(ValueError, msg=bad):
                mpv.apply_sets(load(SRC), [bad])

    def test_multiple_sets_apply_in_order_and_report_each(self):
        new, changes = mpv.apply_sets(load(SRC), ["sim.battery_soc_init=0.8", "sim.battery_soc_init=1.0", "battery_capacity=0.6"])
        self.assertEqual(new["sim"]["value"]["battery_soc_init"], 1.0)
        self.assertEqual([c[0] for c in changes], ["sim.value.battery_soc_init"] * 2 + ["battery_capacity.value"])
        self.assertEqual(changes[1][1:], (0.8, 1.0))

    def test_unknown_path_rejected(self):
        for bad in ("sim.battery_soc_ini=1.0", "no_such_key=1", "sim.value.nope.deeper=1"):
            with self.assertRaises(ValueError, msg=bad):
                mpv.apply_sets(load(SRC), [bad])

    def test_type_mismatch_rejected(self):
        for bad in ('sim.battery_soc_init="full"', "sim.battery_soc_init=true", "sim.battery_soc_init=[1]",
                    'battery_capacity=null'):
            with self.assertRaises(ValueError, msg=bad):
                mpv.apply_sets(load(SRC), [bad])

    def test_value_must_be_json(self):
        for bad in ("sim.battery_soc_init=abc", "sim.battery_soc_init=", "sim.battery_soc_init=1.0.0", "sim.battery_soc_init"):
            with self.assertRaises(ValueError, msg=bad):
                mpv.apply_sets(load(SRC), [bad])

    def test_non_finite_number_rejected(self):
        for bad in ("sim.battery_soc_init=NaN", "sim.battery_soc_init=Infinity"):
            with self.assertRaises(ValueError, msg=bad):
                mpv.apply_sets(load(SRC), [bad])


class Fingerprint(unittest.TestCase):
    def test_source_and_full_charge_fingerprints_match_the_card(self):
        obj = load(SRC)
        self.assertEqual(mpv.values_sha256(obj)[:8], "1bc8fedf")
        new, _ = mpv.apply_sets(obj, ["sim.battery_soc_init=1.0"])
        self.assertEqual(mpv.values_sha256(new)[:8], "06e5f017")

    def test_same_function_as_the_collector_uses(self):
        obj = load(SRC)
        self.assertEqual(mpv.values_sha256(obj), cf_sitl_exec.params_values_sha256(obj))

    def test_profile_string_has_lib_version(self):
        self.assertEqual(mpv.profile_of(load(SRC), 31), "cf_sitl_v1:1bc8fedf:lib31")
        self.assertEqual(mpv.profile_of(load(SRC), None), "cf_sitl_v1:1bc8fedf:lib?")

    def test_lib_version_read_from_header(self):
        self.assertEqual(mpv.read_lib_version(), 31)

    def test_comment_only_edit_keeps_fingerprint(self):
        obj = load(SRC)
        edited = copy.deepcopy(obj)
        edited["battery_capacity"]["basis"] = "改一句注释"
        edited["provenance"] = "改一句来历"
        self.assertEqual(mpv.values_sha256(edited), mpv.values_sha256(obj))

    def test_value_edit_changes_fingerprint(self):
        obj = load(SRC)
        for sets in (["sim.battery_soc_init=0.8"], ["battery_capacity=1.0"]):
            new, _ = mpv.apply_sets(obj, sets)
            self.assertNotEqual(mpv.values_sha256(new), mpv.values_sha256(obj), sets)


class WriteVariant(unittest.TestCase):
    def test_variant_differs_only_at_the_set_path(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "soc100.json")
            info = mpv.make_variant(SRC, out, ["sim.battery_soc_init=1.0"])
            self.assertEqual(leaf_diffs(load(SRC), load(out)), [("sim.value.battery_soc_init", 0.5, 1.0)])
            self.assertEqual(info["profile"], "cf_sitl_v1:06e5f017:lib31")
            self.assertEqual(info["src_profile"], "cf_sitl_v1:1bc8fedf:lib31")
            self.assertEqual(info["changes"], [("sim.value.battery_soc_init", 0.5, 1.0)])

    def test_no_set_copy_has_the_source_fingerprint_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "copy.json")
            info = mpv.make_variant(SRC, out, [])
            self.assertEqual(info["values_sha256"], info["src_values_sha256"])
            self.assertEqual(info["sha256"], info["src_sha256"])
            self.assertEqual(read_bytes(out), read_bytes(SRC), "没有改动：逐字节相同")

    def test_variant_differs_from_source_by_one_line_and_one_token(self):
        """副本是源文件的文本原样复制，只换被改的值：diff 只有一行，且那一行只差 0.5 → 1.0。"""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "soc100.json")
            mpv.make_variant(SRC, out, ["sim.battery_soc_init=1.0"])
            a = read_bytes(SRC).decode("utf-8").splitlines(keepends=True)
            b = read_bytes(out).decode("utf-8").splitlines(keepends=True)
            self.assertEqual(len(a), len(b))
            diff = [(x, y) for x, y in zip(a, b) if x != y]
            self.assertEqual(len(diff), 1)
            self.assertEqual(diff[0][0].replace('"battery_soc_init": 0.5', '"battery_soc_init": 1.0'), diff[0][1])
            self.assertIn('"battery_soc_init": 1.0', diff[0][1])

    def test_output_is_world_readable(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "soc100.json")
            mpv.make_variant(SRC, out, ["sim.battery_soc_init=1.0"])
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o644)

    def test_collector_can_read_the_variant(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "soc100.json")
            info = mpv.make_variant(SRC, out, ["sim.battery_soc_init=1.0"])
            p = cf_sitl_exec.CfSitlParams.from_json(out)
            self.assertEqual(p.values_sha256, info["values_sha256"])
            self.assertEqual(p.sha256, info["sha256"], "输出里报的文件字节 sha 与读回来的一致")
            self.assertEqual(p.profile(), "cf_sitl_v1:06e5f017:lib?")

    def test_source_bytes_untouched(self):
        before = hashlib.sha256(read_bytes(SRC)).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            mpv.make_variant(SRC, os.path.join(d, "v.json"), ["sim.battery_soc_init=1.0"])
        self.assertEqual(hashlib.sha256(read_bytes(SRC)).hexdigest(), before)

    def test_refuses_to_overwrite_source_or_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.json")
            with open(src, "wb") as f:
                f.write(read_bytes(SRC))
            with self.assertRaises(ValueError):
                mpv.make_variant(src, src, [])
            link = os.path.join(d, "link.json")
            os.symlink(src, link)
            with self.assertRaises(ValueError, msg="符号链接指回原文件也不行"):
                mpv.make_variant(src, link, [])
            out = os.path.join(d, "out.json")
            mpv.make_variant(src, out, ["sim.battery_soc_init=1.0"])
            with self.assertRaises(ValueError):
                mpv.make_variant(src, out, ["sim.battery_soc_init=0.9"])
            self.assertEqual(load(out)["sim"]["value"]["battery_soc_init"], 1.0, "拒绝时输出不被改")
            mpv.make_variant(src, out, ["sim.battery_soc_init=0.9"], force=True)
            self.assertEqual(load(out)["sim"]["value"]["battery_soc_init"], 0.9)

    def test_failed_set_leaves_no_output_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "bad.json")
            with self.assertRaises(ValueError):
                mpv.make_variant(SRC, out, ["sim.nope=1"])
            self.assertFalse(os.path.exists(out))
            self.assertEqual(os.listdir(d), [], "也不留临时文件")


class Splice(unittest.TestCase):
    """文本级替换：只动被改的值，字符串里的引号、括号、逗号、冒号、转义不会让定位错位。"""
    TEXT = (
        '{"schema": "x", "a": {"value": 1.5, "basis": "含 \\"引号\\" 与 {大括号} [方括号], 逗号: 冒号 \\\\"},\n'
        ' "b": {"value": [[1.0, 2.0],\n          [3.0, 4.0]], "unit": "m"},\n'
        ' "c": {"value": {"p": 1, "q": {"deep": 2.5}}, "tags": ["{", "}", "\\"", ","]},\n'
        ' "d": {"value": true},\n'
        ' "e": {"value": "s,}"},\n'
        ' "f": {"value": 7}}\n')

    def spans_text(self, path):
        spans = mpv.value_spans(self.TEXT)
        s, e = spans[tuple(path)]
        return self.TEXT[s:e]

    def test_spans_cover_exactly_the_value_text(self):
        self.assertEqual(self.spans_text(["a", "value"]), "1.5")
        self.assertEqual(self.spans_text(["b", "value"]), "[[1.0, 2.0],\n          [3.0, 4.0]]")
        self.assertEqual(self.spans_text(["c", "value", "q", "deep"]), "2.5")
        self.assertEqual(self.spans_text(["c", "value", "q"]), '{"deep": 2.5}')
        self.assertEqual(self.spans_text(["d", "value"]), "true")
        self.assertEqual(self.spans_text(["e", "value"]), '"s,}"')
        self.assertEqual(self.spans_text(["f", "value"]), "7")
        self.assertEqual(self.spans_text(["c", "tags"]), '["{", "}", "\\"", ","]')

    def test_splice_changes_only_the_targets(self):
        obj = json.loads(self.TEXT)
        for sets, replaced in ((["a=2.5"], [("1.5", "2.5")]),
                               (["c.q.deep=3.5", "f=8"], [("2.5", "3.5"), ('"f": {"value": 7}', '"f": {"value": 8}')]),
                               (["b=[[1.0,2.0],[3.0,9.0]]"], [("[[1.0, 2.0],\n          [3.0, 4.0]]", "[[1.0, 2.0], [3.0, 9.0]]")]),
                               (["e=\"t\""], [('"s,}"', '"t"')]),
                               (["d=false"], [("true", "false")])):
            new, trail_changes = mpv._apply(obj, sets)
            got = mpv.splice_values(self.TEXT, obj, trail_changes)
            expect = self.TEXT
            for a, b in replaced:
                self.assertEqual(expect.count(a), 1, (sets, a))
                expect = expect.replace(a, b)
            self.assertEqual(got, expect, sets)
            self.assertEqual(json.loads(got), new, sets)

    def test_same_value_and_repeated_path(self):
        obj = json.loads(self.TEXT)
        _, ch = mpv._apply(obj, ["a=1.5"])
        self.assertEqual(mpv.splice_values(self.TEXT, obj, ch), self.TEXT, "写回原值：文本不动")
        new, ch = mpv._apply(obj, ["a=2.5", "a=3.5"])
        self.assertEqual(mpv.splice_values(self.TEXT, obj, ch), self.TEXT.replace("1.5", "3.5"), "同一路径写两次取最后一次")
        _, ch = mpv._apply(obj, ["a=2.5", "a=1.5"])
        self.assertEqual(mpv.splice_values(self.TEXT, obj, ch), self.TEXT, "改了又改回：文本不动")

    def test_every_member_of_the_real_file_is_located(self):
        text = read_bytes(SRC).decode("utf-8")
        obj = json.loads(text)
        spans = mpv.value_spans(text)
        n = 0

        def walk(node, path):
            nonlocal n
            if isinstance(node, dict):
                for k, v in node.items():
                    s, e = spans[path + (k,)]
                    self.assertEqual(json.loads(text[s:e]), v, path + (k,))
                    n += 1
                    walk(v, path + (k,))
        walk(obj, ())
        self.assertGreater(n, 300, "e1_mini.json 的对象成员数")


class Cli(unittest.TestCase):
    def run_cli(self, *argv):
        return subprocess.run([sys.executable, SCRIPT, *argv], capture_output=True, text=True, timeout=60)

    def test_cli_prints_new_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "soc100.json")
            r = self.run_cli("--out", out, "--set", "sim.battery_soc_init=1.0")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("cf_sitl_v1:06e5f017:lib31", r.stdout)
            self.assertIn("cf_sitl_v1:1bc8fedf:lib31", r.stdout)
            self.assertIn("sim.value.battery_soc_init: 0.5 -> 1.0", r.stdout)
            self.assertEqual(load(out)["sim"]["value"]["battery_soc_init"], 1.0)

    def test_cli_errors_exit_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_cli("--out", os.path.join(d, "x.json"), "--set", "sim.nope=1")
            self.assertEqual(r.returncode, 2)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("sim.nope", r.stderr)
            r = self.run_cli("--set", "sim.battery_soc_init=1.0")
            self.assertEqual(r.returncode, 2, "缺 --out")

    def test_cli_self_check(self):
        r = self.run_cli("--self-check")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_self_check_function_is_quiet_on_success(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(mpv.self_check(), 0)
        self.assertIn("PASS", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
