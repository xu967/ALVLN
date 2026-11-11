#!/usr/bin/env python3
"""run.sh 的静态核对（不执行入口）：头里的任务卡原文与任务卡逐行相同、变体表与任务卡 / sim/tools/test_rule_v2.py 的表一致、
配置区的数字与任务卡一致（hold、yaw_step、seed、K、路线份数、选变体与闸门的阈值）、bash 语法、formal 拒绝覆盖的名单。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_run_sh.py
"""
import importlib.util
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN_SH = os.path.join(HERE, "run.sh")
CARD = os.path.join(REPO, "docs", "briefs", "2026-11-10-phase12-task-cards.md")
sys.path.insert(0, HERE)
import variants as vr  # noqa: E402

with open(RUN_SH, encoding="utf-8") as f:
    SRC = f.read()
with open(CARD, encoding="utf-8") as f:
    CARD_LINES = f.read().split("\n")


def card_block():
    start = next(i for i, ln in enumerate(CARD_LINES) if ln.startswith("- 实验（入口 `experiments/2026-11-11-rule-v2-3dgs/`"))
    end = next(i for i, ln in enumerate(CARD_LINES) if ln.startswith("- 测试（先红后绿，带变异检查）") and i > start)
    return CARD_LINES[start:end]


def header_block():
    lines = SRC.split("\n")
    i = next(k for k, ln in enumerate(lines) if "任务卡原文（docs/briefs" in ln)
    j = next(k for k, ln in enumerate(lines) if ln.startswith("# ---------------- 任务卡原文结束"))
    out = []
    for ln in lines[i + 1:j]:
        assert ln.startswith("#"), ln
        out.append(ln[2:] if ln.startswith("# ") else "")
    return out


def conf(name):
    """配置区里 NAME=值 的值：支持 NAME=值、NAME=${NAME:-值}、NAME=${NAME:-"多行 / 带空格的值"}。"""
    m = re.search(rf'^{name}=\$\{{{name}:-"(.*?)"\}}', SRC, re.M | re.S)
    if m:
        return m.group(1)
    m = re.search(rf"^{name}=\$\{{{name}:-([^}}\s]*)\}}", SRC, re.M)
    if m:
        return m.group(1)
    m = re.search(rf"^{name}=([^\s#;]+)", SRC, re.M)
    if m:
        return m.group(1)
    raise KeyError(name)


class TestHeader(unittest.TestCase):
    def test_card_text_is_pasted_verbatim(self):
        card = [ln.rstrip() for ln in card_block()]
        head = [ln.rstrip() for ln in header_block()]
        self.assertEqual(head, card, "run.sh 头里的任务卡原文与任务卡 467–501 行逐行相同（改任务卡或改头都会让这条红）")
        self.assertEqual(len(card), 35)

    def test_card_block_contains_the_preregistration(self):
        text = "\n".join(header_block())
        for needle in ("预注册（写进入口脚本头，起跑前定死；主会话 11-11 定，不许改）", "高 0.10", "p < 0.05", "不高于 0.10，撞在前不高于 0.08", "低于 0.05，或主臂碰撞率高于 0.15",
                       "介于预测与证伪之间", "到 0.80", "不低于 0.93", "depth 臂只报告", "出生即碰撞的集单列"):
            self.assertIn(needle, text, needle)


class TestConfigZone(unittest.TestCase):
    def test_numbers_match_the_card(self):
        want = {"HOLD": "1.0", "YAW_STEP": "31.7", "V1_R_SAFE": "0.25", "V1_D_LOOK": "1.2", "SEED0_DEV": "9000", "SEED0_HOLDOUT": "5000", "SEED0_FORMAL40": "8000",
                "SEED0_OUTDOOR": "7000", "K_DEV": "1", "K_HOLDOUT": "3", "K_FORMAL40": "3", "K_OUTDOOR": "5", "DEV_NEW_PER_SCENE": "4", "HOLDOUT_PER_SCENE": "8",
                "MIN_HOLDOUT": "24", "MAX_COLLIDE_FIRST": "3", "TIE_SLACK": "1", "GATE_MIN_GAIN": "3", "JITTER_INDOOR": "0.2", "JITTER_OUTDOOR": "0.5",
                "LOOKAHEAD_INDOOR": "1.5", "LOOKAHEAD_OUTDOOR": "3.0", "STEP_CAP_INDOOR": "100", "GUARD_DEG": "45.0", "SIZE_LIMIT_MB": "1500", "NO_GATE": "0", "LANES": "3"}
        for k, v in want.items():
            self.assertEqual(conf(k), v, k)
        self.assertEqual(conf("DEPTH_MAP"), "v2e=v2a v2f=v2d")
        self.assertEqual(conf("GPU"), "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740")
        self.assertEqual(conf("ROUTES_OUTDOOR"), "0 1 2 3 4 5")
        self.assertEqual(conf("INDOOR_SCENES").split(), ["sage3d_841150", "sage3d_840865", "sage3d_841012", "sage3d_840139", "sage3d_839992"])
        self.assertEqual(conf("OUTDOOR_SCENES").split(), ["garden", "wanderland_1MlUucug", "bicycle", "callinstrasse"])

    def test_variant_table_matches_card_and_rule_tests(self):
        table = vr.parse_variants(conf("VARIANTS"))
        spec = importlib.util.spec_from_file_location("test_rule_v2", os.path.join(REPO, "sim", "tools", "test_rule_v2.py"))
        sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual([v["name"] for v in table], list(mod.VARIANTS), "名字与次序")
        for v in table:
            levels, order = mod.VARIANTS[v["name"]]
            self.assertEqual([tuple(x) for x in v["levels"]], [tuple(x) for x in levels], v["name"])
            self.assertEqual(v["order"], order, v["name"])
        # 任务卡里的 markdown 表（头里已逐字抄）：每行 `v2x` | (r, d)、(r, d)* | order
        text = "\n".join(header_block())
        rows = re.findall(r"^\s*\| `(v2[a-f])` \| (.*?) \| (lexi|near) \|", text, re.M)
        self.assertEqual([r[0] for r in rows], [v["name"] for v in table])
        for (name, lv, order), v in zip(rows, table):
            got = [(float(a), float(b), st == "*") for a, b, st in re.findall(r"\(([\d.]+), ([\d.]+)\)(\*?)", lv)]
            self.assertEqual(got, [tuple(x) for x in v["levels"]], name)
            self.assertEqual(order, v["order"], name)

    def test_v1_arm_values_are_k1_frozen(self):
        k1 = os.path.join(REPO, "experiments", "2026-11-10-rule-oracle-3dgs", "run.sh")
        with open(k1, encoding="utf-8") as f:
            k1_src = f.read()
        for pat, why in ((r"^R_SAFE=0\.25", "K1 冻结的 r_safe"), (r"^D_LOOK=1\.2", "K1 冻结的 d_look"), (r"^YAW_STEP_H10=31\.7", "K1 的 yaw_step")):
            self.assertTrue(re.search(pat, k1_src, re.M), why)


class TestScriptShape(unittest.TestCase):
    def test_bash_syntax(self):
        r = subprocess.run(["bash", "-n", RUN_SH], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_override_list_covers_every_overridable_config_var(self):
        """配置区里写成 ${VAR:-默认} 的变量（LANES 除外）必须在「formal 拒绝覆盖」名单里，否则正式起跑时环境里带了它也会悄悄生效。"""
        m = re.search(r"for v in (.*?); do", SRC, re.S)
        listed = set(m.group(1).replace("\\\n", " ").split())
        overridable = set(re.findall(r"^([A-Z][A-Z0-9_]*)=\$\{\1:-", SRC, re.M))
        self.assertEqual(sorted(overridable - listed - {"LANES", "SMOKE", "DRY_RUN"}), [], "没进拒绝覆盖名单的可覆盖变量（SMOKE / DRY_RUN 是模式开关，另有判断）")
        self.assertTrue({"OUT_ROOT", "DRIVER", "K1_ROOT", "PY", "NO_GATE", "FREEZE", "VARIANTS", "DEPTH_MAP"} <= listed)

    def test_formal_guards_exist(self):
        for needle in ("formal 不接受环境变量覆盖", "已跟踪文件有 $DIRTY 个未提交改动", "NO_GATE 只给冒烟用", "FREEZE=tree 只给冒烟与干跑用", "无卡模式，未起跑",
                       "开发路线上 v2 没有比 v1 好", "F0 对拍不一致", "flock 8", "flock -n 9", "kill -TERM -- -$$"):
            self.assertIn(needle, SRC, needle)
        code = [ln for ln in SRC.split("\n") if not ln.lstrip().startswith("#")]
        self.assertFalse([ln for ln in code if re.search(r"\bpkill\b", ln)], "停进程只按 PGID，代码里不许有 pkill")

    def test_queue_lock_is_the_3dgs_pool_lock(self):
        """公共规则 §4（2026-11-11 23:55 起）：3DGS 批量实验拿 gs3d.lock，不排 machine.lock（只管 Gazebo）；阻塞式；持有到脚本退出（fd 8 不关）；车道数不超过 3。"""
        code = "\n".join(re.sub(r"\s+#.*$", "", ln) for ln in SRC.split("\n") if not ln.lstrip().startswith("#"))      # 去掉整行注释与行尾注释
        self.assertEqual(conf("GS3D_LOCK"), "$D/tmp/phase12-cards/gs3d.lock")
        self.assertNotIn("machine.lock", code, "代码里不许再出现 machine.lock")
        self.assertNotIn("MACHINE_LOCK", code)
        self.assertIn('exec 8>> "$GS3D_LOCK"; flock 8', code, "阻塞式拿锁，写法同公共规则 §4")
        self.assertIsNone(re.search(r"flock\s+-\S*[nw]\S*\s+8\b", code), "fd 8 上的 flock 不许带 -n / -w（排队必须阻塞）")
        self.assertNotRegex(code, r"\bexec\s+8>&-", "fd 8 持有到脚本退出，不许中途关")
        self.assertLessEqual(int(conf("LANES")), 3, "3DGS 池一次不超过 3 条车道")
        i_lock, i_dev = code.index('flock 8'), code.index('phase dev')
        self.assertLess(i_lock, i_dev, "拿锁在开发阶段之前")

    def test_cgroup_notes_at_start_and_end(self):
        """公共规则 §4：守护脚本起止各记一次 cgroup 的 usage_usec 与 nr_throttled；「止」要先于 DONE 落盘（fail_done、on_term、正常结束三处）。"""
        code = [ln for ln in SRC.split("\n") if not ln.lstrip().startswith("#")]
        text = "\n".join(code)
        for needle in ("/sys/fs/cgroup/cpu.stat", "usage_usec", "nr_throttled", "cgroup_start", "cgroup_end"):
            self.assertIn(needle, text, needle)
        self.assertTrue(any(re.match(r"fail_done\(\)", ln) and "cgroup_end;" in ln for ln in code), "fail_done 要先记「止」")
        self.assertTrue(any(re.match(r"on_term\(\)", ln) and "cgroup_end;" in ln for ln in code), "on_term 要先记「止」")
        i_end, i_done = text.rindex("cgroup_end\n"), text.rindex('> "$OUT_ROOT/DONE"')
        self.assertLess(i_end, i_done, "正常结束：先记「止」再写 DONE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
