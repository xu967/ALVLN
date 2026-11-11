#!/usr/bin/env python3
"""3DGS 线喂给 prev-action 学生的「上一步动作」必须是真正执行的动作（审核 C2 = G1 = F1，卡 S1a）。

训练口径：preprocess 的 prev_action 取上一行的 action_executed（tools/preprocess.py）；Gazebo 线 sim/tools/collect.py
显式传 prev_action=prev_executed。3DGS 线的 StudentAdapter 以前调 student.act(msg) 不传，student.py 就拿模型自己上一步的
输出，DAgger 的 β 混合换成教师动作、或护栏（OracleGuardedPolicy）改写之后，学生看到的和训练时不是一回事。

PrevActionIsExecuted        审核 F 提议的两条用例（假推理进程，不经 run_episode 直接调策略），原样入库。
PrevActionComposition       护栏套 DAgger：两层都改写时，最外层真正执行的动作才是下一步的 prev-action。
PureImageUnchanged          纯图像模型（prev_dim = 0）：act 的调用与送进推理进程的字节流与改动前逐位相同。
PrevActionEndToEnd          真 run_episode：推理进程收到的 prev 字节 == actions.jsonl 里上一行的 action_executed；
                            manifest 的 student_cond 只在 prev-action 学生上出现，其余策略（纯图像学生、教师、forward、
                            护栏包 forward）的 manifest 键集与 forward 对照集逐位相同。

跑法（要 numpy / PIL，不需要 torch）：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_3dgs_student_prev_action.py
UAV_RGB_ROOT_UNDER_TEST=<另一份代码树> 可以拿同一份用例去跑那份代码（改前红 / 改后绿的对照就是这么出的）。
没设 TMPDIR 时临时目录落 $DATA_ROOT/uav_rgb/tmp（根分区很小，不往 /tmp 写）。
"""
import importlib
import json
import os
import random
import sys
import tempfile
import types
import unittest

import numpy as np

R = os.environ.get("UAV_RGB_ROOT_UNDER_TEST") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "sim", "tools"))
guard = importlib.import_module("sim.3dgs.guard")
c3 = importlib.import_module("sim.3dgs.collect_3dgs")
scene_mod = importlib.import_module("sim.3dgs.scene")
planner = importlib.import_module("sim.3dgs.planner")
from student import StudentPolicy  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
    os.environ["TMPDIR"] = _DATA_TMP   # 子进程也不要写 /tmp


def fake_infer_source(prev_dim, action):
    """假 uavrgb_infer：单帧 2x2x3；prev_dim = 4 时每步多读 1 字节（上一步动作）；恒出 action。"""
    return ('#!/usr/bin/env python3\n'
            'import json, sys\n'
            'print(json.dumps({"in_h": 2, "in_w": 2, "in_c": 3, "num_classes": 4, "text_dim": 0, "prev_action_dim": %d}), flush=True)\n'
            'while sys.stdin.buffer.read(%d):\n'
            '    print(json.dumps({"action": %d, "probs": [0.25] * 4, "infer_ms": 0.1}), flush=True)\n'
            % (prev_dim, 12 + (1 if prev_dim else 0), action))


# 单帧 + prev-action（13 字节一步），学生恒出 FORWARD
FAKE = fake_infer_source(4, 1)


class PrevActionIsExecuted(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        fake = os.path.join(self._td.name, "fake.py")
        with open(fake, "w") as f:
            f.write(FAKE)
        os.chmod(fake, 0o700)
        self.pol = StudentPolicy(fake, fake)
        self.sent = []
        real = self.pol._exchange
        self.pol._exchange = lambda payload, stage: (self.sent.append(payload[-1]), real(payload, stage))[1]
        self.rgb = np.zeros((4, 4, 3), np.uint8)
        self.state = types.SimpleNamespace(x=0.0, y=0.0, yaw=0.0)

    def tearDown(self):
        self.pol.close()
        self._td.cleanup()

    def _drive(self, policy, steps=3):
        executed = [policy(self.rgb, None, self.state)[0] for _ in range(steps)]
        warm = len(self.sent) - steps          # 构造时的 warmup 帧
        return executed, list(self.sent[warm + 1:])

    def test_guard_rewrite_feeds_executed_action(self):
        g = guard.OracleGuardedPolicy(c3.StudentAdapter(self.pol), lambda x, y: (0.0, 10.0), 30.0)
        g.reset()
        executed, prev_seen = self._drive(g)
        self.assertEqual(executed, [guard.YAW_LEFT] * 3)
        self.assertEqual(prev_seen, executed[:-1], "护栏改写后，下一步的 prev-action 应是执行的 YAW_LEFT")

    def test_dagger_beta_feeds_executed_action(self):
        teacher = lambda rgb, depth, state: (3, None)   # noqa: E731  教师恒 YAW_RIGHT
        d = c3.DaggerPolicy.__new__(c3.DaggerPolicy)   # 绕开构造函数里对教师属性的读取，只测 __call__
        d.student, d.expert, d.beta, d.rng = c3.StudentAdapter(self.pol), teacher, 1.0, random.Random(0)
        d.last_expert, d.n_expert_exec = None, 0
        executed, prev_seen = self._drive(d)
        self.assertEqual(executed, [3, 3, 3])
        self.assertEqual(prev_seen, executed[:-1], "β=1 执行教师动作后，prev-action 应是教师的 YAW_RIGHT")


class FakeExpert:
    """DaggerPolicy 要的教师属性 + 恒定动作。"""
    planner_name, stop_radius, n_plan_fail = "astar_occ", 0.5, 0

    def __init__(self, action):
        self.action = action

    def __call__(self, rgb, depth, state):
        return self.action, None


class FakeInferCase(unittest.TestCase):
    """起假推理进程的公共部分：self.student(prev_dim, action) -> (StudentPolicy, sent)，
    sent 记下每步（不含构造时的 warmup）发给进程的整段字节，prev_dim = 4 时最后一个字节就是 prev-action。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self._td.cleanup)
        self._n = 0

    def student(self, prev_dim, action):
        self._n += 1
        path = os.path.join(self._td.name, f"fake{self._n}.py")
        with open(path, "w") as f:
            f.write(fake_infer_source(prev_dim, action))
        os.chmod(path, 0o700)
        pol = StudentPolicy(path, path)
        self.addCleanup(pol.close)
        sent = []
        real = pol._exchange
        pol._exchange = lambda payload, stage: (sent.append(bytes(payload)), real(payload, stage))[1]
        return pol, sent


FRAMES = [np.full((4, 4, 3), 30 * k, np.uint8) for k in range(3)]
STATE = types.SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
LEFT_OF_ME = lambda x, y: (x, y + 10.0)   # noqa: E731  目标恒在正左方：|err| ≈ 90°，护栏出 YAW_LEFT


def drive(policy, frames=FRAMES):
    return [policy(f, None, STATE)[0] for f in frames]


class PrevActionComposition(FakeInferCase):
    def test_outermost_rewrite_wins(self):
        pol, sent = self.student(prev_dim=4, action=1)
        dagger = c3.DaggerPolicy(pol, FakeExpert(3), 1.0, random.Random(0))     # β=1：执行教师的 YAW_RIGHT
        g = guard.OracleGuardedPolicy(dagger, LEFT_OF_ME, 30.0)                  # 护栏再改成 YAW_LEFT
        g.reset()
        executed = drive(g)
        self.assertEqual(executed, [guard.YAW_LEFT] * 3)
        self.assertEqual([p[-1] for p in sent], [0, guard.YAW_LEFT, guard.YAW_LEFT],
                         "两层都改写时，prev-action 取最外层真正执行的动作；第一步是 SAFE（训练口径：首帧 prev=SAFE）")

    def test_guard_that_does_not_trigger_keeps_inner_action(self):
        pol, sent = self.student(prev_dim=4, action=1)
        dagger = c3.DaggerPolicy(pol, FakeExpert(3), 1.0, random.Random(0))
        g = guard.OracleGuardedPolicy(dagger, lambda x, y: (x + 10.0, y), 30.0)  # 目标在正前方：护栏不动
        g.reset()
        self.assertEqual(drive(g), [3, 3, 3])
        self.assertEqual([p[-1] for p in sent], [0, 3, 3])

    def test_beta_zero_prev_is_students_own_output(self):
        pol, sent = self.student(prev_dim=4, action=1)
        dagger = c3.DaggerPolicy(pol, FakeExpert(3), 0.0, random.Random(0))     # β=0：不换，执行的就是学生输出
        dagger.reset()
        self.assertEqual(drive(dagger), [1, 1, 1])
        self.assertEqual([p[-1] for p in sent], [0, 1, 1])

    def test_dagger_reuse_clears_executed_action_at_episode_boundary(self):
        # DaggerPolicy 整轮复用同一个 adapter：上一集末步执行的教师动作不能漏进下一集的首帧（首帧 prev = SAFE）
        pol, sent = self.student(prev_dim=4, action=1)
        dagger = c3.DaggerPolicy(pol, FakeExpert(guard.YAW_RIGHT), 1.0, random.Random(0))
        drive(dagger, FRAMES[:2])
        dagger.reset()
        drive(dagger, FRAMES[:2])
        self.assertEqual([p[-1] for p in sent], [0, guard.YAW_RIGHT, 0, guard.YAW_RIGHT])


class StrictPureImageStudent:
    """纯图像学生的替身：act 只收 msg，传了 prev_action 就 TypeError（现有的各种 FakeStudent 都是这个签名）。"""

    def __init__(self, action=1):
        self.action, self.calls = action, 0

    def reset(self):
        pass

    def act(self, msg):
        self.calls += 1
        return self.action, 1.0, [0.0] * 4


class KwargSpyStudent:
    """prev_dim = 4 的替身：记下 act 收到的关键字参数。"""
    prev_dim = 4

    def __init__(self, action=1):
        self.action, self.kw = action, []

    def reset(self):
        pass

    def act(self, msg, **kw):
        self.kw.append(kw)
        return self.action, 1.0, [0.0] * 4


class PureImageUnchanged(FakeInferCase):
    def test_act_is_called_with_msg_only(self):
        # 无论外面有没有换动作、有没有改写，纯图像学生的 act 都只收到 msg（没抛 TypeError 就说明没传 prev_action）
        s = StrictPureImageStudent(action=1)
        self.assertEqual(drive(c3.StudentAdapter(s)), [1, 1, 1])
        d = c3.DaggerPolicy(s, FakeExpert(3), 1.0, random.Random(0))
        self.assertEqual(drive(d), [3, 3, 3])
        g = guard.OracleGuardedPolicy(c3.StudentAdapter(s), LEFT_OF_ME, 30.0)
        self.assertEqual(drive(g), [guard.YAW_LEFT] * 3)
        self.assertEqual(s.calls, 9)

    def test_student_without_prev_dim_has_no_condition(self):
        # prev_dim 缺省或为 0 都算纯图像：没有条件输入，manifest 不写 student_cond
        for s in (StrictPureImageStudent(), types.SimpleNamespace(prev_dim=0, reset=lambda: None)):
            self.assertIsNone(getattr(c3.StudentAdapter(s), "student_cond", None))
            d = c3.DaggerPolicy(s, FakeExpert(3), 0.5, random.Random(0))
            self.assertIsNone(getattr(d, "student_cond", None))

    def test_prev_action_student_gets_prev_action_keyword(self):
        # 关键字名与 StudentPolicy.act 的形参同名；首步 SAFE，之后是上一步执行的动作
        s = KwargSpyStudent(action=1)
        d = c3.DaggerPolicy(s, FakeExpert(3), 1.0, random.Random(0))
        d.reset()
        drive(d)
        self.assertEqual(s.kw, [{"prev_action": 0}, {"prev_action": 3}, {"prev_action": 3}])
        self.assertEqual(getattr(d, "student_cond", None), "prev-action")

    def test_byte_stream_to_infer_process_is_the_same_with_or_without_rewrites(self):
        # 真 StudentPolicy + 假推理进程（prev_action_dim = 0，一步 12 字节、没有 prev 字节）：
        # 光秃秃的 adapter、β=1 的 DAgger、护栏改写，三种包法送给推理进程的字节流逐位相同，只取决于画面。
        runs = []
        for wrap in ("bare", "dagger", "guard"):
            pol, sent = self.student(prev_dim=0, action=1)
            if wrap == "bare":
                policy = c3.StudentAdapter(pol)
            elif wrap == "dagger":
                policy = c3.DaggerPolicy(pol, FakeExpert(3), 1.0, random.Random(0))
            else:
                policy = guard.OracleGuardedPolicy(c3.StudentAdapter(pol), LEFT_OF_ME, 30.0)
            runs.append((drive(policy), sent))
        self.assertEqual([a for a, _ in runs], [[1, 1, 1], [3, 3, 3], [guard.YAW_LEFT] * 3])
        self.assertEqual([len(p) for p in runs[0][1]], [12, 12, 12])
        self.assertEqual(runs[0][1], runs[1][1])
        self.assertEqual(runs[0][1], runs[2][1])

    def test_byte_stream_equals_direct_act_before_the_change(self):
        # 对照组 = 改动前 adapter 的行为：对同一串画面直接调 StudentPolicy.act(msg)（不传 prev_action）。
        # adapter 外面套护栏改写，纯图像学生收到的字节也必须与对照组逐位相同。
        import base64
        pol, sent = self.student(prev_dim=0, action=1)
        ref, ref_sent = self.student(prev_dim=0, action=1)
        g = guard.OracleGuardedPolicy(c3.StudentAdapter(pol), LEFT_OF_ME, 30.0)
        self.assertEqual(drive(g), [guard.YAW_LEFT] * 3)
        for fr in FRAMES:
            ref.act({"width": 4, "height": 4, "step": 12, "pixelFormatType": "RGB_INT8",
                     "data": base64.b64encode(np.ascontiguousarray(fr).tobytes()).decode()})
        self.assertEqual(sent, ref_sent)
        self.assertEqual([g.trace[i]["inner"] for i in range(3)], [1, 1, 1])      # 学生自己的输出没被改


def _rows(ep_dir):
    with open(os.path.join(ep_dir, "actions.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


class PrevActionEndToEnd(FakeInferCase):
    """合成走廊 + mock 渲染器 + cf_blackbox，真 run_episode。"""

    def setUp(self):
        super().setUp()
        self.tmp = self._td.name
        scene_dir = os.path.join(self.tmp, "scene")
        scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
        self.scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
        self.renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(self.scene, True)
        self.occ = importlib.import_module("sim.3dgs.collision").from_scene(self.scene)
        self.params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
        self.scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}      # 走廊默认 waypoint 在尽头墙里，A* 无路
        self.identity = c3.scene_identity(self.scene)

    def episode(self, name, policy, steps, dagger=None):
        m, err = c3.run_episode(self.scene, os.path.join(self.tmp, name), self.renderer, self.occ, self.params,
                                policy, steps, 0.5, dagger=dagger, split=self.identity["split"], identity=self.identity)
        self.assertIsNone(err)
        return m, _rows(os.path.join(self.tmp, name))

    def test_dagger_prev_byte_is_previous_action_executed(self):
        pol, sent = self.student(prev_dim=4, action=0)                 # 学生恒 SAFE
        expert = planner.OccAstarTeacher(planner.OccPlanner(self.occ), self.scene["waypoint"])
        policy = c3.DaggerPolicy(pol, expert, 0.5, random.Random(0))
        dagger = {"executed_by": "student", "dagger_round": 1, "beta": 0.5}
        for ep in range(2):             # 同一个 DaggerPolicy 连跑两集（collect_3dgs.main 就是这么用的）
            del sent[:]
            m, rows = self.episode(f"d{ep}", policy, 8, dagger)
            executed = [r["action_executed"] for r in rows]
            self.assertTrue(any(a != 0 for a in executed), f"β=0.5 没换过教师动作，用例空转：{executed}")
            self.assertEqual([p[-1] for p in sent], [0] + executed[:-1],
                             f"第 {ep} 集：推理进程收到的 prev 字节应是 SAFE 加上 actions.jsonl 里上一行的 action_executed")
            self.assertEqual(m.get("student_cond"), "prev-action")

    def test_guard_prev_byte_is_previous_action_executed(self):
        pol, sent = self.student(prev_dim=4, action=0)                 # 学生恒 SAFE，护栏把它改成 YAW_LEFT
        g = guard.OracleGuardedPolicy(c3.StudentAdapter(pol), LEFT_OF_ME, 30.0)
        m, rows = self.episode("g", g, 6)
        executed = [r["action_executed"] for r in rows]
        self.assertIn(guard.YAW_LEFT, executed, f"护栏没改写过，用例空转：{executed}")
        self.assertEqual([p[-1] for p in sent], [0] + executed[:-1])
        self.assertEqual(m.get("student_cond"), "prev-action")

    def test_manifest_keys_only_grow_for_prev_action_students(self):
        # 键集以 forward 对照集为基准：纯图像学生（裸 adapter、DAgger、护栏）、教师、护栏包 forward 都与它同键，
        # 只有 prev_dim > 0 的学生多一个 student_cond。
        base, _ = self.episode("fwd", c3.ForwardPolicy(), 2)
        keys = set(base)
        self.assertNotIn("student_cond", keys)
        pure, sent = self.student(prev_dim=0, action=1)
        pure_dagger, _ = self.student(prev_dim=0, action=1)
        prev, _ = self.student(prev_dim=4, action=1)
        same_keys = (
            ("pure_adapter", c3.StudentAdapter(pure)),
            ("pure_dagger", c3.DaggerPolicy(pure_dagger, FakeExpert(1), 0.5, random.Random(0))),
            ("pure_guard", guard.OracleGuardedPolicy(c3.StudentAdapter(pure), LEFT_OF_ME, 30.0)),
            ("guarded_fwd", guard.OracleGuardedPolicy(c3.ForwardPolicy(), LEFT_OF_ME, 0.0)),
            ("teacher", c3.DepthTeacher()))
        for name, policy in same_keys:
            m, _ = self.episode(name, policy, 2)
            self.assertEqual(set(m), keys, name)
        self.assertTrue(all(len(p) == 12 for p in sent), "纯图像学生没有 prev 字节")
        m, _ = self.episode("prev", c3.StudentAdapter(prev), 2)
        self.assertEqual(set(m), keys | {"student_cond"})
        self.assertEqual(m["student_cond"], "prev-action")


if __name__ == "__main__":
    unittest.main(verbosity=2)
