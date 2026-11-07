#!/usr/bin/env python3
"""e1_pipeline.evaluate 的等待顺序：四组各带 1 个 gz 并行跑，一组没写出 eval.json 时必须等其余几组
跑完再报错。中途抛出会让 main 写 DONE(rc=1) 退出，剩下几组的 gz 还在跑；续跑 --from eval 会再起 4 个。

运行（仓库根）：.venv-tools/bin/python tools/test_e1_pipeline.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import e1_pipeline as ep  # noqa: E402


class FakeProc:
    """替身子进程：wait() 记录被等过；write_eval=False 的组模拟 eval_closed_loop 崩在写 eval.json 之前。"""
    waited = []

    def __init__(self, name, edir, write_eval):
        self.name, self.edir, self.write_eval = name, edir, write_eval

    def wait(self):
        FakeProc.waited.append(self.name)
        if self.write_eval:
            os.makedirs(self.edir, exist_ok=True)
            with open(os.path.join(self.edir, "eval.json"), "w", encoding="utf-8") as f:
                json.dump({"episodes_done": 0}, f)
            return 0
        return 1


def main():
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    real_popen, real_ckpt = subprocess.Popen, ep.CFG["old_x3_ckpt"]
    try:
        for rel in ("old_x3.bin", os.path.join("run", "ckpt_best.bin")):   # evaluate 开头 need() 的两个 ckpt
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)) or tmp, exist_ok=True)
            open(os.path.join(tmp, rel), "wb").close()
        ep.CFG["old_x3_ckpt"] = os.path.join(tmp, "old_x3.bin")
        groups = [g[0] for g in ep.eval_groups()]
        broken = groups[0]   # 第一个被 wait 的组没写 eval.json

        def fake_popen(cmd, **_kw):
            edir = cmd[cmd.index("--out") + 1]
            name = os.path.basename(edir)[len("eval-"):]
            return FakeProc(name, edir, write_eval=(name != broken))

        subprocess.Popen = fake_popen
        try:
            ep.evaluate(tmp)
        except RuntimeError as e:
            err = str(e)
        else:
            raise AssertionError("有一组没写 eval.json，evaluate 应报错")
        assert broken in err, err
        assert FakeProc.waited == groups, f"报错前没等齐所有组：等了 {FakeProc.waited}，应等 {groups}"
    finally:
        subprocess.Popen, ep.CFG["old_x3_ckpt"] = real_popen, real_ckpt
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_e1_pipeline: PASS（{len(groups)} 组，{broken} 缺 eval.json 时其余组仍被等完）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
