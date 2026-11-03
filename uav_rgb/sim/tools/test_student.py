#!/usr/bin/env python3
"""student.py 自检：假推理进程验证协议、RGB/RGBA 缩放、异常格式拒绝。"""
import base64
import os
import stat
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student import StudentPolicy

FAKE = '''#!/usr/bin/env python3
import json, sys
print(json.dumps({"in_h": 2, "in_w": 2, "in_c": 3, "num_classes": 4}), flush=True)
while True:
    b = sys.stdin.buffer.read(12)
    if not b:
        break
    if len(b) != 12:
        sys.exit(3)
    print(json.dumps({"action": 2, "probs": [0.1, 0.2, 0.6, 0.1], "infer_ms": 0.5}),
          flush=True)
'''


def gz_msg(w, h, nch, data):
    fmt = {3: "RGB_INT8", 4: "RGBA_INT8"}[nch]
    return {"width": w, "height": h, "step": w * nch, "pixelFormatType": fmt,
            "data": base64.b64encode(data).decode()}


def main():
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "fake_infer.py")
        with open(fake, "w", encoding="utf-8") as f:
            f.write(FAKE)
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)

        ckpt = os.path.join(td, "ckpt.bin")
        with open(ckpt, "wb") as f:
            f.write(b"not-a-real-checkpoint")  # 假进程不读，只校验路径存在

        policy = StudentPolicy(ckpt, fake)
        try:
            assert (policy.h, policy.w, policy.in_c) == (2, 2, 3)

            action, ms, probs = policy.act(gz_msg(4, 4, 3, bytes(range(48))))
            assert action == 2, action
            assert abs(ms - 0.5) < 1e-9, ms
            assert probs == [0.1, 0.2, 0.6, 0.1], probs

            # 4 通道帧必须能转 RGB 后照常推理。
            action, _, _ = policy.act(gz_msg(4, 4, 4, bytes(range(64))))
            assert action == 2

            # 不支持的像素格式必须在 Python 侧就拒绝，不能把坏字节喂给模型。
            bad = gz_msg(4, 4, 3, bytes(48))
            bad["pixelFormatType"] = "GRAY8"
            try:
                policy.act(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("GRAY8 未被拒绝")
        finally:
            policy.close()

    try:
        StudentPolicy("/nonexistent/ckpt.bin", "/nonexistent/uavrgb_infer")
    except RuntimeError:
        pass
    else:
        raise AssertionError("缺失推理程序未报错")

    # 非四类模型必须拒绝：否则越界动作会在 collect 里静默变成悬停。
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "fake3.py")
        with open(fake, "w", encoding="utf-8") as f:
            f.write(FAKE.replace('"num_classes": 4', '"num_classes": 3'))
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        ckpt = os.path.join(td, "ckpt.bin")
        with open(ckpt, "wb") as f:
            f.write(b"x")
        try:
            StudentPolicy(ckpt, fake)
        except RuntimeError as exc:
            assert "3 类" in str(exc), exc
        else:
            raise AssertionError("非四类模型未拒绝")

    # 活进程无 hello、只有半行、warmup 后不回应，都必须有界失败并清理。
    for stage, body in (
        ("hello", "import time; time.sleep(60)"),
        ("hello", "import sys,time; sys.stdout.write('{'); sys.stdout.flush(); time.sleep(60)"),
        ("warmup", "import json,time; print(json.dumps({'in_h':2,'in_w':2,'in_c':3,'num_classes':4}),flush=True); time.sleep(60)"),
    ):
        with tempfile.TemporaryDirectory() as td:
            fake = os.path.join(td, "hung.py")
            with open(fake, "w") as f:
                f.write("#!/usr/bin/env python3\n" + body + "\n")
            os.chmod(fake, 0o700)
            start = time.monotonic()
            try:
                # 进程冷启动在负载较高的桌面/CI 主机上可能超过 300 ms；
                # 1.5 s 仍能验证 hello/warmup 阶段有界超时，同时避免误判启动抖动。
                StudentPolicy(fake, fake, timeout_s=1.5)
            except RuntimeError as exc:
                assert stage in str(exc) and "超时" in str(exc), exc
            else:
                raise AssertionError("挂起进程未超时")
            assert time.monotonic() - start < 5
    with tempfile.TemporaryDirectory() as td:
        fake = os.path.join(td, "act_hung.py")
        with open(fake, "w") as f:
            f.write(FAKE.replace("while True:", "for i in range(2):").replace(
                "    b = sys.stdin.buffer.read(12)",
                "    if i == 1:\n        import time; time.sleep(60)\n    b = sys.stdin.buffer.read(12)"))
        os.chmod(fake, 0o700)
        policy = StudentPolicy(fake, fake, timeout_s=1.5)
        # hello/warmup 已完成后再缩短动作响应 deadline，专门覆盖 act 超时。
        policy.timeout_s = 0.3
        proc = policy.proc
        try:
            policy.act(gz_msg(4, 4, 3, bytes(48)))
        except RuntimeError as exc:
            assert "act" in str(exc) and "超时" in str(exc)
            assert proc.poll() is not None and policy.proc is None
        else:
            raise AssertionError("act 挂起没有超时")
        finally:
            policy.close()
    print("test_student: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
