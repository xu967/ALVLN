#!/usr/bin/env python3
"""student.py 自检：假推理进程验证协议、RGB/RGBA 缩放、异常格式拒绝。"""
import base64
import os
import stat
import sys
import tempfile

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

    print("test_student: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
