#!/usr/bin/env python3
"""软标签拼接拒绝缺失/错位；可选串联真实 C++ 读端，验证同长度错配也会失败。"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from join_soft_targets import HEADER


def run(cmd, ok, contains=""):
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert (p.returncode == 0) == ok, (cmd, p.stdout, p.stderr)
    assert contains in p.stdout + p.stderr, (contains, p.stdout, p.stderr)


def fixture(root):
    ds = root / "dataset"
    ds.mkdir()
    eps = root / "episodes" / "train" / "10"
    eps.mkdir(parents=True)
    (eps / "actions.jsonl").write_text(
        '\n'.join(json.dumps({"step": i, "collision": False}) for i in range(3)) + '\n')
    spans = [{"episode_id": 10, "sample_start": 0, "sample_count": 3,
              "scene_id": 10, "split": "train"}]
    meta = json.dumps({"split_policy": "scene_id", "labels": {"num_classes": 4}, "episodes": spans},
                      separators=(",", ":")).encode()
    images = bytes(range(192)) * 3  # 3x3x8x8
    labels_off = 48 + len(images)
    header = struct.pack("<4sHHIHHBBBBQQQI", b"UAVR", 1, 48, 3, 8, 8,
                         3, 1, 1, 0, 48, labels_off, labels_off + 3, len(meta))
    (ds / "dataset.uavr").write_bytes(header + images + b"\x00\x01\x02" + meta)
    row = {"episode_id": 10, "sample_start": 0, "sample_count": 3, "text_ids": [2, 3]}
    (ds / "instructions.jsonl").write_text(json.dumps(row) + '\n')
    return ds, root / "episodes", row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native-check", help="构建后的 test_soft_targets_identity")
    args = p.parse_args()
    script = str(Path(__file__).with_name("join_soft_targets.py"))
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        ds, episodes, row = fixture(root)
        data = ds / "dataset.uavr"
        side = ds / "instructions.jsonl"
        soft = root / "teacher.npz"
        out = root / "soft.bin"
        logits = {f"10/{i}/logits": np.array([i, 1, 2, 3], dtype=np.float32)
                  for i in range(3)}
        cmd = [sys.executable, script, "--uavr-dir", str(ds), "--episodes-dir", str(episodes),
               "--soft", str(soft), "--out", str(out)]
        np.savez(soft, **{k: v for k, v in logits.items() if k != "10/2/logits"})
        run(cmd, False, "缺失 1/3")
        assert not out.exists(), "缺失标签留下了半成品"
        np.savez(soft, **logits)
        run(cmd, True)
        original = out.read_bytes()
        fields = HEADER.unpack(original[:HEADER.size])
        assert fields[:4] == (b"UAVSOFT1", 3, 4, 1)
        assert fields[4] == hashlib.sha256(data.read_bytes()).digest()
        assert fields[5] == hashlib.sha256(side.read_bytes()).digest()
        assert fields[6] == hashlib.sha256(original[HEADER.size:]).digest()
        assert np.allclose(np.frombuffer(original[HEADER.size:], "<f4").reshape(-1, 4).sum(1), 1)

        # 缺失导出不能破坏先前可用文件。
        np.savez(soft, **{k: v for k, v in logits.items() if k != "10/2/logits"})
        run(cmd, False, "缺失")
        assert out.read_bytes() == original
        np.savez(soft, **logits)
        row["episode_id"] = 11
        side.write_text(json.dumps(row) + '\n')
        run(cmd, False, "与 UAVR 不匹配")
        row["episode_id"] = 10
        side.write_text(json.dumps(row) + '\n')
        bad = dict(logits)
        bad["10/0/logits"] = np.array([np.nan, 1, 2, 3])
        np.savez(soft, **bad)
        run(cmd, False, "有限 logits")
        np.savez(soft, **logits)
        run(cmd + ["--temperature", "inf"], False, "必须为正")
        run(cmd + ["--soft", str(soft), str(soft)], False, "重复键")

        if args.native_check:
            native = [args.native_check, str(data), str(out)]
            run(native, True)
            raw = data.read_bytes()
            changed = bytearray(raw)
            changed[48] ^= 1  # 同字节数，仅一像素不同。
            data.write_bytes(changed)
            run(native, False, "UAVR SHA256 不匹配")
            data.write_bytes(raw)
            row["text_ids"] = [2, 4]  # 相同 episode 与 span 的另一条指令。
            side.write_text(json.dumps(row) + '\n')
            run(native, False, "文本侧车 SHA256 不匹配")
            row["text_ids"] = [2, 3]
            side.write_text(json.dumps(row) + '\n')
            changed = bytearray(original)
            changed[-1] ^= 1
            out.write_bytes(changed)
            run(native, False, "payload SHA256 不匹配")
            out.write_bytes(original[HEADER.size:])
            run(native, False, "UAVSOFT1")
            out.write_bytes(original)
            run(native, True)
        print("PASS: join 缺失/错序/非法/重复/原子输出" +
              (" + C++ 数据/文本/payload 身份校验" if args.native_check else
               "（未执行 C++ 读端；需 --native-check）"))


if __name__ == "__main__":
    main()
