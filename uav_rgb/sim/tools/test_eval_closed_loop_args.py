#!/usr/bin/env python3
"""eval_closed_loop 参数自检：--seeds 解析、缺省集数、格式错误。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ec


def main():
    assert ec.parse_seeds("23, 25:28") == [23, 25, 26, 27], ec.parse_seeds("23, 25:28")
    assert ec.parse_seeds("7") == [7]
    for bad in ("23,abc", "-1", "5:5", "9:3"):
        try:
            ec.parse_seeds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad!r}")
    # 关键回归：传 20 个 seed 不显式给 --episodes 时，目标集数必须是 20，不是 3。
    assert ec.episode_target(None, list(range(20))) == 20
    assert ec.episode_target(None, []) == 3
    assert ec.episode_target(20, [1, 2, 3]) == 20
    print("test_eval_closed_loop_args: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
