#!/usr/bin/env python3
"""Single-entry CLI: honest preflight, scene creation, dashboard and gated binaries."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'artifacts' / 'runs'
RUN_ID = re.compile(r'^[A-Za-z0-9_]{1,70}$')  # identical to tools/dashboard.py
# Exit codes shared with tools/sim_preflight.py
EXIT_USAGE, EXIT_NOT_IMPLEMENTED, EXIT_INPUT, EXIT_PROCESS, EXIT_NOT_BUILT = 2, 3, 4, 5, 6


def probe(cmd: list[str]) -> str:
    """Run a version probe; FOUND in PATH is not evidence that the tool works."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f'probe failed: {type(exc).__name__}'
    lines = (out.stdout or out.stderr).strip().splitlines()
    return (lines[-1] if lines else '(no output)') + ('' if out.returncode == 0 else f' [exit={out.returncode}]')


def doctor() -> int:
    print('=== TinyAerialVLN V7.1 环境自检 ===')
    commands = [('python3', False), ('cmake', True), ('ninja', True),
                ('g++', True), ('nvcc', True), ('gz', True)]
    missing = []
    for cmd, optional in commands:
        found = shutil.which(cmd)
        print(f'{"[FOUND]" if found else "[WARN]"} {cmd}: {found or "未安装/不在 PATH"}')
        if not found and not optional:
            missing.append(cmd)
    probes = {'nvcc': ['nvcc', '--version'], 'gz': ['gz', 'sim', '--versions'],
              'nvidia-smi': ['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader']}
    for cmd, argv in probes.items():
        if shutil.which(cmd):
            print(f'[PROBE] {cmd}: {probe(argv)}')
    print('[INFO] FOUND 只表示可执行文件在 PATH 中；RTX 5090 (sm_120) 需要 CUDA ≥ 12.8，请核对 PROBE 行。')
    for key in ['artifacts', 'tools', 'code', 'assets']:
        print(f'{"[OK]" if (ROOT / key).exists() else "[ERR]"} project/{key}')
    if not (ROOT / 'config' / 'experiment.yaml').exists():
        print('[WARN][CONFIG_NOT_READY] config/experiment.yaml 尚未建立；训练功能锁定。')
    if not any((ROOT / 'build' / name).is_file() for name in ('tinyvln_train', 'tinyvln_eval_bridge')):
        print('[WARN][BINARY_NOT_BUILT] 自研训练/评测器尚未编译。')
    print('[INFO] 此检测不声称 Gazebo/NVIDIA/SG2002 已经适配成功。')
    return 0 if not missing else 2


def gated_binary(name: str, run_id: str | None, mode: str = 'train') -> int:
    if mode == 'train':
        run_id = run_id or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    elif not run_id:
        print('[ERR][RUN_ID_REQUIRED] eval 必须用 --run-id 指定一个已有的训练 run。', file=sys.stderr)
        return EXIT_INPUT
    if not RUN_ID.fullmatch(run_id):  # validate input before probing the environment
        print('[ERR][RUN_ID_INVALID] run ID 只能用 ASCII 字母、数字和下划线，1–70 字符（与看板一致）。', file=sys.stderr)
        return EXIT_INPUT
    exe = ROOT / 'build' / name
    cfg = ROOT / 'config' / 'experiment.yaml'
    if not exe.is_file() or not os.access(exe, os.X_OK):
        print(f'[ERR][BINARY_NOT_BUILT] {exe} 不存在或不可执行。先按开发门禁实现并编译。', file=sys.stderr)
        return EXIT_NOT_BUILT
    if not cfg.is_file():
        print(f'[ERR][CONFIG_NOT_READY] 缺少 {cfg}，不得使用 .planned 占位文件训练。', file=sys.stderr)
        return EXIT_INPUT
    run_dir = RUNS / run_id
    extra: list[str] = []
    if mode == 'train':
        if run_dir.exists():
            print(f'[ERR][RUN_EXISTS] {run_dir} 已存在，禁止覆盖旧实验。', file=sys.stderr)
            return EXIT_INPUT
        out = run_dir
    else:  # eval never overwrites the training run: it writes a sub-directory per evaluation
        if not run_dir.is_dir():
            print(f'[ERR][RUN_NOT_FOUND] {run_dir} 不存在，无法评测。', file=sys.stderr)
            return EXIT_INPUT
        out = run_dir / datetime.now().strftime('eval_%Y%m%d_%H%M%S')
        if out.exists():
            print(f'[ERR][RUN_EXISTS] {out} 已存在，请稍后重试。', file=sys.stderr)
            return EXIT_INPUT
        extra = ['--run-dir', str(run_dir)]
    out.mkdir(parents=True)
    print(f'[INFO] 启动 {name}; run_id={run_id}; output={out}; Ctrl+C 请求终止。')
    try:
        return subprocess.call([str(exe), '--config', str(cfg), *extra, '--output', str(out)], cwd=ROOT)
    except KeyboardInterrupt:
        print('\n[WARN][USER_INTERRUPT] 请求中止；检查是否生成完整 checkpoint。', file=sys.stderr)
        return 130
    except OSError as exc:
        print(f'[ERR][PROCESS_START_FAILED] {exc}', file=sys.stderr)
        return EXIT_PROCESS


def menu() -> int:
    print('TinyAerialVLN V7.1 —— 一键工程入口')
    print('1. 环境自检\n2. 生成简单森林场景\n3. 打开训练指标看板\n4. 启动学生训练（需已编译）\n5. 启动评测（需已编译）\n0. 退出')
    try:
        selection = input('请选择 [0-5]: ').strip()
    except EOFError:
        print('[ERR][NO_INPUT] 无交互终端；使用 ./run.sh doctor 等子命令。', file=sys.stderr)
        return 2
    return {'0': lambda: 0, '1': doctor,
            '2': lambda: subprocess.call([sys.executable, str(ROOT / 'tools/scene_tool.py')], cwd=ROOT),
            '3': lambda: subprocess.call([sys.executable, str(ROOT / 'tools/dashboard.py')], cwd=ROOT),
            '4': lambda: gated_binary('tinyvln_train', None),
            '5': lambda: gated_binary('tinyvln_eval_bridge', input('评测哪个 run ID: ').strip(), 'eval')}.get(
                selection, lambda: print('[ERR][MENU_OPTION_INVALID] 请输入 0..5。') or 2)()


def main() -> int:
    p = argparse.ArgumentParser(description='TinyAerialVLN CLI (no fake training)')
    p.add_argument('command', nargs='?', default=None,
                   choices=['doctor', 'scene', 'dashboard', 'ui', 'train', 'eval', 'board', 'uart'])
    p.add_argument('rest', nargs='*', help='board/uart 的子命令（尚未实现）')
    p.add_argument('--run-id', default=None)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--preset', choices=['forest', 'corridor', 'columns', 'mixed'], default='forest')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--count', type=int, default=24)
    args = p.parse_args()
    if args.command is None:
        return menu()
    if args.command in ('board', 'uart'):  # listed in plan §11, gated by EDGE/HIL
        print(f'[ERR][FEATURE_NOT_IMPLEMENTED] ./run.sh {args.command} {" ".join(args.rest)} 需在 EDGE/HIL 门禁实现后提供。',
              file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    if args.rest:
        p.error(f'unexpected arguments: {" ".join(args.rest)}')
    if args.command == 'doctor':
        return doctor()
    if args.command in ('dashboard', 'ui'):
        return subprocess.call([sys.executable, str(ROOT / 'tools/dashboard.py'), '--port', str(args.port)], cwd=ROOT)
    if args.command == 'scene':
        return subprocess.call([sys.executable, str(ROOT / 'tools/scene_tool.py'), '--preset', args.preset,
                               '--seed', str(args.seed), '--count', str(args.count)], cwd=ROOT)
    if args.command == 'train':
        return gated_binary('tinyvln_train', args.run_id)
    return gated_binary('tinyvln_eval_bridge', args.run_id, 'eval')


if __name__ == '__main__':
    raise SystemExit(main())
