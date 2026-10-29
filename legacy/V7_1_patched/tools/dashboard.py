#!/usr/bin/env python3
"""V7.1 read-only local research cockpit. No training or simulator actions are executed here.

Every status check carries `derived_from`: `file_probe` (computed from local files / binaries)
or `hardcoded` (a hand-maintained label until a gate-evidence file exists). Dashboard does NOT
claim that an available binary, world file or metric proves a validated flight.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'artifacts' / 'runs'
SCENES = ROOT / 'artifacts' / 'scenes'
HTML = ROOT / 'assets' / 'dashboard.html'
RUN_ID = re.compile(r'^[A-Za-z0-9_]{1,70}$')  # tools/launcher.py uses the same rule
MAX_LOG_TAIL_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 2000
MAX_LINE_BYTES = 64 * 1024
MAX_RUNS_LISTED = 80
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 32
PROVENANCE_FILES = ('tools/dashboard.py', 'tools/launcher.py', 'tools/scene_tool.py',
                    'tools/sim_preflight.py', 'assets/dashboard.html')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def all_finite(value, depth=0):
    """True only if every number in a bounded-depth JSON value is finite."""
    if depth > MAX_JSON_DEPTH:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return all(all_finite(v, depth + 1) for v in value)
    return True


def run_ok(path):
    metrics = path / 'metrics.jsonl'
    return bool(RUN_ID.fullmatch(path.name) and path.is_dir() and not path.is_symlink()
                and metrics.is_file() and not metrics.is_symlink())


def all_runs():
    if not RUNS.is_dir():
        return []
    runs = [p for p in RUNS.iterdir() if run_ok(p)]
    runs.sort(key=lambda p: (p / 'metrics.jsonl').stat().st_mtime, reverse=True)  # newest first
    return [p.name for p in runs]


def runs_list():
    return all_runs()[:MAX_RUNS_LISTED]


def metrics_for(name):
    if not RUN_ID.fullmatch(name or '') or not run_ok(RUNS / name):
        raise ValueError('RUN_NOT_FOUND')
    path = RUNS / name / 'metrics.jsonl'
    rows, warnings = deque(maxlen=MAX_EVENTS), []
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))  # no symlink swap after the check
    with os.fdopen(fd, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        offset = max(0, size - MAX_LOG_TAIL_BYTES)
        f.seek(offset)
        if offset:
            f.readline()  # discard incomplete head of bounded tail
            warnings.append('LOG_TAIL_LIMIT: 仅加载最新日志；不代表全量训练历史')
        for number, raw in enumerate(f, 1):
            if not raw.endswith(b'\n'):
                warnings.append('LOG_PARTIAL_WRITE: 正在写入最后一行，暂不展示')
                break
            if len(raw) > MAX_LINE_BYTES:
                warnings.append(f'LOG_LINE_TOO_LONG: near_tail_line={number}')
                continue
            try:
                event = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, ValueError, RecursionError):
                # ValueError also covers the int-digit limit; one bad line must never fail the run
                warnings.append(f'LOG_JSON_INVALID: near_tail_line={number}')
                continue
            if not isinstance(event, dict) or not isinstance(event.get('event'), str):
                warnings.append(f'LOG_EVENT_INVALID: near_tail_line={number}')
                continue
            if not all_finite(event):  # recursive: nested NaN/Inf or over-deep objects
                warnings.append(f'LOG_NONFINITE_OR_TOO_DEEP: near_tail_line={number}')
                continue
            rows.append(event)
    return {'run_id': name, 'events': list(rows), 'warnings': warnings[-15:],
            'truncated': size > MAX_LOG_TAIL_BYTES, 'source': 'real_jsonl_only'}


def scene_list():
    if not SCENES.is_dir():
        return []
    scenes = []
    for path in sorted(SCENES.glob('*.json'))[:80]:
        if path.is_symlink():
            continue
        sdf = path.with_suffix('.sdf')
        try:
            if path.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError('manifest larger than 1 MiB')
            spec = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(spec, dict):
                raise ValueError('manifest root must be a JSON object')
            sha_ok = None
            if sdf.is_file() and isinstance(spec.get('sha256'), str):
                sha_ok = hashlib.sha256(sdf.read_bytes()).hexdigest() == spec['sha256']
            scenes.append({'name': path.stem, 'preset': str(spec.get('preset', 'unknown')),
                           'seed': spec.get('seed'), 'obstacles': spec.get('placed_obstacles', spec.get('requested_obstacles')),
                           'sdf_exists': sdf.is_file(), 'sdf_sha256_ok': sha_ok, 'kind': 'static_geometry_only',
                           'status': 'not_a_flight_simulator'})
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            detail = type(exc).__name__ if isinstance(exc, OSError) else str(exc)[:160]  # no local paths
            scenes.append({'name': path.stem, 'status': 'manifest_invalid',
                           'error_code': 'SCENE_MANIFEST_INVALID', 'detail': detail})
    return scenes


def state_check(key, label, state, detail, command=None, derived_from='file_probe'):
    value = {'key': key, 'label': label, 'state': state, 'detail': detail, 'derived_from': derived_from}
    if command:
        value['command'] = command
    return value


def status():
    found_runs = all_runs()
    scenes = scene_list()
    valid = sum(1 for s in scenes if s.get('status') == 'not_a_flight_simulator')
    invalid = len(scenes) - valid
    gz = shutil.which('gz')
    trainer = (ROOT / 'build/tinyvln_train').is_file()
    config = (ROOT / 'config/experiment.yaml').is_file()
    checks = [
        state_check('dashboard', '本地指标看板', 'ready', '只读服务；只显示已记录的 JSONL',
                    './run.sh dashboard --port 8765', derived_from='hardcoded'),
        state_check('scenes', '程序化静态场景', 'ready' if valid else 'blocked',
                    f'有效 manifest {valid} 份，损坏 {invalid} 份；这不是 RGB 飞行仿真',
                    './run.sh scene --preset forest --seed 42 --count 24'),
        state_check('gazebo', 'Gazebo CLI', 'available_unverified' if gz else 'blocked',
                    '已发现 gz CLI，但动力学/电机/RGB 尚未验证' if gz else '当前机器找不到 gz 命令',
                    './run.sh sim doctor'),
        state_check('physics', '四旋翼物理闭环 GZ0–GZ3', 'not_implemented',
                    '尚无通过测试的四电机→动力学→RGB 运行链路', derived_from='hardcoded'),
        state_check('cf_sil', 'CF-Drone 软件在环', 'not_implemented',
                    '尚未移植 PID 控制核；原 ESP32 固件未改动', derived_from='hardcoded'),
        state_check('trainer', '自研 C++/CUDA 学生训练器', 'available_unverified' if trainer else 'not_implemented',
                    '发现训练文件但尚未通过训练验证' if trainer else '尚无已编译训练器；不提供模拟曲线'),
        state_check('model_config', '训练配置', 'ready' if config else 'blocked',
                    '已提供训练配置（未校验内容）' if config else '仅有 .planned 占位，不得启动训练'),
        state_check('edge', 'SG2002 NPU 部署', 'not_implemented',
                    '未提供设备实测 cvimodel / camera→UART 端到端结果', derived_from='hardcoded'),
        state_check('firmware', 'ESP32 固件', 'frozen',
                    '本次只读参考 CF-Drone.zip；没有修改、烧录或验证新固件', derived_from='hardcoded'),
    ]
    return {'version': '7.1', 'generated_at': utc_now(), 'checks': checks,
            'run_count': len(found_runs), 'runs_listed': min(len(found_runs), MAX_RUNS_LISTED),
            'scene_count': valid, 'scene_invalid_count': invalid,
            'read_only': True, 'no_fake_metrics': True}


def provenance():
    hashes = {}
    for rel in PROVENANCE_FILES:
        try:
            hashes[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        except OSError:
            hashes[rel] = None
    return {'file_sha256': hashes}


def audit():
    return {'schema': 'tinyvln.audit.v1', 'generated_at': utc_now(),
            'project': 'TinyAerialVLN V7.1', 'status': status(), 'provenance': provenance(),
            'evidence_levels': {'ready': '文件或工具存在；不等于物理闭环验证',
                                'available_unverified': '仅发现资源，缺运行验收',
                                'blocked': '缺前置依赖',
                                'not_implemented': '功能尚未实现',
                                'frozen': '明确冻结，本次不修改'},
            'explicit_limitations': [
                '尚未实现 Gazebo RGB + 四旋翼动力学闭环',
                '尚未实现 CF-Drone 软件在环',
                '尚未实现学生训练或教师蒸馏',
                '尚无 SG2002 端到端推理实测',
                '没有修改 CF-Drone 的 ESP32 固件',
                '静态障碍场景不等于可飞行的模拟器',
                'derived_from=hardcoded 的状态由人工维护，不是自动验收结果',
                '本页不支持开始训练或修改设备配置'
            ],
            'verification_commands': ['python3 -m unittest discover -s code/tests -p "test_*.py"',
                                      'python3 -m unittest discover -s sim/tests -p "test_*.py"',
                                      './run.sh doctor', './run.sh sim doctor']}


def serve(port):
    allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}  # blocks DNS-rebinding reads

    class Handler(BaseHTTPRequestHandler):
        def fail(self, code, error_code):
            # The reason phrase goes into the latin-1 status line: keep it an ASCII error code.
            self.send_error(code, error_code)

        def do_GET(self):
            if self.headers.get('Host', '') not in allowed_hosts:
                self.fail(421, 'HOST_NOT_ALLOWED')
                return
            path = urlparse(self.path)
            try:
                if path.path in ('/', '/index.html'):
                    body, ctype, code = HTML.read_bytes(), 'text/html; charset=utf-8', 200
                elif path.path == '/api/health':
                    body, ctype, code = {'status': 'ok', 'read_only': True}, 'application/json', 200
                elif path.path == '/api/runs':
                    total = all_runs()
                    body = {'runs': total[:MAX_RUNS_LISTED], 'total': len(total),
                            'truncated': len(total) > MAX_RUNS_LISTED}
                    ctype, code = 'application/json', 200
                elif path.path == '/api/metrics':
                    name = parse_qs(path.query).get('run', [''])[0]
                    body, ctype, code = metrics_for(name), 'application/json', 200
                elif path.path == '/api/scenes':
                    body, ctype, code = {'scenes': scene_list()}, 'application/json', 200
                elif path.path == '/api/status':
                    body, ctype, code = status(), 'application/json', 200
                elif path.path == '/api/audit':
                    body, ctype, code = audit(), 'application/json', 200
                else:
                    self.fail(404, 'NOT_FOUND'); return
                if not isinstance(body, bytes):
                    body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy',
                                 "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                 "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
                                 "frame-ancestors 'none'")
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except ValueError as exc:
                if str(exc) == 'RUN_NOT_FOUND':
                    self.fail(404, 'RUN_NOT_FOUND')
                else:
                    print(f'[DASHBOARD][DASHBOARD_INTERNAL_ERROR] {exc!r}', file=sys.stderr)
                    self.fail(500, 'DASHBOARD_INTERNAL_ERROR')
            except OSError as exc:
                print(f'[DASHBOARD][DASHBOARD_IO_ERROR] {exc!r}', file=sys.stderr)
                self.fail(500, 'DASHBOARD_IO_ERROR')
            except Exception as exc:  # never drop a connection without a response
                print(f'[DASHBOARD][DASHBOARD_INTERNAL_ERROR] {exc!r}', file=sys.stderr)
                self.fail(500, 'DASHBOARD_INTERNAL_ERROR')

        def do_POST(self):
            self.send_error(405, 'READ_ONLY_DASHBOARD')

        def log_message(self, fmt, *args):
            print('[DASHBOARD]', fmt % args)

    try:
        httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except OSError as exc:
        print(f'[ERR][DASHBOARD_PORT_UNAVAILABLE] 127.0.0.1:{port} 无法监听：{exc.strerror}', file=sys.stderr)
        raise SystemExit(4)
    print(f'[OK] TinyAerialVLN V7.1 看板: http://127.0.0.1:{port}/')
    print('[INFO] 本地只读，不执行训练、仿真或固件改动。Ctrl+C 退出。')
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print('\n[INFO] 看板关闭。')
    finally:
        httpd.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TinyAerialVLN V7.1 local read-only cockpit')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('DASHBOARD_PORT_INVALID: 端口必须介于 1024..65535')
    serve(args.port)
