#!/usr/bin/env python3
"""TinyAerialVLN V7.1 评审（第二版）复现脚本 —— 只用 Python 标准库（3.9+）。

用法:  python3 review_v2_repro.py /path/to/TinyAerialVLN_V7_1_工程
脚本把工程复制到临时目录（目录名保留“工程”二字，用于复现 B04），绝不修改原目录。
每条结果为：存在 / 已修复 / 跳过。对原始 V7.1 与打补丁后的工程各跑一次即可对照。
"""
import contextlib
import http.client
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

GOOD = b'{"event":"train_step","step":1,"total_steps":2,"loss":0.5}\n'
PLAN_CONTRACT = [  # V7_1_RESEARCH_PLAN.md §12.3 的四行示例（时间戳占位已替换）
    '{"kind":"train","run_id":"bc_001","step":1200,"loss_total":0.73,"lr":0.0002,"timestamp_utc":"2026-10-28T00:00:00Z"}',
    '{"kind":"sim_step","episode_id":"s1_042","step":34,"sim_time_ns":1700000000,"camera_ts_ns":1666000000,'
    '"policy_action":"turn_right","collision":false}',
    '{"kind":"eval_episode","episode_id":"s1_042","success":false,"collision":true,"stop_reason":"collision",'
    '"path_length_m":12.4}',
    '{"kind":"hardware","seq":125,"camera_to_uart_ms":62.3,"ack_ok":true,"ttl_expired":false}',
]
FAKE_BINARY = '''#!/usr/bin/env bash
out=""; while [[ $# -gt 0 ]]; do [[ "$1" == "--output" ]] && out="$2"; shift; done
echo '{"event":"train_step","step":1,"total_steps":1,"loss":1.0}' > "$out/metrics.jsonl"
'''
MONITOR_PROGRAM = r'''
#include "tinyvln/training_monitor.hpp"
#include <locale>
struct Comma : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};
int main(int, char** argv) {
  std::locale::global(std::locale(std::locale::classic(), new Comma));
  tinyvln::TrainingMonitor m(argv[1]);
  m.Step(26000, 52000, 0.75, 0.5, 2e-4);
  std::cerr << "\nprobe=" << 0.37 << "\n";
  try { m.Step(26001, 52000, 0.0 / 0.0, 0.5, 2e-4); } catch (const std::exception&) { std::cerr << "THROWN\n"; }
}
'''
RESULTS = []


def report(code, title, present, note=''):
    state = '存在' if present else '已修复'
    RESULTS.append(state)
    print(f'[{code:5s}] {state}  {title}' + (f'  —— {note}' if note else ''))


def skip(code, title, why):
    RESULTS.append('跳过')
    print(f'[{code:5s}] 跳过  {title}  —— {why}')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_port(port, seconds=8.0):
    end = time.time() + seconds
    while time.time() < end:
        try:
            socket.create_connection(('127.0.0.1', port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def http_get(port, path, host=None):
    """Return (status, body); status is None when the server closes without any response."""
    try:
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        conn.request('GET', path, headers={'Host': host} if host else {})
        resp = conn.getresponse()
        return resp.status, resp.read()
    except (OSError, http.client.HTTPException) as exc:
        return None, repr(exc).encode()


def run(args, cwd, env=None, timeout=120):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(root, name, payload):
    d = root / 'artifacts' / 'runs' / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'metrics.jsonl').write_bytes(payload)


def check_scenes(root):
    st = load_module('scene_tool_review', root / 'tools' / 'scene_tool.py')
    clearance = min(min(abs(o['y']) - o['r'] for o in obs)
                    for preset in st.PRESETS for seed in range(300)
                    for obs in [st.create_obstacles(preset, seed, 200)] if obs)
    report('P0-2', '起点→终点直线在所有 preset/seed 下都畅通', clearance > 0.3, f'最小净空 {clearance:.2f} m')
    same = st.make_sdf(st.create_obstacles('forest', 7, 40), 7) == st.make_sdf(st.create_obstacles('mixed', 7, 40), 7)
    report('P0-2b', 'mixed preset 与 forest 输出完全相同', same)
    ignored = [line.strip() for line in (root / '.gitignore').read_text().splitlines()]
    report('B16c', '.gitignore 未忽略 build/', not any(x in ('build/', 'build', '/build/') for x in ignored))


def check_dashboard_http(root):
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(root / 'tools' / 'dashboard.py'), '--port', str(port)],
                            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_port(port):
            skip('B01-7', '看板 HTTP 检查', '看板没有启动')
            return
        status, _ = http_get(port, '/api/audit', host=f'evil.example:{port}')
        report('B05', '伪造 Host 头仍可读取 /api/audit（DNS rebinding）', status == 200, f'HTTP {status}')

        write_run(root, 'r_nested', GOOD + b'{"event":"train_step","extra":{"grad_norm":NaN}}\n')
        status, _ = http_get(port, '/api/metrics?run=r_nested')
        report('B01', '一行嵌套 NaN 让整个 run 的指标接口失效', status != 200, f'HTTP {status}')

        write_run(root, 'r_deep', GOOD + b'{"event":"x","v":' + b'[' * 30000 + b']' * 30000 + b'}\n')
        write_run(root, 'r_bigint', GOOD + b'{"event":"x","v":' + b'9' * 5000 + b'}\n')
        s1, _ = http_get(port, '/api/metrics?run=r_deep')
        s2, _ = http_get(port, '/api/metrics?run=r_bigint')
        report('B02', '深层嵌套 / 超长整数行让接口断连或 404', s1 != 200 or s2 != 200,
               f'deep→{s1 or "无响应"}, bigint→{s2 or "无响应"}')

        write_run(root, 'plan_contract', ('\n'.join(PLAN_CONTRACT) + '\n').encode())
        status, body = http_get(port, '/api/metrics?run=plan_contract')
        shown = len(json.loads(body)['events']) if status == 200 else 0
        report('P1-2', '方案 §12.3 的 JSONL 契约示例被看板全部拒收', shown == 0,
               f'显示 {shown}/4 条（补丁不改契约，需你决定以哪份为准）')

        scenes = root / 'artifacts' / 'scenes'
        scenes.mkdir(parents=True, exist_ok=True)
        bad = scenes / 'zz_listroot.json'
        bad.write_text('[]')
        status, _ = http_get(port, '/api/status')
        report('B03', '根为 [] 的场景 manifest 让 /api/status 断连', status != 200, f'HTTP {status or "无响应"}')
        bad.unlink()

        backup = root / 'artifacts' / '_manifest_backup'
        backup.mkdir(exist_ok=True)
        for f in scenes.glob('*.json'):
            f.rename(backup / f.name)
        (scenes / 'a.json').write_text('{broken')
        (scenes / 'b.json').write_text('{broken')
        status, body = http_get(port, '/api/status')
        state = {c['key']: c for c in json.loads(body)['checks']}['scenes']['state'] if status == 200 else None
        report('B06', 'manifest 全部损坏时 scenes 仍为 ready', state == 'ready', f'state={state}')
        for f in ('a.json', 'b.json'):
            (scenes / f).unlink()
        for f in backup.glob('*.json'):
            f.rename(scenes / f.name)
        backup.rmdir()

        for i in range(85):
            write_run(root, f'bulk_{i:03}', GOOD)
        actual = sum(1 for p in (root / 'artifacts' / 'runs').iterdir() if (p / 'metrics.jsonl').is_file())
        status, body = http_get(port, '/api/status')
        count = json.loads(body)['run_count'] if status == 200 else None
        report('B07', '超过 80 个 run 时计数与列表被静默截断', count != actual, f'run_count={count}，实际 {actual}')
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(root / 'artifacts' / 'runs', ignore_errors=True)


def check_io_error_status_line(root):
    if all(ord(ch) < 256 for ch in str(root)):
        skip('B04', 'OSError 消息含中文路径时无响应', '临时目录名不含非 latin-1 字符')
        return
    dash = load_module('dashboard_review', root / 'tools' / 'dashboard.py')

    def boom(_name):
        raise PermissionError(13, 'Permission denied', str(dash.ROOT / 'artifacts' / 'runs' / 'x' / 'metrics.jsonl'))
    dash.metrics_for = boom
    port = free_port()
    sink = io.StringIO()  # keep the expected server traceback out of the report
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        threading.Thread(target=dash.serve, args=(port,), daemon=True).start()
        wait_port(port)
        status, _ = http_get(port, '/api/metrics?run=x')
        time.sleep(0.2)
    report('B04', 'OSError 消息含中文路径时连接无响应（文档声称 500）', status != 500, f'HTTP {status or "无响应"}')


def check_launcher(root):
    env = dict(os.environ)
    fakebin = root.parent / 'fakebin'
    fakebin.mkdir(exist_ok=True)
    (fakebin / 'nvcc').write_text('#!/bin/sh\nexit 1\n')
    (fakebin / 'nvcc').chmod(0o755)
    env['PATH'] = f'{fakebin}{os.pathsep}{env.get("PATH", "")}'
    r = run([str(root / 'run.sh'), 'doctor'], root, env=env)
    report('B11', 'doctor 只凭 which 标 [OK]（坏掉的 nvcc 也算 OK）', '[OK] nvcc' in r.stdout, f'exit={r.returncode}')

    r = run([str(root / 'run.sh'), 'board', 'doctor'], root)
    report('B10', '方案 §11 的 board/uart 子命令不报 FEATURE_NOT_IMPLEMENTED',
           'FEATURE_NOT_IMPLEMENTED' not in r.stderr, f'exit={r.returncode}')

    build, config = root / 'build', root / 'config'
    build.mkdir(exist_ok=True)
    for name in ('tinyvln_train', 'tinyvln_eval_bridge'):
        (build / name).write_text(FAKE_BINARY)
        (build / name).chmod(0o755)
    (config / 'experiment.yaml').write_text('seed: 1\n')

    r = run([str(root / 'run.sh'), 'train', '--run-id', '实验一'], root)
    report('B08', 'launcher 接受看板永远不显示的非 ASCII run ID', r.returncode == 0, f'exit={r.returncode}')

    run([str(root / 'run.sh'), 'train', '--run-id', 'bc_x'], root)
    r = run([str(root / 'run.sh'), 'eval', '--run-id', 'bc_x'], root)
    report('B09', '按方案 §11 先 train 再 eval 同一 run-id 会失败', r.returncode != 0,
           f'eval exit={r.returncode} {r.stderr.strip()[:40]}')

    r = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_infrastructure.py',
             '-k', 'test_refuses_nonexistent_training_binary'], root)
    launched = (root / 'artifacts' / 'runs' / 'forbidden_test').exists()
    report('B16b', '训练器编译后，现有测试会真的启动训练并失败', launched or 'FAILED' in r.stderr,
           '已创建 runs/forbidden_test' if launched else '')
    shutil.rmtree(build, ignore_errors=True)
    (config / 'experiment.yaml').unlink()


def check_sim_preflight(root):
    world = root.parent / 'velocity_only.sdf'
    world.write_text('<sdf version="1.9"><world name="w"><plugin filename="gz-sim-multicopter-control-system" '
                     'name="gz::sim::systems::MulticopterVelocityControl"/></world></sdf>')
    r = run([str(root / 'run.sh'), 'sim', 'smoke', '--world', str(world), '--dry-run'], root)
    report('B15', '只有速度控制插件的 world 通过“电机模型”检查', 'GZ_MOTOR_MODEL_MISSING' not in r.stderr,
           f'exit={r.returncode}')


def check_monitor_header(root):
    if not shutil.which('g++'):
        skip('B12-14', 'training_monitor.hpp', '未找到 g++')
        return
    work = root.parent / 'hpp'
    work.mkdir(exist_ok=True)
    (work / 't.cpp').write_text(MONITOR_PROGRAM)
    build = run(['g++', '-std=c++17', '-I', str(root / 'code' / 'include'), str(work / 't.cpp'),
                 '-o', str(work / 't')], root)
    if build.returncode:
        skip('B12-14', 'training_monitor.hpp', '编译失败：' + build.stderr[:80])
        return
    log = work / 'm.jsonl'
    r = run([str(work / 't'), str(log)], root)
    lines = log.read_text().splitlines()
    valid = True
    try:
        json.loads(lines[0])
    except ValueError:
        valid = False
    report('B13', '全局 locale 带千分位/逗号小数时写出非法 JSON', not valid, lines[0][:60])
    report('B12', 'Step() 之后 std::cerr 被改成 fixed+precision(0)', 'probe=0.37' not in r.stderr,
           r.stderr.split('probe=')[-1].split('\n')[0] if 'probe=' in r.stderr else '')
    report('B14', 'NaN 时先抛异常、不写错误事件', not any('TRAIN_NONFINITE_METRIC' in line for line in lines))


def check_fresh_clone(source):
    if not shutil.which('git'):
        skip('B16a', '全新 git clone 后测试失败', '未找到 git')
        return
    base = Path(tempfile.mkdtemp(prefix='tinyvln_clone_'))
    repo = base / 'repo'
    shutil.copytree(source, repo, ignore=shutil.ignore_patterns('__pycache__', 'build', 'runs'))
    git = ['git', '-c', 'user.name=review', '-c', 'user.email=review@example.invalid', '-c', 'init.defaultBranch=main']
    run(git + ['init', '-q'], repo)
    run(git + ['add', '-A'], repo)
    run(git + ['commit', '-q', '-m', 'snapshot'], repo)
    run(['git', 'clone', '-q', str(repo), str(base / 'clone')], base)
    clone = base / 'clone'
    (clone / 'run.sh').chmod(0o755)
    a = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'code/tests', '-p', 'test_*.py'], clone)
    b = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'sim/tests', '-p', 'test_*.py'], clone)
    broken = 'FAILED' in a.stderr or 'FAILED' in b.stderr
    report('B16a', '全新 git clone（artifacts/ 被忽略）后测试失败', broken,
           f"code/tests: {a.stderr.strip().splitlines()[-1]}; sim/tests: {b.stderr.strip().splitlines()[-1]}")
    shutil.rmtree(base, ignore_errors=True)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    source = Path(sys.argv[1]).resolve()
    if not (source / 'tools' / 'dashboard.py').is_file():
        print(f'[ERR] {source} 不是 TinyAerialVLN 工程根目录')
        return 2
    tmp = Path(tempfile.mkdtemp(prefix='tinyvln_review_'))
    root = tmp / 'TinyAerialVLN_V7_1_工程'
    shutil.copytree(source, root, ignore=shutil.ignore_patterns('__pycache__', 'build', 'runs'))
    (root / 'run.sh').chmod(0o755)
    print(f'被测工程: {source}\n临时副本: {root}\n')
    check_scenes(root)
    check_dashboard_http(root)
    check_io_error_status_line(root)
    check_launcher(root)
    check_sim_preflight(root)
    check_monitor_header(root)
    check_fresh_clone(source)
    print(f"\n汇总：存在 {RESULTS.count('存在')}，已修复 {RESULTS.count('已修复')}，跳过 {RESULTS.count('跳过')}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
