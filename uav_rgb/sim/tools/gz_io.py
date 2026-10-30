#!/usr/bin/env python3
"""常驻 gz topic 订阅 / 发布：一个 `gz topic -e --json-output` 进程 + 后台读线程 + 环形缓冲。

为什么：每帧 spawn 一次 `gz topic -e -n 1` 本机约 250 ms（4 个话题 ≈ 1 s/步）；
`gz topic -p` 每次固定 sleep 0.8 s 再发布（实测 ~0.96 s）。常驻订阅后取帧是内存
操作；常驻 publisher 后每条指令是内存级 publish（实测 <0.1 ms）。缓冲是
latest-wins（deque(maxlen=N)），不做队列积压：30 Hz 下积压 100 帧就是 3 s 延迟。

用法：
    sub = TopicSub("/uav/camera")
    msg = sub.latest_after(stamp_ns, timeout=2.0)   # 无数据/进程退出/超时都返回 None
    sub.close()

    pub = open_twist_publisher("/X3/gazebo/command/twist")   # 建不起来抛 PublisherUnavailable
    h = pub.start(0.3, 0.0, 0.05, 0.0)   # 发布并返回完成时刻/发布句柄
    pub.wait(h)
    pub.close()

自检（不需要 sim）：
    python3 gz_io.py --self-check
"""
import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time

PUBLISHER_START_TIMEOUT = 5.0   # 等订阅者发现的上限；超时即判 publisher 不可用
PUBLISH_TIMEOUT = 2.0           # helper 单条指令等 ack 的上限
HELPER_ENV = "UAVRGB_GZ_PYTHON"  # 覆盖 helper 解释器（默认 /usr/bin/python3）
DEFAULT_GZ_PYTHON = "/usr/bin/python3"


class PublisherUnavailable(RuntimeError):
    """常驻 twist publisher 建不起来；定量采集应失败，不要回退 `gz topic -p` CLI。"""


def header_stamp_ns(msg):
    """gz.msgs 的 header.stamp -> 纳秒；Pose_V 的戳在 pose[0].header.stamp。

    缺字段返回 0（latest_after 用严格大于，0 永远不会被误认为有效帧）。
    """
    try:
        stamp = msg["header"]["stamp"]
    except (KeyError, TypeError):
        try:
            stamp = msg["pose"][0]["header"]["stamp"]
        except (KeyError, TypeError, IndexError):
            return 0
    try:
        return int(stamp.get("sec", 0)) * 10**9 + int(stamp.get("nsec", 0))
    except (AttributeError, ValueError):
        return 0


class TopicSub:
    """一个话题的常驻订阅。线程安全：latest/frames/latest_after 都加锁。

    stamp_fn 决定 latest_after 用哪个时间戳比较（默认 header.stamp；
    /model/x3/pose 是 Pose_V，时间戳在 pose[0].header.stamp，由调用方传入）。
    """

    def __init__(self, topic, buf=4, stamp_fn=header_stamp_ns):
        self.topic = topic
        self._stamp = stamp_fn
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frames = collections.deque(maxlen=buf)   # (seq, msg)，seq 单调递增
        self._seq = 0
        self._eof = False
        self._proc = subprocess.Popen(
            ["gz", "topic", "-e", "-t", topic, "--json-output"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read_loop, daemon=True,
                                        name="gz-sub:" + topic)
        self._thread.start()

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue                      # 坏行（如半截 JSON）跳过
                with self._cond:
                    self._seq += 1
                    self._frames.append((self._seq, msg))
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def healthy(self):
        """订阅进程还活着且读线程没结束。注意：进程活着但话题静默时也是 True。"""
        return not self._eof and self._proc.poll() is None

    def latest(self):
        """最新一帧（dict）或 None。"""
        with self._lock:
            return self._frames[-1][1] if self._frames else None

    def frames(self):
        """缓冲快照 [(seq, msg), ...]，旧 -> 新。"""
        with self._lock:
            return list(self._frames)

    def latest_after(self, stamp_ns, timeout=0.0):
        """sim-time 戳严格大于 stamp_ns 的最新一帧；无数据最多等 timeout 秒。

        进程退出或超时返回 None（不会永久阻塞）。
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._cond:
                for _, msg in reversed(self._frames):
                    if self._stamp(msg) > stamp_ns:
                        return msg
                if self._eof:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def close(self):
        """terminate -> wait -> kill 兜底，再收读线程。可重复调用。"""
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._thread.join(timeout=2.0)
        try:
            self._proc.stdout.close()
        except OSError:
            pass


# ---------------------------------------------------------------- 常驻发布

class _InProcTwistPublisher:
    """进程内 gz.transport13 publisher：publish 只写共享内存，实测 <0.1 ms。"""

    mode = "gz.transport13（进程内）"

    def __init__(self, topic, connect_timeout=PUBLISHER_START_TIMEOUT):
        from gz.transport13 import Node
        from gz.msgs10.twist_pb2 import Twist
        self._Twist = Twist
        self._node = Node()
        self._pub = self._node.advertise(topic, Twist)
        # 发现完成前的 publish 会丢；等订阅者出现（sim 已在跑时约 0.2 s）。
        deadline = time.monotonic() + connect_timeout
        while not self._pub.has_connections():
            if time.monotonic() >= deadline:
                raise PublisherUnavailable(
                    f"{topic}: {connect_timeout:.0f}s 内没有订阅者连接")
            time.sleep(0.02)

    def start(self, x, y, z, wz):
        msg = self._Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = x, y, z
        msg.angular.z = wz
        self._pub.publish(msg)
        return time.perf_counter()          # 发布完成时刻

    def wait(self, handle):
        return handle

    def abort(self, handle):
        pass                                 # 进程内 publish 同步完成，无进程可收

    def close(self):
        self._pub = None
        self._node = None


class _HelperTwistPublisher:
    """当前解释器没有 gz.transport13 时，用带该模块的解释器跑一个常驻发布子进程。

    父进程（例如 conda python）通过 stdin/stdout 发指令、收 ack；每条约 0.2 ms。
    子进程忽略 SIGINT，生命周期由 stdin 的 "q"/EOF 控制，父进程死了也会自行退出。
    """

    def __init__(self, topic, connect_timeout=PUBLISHER_START_TIMEOUT):
        exe = os.environ.get(HELPER_ENV, DEFAULT_GZ_PYTHON)
        try:
            self._proc = subprocess.Popen(
                [exe, "-I", os.path.abspath(__file__), "--pub-server", topic],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError as e:
            raise PublisherUnavailable(f"启动 {exe} 失败: {e}") from e
        self.mode = f"gz.transport13（helper: {exe}）"
        self._closed = False
        self._acks = queue.Queue()
        threading.Thread(target=self._pump, daemon=True,
                         name="gz-pub-helper").start()
        ready = self._readline(connect_timeout)
        if ready != "ready":
            self.close()
            raise PublisherUnavailable(
                f"{exe} helper 未就绪: {ready or '超时/已退出'}")

    def _pump(self):
        try:
            for line in self._proc.stdout:
                self._acks.put(line.strip())
        finally:
            self._acks.put(None)

    def _readline(self, timeout):
        try:
            return self._acks.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self, x, y, z, wz):
        if self._closed or self._proc.poll() is not None:
            raise RuntimeError("publisher helper 已退出")
        try:
            self._proc.stdin.write(f"{x:.6f} {y:.6f} {z:.6f} {wz:.6f}\n")
            self._proc.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"写 publisher helper 失败: {e}") from e
        reply = self._readline(PUBLISH_TIMEOUT)
        if reply != "ok":
            # 超时/进程退出后不能再信任这条通道，后续 publish 一律报错走 CLI 兜底。
            self.close()
            raise RuntimeError(f"publisher helper 发布失败: {reply or '超时/已退出'}")
        return time.perf_counter()          # 发布完成时刻

    def wait(self, handle):
        return handle

    def abort(self, handle):
        pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdin.write("q\n")
            self._proc.stdin.flush()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        try:
            self._proc.stdin.close()
        except OSError:
            pass


def open_twist_publisher(topic, connect_timeout=PUBLISHER_START_TIMEOUT):
    """建常驻 twist publisher：优先当前解释器内，否者用 helper 子进程。

    都失败抛 PublisherUnavailable。定量采集应把失败当错误：`gz topic -p` CLI
    每条约 1 s，动作窗口会失真；CLI 只保留给零速安全网。
    """
    try:
        from gz.transport13 import Node  # noqa: F401 - 只探测能否导入
    except Exception as e:  # noqa: BLE001 - 缺模块/共享库加载失败都算不可用
        inproc_err = f"当前解释器无 gz.transport13（{e}）"
    else:
        return _InProcTwistPublisher(topic, connect_timeout)
    try:
        pub = _HelperTwistPublisher(topic, connect_timeout)
    except Exception as e:  # noqa: BLE001 - helper 任何启动失败都回退
        raise PublisherUnavailable(
            f"{inproc_err}；helper 不可用（{e}）") from e
    pub.warning = f"{inproc_err}，改用 helper 子进程发布"
    return pub


def _pub_server(topic, connect_timeout=PUBLISHER_START_TIMEOUT):
    """helper 子进程入口：advertise 后从 stdin 逐行收 twist，发布并回 ack。"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # 收尾交给父进程；父死 EOF 即退
    from gz.transport13 import Node
    from gz.msgs10.twist_pb2 import Twist
    node = Node()
    pub = node.advertise(topic, Twist)
    deadline = time.monotonic() + connect_timeout
    while not pub.has_connections():
        if time.monotonic() >= deadline:
            print(f"error {topic}: {connect_timeout:.0f}s 内没有订阅者连接",
                  flush=True)
            return 1
        time.sleep(0.02)
    print("ready", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "q":
            break
        if not line:
            continue
        try:
            x, y, z, wz = (float(v) for v in line.split())
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.linear.z = x, y, z
            msg.angular.z = wz
            pub.publish(msg)
            print("ok", flush=True)
        except Exception as e:  # noqa: BLE001 - 错误回给父进程，不静默
            print(f"error {type(e).__name__}: {e}", flush=True)
    return 0


def _self_check():
    """不依赖 sim：假 stdout 验证坏行跳过 / 环形缓冲上限 / latest_after / EOF。"""
    import io

    class FakeProc:
        def __init__(self, text):
            self.stdout = io.StringIO(text)

        def poll(self):
            return 0            # 流读完即视为进程退出

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    lines = 'not json\n{"n": 1}\n\n{"n": 2}\n{"n": 3}\n{"n": 4}\n'
    orig = subprocess.Popen
    subprocess.Popen = lambda *a, **k: FakeProc(lines)
    try:
        sub = TopicSub("/fake", buf=2, stamp_fn=lambda m: m["n"])
        try:
            import time as _time
            for _ in range(100):
                if sub._eof:
                    break
                _time.sleep(0.01)
            assert sub.latest() == {"n": 4}, sub.latest()
            assert sub.latest_after(2) == {"n": 4}, sub.latest_after(2)
            assert sub.latest_after(0, timeout=0.2) == {"n": 4}
            t0 = _time.monotonic()
            assert sub.latest_after(4, timeout=0.2) is None      # 无新帧 -> 超时返回
            assert _time.monotonic() - t0 < 0.5
            assert [m["n"] for _, m in sub.frames()] == [3, 4]   # 只留最近 2 帧
            assert not sub.healthy()                             # EOF -> unhealthy
        finally:
            sub.close()
    finally:
        subprocess.Popen = orig

    # helper 发布通道：用假解释器（只讲协议）验证 ready/ack/错误/close，不需要 gz。
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gz_io_selfcheck_")
    old_env = os.environ.get(HELPER_ENV)
    try:
        fake = os.path.join(tmp, "fake_pub_server.py")
        with open(fake, "w") as f:
            f.write("import sys\n"
                    "print('ready', flush=True)\n"
                    "for line in sys.stdin:\n"
                    "    line = line.strip()\n"
                    "    if line == 'q':\n"
                    "        break\n"
                    "    vals = [float(v) for v in line.split()]\n"
                    "    print('error boom' if vals[0] < 0 else 'ok', flush=True)\n")
        wrapper = os.path.join(tmp, "fake_python.sh")
        with open(wrapper, "w") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" "{fake}"\n')
        os.chmod(wrapper, 0o755)
        os.environ[HELPER_ENV] = wrapper
        pub = _HelperTwistPublisher("/fake/twist", connect_timeout=2.0)
        try:
            h = pub.start(0.3, 0.0, 0.05, 0.0)
            assert pub.wait(h) == h
            try:
                pub.start(-1.0, 0.0, 0.0, 0.0)     # 假 helper 回 error
                raise AssertionError("helper 错误应答应抛 RuntimeError")
            except RuntimeError:
                pass
        finally:
            pub.close()
            pub.close()                            # 幂等
        try:
            pub.start(0.0, 0.0, 0.0, 0.0)          # 关闭后必须报错，由调用方兜底
            raise AssertionError("关闭后 publish 应抛 RuntimeError")
        except RuntimeError:
            pass
    finally:
        if old_env is None:
            os.environ.pop(HELPER_ENV, None)
        else:
            os.environ[HELPER_ENV] = old_env
        shutil.rmtree(tmp, ignore_errors=True)
    print("gz_io self-check: 全部通过")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
        raise SystemExit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "--pub-server":
        try:
            raise SystemExit(_pub_server(sys.argv[2]))
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - 启动失败回给父进程
            print(f"error {type(e).__name__}: {e}", flush=True)
            raise SystemExit(1)
    topic = sys.argv[1] if len(sys.argv) > 1 else "/uav/camera"
    sub = TopicSub(topic)
    try:
        msg = sub.latest_after(0, timeout=5.0)
        if msg is None:
            print(f"{topic}: 5s 内无消息 (healthy={sub.healthy()})")
            raise SystemExit(1)
        print(f"{topic}: stamp={header_stamp_ns(msg)} healthy={sub.healthy()} "
              f"frames={len(sub.frames())}")
    finally:
        sub.close()
