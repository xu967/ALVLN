"""tools/test_preprocess_*.py 共用的测试辅助（卡 T2a，审核 P-10）。

名字不以 test_ 开头：tools/check.sh 的 `tools/test_*.py` 通配不会把它当测试跑。只放三样东西：

1. SharedFixture：只读共享夹具。在 TMPDIR 下造一次；用例只许拷出去再改（copy_to）；每个用例结束与模块结束时
   核对夹具内容的哈希没变（verify），谁改了夹具会在谁的用例上红；模块结束清掉（cleanup）。
2. run_preprocess：进程内调 preprocess.main()，返回与 subprocess.run(capture_output=True, text=True) 同形的
   CompletedProcess，省掉每次起解释器、导 numpy / Pillow 的开销（P-10：打包四个文件 84 s 都花在这里）。
   对齐「python tools/preprocess.py ...」的可观察行为：sys.argv[0] 取脚本名（argparse 的 prog）；
   SystemExit 的整数码、None、字符串码（解释器把字符串打到 stderr，退出码 1）折成退出码；
   未捕获的异常把 traceback 写进 stderr，退出码 1；调用前后恢复 sys.argv、工作目录与环境变量。
3. run_preprocess_cli：真子进程。每个测试文件留两个用例走它，锁命令行本身的退出码与标准错误的关键字；
   其余用例走进程内版，两者的等价靠这批用例与「全部用例改走真子进程也全绿」的对拍保证。

读 preprocess.py 的 main()：它不收参数，用 argparse 读 sys.argv[1:]；成功返回 0，die() 是 sys.exit("preprocess: ...")，
argparse 的 error() 是退出码 2。模块级没有会跨调用残留的可变状态（没有缓存与全局累加器），重复调用 main() 互不影响。
"""
import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

PY = sys.executable
PREPROCESS = TOOLS / "preprocess.py"


# ---------------------------------------------------------------- 只读共享夹具
def _manifest(root):
    """目录下每个文件的 sha256（目录记 <dir>），键是相对路径。"""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "<dir>"
    return out


class SharedFixture:
    """只读共享夹具：build(root) 造一次，用例只许 copy_to 出去改，verify() 核对内容哈希没变。"""

    def __init__(self, tag, build):
        self._tmp = tempfile.TemporaryDirectory(prefix="t2a-%s-" % tag)
        self.path = Path(self._tmp.name)
        try:
            build(self.path)
        except BaseException:
            self._tmp.cleanup()
            raise
        self._manifest = _manifest(self.path)
        self.digest = hashlib.sha256(
            "\n".join("%s %s" % kv for kv in sorted(self._manifest.items())).encode()).hexdigest()

    def copy_to(self, dst):
        shutil.copytree(self.path, dst)

    def verify(self):
        now = _manifest(self.path)
        if now == self._manifest:
            return
        changed = sorted(k for k in now.keys() & self._manifest.keys() if now[k] != self._manifest[k])
        raise AssertionError("共享夹具 %s 的内容变了（用例只许拷出去改）：改动 %s；新增 %s；删除 %s" % (
            self.path, changed, sorted(now.keys() - self._manifest.keys()),
            sorted(self._manifest.keys() - now.keys())))

    def cleanup(self):
        self._tmp.cleanup()


def make_synthetic(root):
    """在 root 下造 3 条合成 episode（ep1 / ep2 / ep3，15 个样本）。进程内调 make_synthetic_episode.main()，
    产物与 `python tools/make_synthetic_episode.py --out root` 逐字节相同。"""
    import make_synthetic_episode
    r = _run_main(make_synthetic_episode.main, "make_synthetic_episode.py", ["--out", str(root)])
    if r.returncode != 0:
        raise RuntimeError("合成 episode 生成失败：rc=%d\n%s%s" % (r.returncode, r.stdout, r.stderr))


def synthetic_fixture():
    """只含原始合成 episode 的共享夹具；要在夹具里再加工的文件自己写 build(root)，开头调 make_synthetic(root)。"""
    return SharedFixture("synthetic", make_synthetic)


# ---------------------------------------------------------------- 进程内入口
def _exit_status(code, err):
    """SystemExit.code（或 main() 的返回值）按解释器的规则折成退出码。"""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    err.write("%s\n" % (code,))   # sys.exit("msg")：解释器把 msg 打到 stderr，退出码 1
    return 1


def _run_main(main, prog, args):
    out, err = io.StringIO(), io.StringIO()
    saved_argv, saved_cwd, saved_env = sys.argv, os.getcwd(), dict(os.environ)
    sys.argv = [prog, *map(str, args)]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = _exit_status(main(), err)
            except SystemExit as exc:
                rc = _exit_status(exc.code, err)
            except Exception:
                traceback.print_exc(file=err)
                rc = 1
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        for key in set(os.environ) - set(saved_env):
            del os.environ[key]
        for key, val in saved_env.items():
            if os.environ.get(key) != val:
                os.environ[key] = val
    return subprocess.CompletedProcess([prog, *map(str, args)], rc, out.getvalue(), err.getvalue())


def run_preprocess(*args):
    """进程内等价于 `python tools/preprocess.py <args>`。"""
    import preprocess
    return _run_main(preprocess.main, "preprocess.py", args)


def run_preprocess_cli(*args, script=None):
    """真子进程：`python tools/preprocess.py <args>`；script 可指到别的副本（对拍旧版输出用）。"""
    return subprocess.run([PY, str(script or PREPROCESS), *map(str, args)], capture_output=True, text=True)
