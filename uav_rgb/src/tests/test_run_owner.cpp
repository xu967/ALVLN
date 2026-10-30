#include "uavrgb/run_owner.hpp"

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const std::string& message) {
  if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template <class F>
void rejected(F operation, const char* message) {
  bool threw = false;
  try { operation(); } catch (const std::exception&) { threw = true; }
  check(threw, message);
}
struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() / "uavrgb_owner_XXXXXX").string();
    char* result = ::mkdtemp(pattern.data());
    if (!result) throw std::runtime_error("mkdtemp 失败");
    path = result;
  }
  ~TemporaryDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
int child_status(pid_t child) {
  if (child < 0) throw std::runtime_error("fork 失败");
  int status = 0;
  pid_t result;
  do { result = ::waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
  if (result != child || !WIFEXITED(status)) return -1;
  return WEXITSTATUS(status);
}
}  // namespace

int main(int argc, char** argv) {
  using uavrgb::RunOwner;
  namespace fs = std::filesystem;
  if (argc == 3 && std::string(argv[1]) == "--exec-probe") {
    try { RunOwner owner(argv[2]); return 0; } catch (...) { return 2; }
  }
  try {
    TemporaryDirectory temporary;
    const auto output = temporary.path / "run";
    {
      RunOwner owner(output);
      check(owner.directory() == fs::canonical(output), "输出路径规范化");
      rejected([&] { RunOwner duplicate(output); }, "同进程重复 writer 拒绝");
      fs::create_directory_symlink(output, temporary.path / "alias");
      rejected([&] { RunOwner alias(temporary.path / "alias"); }, "路径别名不能绕开锁");
      const pid_t child = ::fork();
      if (child == 0) {
        try { RunOwner duplicate(output, true); ::_exit(3); }
        catch (const std::exception&) { ::_exit(0); }
      }
      check(child_status(child) == 0, "另一个进程不能同时持有同目录锁");
      RunOwner moved(std::move(owner));
      rejected([&] { RunOwner duplicate(output); }, "move 后仍持锁");
      RunOwner assigned(temporary.path / "other");
      assigned = std::move(moved);
      rejected([&] { RunOwner duplicate(output); }, "move assignment 转移锁");
      RunOwner released(temporary.path / "other");
    }
    { RunOwner reacquired(output); }
    check(fs::exists(output / ".run.lock"), "锁文件可以保留，不能据其存在判锁");

    // --resume 同目录判定：RunOwner 用它决定能否复用非空 out。
    check(uavrgb::same_run_dir(output, output / "ckpt.bin"), "同目录 resume 判定");
    check(!uavrgb::same_run_dir(temporary.path / "other_out", output / "ckpt.bin"),
          "跨目录 resume 判定");
    check(uavrgb::same_run_dir(output, fs::relative(output / "ckpt.bin")),
          "相对路径 resume 判定");
    check(!uavrgb::same_run_dir(output, ""), "空 resume 不算同目录");
    fs::create_directory_symlink(output, temporary.path / "outlink");
    check(uavrgb::same_run_dir(temporary.path / "outlink", output / "ckpt.bin"),
          "软链 out 规范化后判同目录");
    try { RunOwner owner(output); throw std::runtime_error("受控异常"); }
    catch (const std::exception&) {}
    { RunOwner after_exception(output); }

    std::ofstream(output / "results.csv") << "existing result\n";
    rejected([&] { RunOwner nonempty(output); }, "非 resume 拒绝非空目录");
    { RunOwner resume(output, true); }
    std::ifstream original(output / "results.csv");
    std::string line;
    std::getline(original, line);
    check(line == "existing result", "拒绝和 resume 均不改写已有成果");
    rejected([&] { RunOwner empty(""); }, "空目录参数拒绝");

    const auto abnormal = temporary.path / "abnormal";
    const pid_t crashing = ::fork();
    if (crashing == 0) {
      try { RunOwner owner(abnormal); ::_exit(0); } catch (...) { ::_exit(4); }
    }
    check(child_status(crashing) == 0, "子进程持锁后直接退出");
    { RunOwner after_exit(abnormal); }

    const auto exec_path = temporary.path / "exec";
    const pid_t executing = ::fork();
    if (executing == 0) {
      try {
        RunOwner owner(exec_path);
        ::execl("/proc/self/exe", "test_run_owner", "--exec-probe", exec_path.c_str(),
                static_cast<char*>(nullptr));
        ::_exit(5);
      } catch (...) { ::_exit(6); }
    }
    check(child_status(executing) == 0, "O_CLOEXEC 防止 exec 意外继承锁");

    const auto unsafe = temporary.path / "unsafe";
    fs::create_directory(unsafe);
    fs::create_symlink(output / "results.csv", unsafe / ".run.lock");
    rejected([&] { RunOwner symlink_lock(unsafe, true); }, "拒绝锁文件符号链接");
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
  }
  std::cout << "test_run_owner: " << (failures == 0 ? "PASS" : "FAIL") << '\n';
  return failures == 0 ? 0 : 1;
}
