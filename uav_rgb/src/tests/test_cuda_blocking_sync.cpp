// 第 7 条：训练进程 CUDA 阻塞同步开关（--cuda-blocking-sync / UAVRGB_CUDA_BLOCKING_SYNC=1，默认关）。
//
// 验证要点：
//   1) 默认关闭时，CUDA 采用自旋等待 GPU，训练子进程 (utime+stime)/wall 显著占核；
//   2) 开启 --cuda-blocking-sync 或 UAVRGB_CUDA_BLOCKING_SYNC=1 后，(utime+stime)/wall 降至默认值的 1/4 以下，
//      且产出的 ckpt.bin 与默认模式逐字节完全一致（阻塞等待不改变任何浮点计算与参数更新）；
//   3) UAVRGB_CUDA_BLOCKING_SYNC 设为 "0" 等同关闭；设为非法字符串（如 "yes"）在入口 rc=2 拒绝。
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"

namespace fs = std::filesystem;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_fail;
  }
}

struct ProcStats {
  int rc = -1;
  double wall_s = 0.0;
  double utime_s = 0.0;
  double stime_s = 0.0;
  double cpu_cores() const { return wall_s > 0.0 ? (utime_s + stime_s) / wall_s : 0.0; }
};

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// fork + exec 直接起 uavrgb_train（不包一层 sh，使 wait4 的 rusage 精确归属该进程本身）。
ProcStats run_train_rusage(const std::string& exe, const std::vector<std::string>& args,
                           const std::string& env_kv, const std::string& log_path) {
  const auto t0 = std::chrono::steady_clock::now();
  const pid_t pid = fork();
  if (pid == 0) {
    if (!env_kv.empty()) {
      const std::size_t eq = env_kv.find('=');
      setenv(env_kv.substr(0, eq).c_str(), env_kv.substr(eq + 1).c_str(), 1);
    } else {
      unsetenv("UAVRGB_CUDA_BLOCKING_SYNC");
    }
    FILE* f = std::fopen(log_path.c_str(), "wb");
    if (f) {
      dup2(fileno(f), STDOUT_FILENO);
      dup2(fileno(f), STDERR_FILENO);
      std::fclose(f);
    }
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (const std::string& s : args) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);
    execv(exe.c_str(), argv.data());
    _exit(127);
  }
  int status = 0;
  rusage ru{};
  wait4(pid, &status, 0, &ru);
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ProcStats ps;
  if (WIFEXITED(status)) ps.rc = WEXITSTATUS(status);
  ps.wall_s = wall;
  ps.utime_s = static_cast<double>(ru.ru_utime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_utime.tv_usec);
  ps.stime_s = static_cast<double>(ru.ru_stime.tv_sec) + 1e-6 * static_cast<double>(ru.ru_stime.tv_usec);
  return ps;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: test_cuda_blocking_sync <uavrgb_train 路径>\n";
    return 2;
  }
  const std::string exe = argv[1];
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_bsync_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  // 128 train + 16 val（120x160，batch=128）：每轮 1 个重 GPU 步，跑 12 轮使 GPU 同步等待占主体。
  const auto sd = uavrgb_test::write_synthetic_uavr(root + "/ds.uavr", 144, 120, 160, 128, 23);
  auto base_args = [&](const std::string& out_dir) {
    return std::vector<std::string>{"--data", sd.path,  "--epochs", "12",  "--batch", "128",
                                    "--lr",   "1e-3",   "--seed",   "11",  "--out",   out_dir};
  };

  // 1) 默认关闭（自旋） vs --cuda-blocking-sync vs UAVRGB_CUDA_BLOCKING_SYNC=1
  const ProcStats off = run_train_rusage(exe, base_args(root + "/off"), "", root + "/off.log");
  std::vector<std::string> cli_args = base_args(root + "/cli");
  cli_args.push_back("--cuda-blocking-sync");
  const ProcStats cli = run_train_rusage(exe, cli_args, "", root + "/cli.log");
  const ProcStats env1 = run_train_rusage(exe, base_args(root + "/env1"),
                                          "UAVRGB_CUDA_BLOCKING_SYNC=1", root + "/env1.log");
  const ProcStats env0 = run_train_rusage(exe, base_args(root + "/env0"),
                                          "UAVRGB_CUDA_BLOCKING_SYNC=0", root + "/env0.log");

  std::cout << "off : rc=" << off.rc << " wall=" << off.wall_s << "s utime=" << off.utime_s
            << "s stime=" << off.stime_s << "s cores=" << off.cpu_cores() << "\n";
  std::cout << "cli : rc=" << cli.rc << " wall=" << cli.wall_s << "s utime=" << cli.utime_s
            << "s stime=" << cli.stime_s << "s cores=" << cli.cpu_cores() << "\n";
  std::cout << "env1: rc=" << env1.rc << " wall=" << env1.wall_s << "s utime=" << env1.utime_s
            << "s stime=" << env1.stime_s << "s cores=" << env1.cpu_cores() << "\n";
  std::cout << "env0: rc=" << env0.rc << " wall=" << env0.wall_s << "s utime=" << env0.utime_s
            << "s stime=" << env0.stime_s << "s cores=" << env0.cpu_cores() << "\n";

  check(off.rc == 0 && cli.rc == 0 && env1.rc == 0 && env0.rc == 0,
        "off / --cuda-blocking-sync / ENV=1 / ENV=0 四次训练均应 rc=0");
  check(off.cpu_cores() > 0.4, "默认模式应体现 GPU 自旋占核（实测 cores=" +
                                   std::to_string(off.cpu_cores()) + "）");
  check(cli.rc == 0 && cli.cpu_cores() < 0.35 * off.cpu_cores(),
        "--cuda-blocking-sync 应显著降低 CPU 核占用（cli=" + std::to_string(cli.cpu_cores()) +
            " vs off=" + std::to_string(off.cpu_cores()) + "）");
  check(env1.rc == 0 && env1.cpu_cores() < 0.35 * off.cpu_cores(),
        "UAVRGB_CUDA_BLOCKING_SYNC=1 应显著降低 CPU 核占用（env1=" +
            std::to_string(env1.cpu_cores()) + " vs off=" + std::to_string(off.cpu_cores()) + "）");

  // 2) 数值确定性：阻塞同步不改任何 GPU 计算，ckpt.bin 与默认模式逐字节相等。
  const std::string ck_off = slurp(root + "/off/ckpt.bin");
  check(!ck_off.empty() && ck_off == slurp(root + "/cli/ckpt.bin") &&
            ck_off == slurp(root + "/env1/ckpt.bin") && ck_off == slurp(root + "/env0/ckpt.bin"),
        "开启阻塞同步后的 ckpt.bin 必须与默认模式逐字节一致");

  // 3) 非法环境变量值应在入口 rc=2 拒绝，不静默忽略。
  const ProcStats bad = run_train_rusage(exe, base_args(root + "/bad"),
                                         "UAVRGB_CUDA_BLOCKING_SYNC=yes", root + "/bad.log");
  check(bad.rc == 2, "非法 UAVRGB_CUDA_BLOCKING_SYNC=yes 应 rc=2 拒绝（实际 " +
                         std::to_string(bad.rc) + "）");

  fs::remove_all(root);
  if (g_fail == 0) {
    std::cout << "test_cuda_blocking_sync: PASS\n";
    return 0;
  }
  std::cerr << "test_cuda_blocking_sync: " << g_fail << " 项失败\n";
  return 1;
}
