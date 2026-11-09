// uavrgb_train 命令行入口的退出码回归（argv[1] = 可执行文件路径，CMake 用 $<TARGET_FILE> 传入）：
//   B4   eval-only 读到坏的训练 config.json 必须是 rc=1 的报错，不能 std::terminate（SIGABRT）；
//   #3   eval-only 的 --out 指向另一个已完成 run 目录（有 config.json 且 ckpt 不在其中）→ rc=2，
//        否则一次手滑会把别的模型的指标追加进那个 run 的 metrics.jsonl；
//   #7   --seed 负数 / 超过 uint32 → rc=2（std::stoul 会把 -1 静默折成 4294967295）；
//   #8   --eval-only 与 --resume 同给 → rc=2（resume 对评估无意义，多半是命令拼错）。
// 训练只跑 1 epoch 的 24 样本小集，每个场景都走真实二进制。
#include <sys/wait.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

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

struct Run {
  int rc = -1;        // 正常退出时的退出码；被信号杀死时 -1
  int sig = 0;        // 终止信号
  std::string output;  // stdout+stderr
};

Run run_cli(const std::string& exe, const std::string& args, const std::string& log) {
  const int st = std::system((exe + " " + args + " >" + log + " 2>&1").c_str());
  Run r;
  if (WIFEXITED(st)) r.rc = WEXITSTATUS(st);
  if (WIFSIGNALED(st)) r.sig = WTERMSIG(st);
  std::ifstream f(log);
  std::ostringstream os;
  os << f.rdbuf();
  r.output = os.str();
  return r;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  std::ostringstream os;
  os << f.rdbuf();
  return os.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: test_train_cli <uavrgb_train 路径>\n";
    return 2;
  }
  const std::string exe = argv[1];
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_train_cli_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);
  const auto sd = uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", 24, 120, 160, 16, 9, 4);
  const std::string common = "--data " + sd.path + " --epochs 1 --batch 8 --lr 1e-3";

  // 两个独立完成的 run：run_a 是被评估的权重来源，run_b 是「别人的」run 目录。
  Run a = run_cli(exe, common + " --seed 3 --out " + root + "/run_a", root + "/a.log");
  check(a.rc == 0, "基线训练 run_a 应成功: " + a.output);
  Run b = run_cli(exe, common + " --seed 4 --out " + root + "/run_b", root + "/b.log");
  check(b.rc == 0, "基线训练 run_b 应成功: " + b.output);
  if (g_fail) {
    std::cerr << "test_train_cli: FAIL（基线训练失败，后续场景无法构造）\n";
    return 1;
  }
  const std::string ckpt_a = root + "/run_a/ckpt.bin";

  // #3 同目录 / 新目录评估放行；别的已完成 run 目录拒绝。
  {
    Run same = run_cli(exe, "--data " + sd.path + " --eval-only " + ckpt_a + " --split val --out " +
                                root + "/run_a", root + "/e_same.log");
    check(same.rc == 0, "eval-only 写回 ckpt 自己的 run 目录应放行: " + same.output);
    Run fresh = run_cli(exe, "--data " + sd.path + " --eval-only " + ckpt_a + " --split val --out " +
                                 root + "/eval_fresh", root + "/e_fresh.log");
    check(fresh.rc == 0, "eval-only 写到新目录应放行: " + fresh.output);
    const std::string b_metrics_before = read_file(root + "/run_b/metrics.jsonl");
    Run other = run_cli(exe, "--data " + sd.path + " --eval-only " + ckpt_a + " --split val --out " +
                                 root + "/run_b", root + "/e_other.log");
    check(other.rc == 2, "eval-only 把 run_a 的 ckpt 评进 run_b 目录应 rc=2（实际 " +
                             std::to_string(other.rc) + "）");
    check(read_file(root + "/run_b/metrics.jsonl") == b_metrics_before,
          "被拒绝的 eval-only 不得改动 run_b/metrics.jsonl");
  }

  // #8 eval-only 与 resume 同给。
  {
    Run r = run_cli(exe, "--data " + sd.path + " --eval-only " + ckpt_a + " --resume " + ckpt_a +
                             " --split val --out " + root + "/eval_resume", root + "/e_resume.log");
    check(r.rc == 2, "--eval-only 与 --resume 同给应 rc=2（实际 " + std::to_string(r.rc) + "）");
  }

  // #7 seed 取值范围：训练参数齐全，拒绝只能来自 --seed 本身。
  {
    Run neg = run_cli(exe, common + " --seed -1 --out " + root + "/run_seed_neg", root + "/s_neg.log");
    check(neg.rc == 2 && !fs::exists(root + "/run_seed_neg/config.json"),
          "--seed -1 应在入口拒绝（实际 rc " + std::to_string(neg.rc) + "）");
    Run big = run_cli(exe, common + " --seed 4294967296 --out " + root + "/run_seed_big",
                      root + "/s_big.log");
    check(big.rc == 2 && !fs::exists(root + "/run_seed_big/config.json"),
          "--seed 4294967296 超过 uint32 应在入口拒绝（实际 rc " + std::to_string(big.rc) + "）");
  }

  // B4 坏 config.json（lr 不是数字）：eval-only 继承 drop_collision 时读到它，必须是报错退出而非 abort。
  {
    const std::string cfg_path = root + "/run_a/config.json";
    std::string js = read_file(cfg_path);
    const std::size_t p = js.find("\"lr\":");
    check(p != std::string::npos, "run_a/config.json 应有 lr 字段");
    if (p != std::string::npos) {
      const std::size_t e = js.find(',', p);
      js.replace(p, e - p, "\"lr\":abc");
      std::ofstream(cfg_path, std::ios::trunc) << js;
    }
    Run r = run_cli(exe, "--data " + sd.path + " --eval-only " + ckpt_a + " --split val --out " +
                             root + "/eval_badcfg", root + "/e_badcfg.log");
    check(r.sig == 0, "坏 config.json 让 eval-only 被信号 " + std::to_string(r.sig) + " 杀死（未捕获异常）");
    check(r.rc == 1, "坏 config.json 的 eval-only 应 rc=1（实际 " + std::to_string(r.rc) + "）: " + r.output);
  }

  fs::remove_all(root);
  if (g_fail) {
    std::cerr << "test_train_cli: FAIL (" << g_fail << ")\n";
    return 1;
  }
  std::cout << "test_train_cli: PASS\n";
  return 0;
}
