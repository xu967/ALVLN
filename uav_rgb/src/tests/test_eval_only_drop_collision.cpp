// C6：--eval-only 优先从同目录 config.json 或 checkpoint v5 训练契约读取 drop_collision 口径；
//     两处均无记录或互相冲突时直接报错（rc=2），不再静默回落到命令行缺省 false（卡 C1a 第 5 条）。
//
// 场景覆盖：
//   1) 目录外平均/交付权重（无 config.json，但 v6 ckpt 含有效 v5 契约 drop_collision=true）：
//      - 不传 --drop-collision：自动从契约继承 true（val 剔除 2 个碰撞步，评 4 帧；改前静默评 6 帧）；
//      - 显式传 --drop-collision（同 E2E-1 对拍脚本）：与契约一致，rc=0 放行；
//   2) 目录外权重契约 drop_collision=false：不传时评 6 帧；误传 --drop-collision 时 rc=2 拒绝；
//   3) 既无含 drop_collision 的 config.json、又无有效 v5 契约（全零契约 ckpt，或剪成 v4 的旧 ckpt）：
//      --eval-only 直接 rc=2 报错，不再只打警告静默回落；
//   4) 同目录 config.json 与 ckpt 契约的 drop_collision 矛盾时 rc=2 拒绝。
#include <sys/wait.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/model.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_fail;
  }
}

struct RunOut {
  int rc = -1;
  std::string out;
};

RunOut run_cli(const std::string& exe, const std::string& args, const std::string& log_path) {
  const int st = std::system((exe + " " + args + " >" + log_path + " 2>&1").c_str());
  std::ifstream f(log_path);
  return {WIFEXITED(st) ? WEXITSTATUS(st) : -1,
          std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>())};
}

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: test_eval_only_drop_collision <uavrgb_train 路径>\n";
    return 2;
  }
  const std::string exe = argv[1];
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_c6_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  // train 8 帧（无碰撞），val 6 帧（其中 2 帧标记碰撞 '1' -> drop=true 时 n=4，drop=false 时 n=6）。
  const std::vector<int> tr_y = {0, 1, 2, 3, 0, 1, 2, 3};
  const std::vector<int> va_y = {0, 1, 2, 3, 1, 1};
  const std::vector<std::uint8_t> coll = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
  const auto sd = uavrgb_test::write_class_conditional_uavr(
      root + "/ds.uavr", 16, 16, tr_y, va_y, {0.2f, 0.4f, 0.6f, 0.8f}, 0.02f, 0.02f, 19, coll);

  // 训练两份源模型：drop_collision=true 与 drop_collision=false。
  const RunOut tr_drop = run_cli(
      exe, "--data " + sd.path + " --epochs 1 --batch 4 --lr 1e-3 --seed 3 --drop-collision --out " +
               root + "/run_drop",
      root + "/tr_drop.log");
  const RunOut tr_keep = run_cli(
      exe, "--data " + sd.path + " --epochs 1 --batch 4 --lr 1e-3 --seed 3 --out " + root + "/run_keep",
      root + "/tr_keep.log");
  check(tr_drop.rc == 0 && tr_keep.rc == 0, "两份源训练应成功");

  // 1) 目录外平均/交付权重（无 config.json，契约记 drop_collision=true）
  fs::create_directories(root + "/avg_drop");
  fs::copy_file(root + "/run_drop/ckpt.bin", root + "/avg_drop/cal_swa.bin");
  {
    const RunOut e_auto = run_cli(exe, "--data " + sd.path + " --eval-only " + root +
                                           "/avg_drop/cal_swa.bin --split val --out " + root + "/ev1a",
                                  root + "/ev1a.log");
    check(e_auto.rc == 0, "1a) 目录外 v6 权重不传 --drop-collision 应从契约继承并成功: " + e_auto.out);
    check(e_auto.out.find("] n=4") != std::string::npos,
          "1a) 契约 drop_collision=true 时 val 应剔除碰撞步只评 n=4 帧（实际: " + e_auto.out + "）");

    const RunOut e_exp = run_cli(
        exe, "--data " + sd.path + " --eval-only " + root +
                 "/avg_drop/cal_swa.bin --split val --drop-collision --out " + root + "/ev1b",
        root + "/ev1b.log");
    check(e_exp.rc == 0 && e_exp.out.find("] n=4") != std::string::npos,
          "1b) 显式传与契约一致的 --drop-collision 应放行且评 n=4 帧");
  }

  // 2) 目录外权重契约 drop_collision=false：不传评 n=6，误传 --drop-collision 必须拒绝（rc=2）
  fs::create_directories(root + "/avg_keep");
  fs::copy_file(root + "/run_keep/ckpt.bin", root + "/avg_keep/cal_swa.bin");
  {
    const RunOut e_ok = run_cli(exe, "--data " + sd.path + " --eval-only " + root +
                                         "/avg_keep/cal_swa.bin --split val --out " + root + "/ev2a",
                                root + "/ev2a.log");
    check(e_ok.rc == 0 && e_ok.out.find("] n=6") != std::string::npos,
          "2a) 契约 drop_collision=false 时应评全部 n=6 帧");

    const RunOut e_bad = run_cli(
        exe, "--data " + sd.path + " --eval-only " + root +
                 "/avg_keep/cal_swa.bin --split val --drop-collision --out " + root + "/ev2b",
        root + "/ev2b.log");
    check(e_bad.rc == 2,
          "2b) 契约 drop_collision=false 但传 --drop-collision 应 rc=2 拒绝（实际 " +
              std::to_string(e_bad.rc) + "）");
  }

  // 3) 读不到碰撞过滤口径（无 config.json 且无有效 v5 契约）时必须报错，不再静默回落
  {
    // 3a) 直接 save_checkpoint 写出的全零契约 v6 ckpt
    fs::create_directories(root + "/zero_contract");
    CnnModel m = make_cnn(16, 16, 1, 4);
    TrainState st = make_train_state(m, 1e-3f);
    save_checkpoint(root + "/zero_contract/ckpt.bin", m, st);
    const RunOut e_zero = run_cli(exe, "--data " + sd.path + " --eval-only " + root +
                                           "/zero_contract/ckpt.bin --split val --out " + root + "/ev3a",
                                  root + "/ev3a.log");
    check(e_zero.rc == 2,
          "3a) 无 config.json 且契约全零时 --eval-only 必须 rc=2 拒绝静默回落（实际 " +
              std::to_string(e_zero.rc) + "）");

    // 3b) 剪成 v4（无契约段）且同目录 config.json 缺 drop_collision 字段
    fs::create_directories(root + "/v4_dir");
    std::string raw = slurp(root + "/run_drop/ckpt.bin");
    constexpr std::size_t kContractOff = 52;  // class_weights 为空：44 + best(4) + text_dim(4) = 52
    raw.erase(kContractOff, 113 + 8);         // 去掉 v5 契约(113B) 与 v6 字段(8B)
    const std::uint32_t v4 = 4;
    std::memcpy(raw.data() + 8, &v4, 4);
    {
      std::ofstream out(root + "/v4_dir/ckpt.bin", std::ios::binary | std::ios::trunc);
      out.write(raw.data(), static_cast<std::streamsize>(raw.size()));
    }
    std::ofstream(root + "/v4_dir/config.json") << "{\"schema\":\"uavrgb.train_config.v1\",\"lr\":0.001}\n";
    const RunOut e_v4 = run_cli(exe, "--data " + sd.path + " --eval-only " + root +
                                         "/v4_dir/ckpt.bin --split val --out " + root + "/ev3b",
                                root + "/ev3b.log");
    check(e_v4.rc == 2,
          "3b) v4 无契约且 config 缺 drop_collision 时必须 rc=2 拒绝静默回落（实际 " +
              std::to_string(e_v4.rc) + "）");
  }

  // 4) 同目录 config.json 与 checkpoint 契约的 drop_collision 互相矛盾时拒绝（rc=2）
  {
    fs::create_directories(root + "/conflict");
    fs::copy_file(root + "/run_drop/ckpt.bin", root + "/conflict/ckpt.bin");
    std::string js = slurp(root + "/run_drop/config.json");
    const std::string pat = "\"drop_collision\":true";
    const std::size_t p = js.find(pat);
    if (p != std::string::npos) js.replace(p, pat.size(), "\"drop_collision\":false");
    std::ofstream(root + "/conflict/config.json", std::ios::trunc) << js;
    const RunOut e_cf = run_cli(exe, "--data " + sd.path + " --eval-only " + root +
                                         "/conflict/ckpt.bin --split val --out " + root + "/ev4",
                                root + "/ev4.log");
    check(e_cf.rc == 2, "4) config.json 与 ckpt 契约的 drop_collision 矛盾应 rc=2 拒绝");
  }

  fs::remove_all(root);
  if (g_fail == 0) {
    std::cout << "test_eval_only_drop_collision: PASS\n";
    return 0;
  }
  std::cerr << "test_eval_only_drop_collision: " << g_fail << " 项失败\n";
  return 1;
}
