// D6：续训（--resume）时继承源 run 的 --max-grad-norm（卡 C1a 第 4 条）。
//
// 验证要点：
//   1) 源 run 设 --max-grad-norm 0.01（强裁剪），续训不传 --max-grad-norm 时：
//      - config.resume.json 记录 "max_grad_norm":0.01（改前记成默认 5）；
//      - 产出 ckpt.bin 与显式传 --max-grad-norm 0.01 逐字节一致，且与 --max-grad-norm 5 不同；
//   2) 续训显式传 --max-grad-norm 1.5 时，以 CLI 显式值为准（漏设 max_grad_norm_set 会变红）；
//   3) 源 config.json 的 max_grad_norm <= 0 时拒绝续训，不能绕过入口校验把梯度静默清零；
//   4) 旧 config.json 缺 max_grad_norm 字段时兼容放行，沿用默认 5。
#include <sys/wait.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
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

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int run_cli(const std::string& exe, const std::string& args, const std::string& log_path) {
  const int st = std::system((exe + " " + args + " >" + log_path + " 2>&1").c_str());
  return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: test_resume_grad_norm <uavrgb_train 路径>\n";
    return 2;
  }
  const std::string exe = argv[1];
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_resume_gn_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  const auto sd = uavrgb_test::write_synthetic_uavr(root + "/ds.uavr", 20, 16, 16, 16, 13);
  const std::string src_dir = root + "/src";
  const int rc0 = run_cli(exe,
                          "--data " + sd.path +
                              " --epochs 1 --batch 4 --lr 1e-2 --seed 7 --max-grad-norm 0.03125 --out " +
                              src_dir,
                          root + "/src.log");
  check(rc0 == 0, "源训练应成功");
  const std::string src_ckpt = src_dir + "/ckpt.bin";

  // 1) 未显式传 --max-grad-norm：必须继承 0.03125，且权重更新轨迹与显式 0.03125 一致、与 5 不同。
  const int rc_inh = run_cli(exe, "--data " + sd.path + " --epochs 1 --resume " + src_ckpt +
                                      " --out " + root + "/inh",
                             root + "/inh.log");
  const int rc_001 = run_cli(exe, "--data " + sd.path + " --epochs 1 --resume " + src_ckpt +
                                      " --max-grad-norm 0.03125 --out " + root + "/exp001",
                             root + "/exp001.log");
  const int rc_5 = run_cli(exe, "--data " + sd.path + " --epochs 1 --resume " + src_ckpt +
                                    " --max-grad-norm 5 --out " + root + "/exp5",
                           root + "/exp5.log");
  check(rc_inh == 0 && rc_001 == 0 && rc_5 == 0, "三组续训均应 rc=0");

  const std::string inh_cfg = slurp(root + "/inh/config.resume.json");
  check(inh_cfg.find("\"max_grad_norm\":0.03125") != std::string::npos,
        "未传 --max-grad-norm 的续训应继承 0.03125（实际 config.resume.json: " + inh_cfg + "）");
  const std::string b_inh = slurp(root + "/inh/ckpt.bin");
  const std::string b_001 = slurp(root + "/exp001/ckpt.bin");
  const std::string b_5 = slurp(root + "/exp5/ckpt.bin");
  check(!b_inh.empty() && b_inh == b_001, "继承 0.03125 的续训 ckpt.bin 应与显式 0.03125 逐字节相同");
  check(b_inh != b_5, "继承 0.03125 的续训 ckpt.bin 应与 max_grad_norm=5 不同（裁剪实际生效）");

  // 2) CLI 显式覆盖优先于源 config。
  const int rc_ovr = run_cli(exe, "--data " + sd.path + " --epochs 1 --resume " + src_ckpt +
                                      " --max-grad-norm 1.5 --out " + root + "/ovr",
                             root + "/ovr.log");
  check(rc_ovr == 0, "显式覆盖 --max-grad-norm 1.5 续训应 rc=0");
  const std::string ovr_cfg = slurp(root + "/ovr/config.resume.json");
  check(ovr_cfg.find("\"max_grad_norm\":1.5") != std::string::npos,
        "CLI 显式 --max-grad-norm 1.5 应覆盖源 config 的 0.03125");

  // 3) 源 config.json 的 max_grad_norm <= 0 时拒绝续训（防绕过入口检查把梯度静默清零）。
  {
    const std::string bad_src = root + "/bad_src";
    fs::create_directories(bad_src);
    fs::copy_file(src_ckpt, bad_src + "/ckpt.bin");
    fs::copy_file(src_dir + "/ckpt_best.bin", bad_src + "/ckpt_best.bin");
    std::string js = slurp(src_dir + "/config.json");
    const std::string pat = "\"max_grad_norm\":0.03125";
    const std::size_t p = js.find(pat);
    check(p != std::string::npos, "源 config.json 应含 max_grad_norm:0.03125");
    if (p != std::string::npos) js.replace(p, pat.size(), "\"max_grad_norm\":0");
    std::ofstream(bad_src + "/config.json", std::ios::trunc) << js;
    const int rc_bad = run_cli(exe, "--data " + sd.path + " --epochs 1 --resume " + bad_src +
                                        "/ckpt.bin --out " + root + "/bad_out",
                               root + "/bad.log");
    check(rc_bad != 0, "源 config max_grad_norm=0 的续训必须报错拒绝");
  }

  // 4) 旧源 config.json 缺 max_grad_norm 字段时兼容放行，落盘默认 5。
  {
    const std::string old_src = root + "/old_src";
    fs::create_directories(old_src);
    fs::copy_file(src_ckpt, old_src + "/ckpt.bin");
    fs::copy_file(src_dir + "/ckpt_best.bin", old_src + "/ckpt_best.bin");
    std::ofstream(old_src + "/config.json")
        << "{\"schema\":\"uavrgb.train_config.v1\",\"batch\":4,\"lr\":0.01,"
           "\"seed\":7,\"drop_collision\":false}\n";
    const int rc_old = run_cli(exe, "--data " + sd.path + " --epochs 1 --resume " + old_src +
                                        "/ckpt.bin --out " + root + "/old_out",
                               root + "/old.log");
    check(rc_old == 0, "缺 max_grad_norm 字段的旧 config 应兼容续训");
    check(slurp(root + "/old_out/config.resume.json").find("\"max_grad_norm\":5") !=
              std::string::npos,
          "旧 config 无该字段时续训应使用默认 max_grad_norm=5");
  }

  fs::remove_all(root);
  if (g_fail == 0) {
    std::cout << "test_resume_grad_norm: PASS\n";
    return 0;
  }
  std::cerr << "test_resume_grad_norm: " << g_fail << " 项失败\n";
  return 1;
}
