// 常驻推理循环自检：帧协议、同帧确定性、截断帧报错；不依赖 checkpoint。
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "uavrgb/action.hpp"
#include "uavrgb/infer.hpp"
#include "uavrgb/model.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}

}  // namespace

int main() {
  using namespace uavrgb;

  const CnnModel model = make_cnn(8, 8, 7, kNumActions);
  const std::size_t frame_bytes = 8 * 8 * 3;
  std::string frame(frame_bytes, '\0');
  for (std::size_t i = 0; i < frame.size(); ++i) {
    frame[i] = static_cast<char>((i * 37) % 256);
  }

  auto run = [&](const std::string& input) {
    std::istringstream in(input);
    std::ostringstream out;
    const std::size_t n = run_infer_loop(model, in, out);
    return std::pair<std::size_t, std::string>(n, out.str());
  };

  const auto first = run(frame);
  const auto second = run(frame);
  check(first.first == 1 && second.first == 1, "单帧输入应处理 1 帧");

  const std::string key = ",\"infer_ms\":";
  const std::size_t cut1 = first.second.find(key);
  const std::size_t cut2 = second.second.find(key);
  check(cut1 != std::string::npos && cut2 != std::string::npos, "输出缺 infer_ms");
  check(cut1 == cut2 && first.second.substr(0, cut1) == second.second.substr(0, cut2),
        "同帧两次输出（除耗时外）不一致");

  const std::string head = first.second.substr(0, cut1);
  const std::string action_key = "\"action\":";
  const std::size_t ap = head.find(action_key);
  check(ap != std::string::npos, "输出缺 action");
  if (ap != std::string::npos) {
    const int action = std::stoi(head.substr(ap + action_key.size()));
    check(action >= 0 && action < kNumActions, "action 越界");
  }

  const std::string probs_key = "\"probs\":[";
  const std::size_t pp = head.find(probs_key);
  const std::size_t pe = head.find(']', pp == std::string::npos ? 0 : pp);
  check(pp != std::string::npos && pe != std::string::npos, "输出缺 probs 数组");
  if (pp != std::string::npos && pe != std::string::npos) {
    const std::string body = head.substr(pp + probs_key.size(), pe - pp - probs_key.size());
    std::size_t values = body.empty() ? 0 : 1;
    for (char ch : body) values += (ch == ',') ? 1 : 0;
    check(values == static_cast<std::size_t>(kNumActions), "probs 数量应等于动作数");
  }

  {
    std::istringstream in(frame);
    std::ostringstream out;
    const std::size_t n = run_infer_loop(model, in, out, {1, 2, 3});
    check(n == 1, "带 text_ids 应处理 1 帧");
    const std::string with_text = out.str();
    const std::size_t tcut = with_text.find(key);
    check(tcut != std::string::npos && with_text.substr(0, tcut) == head,
          "L0 text_ids 不应改变动作输出");
  }

  check(run("").first == 0, "空输入应返回 0 帧");

  bool threw = false;
  try {
    (void)run(frame.substr(0, frame.size() - 1));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "截断帧未抛 runtime_error");

  // ---- STUDENT-P2 协议：6 通道双帧 + prev-action 字节 ----
  {
    const CnnModel p2 = make_cnn(8, 8, 7, kNumActions, 0, /*in_c=*/6, /*aux_classes=*/8, /*prev_dim=*/4);
    std::string prev_frame(frame_bytes, '\0');
    for (std::size_t i = 0; i < prev_frame.size(); ++i) prev_frame[i] = static_cast<char>((i * 11 + 5) % 256);
    auto run_p2 = [&](const std::string& input) {
      std::istringstream in(input);
      std::ostringstream out;
      const std::size_t n = run_infer_loop(p2, in, out);
      return std::pair<std::size_t, std::string>(n, out.str());
    };
    const std::string step = frame + prev_frame + std::string(1, '\x02');
    const auto a = run_p2(step);
    const auto b = run_p2(step);
    check(a.first == 1 && b.first == 1, "双帧+prev 一步应处理 1 帧");
    const std::size_t ca = a.second.find(key), cb = b.second.find(key);
    check(ca != std::string::npos && a.second.substr(0, ca) == b.second.substr(0, cb), "双帧同输入两次输出一致");
    // 上一帧或 prev-action 变化必须影响输出（否则第二帧/条件根本没接进网络）。
    const auto other_prev = run_p2(frame + frame + std::string(1, '\x02'));
    const auto other_act = run_p2(frame + prev_frame + std::string(1, '\x00'));
    check(other_prev.second.substr(0, other_prev.second.find(key)) != a.second.substr(0, ca),
          "换上一帧应改变输出");
    check(other_act.second.substr(0, other_act.second.find(key)) != a.second.substr(0, ca),
          "换 prev-action 应改变输出");
    check(run_p2(step + step).first == 2, "两步连续输入应处理 2 帧");
    bool t1 = false, t2 = false;
    try {
      (void)run_p2(frame + prev_frame);  // 缺 prev-action 字节
    } catch (const std::runtime_error&) {
      t1 = true;
    }
    try {
      (void)run_p2(frame + prev_frame + std::string(1, '\x04'));  // prev >= prev_dim
    } catch (const std::runtime_error&) {
      t2 = true;
    }
    check(t1, "缺 prev-action 字节未抛 runtime_error");
    check(t2, "prev-action 越界未抛 runtime_error");
    // 只有双帧、不接 prev-action：payload 恰为两帧。
    const CnnModel two = make_cnn(8, 8, 7, kNumActions, 0, 6);
    std::istringstream in(frame + prev_frame);
    std::ostringstream out;
    check(run_infer_loop(two, in, out) == 1, "纯双帧模型一步 = 两帧、无 prev 字节");
  }

  if (g_failures == 0) {
    std::cout << "test_infer_io: PASS\n";
    return 0;
  }
  std::cerr << "test_infer_io: " << g_failures << " 项失败\n";
  return 1;
}
