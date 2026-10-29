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

  check(run("").first == 0, "空输入应返回 0 帧");

  bool threw = false;
  try {
    (void)run(frame.substr(0, frame.size() - 1));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "截断帧未抛 runtime_error");

  if (g_failures == 0) {
    std::cout << "test_infer_io: PASS\n";
    return 0;
  }
  std::cerr << "test_infer_io: " << g_failures << " 项失败\n";
  return 1;
}
