// 宏指标自检：无外部测试框架，返回 0 表示通过。
//
// 为什么要单独测：macro_f1 / balanced_accuracy 是 docs/EVAL_STANDARD.md 强制要求的
// 报告口径，而 test_trainer 只比对 metrics.jsonl 的键名，不校验数值。混淆矩阵一旦
// 把行列写反（conf.add(pred, truth)），准确率只看对角线、macro_f1 转置不变，都发现
// 不了；只有 balanced_accuracy 与每类召回会变。
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "uavrgb/eval_metrics.hpp"

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++failures;
  }
}

void expect_near(double got, double want, const char* msg) {
  if (std::fabs(got - want) > 1e-12) {
    std::fprintf(stderr, "FAIL: %s (got %.17g, want %.17g)\n", msg, got, want);
    ++failures;
  }
}

// 手算基准矩阵（行 = 真值，列 = 预测）：
//            pred0 pred1 pred2   行和   召回
//   truth0     8     2     0      10    8/10 = 4/5
//   truth1     4     4     2      10    4/10 = 2/5
//   truth2     0     0     5       5    5/5  = 1
//   列和      12     6     7
//   精确率   2/3    2/3   5/7
// 每类 F1 = 8/11, 1/2, 5/6 -> macro_f1 = 68/99
// balanced_accuracy = (4/5 + 2/5 + 1)/3 = 11/15
// 若行列转置：macro_f1 仍是 68/99（P/R 互换），但 balanced_accuracy 变成 43/63，
// 每类召回变成 (2/3, 2/3, 5/7)。所以下面两条断言必须同时存在。
uavrgb::Confusion hand_matrix() {
  uavrgb::Confusion c;
  c.reset(3);
  const int m[3][3] = {{8, 2, 0}, {4, 4, 2}, {0, 0, 5}};
  for (int t = 0; t < 3; ++t) {
    for (int p = 0; p < 3; ++p) {
      for (int k = 0; k < m[t][p]; ++k) c.add(t, p);
    }
  }
  return c;
}

}  // namespace

int main() {
  using uavrgb::balanced_accuracy;
  using uavrgb::macro_f1;
  using uavrgb::per_class_recall;

  {
    const uavrgb::Confusion c = hand_matrix();
    // 方向契约：at(真值, 预测)。
    expect(c.at(0, 1) == 2 && c.at(1, 0) == 4, "混淆矩阵方向 at(truth, pred)");

    const std::vector<double> r = per_class_recall(c);
    expect(r.size() == 3, "per_class_recall 长度 = classes");
    expect_near(r[0], 0.8, "recall[0]");
    expect_near(r[1], 0.4, "recall[1]");
    expect_near(r[2], 1.0, "recall[2]");

    expect_near(balanced_accuracy(c), 11.0 / 15.0, "balanced_accuracy");
    expect_near(macro_f1(c), 68.0 / 99.0, "macro_f1");
  }

  // 空类（无真值）与无预测类都要记 0，不能除零。
  {
    uavrgb::Confusion c;
    c.reset(3);
    c.add(1, 1);
    c.add(1, 1);
    c.add(1, 1);
    c.add(2, 1);
    c.add(2, 1);
    const std::vector<double> r = per_class_recall(c);
    expect_near(r[0], 0.0, "空类 recall 记 0");
    expect_near(r[1], 1.0, "全对类 recall 1");
    expect_near(r[2], 0.0, "全错类 recall 0");
    expect_near(balanced_accuracy(c), 1.0 / 3.0, "含空类的 balanced_accuracy");
    // 类 1：P=3/5, R=1 -> F1=0.75；类 0 与类 2 都没有预测命中 -> 0。
    expect_near(macro_f1(c), 0.25, "含空类的 macro_f1");
  }

  // 0 类：不能除零，直接返回 0。
  {
    uavrgb::Confusion c;
    c.reset(0);
    expect_near(macro_f1(c), 0.0, "0 类的 macro_f1");
    expect_near(balanced_accuracy(c), 0.0, "0 类的 balanced_accuracy");
    expect(per_class_recall(c).empty(), "0 类的 per_class_recall 为空");
  }

  if (failures == 0) std::printf("test_eval_metrics: PASS\n");
  return failures == 0 ? 0 : 1;
}
