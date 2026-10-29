// 测量真实 CNN 单步执行链；不含 epoch 验证/checkpoint/日志写盘。
// 用现有 UAVR 读取器和 CUDA 算子，2 步热身后 cudaProfilerStart/Stop 控制 Nsight 窗口。
// 所有时长都是 host 墙钟：当前算子同步返回，不能用于未来异步引擎的 kernel 计时。
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/dataset.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/softmax_cuda.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static int positive_int(const char* value) {
  std::size_t used = 0;
  const int n = std::stoi(value, &used);
  if (used != std::string(value).size() || n <= 0) throw std::invalid_argument("参数须为正整数");
  return n;
}

static void cuda_check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "用法: pipeline_profile DATASET.uavr BATCH MEASURED_STEPS\n");
    return 2;
  }
  try {
    const int batch = positive_int(argv[2]);
    const int steps = positive_int(argv[3]);
    auto ds = uavrgb::load_uavr(argv[1]);
    if (ds.train_idx.size() < static_cast<std::size_t>(batch)) {
      throw std::invalid_argument("train 样本数小于 batch");
    }
    auto model = uavrgb::make_cnn(ds.h, ds.w, 1234);
    auto state = uavrgb::make_train_state(model, 0.001f);
    auto params = uavrgb::cnn_params(model);
    auto order = ds.train_idx;
    std::mt19937 rng(1234);
    uavrgb::Tensor x;
    std::array<double, 6> sum{};
    const char* names[] = {"load", "forward", "loss_grad", "backward", "clip", "adam"};
    double total_ms = 0;
    float last_loss = 0;
    for (int step = -2; step < steps; ++step) {
      if (step == 0) cuda_check(cudaProfilerStart());
      const auto begin = Clock::now();
      auto previous = begin;
      auto mark = [&](int stage) {
        const auto now = Clock::now();
        if (step >= 0) sum[stage] += std::chrono::duration<double, std::milli>(now - previous).count();
        previous = now;
      };
      {
        std::shuffle(order.begin(), order.end(), rng);
        const std::vector<std::uint32_t> indices(order.begin(), order.begin() + batch);
        std::vector<int> labels;
        labels.reserve(batch);
        for (auto i : indices) labels.push_back(ds.labels[i]);
        ds.load_images(indices, x);
        mark(0);
        uavrgb::CnnCache cache;
        const auto logits = uavrgb::forward(model, x, &cache);
        mark(1);
        last_loss = uavrgb::softmax_ce_loss_cuda(logits, labels);
        const auto dl = uavrgb::softmax_ce_grad_cuda(logits, labels);
        mark(2);
        auto grad = uavrgb::backward(model, cache, dl);
        mark(3);
        auto tensors = uavrgb::cnn_grad_tensors(grad);
        uavrgb::clip_grad_norm_cuda(tensors, 5.0f);
        mark(4);
        for (std::size_t i = 0; i < params.size(); ++i) {
          uavrgb::adam_step_cuda(state.adam[i], *params[i], *tensors[i]);
        }
        mark(5);
      }
      if (step >= 0) total_ms += std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
      if (!std::isfinite(last_loss)) throw std::runtime_error("非有限 loss，拒绝输出成功基准");
    }
    cuda_check(cudaProfilerStop());
    for (const auto* param : params) {
      for (std::size_t i = 0; i < param->numel(); ++i) {
        if (!std::isfinite((*param)[i])) throw std::runtime_error("非有限参数");
      }
    }
    std::printf("{\"batch\":%d,\"steps\":%d,\"height\":%d,\"width\":%d,"
                "\"step_ms\":%.6f,\"samples_per_sec\":%.6f,\"stage_ms\":{",
                batch, steps, ds.h, ds.w, total_ms / steps, 1000.0 * batch * steps / total_ms);
    for (std::size_t i = 0; i < sum.size(); ++i) {
      std::printf("%s\"%s\":%.6f", i ? "," : "", names[i], sum[i] / steps);
    }
    const double teardown = total_ms - std::accumulate(sum.begin(), sum.end(), 0.0);
    std::printf("},\"teardown_ms\":%.6f,\"last_loss\":%.9g}\n", teardown / steps, last_loss);
    return 0;
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "profile: %s\n", exc.what());
    return 1;
  }
}
