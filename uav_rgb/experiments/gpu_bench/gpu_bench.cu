// Conv2d 前向/反向吞吐 + device 显存基准（RTX 2080 Ti, sm_75）。
//
// 测的是 src/ 里现成的 parity 版 CUDA 包装（conv_cuda.hpp），该实现：
//   * 每次调用 cudaMalloc/cudaFree 全部设备缓冲（前向 4 个，反向 6 个）；
//   * 每个 kernel 后 cudaDeviceSynchronize；
//   * 每次调用全量 H2D(x/W/b) + D2H(y 或 dW/db/dx)。
// 因此原始数字 = 该包装的真实吞吐（偏低）；程序同时测量 host 清零、设备分配、
// pageable 拷贝等分量，反推“去掉 malloc/free 与同步后”的估算值（明确标 ~EST）。
// 仅 sm_75，不依赖 cuBLAS/cuDNN/PyTorch。
#include "uavrgb/conv.hpp"
#include "uavrgb/conv_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

using uavrgb::Conv2d;
using uavrgb::ConvGrad;
using uavrgb::Tensor;

constexpr int kImgH = 128;
constexpr int kImgW = 128;
constexpr int kKernel = 3;
constexpr int kPad = 1;  // 保持 128x128 输出（典型 CNN 同尺寸卷积）
constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kEsSyncMs = 0.01;  // 估算：每次 cudaDeviceSynchronize ~10 us
constexpr double kReserveMiB = 512.0;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err_));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

struct Layer {
  int in;
  int out;
};

const Layer kLayers[] = {{3, 16}, {16, 32}, {32, 64}};
const int kBatches[] = {1, 8, 32, 128, 256};

static volatile std::size_t g_sink = 0;

static double now_ms() {
  const auto t = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(t).count();
}

static std::size_t act_bytes(int n, int ch) {
  return static_cast<std::size_t>(n) * ch * kImgH * kImgW * sizeof(float);
}
static std::size_t weight_bytes(int in, int out) {
  return static_cast<std::size_t>(out) * in * kKernel * kKernel * sizeof(float);
}
static std::size_t bias_bytes(int out) {
  return static_cast<std::size_t>(out) * sizeof(float);
}

// 采样调用期间的 device 空闲显存最低值，得到真实峰值占用。
// 注意：包装在调用返回前已 cudaFree，单纯“前/后 cudaMemGetInfo 相减”恒为 0，
// 因此必须采样；采样粒度为 200 us，极小配置可能漏采（表里同时给了解析精确值）。
class PeakMem {
 public:
  void start() {
    CUDA_CHECK(cudaMemGetInfo(&free0_, &total_));
    min_free_ = free0_;
    stop_ = false;
    th_ = std::thread([this] {
      while (!stop_.load(std::memory_order_relaxed)) {
        std::size_t f = 0;
        std::size_t t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess && f < min_free_) min_free_ = f;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
  }
  double stop_mib() {
    stop_ = true;
    th_.join();
    return free0_ > min_free_ ? static_cast<double>(free0_ - min_free_) / kMiB : 0.0;
  }

 private:
  std::size_t free0_ = 0;
  std::size_t total_ = 0;
  std::size_t min_free_ = 0;
  std::atomic<bool> stop_{false};
  std::thread th_;
};

// 包装里 Tensor::zeros 的 host 侧清零/分配开销（每次调用都会做）。
// volatile 读首/中/尾元素，防止 -O3 把“分配+清零后立即释放”整体消除。
static double host_zero_ms(const std::vector<std::int64_t>& dims, int reps) {
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    const double t = now_ms();
    Tensor z = Tensor::zeros(dims);
    volatile float* p = z.data();
    const std::size_t nmel = z.numel();
    const volatile float v = p[0] + p[nmel / 2] + p[nmel - 1];
    (void)v;
    best = std::min(best, now_ms() - t);
  }
  return best;
}

// 包装里同一调用内连续的 cudaMalloc（返回时再全部 cudaFree）。计时含 free。
static double dev_alloc_free_ms(const std::vector<std::size_t>& bytes, int reps) {
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    std::vector<void*> p(bytes.size(), nullptr);
    const double t = now_ms();
    for (std::size_t i = 0; i < bytes.size(); ++i) CUDA_CHECK(cudaMalloc(&p[i], bytes[i]));
    for (void* q : p) CUDA_CHECK(cudaFree(q));
    best = std::min(best, now_ms() - t);
  }
  return best;
}

// pageable host <-> device 拷贝耗时（与包装中的 Tensor::data() 一致）。
static double h2d_ms(const void* src, std::size_t bytes, int reps) {
  void* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t = now_ms();
    CUDA_CHECK(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice));
    best = std::min(best, now_ms() - t);
  }
  CUDA_CHECK(cudaFree(d));
  return best;
}

static double d2h_ms(void* dst, std::size_t bytes, int reps) {
  void* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t = now_ms();
    CUDA_CHECK(cudaMemcpy(dst, d, bytes, cudaMemcpyDeviceToHost));
    best = std::min(best, now_ms() - t);
  }
  CUDA_CHECK(cudaFree(d));
  return best;
}

struct Row {
  int in_ch = 0;
  int out_ch = 0;
  int n = 0;
  double fwd_ms = 0;
  double bwd_ms = 0;
  double fwd_min_ms = 0;
  double bwd_min_ms = 0;
  double fwd_sps = 0;
  double bwd_sps = 0;
  double fwd_tf = 0;
  double bwd_tf = 0;
  double peak_fwd_mib = 0;   // 解析精确峰值 = 同调用内同时存活的设备缓冲之和
  double peak_bwd_mib = 0;
  double samp_fwd_mib = 0;   // cudaMemGetInfo 采样峰值
  double samp_bwd_mib = 0;
  double hz_fwd = 0;         // host Tensor::zeros 开销
  double hz_bwd = 0;
  double da_fwd = 0;         // cudaMalloc+cudaFree 开销
  double da_bwd = 0;
  double cp_x = 0;           // H2D x
  double cp_y = 0;           // D2H y
  double cp_dy = 0;          // H2D dy
  double cp_dx = 0;          // D2H dx
  double kf_ms = 0;          // 反推 kernel-only（含 launch/同步残差，~EST）
  double kb_ms = 0;
  double opt_fwd_sps = 0;    // 复用缓冲/去同步，但保留每次 H2D/D2H（~EST）
  double opt_bwd_sps = 0;
  double kern_fwd_sps = 0;   // 数据常驻显存、只算 kernel（~EST 上界）
  double kern_bwd_sps = 0;
};

static Row run_config(int in_ch, int out_ch, int n, int iters) {
  Row r;
  r.in_ch = in_ch;
  r.out_ch = out_ch;
  r.n = n;
  const std::size_t x_bytes = act_bytes(n, in_ch);
  const std::size_t y_bytes = act_bytes(n, out_ch);
  const std::size_t w_bytes = weight_bytes(in_ch, out_ch);
  const std::size_t b_bytes = bias_bytes(out_ch);

  std::fprintf(stderr, "  running %d->%d N=%d ...\n", in_ch, out_ch, n);
  std::fflush(stderr);

  Tensor x = Tensor::zeros({n, in_ch, kImgH, kImgW});
  Tensor dy_host = Tensor::zeros({n, out_ch, kImgH, kImgW});
  Conv2d m(in_ch, out_ch, kKernel, kKernel, kPad);

  // 暖机同时测峰值显存（不计时）。
  PeakMem sampler;
  sampler.start();
  Tensor y = uavrgb::conv2d_forward_cuda(m, x);
  r.samp_fwd_mib = sampler.stop_mib();

  sampler.start();
  ConvGrad g = uavrgb::conv2d_backward_cuda(m, x, dy_host);
  r.samp_bwd_mib = sampler.stop_mib();

  // 包装开销分量（各取 2 次最小值）。
  r.hz_fwd = host_zero_ms({n, out_ch, kImgH, kImgW}, 2);
  r.hz_bwd = host_zero_ms({n, in_ch, kImgH, kImgW}, 2) +
             host_zero_ms({out_ch, in_ch, kKernel, kKernel}, 1) + host_zero_ms({out_ch}, 1);
  r.da_fwd = dev_alloc_free_ms({x_bytes, w_bytes, b_bytes, y_bytes}, 2);
  r.da_bwd = dev_alloc_free_ms({x_bytes, w_bytes, y_bytes, w_bytes, b_bytes, x_bytes}, 2);
  r.cp_x = h2d_ms(x.data(), x_bytes, 2);
  r.cp_y = d2h_ms(y.data(), y_bytes, 2);
  r.cp_dy = h2d_ms(dy_host.data(), y_bytes, 2);
  r.cp_dx = d2h_ms(g.dx.data(), x_bytes, 2);

  // 计时：每次都是完整的包装调用（含 malloc/free、同步、全量拷贝）。
  double fwd_sum = 0;
  double bwd_sum = 0;
  double fwd_min = 1e18;
  double bwd_min = 1e18;
  for (int i = 0; i < iters; ++i) {
    const double t = now_ms();
    Tensor yy = uavrgb::conv2d_forward_cuda(m, x);
    const double dt = now_ms() - t;
    fwd_sum += dt;
    fwd_min = std::min(fwd_min, dt);
    g_sink = g_sink + yy.numel();
  }
  for (int i = 0; i < iters; ++i) {
    const double t = now_ms();
    ConvGrad gg = uavrgb::conv2d_backward_cuda(m, x, dy_host);
    const double dt = now_ms() - t;
    bwd_sum += dt;
    bwd_min = std::min(bwd_min, dt);
    g_sink = g_sink + gg.dx.numel();
  }

  r.fwd_ms = fwd_sum / iters;
  r.bwd_ms = bwd_sum / iters;
  r.fwd_min_ms = fwd_min;
  r.bwd_min_ms = bwd_min;
  r.fwd_sps = static_cast<double>(n) / (r.fwd_ms / 1000.0);
  r.bwd_sps = static_cast<double>(n) / (r.bwd_ms / 1000.0);

  const double fwd_flops =
      2.0 * n * out_ch * kImgH * kImgW * in_ch * kKernel * kKernel;
  const double bwd_flops = 2.0 * fwd_flops;  // dW + dx ≈ 2x 前向 MACs
  r.fwd_tf = fwd_flops / (r.fwd_ms / 1000.0) / 1e12;
  r.bwd_tf = bwd_flops / (r.bwd_ms / 1000.0) / 1e12;

  r.peak_fwd_mib = static_cast<double>(x_bytes + w_bytes + b_bytes + y_bytes) / kMiB;
  r.peak_bwd_mib = static_cast<double>(2 * x_bytes + 2 * w_bytes + b_bytes + y_bytes) / kMiB;

  const double copies_fwd = r.cp_x + r.cp_y;
  const double copies_bwd = r.cp_x + r.cp_dy + r.cp_dx;
  // 估算用最快一次（min）总时间 减 各分量的 min，避免把一次性的调度抖动算进 kernel。
  r.kf_ms = std::max(0.0, r.fwd_min_ms - (r.hz_fwd + r.da_fwd + copies_fwd + kEsSyncMs));
  r.kb_ms = std::max(0.0, r.bwd_min_ms - (r.hz_bwd + r.da_bwd + copies_bwd + 3 * kEsSyncMs));
  // ~EST：去掉 malloc/free、host 清零和每 kernel 同步，保留每次 H2D/D2H。
  r.opt_fwd_sps = static_cast<double>(n) / ((r.kf_ms + copies_fwd) / 1000.0);
  r.opt_bwd_sps = static_cast<double>(n) / ((r.kb_ms + copies_bwd) / 1000.0);
  // ~EST 上界：整网常驻显存，层间不搬数据，只算 kernel。
  r.kern_fwd_sps = r.kf_ms > 0 ? static_cast<double>(n) / (r.kf_ms / 1000.0) : 0.0;
  r.kern_bwd_sps = r.kb_ms > 0 ? static_cast<double>(n) / (r.kb_ms / 1000.0) : 0.0;
  return r;
}

static void print_raw(const std::vector<Row>& rows, int iters) {
  std::printf("\n--- raw throughput: mean of %d timed iters (wrapper: malloc/free + sync per call) ---\n",
              iters);
  std::printf("%-8s %5s %10s %10s %12s %8s %10s %10s %12s %8s %11s %11s %11s %11s\n", "layer",
              "N", "fwd_ms", "fwd_min", "fwd_samp/s", "fwd_TF", "bwd_ms", "bwd_min",
              "bwd_samp/s", "bwd_TF", "peakF_MiB", "peakB_MiB", "sampF_MiB", "sampB_MiB");
  for (const Row& r : rows) {
    std::printf("%-8s %5d %10.2f %10.2f %12.1f %8.3f %10.2f %10.2f %12.1f %8.3f %11.2f %11.2f %11.2f %11.2f\n",
                (std::to_string(r.in_ch) + "->" + std::to_string(r.out_ch)).c_str(), r.n,
                r.fwd_ms, r.fwd_min_ms, r.fwd_sps, r.fwd_tf,
                r.bwd_ms, r.bwd_min_ms, r.bwd_sps, r.bwd_tf,
                r.peak_fwd_mib, r.peak_bwd_mib, r.samp_fwd_mib, r.samp_bwd_mib);
  }
  std::printf("note: peak*_MiB = exact analytic peak (simultaneously live device buffers); "
              "samp*_MiB = cudaMemGetInfo 200us sampling (may miss very short calls).\n");
}

static void print_estimates(const std::vector<Row>& rows) {
  std::printf("\n--- ~EST (from best-case min components): remove malloc/free + host zero + per-kernel sync,\n");
  std::printf("    'opt*' keeps per-call H2D/D2H; 'kern*' assumes device-resident data (upper bound) ---\n");
  std::printf("%-8s %5s %11s %11s %14s %14s %15s %15s\n", "layer", "N", "kernelF_ms",
              "kernelB_ms", "optF_samp/s", "optB_samp/s", "kernF_samp/s", "kernB_samp/s");
  for (const Row& r : rows) {
    std::printf("%-8s %5d %11.3f %11.3f %14.1f %14.1f %15.1f %15.1f\n",
                (std::to_string(r.in_ch) + "->" + std::to_string(r.out_ch)).c_str(), r.n,
                r.kf_ms, r.kb_ms, r.opt_fwd_sps, r.opt_bwd_sps, r.kern_fwd_sps, r.kern_bwd_sps);
  }
  std::printf("~EST is derived from measured components, not a reimplementation; see RESULTS.md caveats.\n");
}

static void print_decomposition(const std::vector<Row>& rows) {
  std::printf("\n--- measured cost decomposition per wrapper call, best-case/min (ms) ---\n");
  for (const Row& r : rows) {
    std::printf("[%d->%d N=%d] fwd %7.2f = host_zero %6.2f + devalloc %5.2f + copies %7.2f + kernel~%7.2f\n",
                r.in_ch, r.out_ch, r.n, r.fwd_min_ms, r.hz_fwd, r.da_fwd, r.cp_x + r.cp_y, r.kf_ms);
    std::printf("%22s bwd %7.2f = host_zero %6.2f + devalloc %5.2f + copies %7.2f + kernel~%7.2f\n",
                "", r.bwd_min_ms, r.hz_bwd, r.da_bwd, r.cp_x + r.cp_dy + r.cp_dx, r.kb_ms);
  }
}

static void print_pcie(const std::vector<Row>& rows) {
  const Row* pick = nullptr;
  for (const Row& r : rows) {
    if (r.in_ch == 32 && r.out_ch == 64 && r.n == 256) pick = &r;
  }
  if (!pick) return;
  const double h2d_gbs = (act_bytes(pick->n, pick->in_ch) / (pick->cp_x / 1000.0)) / 1e9;
  const double d2h_gbs = (act_bytes(pick->n, pick->out_ch) / (pick->cp_y / 1000.0)) / 1e9;
  std::printf("\nmeasured pageable PCIe (32->64, N=256): H2D %.2f GB/s, D2H %.2f GB/s\n", h2d_gbs,
              d2h_gbs);
}

static void print_capacity(double budget_mib) {
  std::printf("\n--- 22 GiB capacity per layer (budget %.0f MiB = free-at-start minus %.0f MiB reserve) ---\n",
              budget_mib, kReserveMiB);
  std::printf("%-8s %14s %17s %15s %14s %17s %15s\n", "layer", "peakF@1_MiB",
              "fwd_bytes/sample", "max_batch_fwd", "peakB@1_MiB", "bwd_bytes/sample",
              "max_batch_bwd");
  const double budget_bytes = budget_mib * kMiB;
  for (const Layer& l : kLayers) {
    const std::size_t w = weight_bytes(l.in, l.out);
    const std::size_t b = bias_bytes(l.out);
    const double fwd_per_sample =
        static_cast<double>(act_bytes(1, l.in) + act_bytes(1, l.out));
    const double bwd_per_sample = static_cast<double>(2 * act_bytes(1, l.in) + act_bytes(1, l.out));
    const double peak_f = static_cast<double>(act_bytes(1, l.in) + act_bytes(1, l.out) + w + b) / kMiB;
    const double peak_b =
        static_cast<double>(2 * act_bytes(1, l.in) + act_bytes(1, l.out) + 2 * w + b) / kMiB;
    const long long max_f = static_cast<long long>((budget_bytes - w - b) / fwd_per_sample);
    const long long max_b = static_cast<long long>((budget_bytes - 2 * w - b) / bwd_per_sample);
    std::printf("%-8s %14.2f %17.0f %15lld %14.2f %17.0f %15lld\n",
                (std::to_string(l.in) + "->" + std::to_string(l.out)).c_str(), peak_f,
                fwd_per_sample, max_f, peak_b, bwd_per_sample, max_b);
  }
  std::printf("max_batch = one layer in isolation. Full 3-layer training graph needs more; see RESULTS.md.\n");
}

// 自检：小形状上 CPU 参考 vs CUDA 前向，确认链接的是正确实现。
static void sanity_check() {
  const int n = 2;
  const int in_ch = 3;
  const int out_ch = 8;
  Tensor x = Tensor::zeros({n, in_ch, 16, 16});
  for (std::size_t i = 0; i < x.numel(); ++i) x[i] = 0.5f * std::sin(0.1f * static_cast<float>(i));
  Conv2d m(in_ch, out_ch, 3, 3, 1);
  for (std::size_t i = 0; i < m.W.numel(); ++i)
    m.W[i] = 0.1f * std::cos(0.07f * static_cast<float>(i));
  for (int i = 0; i < out_ch; ++i) m.b[i] = 0.01f * static_cast<float>(i);
  const Tensor yc = uavrgb::conv2d_forward(m, x);
  const Tensor yg = uavrgb::conv2d_forward_cuda(m, x);
  double max_err = 0;
  for (std::size_t i = 0; i < yc.numel(); ++i)
    max_err = std::max(max_err, std::fabs(static_cast<double>(yc[i] - yg[i])));
  if (max_err > 1e-4) {
    std::fprintf(stderr, "sanity FAIL: CPU/CUDA conv max_abs_err=%.3e\n", max_err);
    std::exit(EXIT_FAILURE);
  }
  std::printf("sanity CPU-vs-CUDA conv forward (2x3x16x16 -> 8ch): max_abs_err=%.3e PASS\n", max_err);
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::max(1, std::atoi(argv[1])) : 3;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int rt_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&rt_version));
  std::size_t free0 = 0;
  std::size_t total0 = 0;
  CUDA_CHECK(cudaMemGetInfo(&free0, &total0));
  const double total_mib = static_cast<double>(total0) / kMiB;
  const double used_mib = static_cast<double>(total0 - free0) / kMiB;
  const double budget_mib = static_cast<double>(free0) / kMiB - kReserveMiB;

  std::printf("=== uav_rgb GPU benchmark: Conv2d fwd/bwd (parity CUDA wrapper) ===\n");
  std::printf("device: %s, sm_%d%d, %d SM, %.0f MiB total, %.0f MiB free at start (%.0f MiB in use)\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, total_mib,
              static_cast<double>(free0) / kMiB, used_mib);
  std::printf("CUDA runtime %d.%d, driver device clock %.2f GHz\n", rt_version / 1000,
              (rt_version % 1000) / 10, prop.clockRate / 1e6);
  std::printf("shape: NCHW fp32, 128x128 input, 3x3 kernels, pad=1, stride=1; layers 3->16, 16->32, 32->64\n");
  std::printf("batches: 1, 8, 32, 128, 256; timed iters per cell = %d (plus 1 warmup)\n", iters);
  std::printf("caveat: wrapper cudaMalloc/cudaFree every call and cudaDeviceSynchronize after every kernel;\n");
  std::printf("        raw numbers are that implementation's real throughput, not peak kernel throughput.\n");

  sanity_check();

  std::vector<Row> rows;
  rows.reserve(sizeof(kLayers) / sizeof(kLayers[0]) * (sizeof(kBatches) / sizeof(kBatches[0])));
  for (const Layer& l : kLayers) {
    for (int n : kBatches) rows.push_back(run_config(l.in, l.out, n, iters));
  }

  print_raw(rows, iters);
  print_estimates(rows);
  print_decomposition(rows);
  print_pcie(rows);
  print_capacity(budget_mib);

  std::printf("\nRESULT: PASS\n");
  return 0;
}
