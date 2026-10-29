// CUDA 冒烟测试: 向量加法 + 设备枚举
// 目标架构: sm_75 (Turing; 本机 RTX 2080 Ti)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err));                                         \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

static void printDeviceInfo() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  printf("=== CUDA device enumeration ===\n");
  printf("visible device count: %d\n", count);
  for (int d = 0; d < count; ++d) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, d));
    printf("GPU %d: %s\n", d, p.name);
    printf("  compute capability : %d.%d\n", p.major, p.minor);
    printf("  SM count           : %d\n", p.multiProcessorCount);
    printf("  global memory      : %.2f GiB\n", p.totalGlobalMem / 1073741824.0);
    printf("  memory clock/bus   : %.0f MHz / %d bit\n", p.memoryClockRate / 1000.0, p.memoryBusWidth);
    printf("  warp size          : %d\n", p.warpSize);
  }
}

int main() {
  printDeviceInfo();

  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  int runtimeVer = 0, driverVer = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVer));
  CUDA_CHECK(cudaDriverGetVersion(&driverVer));
  printf("CUDA runtime version : %d.%d\n", runtimeVer / 1000, (runtimeVer % 1000) / 10);
  printf("CUDA driver version  : %d.%d\n", driverVer / 1000, (driverVer % 1000) / 10);

  const int n = 1 << 20;  // 1M elements
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i) * 0.5f;
    h_b[i] = static_cast<float>(i) * 2.0f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  // 计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < 10; ++it) {
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  double maxErr = 0.0;
  for (int i = 0; i < n; ++i) {
    double ref = static_cast<double>(i) * 0.5 + static_cast<double>(i) * 2.0;
    maxErr = std::fmax(maxErr, std::fabs(static_cast<double>(h_c[i]) - ref));
  }

  printf("=== vectorAdd (n=%d, blocks=%d, threads=%d) ===\n", n, blocks, threads);
  printf("spot check: c[0]=%.3f c[1]=%.3f c[%d]=%.3f\n", h_c[0], h_c[1], n - 1, h_c[n - 1]);
  printf("expected  :           %.3f          %.3f              %.3f\n",
         static_cast<double>(0) * 2.5, static_cast<double>(1) * 2.5, static_cast<double>(n - 1) * 2.5);
  printf("max abs error: %.6e\n", maxErr);
  printf("avg kernel time (10 iters): %.4f ms\n", ms / 10.0);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  bool pass = (maxErr < 1e-3) && (n > 0);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
