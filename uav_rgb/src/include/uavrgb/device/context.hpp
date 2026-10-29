#pragma once
// 每个执行器独占显式非默认流；异步错误必须经 finish 检查。
#include <cuda_runtime_api.h>

namespace uavrgb::device {

void check_cuda(cudaError_t result, const char* operation);

class Context {
 public:
  explicit Context(int device = 0);
  ~Context() noexcept;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(Context&&) = delete;

  int device() const noexcept { return device_; }
  cudaStream_t stream() const noexcept { return stream_; }
  bool failed() const noexcept { return failed_; }
  void activate();
  void check(cudaError_t result, const char* operation);
  void finish();

 private:
  int device_;
  cudaStream_t stream_ = nullptr;
  bool failed_ = false;
};

class Event {
 public:
  // 依赖事件默认禁用计时；性能测量显式创建 timing=true 的事件。
  explicit Event(Context& context, bool timing = false);
  ~Event() noexcept;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;

  cudaEvent_t get() const noexcept { return event_; }
  int device() const noexcept { return device_; }
  void record(Context& context);
  void wait(Context& context) const;
  // 同步错误同时标记调用方 context FAILED，禁止继续提交。
  void synchronize(Context& context) const;

 private:
  int device_ = -1;
  cudaEvent_t event_ = nullptr;
  bool recorded_ = false;
};

}  // namespace uavrgb::device
