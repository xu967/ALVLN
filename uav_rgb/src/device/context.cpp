// 显式 CUDA stream/event 生命周期；析构只负责清理，错误由 finish 报告。
#include "uavrgb/device/context.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace uavrgb::device {
namespace {

class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) check_cuda(cudaSetDevice(device), "cudaSetDevice");
  }
  ~DeviceScope() noexcept { (void)cudaSetDevice(previous_); }
 private:
  int previous_ = -1;
};

template <class F> void cleanup(int device, F release) noexcept {
  int previous = -1;
  (void)cudaGetDevice(&previous);
  if (cudaSetDevice(device) == cudaSuccess) release();
  if (previous >= 0) (void)cudaSetDevice(previous);
}

void require_event(cudaEvent_t event, bool recorded) {
  if (event == nullptr || !recorded) {
    throw std::logic_error("device event: 尚未 record 或已移出");
  }
}

}  // namespace

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

Context::Context(int device) : device_(device) {
  if (device < 0) throw std::invalid_argument("device context: device 不能为负");
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (device >= count) throw std::invalid_argument("device context: device 超出可用设备范围");
  DeviceScope scope(device_);
  check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags");
}

Context::~Context() noexcept {
  if (stream_ != nullptr) {
    cleanup(device_, [this] {
      (void)cudaStreamSynchronize(stream_);
      (void)cudaStreamDestroy(stream_);
    });
  }
}

void Context::activate() {
  if (failed_) throw std::logic_error("device context: FAILED，禁止继续提交");
  check(cudaSetDevice(device_), "cudaSetDevice");
}

void Context::check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) failed_ = true;
  check_cuda(result, operation);
}

void Context::finish() {
  activate();
  check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

Event::Event(Context& context, bool timing) : device_(context.device()) {
  context.activate();
  context.check(cudaEventCreateWithFlags(&event_, timing ? cudaEventDefault : cudaEventDisableTiming),
                "cudaEventCreateWithFlags");
}

Event::~Event() noexcept {
  if (event_ != nullptr) cleanup(device_, [this] { (void)cudaEventDestroy(event_); });
}

Event::Event(Event&& other) noexcept
    : device_(std::exchange(other.device_, -1)),
      event_(std::exchange(other.event_, nullptr)),
      recorded_(std::exchange(other.recorded_, false)) {}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    Event previous(std::move(other));
    std::swap(device_, previous.device_);
    std::swap(event_, previous.event_);
    std::swap(recorded_, previous.recorded_);
  }
  return *this;
}

void Event::record(Context& context) {
  if (event_ == nullptr || context.device() != device_) {
    throw std::invalid_argument("device event: event/context device 不符或 event 已移出");
  }
  context.activate();
  context.check(cudaEventRecord(event_, context.stream()), "cudaEventRecord");
  recorded_ = true;
}

void Event::wait(Context& context) const {
  require_event(event_, recorded_);
  if (context.device() != device_) throw std::invalid_argument("device event: device 不符");
  context.activate();
  context.check(cudaStreamWaitEvent(context.stream(), event_, 0), "cudaStreamWaitEvent");
}

void Event::synchronize(Context& context) const {
  require_event(event_, recorded_);
  if (context.device() != device_) throw std::invalid_argument("device event: device 不符");
  context.activate();
  context.check(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

}  // namespace uavrgb::device
