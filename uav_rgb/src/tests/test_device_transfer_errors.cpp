// 真实上传/D2H完成后注入返回值，检查异常收尾，不损坏实际CUDA上下文。
#include "device_state_test_helpers.hpp"

#include <utility>

using namespace device_state_test;

namespace {
bool fail_sync = false;
int fail_set_after = 0;
bool set_consumed = false, sync_consumed = false;
int sync_calls = 0;
}

extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  ++sync_calls;
  const auto result = __real_cudaStreamSynchronize(stream);
  if (result != cudaSuccess) return result;
  if (std::exchange(fail_sync, false)) {
    sync_consumed = true;
    return cudaErrorLaunchFailure;
  }
  return result;
}

extern "C" cudaError_t __real_cudaSetDevice(int);
extern "C" cudaError_t __wrap_cudaSetDevice(int device) {
  const auto result = __real_cudaSetDevice(device);
  if (result != cudaSuccess) return result;
  if (fail_set_after > 0 && --fail_set_after == 0) {
    set_consumed = true;
    return cudaErrorUnknown;
  }
  return result;
}

namespace {
void test_cleanup(bool inject_set, bool inject_sync) {
  Context context;
  const auto model = make_cnn(8, 16, 73);
  const auto state = make_train_state(model, 0.001f);
  DeviceTrainState device(context, model, state);
  device.adam_m() = DeviceBuffer(sizeof(float), context.device());
  // 第一次set属于params D2H；m容量错误后的第二次set属于异常收尾。
  fail_set_after = inject_set ? 2 : 0;
  fail_sync = inject_sync;
  set_consumed = sync_consumed = false;
  sync_calls = 0;
  bool original_preserved = false;
  try {
    (void)device.snapshot(context);
  } catch (const std::invalid_argument& error) {
    original_preserved = std::string(error.what()) == "device copy: 空指针或容量不足";
  }
  require(original_preserved, "必须保留首次容量异常的类型与信息");
  require(set_consumed == inject_set && sync_consumed == inject_sync &&
              !fail_sync && fail_set_after == 0, "收尾错误注入未命中");
  require(sync_calls == 1, "set失败后仍须尝试排空已提交的D2H");
  require(context.failed() == (inject_set || inject_sync), "收尾FAILED状态错误");
  if (context.failed()) {
    rejects([&] { context.activate(); }, "收尾失败后activate");
    rejects([&] { context.finish(); }, "收尾失败后finish");
  } else {
    context.activate();
    context.finish();
  }
}
}

int main() {
  try {
    test_cleanup(false, true);
    test_cleanup(true, false);
    test_cleanup(true, true);
    test_cleanup(false, false);
    std::cout << "test_device_transfer_errors: PASS (sync/set/both/host-only)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_device_transfer_errors: FAIL: " << error.what() << '\n';
    return 1;
  }
}
