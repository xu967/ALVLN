// 第38节首批验收：布局、设备资源生命周期、状态往返及无污染失败。
#include "device_state_test_helpers.hpp"

#include <array>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace device_state_test;

namespace { bool inject_event_sync_failure = false; }
extern "C" cudaError_t __real_cudaEventSynchronize(cudaEvent_t event);
extern "C" cudaError_t __wrap_cudaEventSynchronize(cudaEvent_t event) {
  if (std::exchange(inject_event_sync_failure, false)) return cudaErrorLaunchFailure;
  return __real_cudaEventSynchronize(event);
}

namespace {
void test_arithmetic_and_layout() {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  require(checked_add(maximum, 0) == maximum && checked_multiply(maximum, 0) == 0,
          "checked 边界值错误");
  require(checked_align_up(0) == 0 && checked_align_up(1) == 256 &&
              checked_align_up(256) == 256, "256 字节对齐错误");
  rejects([&] { checked_add(maximum, 1); }, "加法溢出");
  rejects([&] { checked_multiply(maximum, sizeof(float)); }, "乘法溢出");
  rejects([&] { checked_align_up(maximum); }, "对齐溢出");
  rejects([] { checked_align_up(1, 0); }, "零对齐");
  rejects([] { checked_align_up(1, 3); }, "非2次幂对齐");
  LayoutBuilder arena;
  require(arena.reserve<float>(1) == 0 && arena.reserve<double>(1) == 256 &&
              arena.total_bytes() == 264, "arena offset 错误");
  rejects([&] { arena.reserve<float>(maximum); }, "arena 容量溢出");
  require(arena.total_bytes() == 264, "失败 reserve 污染 arena");
  LayoutBuilder full;
  full.reserve<unsigned char>(maximum);
  rejects([&] { full.reserve<float>(1); }, "arena 总量溢出");

  const auto host = fixture();
  const auto layout = make_parameter_layout(host.model);
  const std::array<std::size_t, 8> offsets{0, 432, 448, 5056, 5088, 23520, 23584, 100384};
  const std::array<std::size_t, 8> counts{432, 16, 4608, 32, 18432, 64, 76800, 4};
  const std::array<const char*, 8> names{"c1.W", "c1.b", "c2.W", "c2.b",
                                         "c3.W", "c3.b", "fc.W", "fc.b"};
  const auto parameters = cnn_params(host.model);
  require(layout.count == 100388, "120x160 参数总量错误");
  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto& segment = layout.segments[i];
    require(segment.name == names[i] && segment.offset == offsets[i] &&
                segment.count == counts[i] && segment.shape == parameters[i]->dims(),
            "布局段错误 index=" + std::to_string(i));
  }
  const auto small = make_cnn(16, 24, 73);
  require(make_parameter_layout(small).count == num_params(small), "小分辨率布局不能硬编码");
  std::cout << "layout: 100388 parameters / 8 segments / checked overflow PASS\n";
}

void test_buffers_and_events() {
  static_assert(!std::is_copy_constructible_v<DeviceBuffer> &&
                !std::is_copy_constructible_v<PinnedBuffer> &&
                !std::is_copy_constructible_v<Event> &&
                std::is_nothrow_destructible_v<Context>);
  Context context;
  unsigned flags = 0;
  check_cuda(cudaStreamGetFlags(context.stream(), &flags), "stream flags");
  require((flags & cudaStreamNonBlocking) != 0, "compute stream 必须 non-blocking");
  DeviceBuffer buffer(4 * sizeof(float), context.device());
  PinnedBuffer source(4 * sizeof(float)), destination(4 * sizeof(float));
  auto* values = source.at<float>(0, 4);
  for (int i = 0; i < 4; ++i) values[i] = static_cast<float>(i) * 0.25f;
  rejects([&] { buffer.at<float>(1, 1); }, "未对齐 device offset");
  rejects([&] { source.at<float>(1, 1); }, "未对齐 pinned offset");
  rejects([&] { checked_at<float>(static_cast<unsigned char*>(source.data()) + 1,
                                  source.size_bytes() - 1, 0, 1); }, "未对齐基址");
  rejects([&] { buffer.at<float>(buffer.size_bytes(), 1); }, "device 容量不足");
  rejects([&] { destination.at<double>(0, 3); }, "pinned 容量不足");
  rejects([&] { buffer.at<float>(std::numeric_limits<std::size_t>::max(), 1); }, "offset 溢出");
  rejects([&] { buffer.copy_from_host(context, nullptr, sizeof(float)); }, "空 H2D 源");
  rejects([&] { buffer.copy_to_host(context, nullptr, sizeof(float)); }, "空 D2H 目标");
  rejects([&] { buffer.copy_from_host(context, values, buffer.size_bytes() + 1); }, "H2D 超容量");
  require(!context.failed(), "前置参数错误不应污染 context");
  Event ready(context);
  rejects([&] { ready.wait(context); }, "未 record event.wait");
  rejects([&] { ready.synchronize(context); }, "未 record event.synchronize");
  buffer.copy_from_host(context, values, source.size_bytes());
  ready.record(context);
  Context consumer(context.device());
  ready.wait(consumer);
  buffer.copy_to_host(consumer, destination.data(), destination.size_bytes());
  consumer.finish();
  context.finish();
  require(std::memcmp(source.data(), destination.data(), source.size_bytes()) == 0,
          "跨 stream event 顺序/复制错误");
  auto* original = buffer.data();
  DeviceBuffer moved(std::move(buffer));
  require(buffer.data() == nullptr && buffer.size_bytes() == 0 && buffer.device() == -1 &&
              moved.data() == original, "DeviceBuffer move 构造错误");
  DeviceBuffer assigned(sizeof(float));
  assigned = std::move(moved);
  require(moved.data() == nullptr && moved.size_bytes() == 0 && assigned.data() == original,
          "DeviceBuffer move 赋值错误");
  original = source.data();
  PinnedBuffer pinned(std::move(source)), pinned_assigned(sizeof(float));
  require(source.data() == nullptr && source.size_bytes() == 0, "PinnedBuffer move 构造错误");
  pinned_assigned = std::move(pinned);
  require(pinned.data() == nullptr && pinned.size_bytes() == 0 && pinned_assigned.data() == original,
          "PinnedBuffer move 赋值错误");
  Event event_moved(std::move(ready));
  require(ready.get() == nullptr && ready.device() == -1, "Event move 构造错误");
  Event event_assigned(context);
  event_assigned = std::move(event_moved);
  require(event_moved.get() == nullptr, "Event move 赋值错误");
  event_assigned.synchronize(context);
  rejects([&] { ready.synchronize(context); }, "moved-from event");
  rejects([&] { buffer.at<float>(0, 1); }, "moved-from buffer");
  rejects([] { Context invalid(-1); }, "负 device");
  int devices = 0;
  check_cuda(cudaGetDeviceCount(&devices), "device count");
  rejects([&] { Context invalid(devices); }, "越界 device");
  // 只清除本用例主动触发的 invalid-device，避免给后续 launch 留下旧错误。
  const auto device_error = cudaGetLastError();
  require(device_error == cudaSuccess || device_error == cudaErrorInvalidDevice,
          "非法 device 测试留下非预期 CUDA 错误");
  rejects([] { DeviceBuffer invalid(sizeof(float), -1); }, "buffer 负 device");
  Context failed;
  Event failed_event(failed);
  failed_event.record(failed);
  inject_event_sync_failure = true;
  rejects([&] { failed_event.synchronize(failed); }, "event同步 CUDA 错误传播");
  require(!inject_event_sync_failure && failed.failed(), "event同步错误未标记 context FAILED");
  rejects([&] { failed.activate(); }, "FAILED context.activate");
  rejects([&] { failed.finish(); }, "FAILED context.finish");
  for (int i = 0; i < 3; ++i) {
    Context repeated;
    Event event(repeated, true);
    event.record(repeated);
    event.synchronize(repeated);
    repeated.finish();
  }
  std::cout << "buffer/context/event: move / bounds / stream dependency PASS\n";
}

void test_state_roundtrip() {
  const auto original = fixture();
  Context context;
  HostSnapshot fresh{make_cnn(8, 16, 73), {}};
  fresh.state = make_train_state(fresh.model, 0.001f);
  DeviceTrainState fresh_device(context, fresh.model, fresh.state);
  same_snapshot(fresh, fresh_device.snapshot(context));
  DeviceTrainState device(context, original.model, original.state);
  require(device.step() == 17 && device.layout().count == 100388 &&
              device.device() == context.device(), "device 元数据错误");
  same_snapshot(original, device.snapshot(context));
  std::vector<float> gradients(device.layout().count, 1.0f);
  device.grads().copy_to_host(context, gradients.data(), gradients.size() * sizeof(float));
  context.finish();
  for (float value : gradients) require(value == 0.0f, "新状态梯度未清零");
  auto snapshot = device.snapshot(context);
  snapshot.model.fc.b[0] += 1.0f;
  snapshot.state.adam[0].m[0] += 1.0f;
  same_snapshot(original, device.snapshot(context));

  using Mutation = std::function<void(HostSnapshot&)>;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<std::pair<const char*, Mutation>> invalid{
      {"c1.W rank", [](auto& h) { h.model.c1.W = Tensor({16, 3, 9}); }},
      {"fc.W transpose", [](auto& h) { h.model.fc.W = Tensor({19200, 4}); }},
      {"fc.in", [](auto& h) { --h.model.fc.in; }},
      {"H indivisible", [](auto& h) { h.model.in_h = 7; }},
      {"conv contract", [](auto& h) { h.model.c2.pad = 0; }},
      {"params NaN", [=](auto& h) { h.model.c3.b[0] = nan; }},
      {"Adam count", [](auto& h) { h.state.adam.pop_back(); }},
      {"Adam m shape", [](auto& h) { h.state.adam[0].m = Tensor({432}); }},
      {"Adam v shape", [](auto& h) { h.state.adam[0].v = Tensor({432}); }},
      {"Adam m Inf", [=](auto& h) { h.state.adam[5].m[0] = inf; }},
      {"Adam v NaN", [=](auto& h) { h.state.adam[6].v[0] = nan; }},
      {"Adam v negative", [](auto& h) { h.state.adam[7].v[0] = -1.0f; }},
      {"Adam lr differs", [](auto& h) { h.state.adam[1].lr *= 2.0f; }},
      {"Adam beta1 differs", [](auto& h) { h.state.adam[2].beta1 = 0.8f; }},
      {"Adam beta2 differs", [](auto& h) { h.state.adam[3].beta2 = 0.8f; }},
      {"Adam eps differs", [](auto& h) { h.state.adam[4].eps *= 2.0f; }},
      {"Adam t differs", [](auto& h) { ++h.state.adam[5].t; }},
      {"epoch negative", [](auto& h) { h.state.epoch = -1; }},
      {"weights length", [](auto& h) { h.state.class_weights.pop_back(); }},
      {"weights negative", [](auto& h) { h.state.class_weights[1] = -1.0f; }},
      {"weights NaN", [=](auto& h) { h.state.class_weights[2] = nan; }},
      {"weights zero", [](auto& h) { h.state.class_weights.assign(4, 0.0f); }}};
  for (const auto& [name, mutate] : invalid) {
    auto bad = original;
    mutate(bad);
    rejects([&] { device.import(context, bad.model, bad.state); }, name);
    same_snapshot(original, device.snapshot(context));
  }
  for (const auto& field : {"lr", "beta1", "beta2", "eps", "t"}) {
    auto bad = original;
    for (auto& adam : bad.state.adam) {
      const std::string name(field);
      if (name == "lr") adam.lr = inf;
      if (name == "beta1") adam.beta1 = 1.0f;
      if (name == "beta2") adam.beta2 = -1.0f;
      if (name == "eps") adam.eps = 0.0f;
      if (name == "t") adam.t = -1;
    }
    rejects([&] { device.import(context, bad.model, bad.state); }, std::string("invalid ") + field);
    same_snapshot(original, device.snapshot(context));
  }
  auto alternate = original;
  alternate.model.fc.b[0] += 2.0f;
  for (auto& adam : alternate.state.adam) ++adam.t;
  device.import(context, alternate.model, alternate.state);
  same_snapshot(alternate, device.snapshot(context));
  DeviceTrainState moved(std::move(device));
  require(device.params().data() == nullptr && device.grads().data() == nullptr &&
              device.adam_m().data() == nullptr && device.adam_v().data() == nullptr,
          "state move 构造未清空源 buffer");
  rejects([&] { device.snapshot(context); }, "moved-from state");
  DeviceTrainState assigned(context, original.model, original.state);
  assigned = std::move(moved);
  require(moved.params().data() == nullptr, "state move 赋值未清空源 buffer");
  same_snapshot(alternate, assigned.snapshot(context));
  Context failed;
  rejects([&] { failed.check(cudaErrorInvalidValue, "injected import context error"); },
          "导入前 context 失败");
  const auto* saved_parameters = assigned.params().data();
  rejects([&] { assigned.import(failed, original.model, original.state); }, "FAILED context 导入");
  require(assigned.params().data() == saved_parameters, "失败 context 导入替换了活跃参数");
  same_snapshot(alternate, assigned.snapshot(context));
  DeviceTrainState empty;
  rejects([&] { empty.snapshot(context); }, "empty state snapshot");
  {
    DeviceTrainState partial(context, original.model, original.state);
    partial.adam_m() = DeviceBuffer(sizeof(float), context.device());
    // params D2H已提交，随后m容量校验失败；异常退出须排空后才释放pinned目标。
    rejects([&] { partial.snapshot(context); }, "snapshot中途失败");
    require(!context.failed(), "host容量错误不应污染 context");
    context.finish();
  }
  device.import(context, original.model, original.state);
  same_snapshot(original, device.snapshot(context));
  context.finish();
  std::cout << "state: p/m/v bitwise roundtrip / independent snapshot / rejected imports PASS\n";
}
}  // namespace

int main() {
  try {
    test_arithmetic_and_layout();
    test_buffers_and_events();
    test_state_roundtrip();
    std::cout << "test_device_state: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_device_state: FAIL: " << error.what() << '\n';
    return 1;
  }
}
