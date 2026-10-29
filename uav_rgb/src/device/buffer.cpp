// 设备/锁页存储唯一所有权；热路径复制不分配、不隐式同步。
#include "uavrgb/device/buffer.hpp"

#include <limits>

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

void release_device(void* data, int device) noexcept {
  if (data == nullptr) return;
  int previous = -1;
  (void)cudaGetDevice(&previous);
  if (cudaSetDevice(device) == cudaSuccess) (void)cudaFree(data);
  if (previous >= 0) (void)cudaSetDevice(previous);
}

void check_copy(const DeviceBuffer& buffer, Context& context, const void* host,
                std::size_t bytes, std::size_t offset) {
  if (buffer.device() != context.device()) {
    throw std::invalid_argument("device copy: buffer/context device 不符");
  }
  if (checked_add(offset, bytes) > buffer.size_bytes() ||
      (bytes != 0 && (host == nullptr || buffer.data() == nullptr))) {
    throw std::invalid_argument("device copy: 空指针或容量不足");
  }
  context.activate();
}

}  // namespace

std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::overflow_error("device layout: 加法溢出");
  }
  return a + b;
}

std::size_t checked_multiply(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::overflow_error("device layout: 乘法溢出");
  }
  return a * b;
}

std::size_t checked_align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("device layout: alignment 必须为非零2次幂");
  }
  const auto remainder = value % alignment;
  return remainder == 0 ? value : checked_add(value, alignment - remainder);
}

DeviceBuffer::DeviceBuffer(std::size_t bytes, int device) : bytes_(bytes), device_(device) {
  if (device < 0) throw std::invalid_argument("device buffer: device 不能为负");
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (device >= count) throw std::invalid_argument("device buffer: device 超出可用设备范围");
  DeviceScope scope(device_);
  if (bytes_ != 0) check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
}

DeviceBuffer::~DeviceBuffer() noexcept { release_device(data_, device_); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    DeviceBuffer previous(std::move(other));
    swap(previous);
  }
  return *this;
}

void DeviceBuffer::swap(DeviceBuffer& other) noexcept {
  std::swap(data_, other.data_);
  std::swap(bytes_, other.bytes_);
  std::swap(device_, other.device_);
}

void DeviceBuffer::copy_from_host(Context& context, const void* source, std::size_t bytes,
                                   std::size_t offset) {
  check_copy(*this, context, source, bytes, offset);
  if (bytes == 0) return;
  context.check(cudaMemcpyAsync(static_cast<unsigned char*>(data_) + offset, source, bytes,
                                cudaMemcpyHostToDevice, context.stream()), "cudaMemcpyAsync H2D");
}

void DeviceBuffer::copy_to_host(Context& context, void* destination, std::size_t bytes,
                                 std::size_t offset) const {
  check_copy(*this, context, destination, bytes, offset);
  if (bytes == 0) return;
  context.check(cudaMemcpyAsync(destination, static_cast<const unsigned char*>(data_) + offset,
                                bytes, cudaMemcpyDeviceToHost, context.stream()), "cudaMemcpyAsync D2H");
}

PinnedBuffer::PinnedBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes_ != 0) check_cuda(cudaMallocHost(&data_, bytes_), "cudaMallocHost");
}

PinnedBuffer::~PinnedBuffer() noexcept {
  if (data_ != nullptr) (void)cudaFreeHost(data_);
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept { swap(other); }

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
  if (this != &other) {
    PinnedBuffer previous(std::move(other));
    swap(previous);
  }
  return *this;
}

void PinnedBuffer::swap(PinnedBuffer& other) noexcept {
  std::swap(data_, other.data_);
  std::swap(bytes_, other.bytes_);
}

}  // namespace uavrgb::device
