#pragma once
// buffer 唯一持有存储；异步复制完成前调用者须保持两端存储有效。
#include "uavrgb/device/context.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace uavrgb::device {

std::size_t checked_add(std::size_t a, std::size_t b);
std::size_t checked_multiply(std::size_t a, std::size_t b);
std::size_t checked_align_up(std::size_t value, std::size_t alignment = 256);

class LayoutBuilder {
 public:
  template <class T> std::size_t reserve(std::size_t count) {
    const auto bytes = checked_multiply(count, sizeof(T));
    const auto offset = checked_align_up(total_, 256);
    const auto next = checked_add(offset, bytes);
    total_ = next;
    return offset;
  }
  std::size_t total_bytes() const noexcept { return total_; }

 private:
  std::size_t total_ = 0;
};

// checked_at 不触及设备内容，只检查类型对齐与 buffer 容量。
template <class T> T* checked_at(void* data, std::size_t capacity,
                                std::size_t offset, std::size_t count) {
  const auto end = checked_add(offset, checked_multiply(count, sizeof(T)));
  if (data == nullptr || reinterpret_cast<std::uintptr_t>(data) % alignof(T) != 0 ||
      offset % alignof(T) != 0 || end > capacity) {
    throw std::invalid_argument("device buffer: 对齐、容量或空指针非法");
  }
  return reinterpret_cast<T*>(static_cast<unsigned char*>(data) + offset);
}

class DeviceBuffer {
 public:
  DeviceBuffer() noexcept = default;
  explicit DeviceBuffer(std::size_t bytes, int device = 0);
  ~DeviceBuffer() noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  std::size_t size_bytes() const noexcept { return bytes_; }
  int device() const noexcept { return device_; }
  template <class T> T* at(std::size_t offset, std::size_t count) {
    return checked_at<T>(data_, bytes_, offset, count);
  }
  template <class T> const T* at(std::size_t offset, std::size_t count) const {
    return checked_at<const T>(const_cast<void*>(data_), bytes_, offset, count);
  }
  // 仅提交到 context.stream；读取结果或释放存储前显式 finish。
  void copy_from_host(Context& context, const void* source, std::size_t bytes,
                      std::size_t offset = 0);
  void copy_to_host(Context& context, void* destination, std::size_t bytes,
                    std::size_t offset = 0) const;
  void swap(DeviceBuffer& other) noexcept;

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  int device_ = -1;
};

class PinnedBuffer {
 public:
  PinnedBuffer() noexcept = default;
  explicit PinnedBuffer(std::size_t bytes);
  ~PinnedBuffer() noexcept;
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& other) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  std::size_t size_bytes() const noexcept { return bytes_; }
  template <class T> T* at(std::size_t offset, std::size_t count) {
    return checked_at<T>(data_, bytes_, offset, count);
  }
  template <class T> const T* at(std::size_t offset, std::size_t count) const {
    return checked_at<const T>(const_cast<void*>(data_), bytes_, offset, count);
  }
  void swap(PinnedBuffer& other) noexcept;

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace uavrgb::device
