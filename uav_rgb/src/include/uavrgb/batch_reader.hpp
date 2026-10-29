#pragma once
// 每个 reader 独占 fd 和 uint8 scratch；同一实例不能并发 read_images。
#include "uavrgb/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace uavrgb {

// 读满才返回；处理 EINTR/短读，EOF 和 off_t 越界抛异常。
void read_exact_at(int fd, std::uint64_t offset, void* destination, std::size_t count);

class BatchReader {
 public:
  explicit BatchReader(const std::string& path, LoadOptions options = {});
  ~BatchReader() noexcept;
  BatchReader(const BatchReader&) = delete;
  BatchReader& operator=(const BatchReader&) = delete;
  BatchReader(BatchReader&& other) noexcept;
  BatchReader& operator=(BatchReader&& other) noexcept;

  const UavrDataset& metadata() const;
  // 身份/元信息指纹，不是内容 SHA256；时间戳不变的原地写入无法保证检出。
  std::string fingerprint() const;
  void verify_unchanged() const;
  void read_images(const std::vector<std::uint32_t>& indices, float* destination,
                   std::size_t capacity);

 private:
  int fd_ = -1;
  std::string path_;
  struct stat initial_ {};
  std::unique_ptr<UavrDataset> metadata_;
  std::vector<unsigned char> scratch_;
};

}  // namespace uavrgb
