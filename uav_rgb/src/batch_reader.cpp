// 独立 pread reader：保留既有 UAVR 校验，避免共享 ifstream 的 seek 状态。
#include "uavrgb/batch_reader.hpp"

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace uavrgb {
namespace {

[[noreturn]] void io_error(const char* operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

bool same_file(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
         a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
         a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}

std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::overflow_error("BatchReader: 元素数乘法溢出");
  }
  return a * b;
}

}  // namespace

void read_exact_at(int fd, std::uint64_t offset, void* destination, std::size_t count) {
  const auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (offset > max_offset || (count != 0 && count - 1 > max_offset - offset)) {
    throw std::overflow_error("pread: 偏移超出 off_t");
  }
  if (count != 0 && destination == nullptr) {
    throw std::invalid_argument("pread: 非空读取的目标指针为空");
  }
  auto* dst = static_cast<unsigned char*>(destination);
  std::size_t done = 0;
  while (done < count) {
    const auto chunk = std::min(count - done,
                               static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t n = ::pread(fd, dst + done, chunk, static_cast<off_t>(offset + done));
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) io_error("pread");
    if (n == 0) throw std::runtime_error("pread: 数据集读取不足/EOF");
    done += static_cast<std::size_t>(n);
  }
}

BatchReader::BatchReader(const std::string& path, LoadOptions options)
    : path_(std::filesystem::absolute(path).lexically_normal().string()) {
  // 先锁定文件身份，元数据校验之后再核验，避免校验与实际读取混用版本。
  fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) io_error("BatchReader open");
  try {
    if (::fstat(fd_, &initial_) != 0) io_error("BatchReader fstat");
    if (!S_ISREG(initial_.st_mode)) {
      throw std::invalid_argument("BatchReader: 只支持普通文件");
    }
    metadata_ = std::make_unique<UavrDataset>(load_uavr(path_, options));
    metadata_->file.close();
    verify_unchanged();
    scratch_.resize(checked_product(checked_product(metadata_->c, metadata_->h), metadata_->w));
  } catch (...) {
    ::close(fd_);
    fd_ = -1;
    throw;
  }
}

BatchReader::~BatchReader() noexcept {
  if (fd_ >= 0) ::close(fd_);
}

BatchReader::BatchReader(BatchReader&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), path_(std::move(other.path_)), initial_(other.initial_),
      metadata_(std::move(other.metadata_)), scratch_(std::move(other.scratch_)) {}

BatchReader& BatchReader::operator=(BatchReader&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = std::exchange(other.fd_, -1);
    path_ = std::move(other.path_);
    initial_ = other.initial_;
    metadata_ = std::move(other.metadata_);
    scratch_ = std::move(other.scratch_);
  }
  return *this;
}

const UavrDataset& BatchReader::metadata() const {
  if (!metadata_) throw std::logic_error("BatchReader: 已移动的 reader");
  return *metadata_;
}

std::string BatchReader::fingerprint() const {
  verify_unchanged();
  return std::to_string(initial_.st_dev) + ":" + std::to_string(initial_.st_ino) + ":" +
         std::to_string(initial_.st_size) + ":" + std::to_string(initial_.st_mtim.tv_sec) +
         ":" + std::to_string(initial_.st_mtim.tv_nsec) + ":" +
         std::to_string(initial_.st_ctim.tv_sec) + ":" +
         std::to_string(initial_.st_ctim.tv_nsec);
}

void BatchReader::verify_unchanged() const {
  if (fd_ < 0) throw std::logic_error("BatchReader: 已移动的 reader");
  struct stat current {}, named {};
  if (::fstat(fd_, &current) != 0) io_error("BatchReader fstat");
  if (::stat(path_.c_str(), &named) != 0) io_error("BatchReader stat");
  if (!same_file(initial_, current) || !same_file(initial_, named)) {
    throw std::runtime_error("BatchReader: 数据文件在运行期间被替换、截断或修改");
  }
}

void BatchReader::read_images(const std::vector<std::uint32_t>& indices, float* destination,
                              std::size_t capacity) {
  verify_unchanged();
  const std::size_t plane = scratch_.size();
  const auto count = checked_product(indices.size(), plane);
  if (count > capacity || (count != 0 && destination == nullptr)) {
    throw std::invalid_argument("BatchReader: batch 目标容量不足或为空");
  }
  for (const auto index : indices) {
    if (index >= static_cast<std::uint32_t>(metadata_->n)) {
      throw std::out_of_range("BatchReader: 样本下标越界");
    }
  }
  const float inv = 1.0f / 255.0f;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const auto relative = checked_product(indices[i], plane);
    if (relative > std::numeric_limits<std::uint64_t>::max() - metadata_->images_off) {
      throw std::overflow_error("BatchReader: 样本偏移加法溢出");
    }
    read_exact_at(fd_, metadata_->images_off + relative, scratch_.data(), plane);
    float* dst = destination + i * plane;
    for (std::size_t j = 0; j < plane; ++j) dst[j] = static_cast<float>(scratch_[j]) * inv;
  }
  verify_unchanged();
}

}  // namespace uavrgb
