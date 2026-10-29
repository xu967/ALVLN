// GNU ld wrap 只注入系统调用结果；实际循环、校验、归一化均走生产 reader。
#include "synthetic_dataset.hpp"
#include "uavrgb/batch_reader.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace {
bool inject_eintr = false;
bool inject_eof = false;
std::size_t read_limit = std::numeric_limits<std::size_t>::max();
int interruptions = 0;
int short_reads = 0;
int failures = 0;

void check(bool ok, const std::string& message) {
  if (!ok) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

template <class F>
void rejected(F operation, const char* message) {
  bool threw = false;
  try { operation(); } catch (const std::exception&) { threw = true; }
  check(threw, message);
}

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() / "uavrgb_reader_XXXXXX").string();
    char* result = ::mkdtemp(pattern.data());
    if (!result) throw std::runtime_error("mkdtemp 失败");
    path = result;
  }
  ~TemporaryDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
}  // namespace

extern "C" ssize_t __real_pread(int fd, void* buffer, size_t count, off_t offset);
extern "C" ssize_t __wrap_pread(int fd, void* buffer, size_t count, off_t offset) {
  if (inject_eintr) {
    inject_eintr = false;
    ++interruptions;
    errno = EINTR;
    return -1;
  }
  if (inject_eof) return 0;
  if (count > read_limit) ++short_reads;
  return __real_pread(fd, buffer, std::min(count, read_limit), offset);
}

int main() {
  using namespace uavrgb;
  namespace fs = std::filesystem;
  try {
    TemporaryDirectory temporary;
    const auto source = temporary.path / "samples.uavr";
    uavrgb_test::write_synthetic_uavr(source.string(), 7, 8, 16, 5, 987);
    const auto reference = load_uavr(source.string());
    BatchReader reader(source.string());
    check(reader.metadata().train_idx == reference.train_idx &&
          reader.metadata().val_idx == reference.val_idx &&
          reader.metadata().labels == reference.labels, "保留原元数据和标签");
    const std::vector<std::uint32_t> order{6, 0, 5, 2, 2, 1, 4};
    Tensor expected;
    reference.load_images(order, expected);
    std::vector<float> actual(expected.numel() + 3, -7.0f);
    const std::string identity = reader.fingerprint();
    inject_eintr = true;
    read_limit = 17;
    reader.read_images(order, actual.data(), expected.numel());
    read_limit = std::numeric_limits<std::size_t>::max();
    check(interruptions == 1 && short_reads > 1, "生产读取处理 EINTR 与连续短读");
    check(std::memcmp(actual.data(), expected.data(), expected.numel() * sizeof(float)) == 0,
          "乱序、重复样本和 uint8/255 与旧读取逐 bit 相同");
    check(actual.back() == -7.0f && actual[expected.numel()] == -7.0f, "不写超有效容量");
    check(reader.fingerprint() == identity, "正常读取不改变身份指纹");

    std::fill(actual.begin(), actual.end(), -9.0f);
    rejected([&] { reader.read_images(order, actual.data(), expected.numel() - 1); }, "容量不足拒绝");
    check(actual.front() == -9.0f, "容量失败先校验后写入");
    rejected([&] { reader.read_images({0, 7}, actual.data(), actual.size()); }, "下标越界拒绝");
    check(actual.front() == -9.0f, "任一下标非法前不写 batch");
    rejected([&] { reader.read_images({0}, nullptr, actual.size()); }, "空目标拒绝");
    reader.read_images({}, nullptr, 0);
    inject_eof = true;
    rejected([&] { reader.read_images({0}, actual.data(), actual.size()); }, "读取不足拒绝而非补零");
    inject_eof = false;
    rejected([&] { read_exact_at(-1, 0, actual.data(), 1); }, "pread 系统错误不吞掉");
    rejected([&] { read_exact_at(-1, std::numeric_limits<std::uint64_t>::max(), nullptr, 0); },
             "off_t 越界拒绝");
    rejected([&] { read_exact_at(-1, std::numeric_limits<off_t>::max(), actual.data(), 2); },
             "偏移加长度越界拒绝");

    BatchReader moved(std::move(reader));
    rejected([&] { reader.verify_unchanged(); }, "移动后源 reader 无所有权");
    moved.read_images({0}, actual.data(), actual.size());
    const auto second = temporary.path / "second.uavr";
    fs::copy_file(source, second);
    BatchReader assigned(second.string());
    assigned = std::move(moved);
    rejected([&] { moved.verify_unchanged(); }, "move assignment 后源 reader 无所有权");
    assigned.read_images({0}, actual.data(), actual.size());

    const auto replacement = temporary.path / "replacement.uavr";
    fs::copy_file(source, replacement);
    fs::rename(replacement, source);
    rejected([&] { assigned.read_images({0}, actual.data(), actual.size()); }, "同内容路径替换仍拒绝");

    BatchReader truncation(source.string());
    fs::resize_file(source, fs::file_size(source) - 1);
    rejected([&] { truncation.verify_unchanged(); }, "截断拒绝");
    rejected([&] { BatchReader invalid(source.string()); }, "构造保留 UAVR 元数据校验");

    uavrgb_test::write_synthetic_uavr(source.string(), 7, 8, 16, 5, 987);
    BatchReader changed(source.string());
    const int writable = ::open(source.c_str(), O_WRONLY | O_CLOEXEC);
    if (writable < 0) throw std::runtime_error("修改注入 open 失败");
    const unsigned char byte = 127;
    const bool written = ::pwrite(writable, &byte, 1, 48) == 1;
    ::close(writable);
    check(written, "修改注入完成");
    // 显式改变 mtime，避免同一文件系统 tick 的写入未反映到 stat。
    struct stat modified {};
    if (::stat(source.c_str(), &modified) != 0) throw std::runtime_error("修改注入 stat 失败");
    struct timespec times[2]{{0, UTIME_OMIT}, modified.st_mtim};
    ++times[1].tv_sec;
    check(::utimensat(AT_FDCWD, source.c_str(), times, 0) == 0, "确定性修改 mtime");
    rejected([&] { changed.verify_unchanged(); }, "同长度原地修改拒绝");
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
  }
  std::cout << "test_batch_reader: " << (failures == 0 ? "PASS" : "FAIL") << '\n';
  return failures == 0 ? 0 : 1;
}
