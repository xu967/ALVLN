#pragma once
// Linux 本地文件系统单 writer 锁；resume 来源/配置由训练控制层另行验证。
#include <filesystem>

namespace uavrgb {

// resume 是否就在 out 目录内（规范化后比较父目录）；resume 为空返回 false。
bool same_run_dir(const std::filesystem::path& out_dir,
                  const std::filesystem::path& resume_path);

class RunOwner {
 public:
  explicit RunOwner(const std::filesystem::path& output_directory, bool resume = false);
  ~RunOwner() noexcept;
  RunOwner(const RunOwner&) = delete;
  RunOwner& operator=(const RunOwner&) = delete;
  RunOwner(RunOwner&& other) noexcept;
  RunOwner& operator=(RunOwner&& other) noexcept;
  const std::filesystem::path& directory() const { return directory_; }

 private:
  std::filesystem::path directory_;
  int fd_ = -1;
};

}  // namespace uavrgb
