#include "uavrgb/run_owner.hpp"

#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <sys/file.h>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace uavrgb {

RunOwner::RunOwner(const std::filesystem::path& output_directory, bool resume) {
  if (output_directory.empty()) throw std::invalid_argument("RunOwner: 输出目录不能为空");
  std::filesystem::create_directories(output_directory);
  directory_ = std::filesystem::canonical(output_directory);
  fd_ = ::open((directory_ / ".run.lock").c_str(),
               O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "RunOwner open");
  try {
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      if (errno == EWOULDBLOCK) throw std::runtime_error("该目录已有训练 writer");
      throw std::system_error(errno, std::generic_category(), "RunOwner flock");
    }
    if (!resume) {
      for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        if (entry.path().filename() != ".run.lock") {
          throw std::runtime_error("RunOwner: 非空输出目录要求显式 resume");
        }
      }
    }
  } catch (...) {
    ::close(fd_);
    fd_ = -1;
    throw;
  }
}

RunOwner::~RunOwner() noexcept {
  if (fd_ >= 0) ::close(fd_);
}

RunOwner::RunOwner(RunOwner&& other) noexcept
    : directory_(std::move(other.directory_)), fd_(std::exchange(other.fd_, -1)) {}

RunOwner& RunOwner::operator=(RunOwner&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) ::close(fd_);
    directory_ = std::move(other.directory_);
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

}  // namespace uavrgb
