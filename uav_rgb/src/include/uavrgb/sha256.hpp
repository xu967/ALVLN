#pragma once
#include <array>
#include <cstdint>
#include <istream>
#include <string>

namespace uavrgb {
using Sha256 = std::array<std::uint8_t, 32>;
Sha256 sha256(std::istream& input);
Sha256 sha256_file(const std::string& path);
std::string sha256_hex(const Sha256& h);  // 64 位小写十六进制
}  // namespace uavrgb
