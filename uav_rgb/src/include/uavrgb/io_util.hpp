#pragma once
// 小端整数/浮点解析与二进制定长读写（仅依赖 C++17 标准库，独立校验工具也可直接 include）。
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace uavrgb {

inline std::uint16_t le16(const void* ptr) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(ptr);
  return static_cast<std::uint16_t>(p[0] | (static_cast<std::uint16_t>(p[1]) << 8));
}

inline std::uint32_t le32(const void* ptr) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(ptr);
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t le64(const void* ptr) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(ptr);
  return static_cast<std::uint64_t>(le32(p)) |
         (static_cast<std::uint64_t>(le32(p + 4)) << 32);
}

inline float le_f32(const void* ptr) noexcept {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "IEEE754 float32 required");
  const std::uint32_t u = le32(ptr);
  float f = 0.0f;
  std::memcpy(&f, &u, 4);
  return f;
}

template <typename T>
inline void put_le(std::ostream& out, T v) {
  static_assert(std::is_integral_v<T> || std::is_same_v<T, float>,
                "put_le supports integral types and float");
  if constexpr (std::is_same_v<T, float>) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "IEEE754 float32 required");
    std::uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    put_le<std::uint32_t>(out, u);
  } else {
    using U = std::make_unsigned_t<T>;
    const auto u = static_cast<std::uint64_t>(static_cast<U>(v));
    char buf[sizeof(T)];
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      buf[i] = static_cast<char>((u >> (8 * i)) & 0xffu);
    }
    out.write(buf, static_cast<std::streamsize>(sizeof(T)));
  }
}

template <typename T>
inline bool get_le(std::istream& in, T& out) {
  static_assert(std::is_integral_v<T> || std::is_same_v<T, float>,
                "get_le supports integral types and float");
  unsigned char buf[sizeof(T)]{};
  in.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(sizeof(T)));
  if (in.gcount() != static_cast<std::streamsize>(sizeof(T))) return false;
  if constexpr (std::is_same_v<T, float>) {
    out = le_f32(buf);
  } else if constexpr (sizeof(T) == 1) {
    out = static_cast<T>(buf[0]);
  } else if constexpr (sizeof(T) == 2) {
    out = static_cast<T>(le16(buf));
  } else if constexpr (sizeof(T) == 4) {
    out = static_cast<T>(le32(buf));
  } else if constexpr (sizeof(T) == 8) {
    out = static_cast<T>(le64(buf));
  }
  return true;
}

template <typename T, typename Err = std::invalid_argument>
inline void get_le(std::istream& in, T& out, const char* what,
                   const char* prefix = "checkpoint") {
  if (!get_le<T>(in, out)) {
    throw Err(std::string(prefix) + ": 读取不足: " + what);
  }
}

inline bool read_exact_ok(std::istream& in, std::uint64_t off, void* dst, std::size_t len) {
  in.clear();
  in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
  if (!in) return false;
  in.read(static_cast<char*>(dst), static_cast<std::streamsize>(len));
  return static_cast<std::size_t>(in.gcount()) == len;
}

// 读整个区间；长度不足即报错（截断文件不允许静默当成零填充）。
inline void read_exact(std::istream& in, std::uint64_t off, void* dst, std::size_t len,
                       const char* what, const char* prefix = "UAVR") {
  in.clear();
  in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
  if (!in) throw std::runtime_error(std::string(prefix) + ": seek 失败: " + what);
  in.read(static_cast<char*>(dst), static_cast<std::streamsize>(len));
  if (static_cast<std::size_t>(in.gcount()) != len) {
    throw std::runtime_error(std::string(prefix) + ": 读取不足: " + what);
  }
}

// 统一 JSON 字符串转义（替代散落在 trainer / class_weights / metrics_json 的三套 if-else 链）。
inline std::string json_quote(const std::string& s) {
  std::string o = "\"";
  o.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"':  o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n";  break;
      case '\r': o += "\\r";  break;
      case '\t': o += "\\t";  break;
      default:
        if (c < 0x20) {
          char b[7];
          std::snprintf(b, sizeof b, "\\u%04x", static_cast<unsigned>(c));
          o += b;
        } else {
          o += static_cast<char>(c);
        }
        break;
    }
  }
  o += '"';
  return o;
}

}  // namespace uavrgb
