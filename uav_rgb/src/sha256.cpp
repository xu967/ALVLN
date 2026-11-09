// FIPS 180-4 SHA-256；流式读取，数据集无需整份载入内存。
#include "uavrgb/sha256.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace uavrgb {
namespace {
constexpr std::uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

std::uint32_t ror(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void block(std::array<std::uint32_t, 8>& h, const std::uint8_t* p) {
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (std::uint32_t(p[4*i]) << 24) | (std::uint32_t(p[4*i+1]) << 16) |
           (std::uint32_t(p[4*i+2]) << 8) | p[4*i+3];
  }
  for (int i = 16; i < 64; ++i) {
    const auto x = w[i-15], y = w[i-2];
    w[i] = w[i-16] + (ror(x,7) ^ ror(x,18) ^ (x>>3)) + w[i-7] +
           (ror(y,17) ^ ror(y,19) ^ (y>>10));
  }
  auto a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], v=h[7];
  for (int i = 0; i < 64; ++i) {
    const auto t1 = v + (ror(e,6)^ror(e,11)^ror(e,25)) + ((e&f)^((~e)&g)) + k[i] + w[i];
    const auto t2 = (ror(a,2)^ror(a,13)^ror(a,22)) + ((a&b)^(a&c)^(b&c));
    v=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=v;
}
}  // namespace

Sha256 sha256(std::istream& input) {
  std::array<std::uint32_t, 8> h = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                  0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::uint64_t bytes = 0;
  std::array<std::uint8_t, 65536> buf{};
  std::size_t tail = 0;
  for (;;) {
    input.read(reinterpret_cast<char*>(buf.data()), buf.size());
    const auto n = static_cast<std::size_t>(input.gcount());
    if (input.bad() || (input.fail() && !input.eof())) throw std::runtime_error("SHA256: 读取失败");
    bytes += n;
    const auto complete = n / 64 * 64;
    for (std::size_t i = 0; i < complete; i += 64) block(h, buf.data()+i);
    if (n < buf.size()) {
      tail = n % 64;
      std::memmove(buf.data(), buf.data()+complete, tail);
      break;
    }
  }
  buf[tail++] = 0x80;
  const auto padded = tail <= 56 ? 64u : 128u;
  std::fill(buf.begin()+tail, buf.begin()+padded, 0);
  const auto bits = bytes * 8;
  for (int i = 0; i < 8; ++i) buf[padded-1-i] = static_cast<std::uint8_t>(bits >> (8*i));
  for (std::size_t i = 0; i < padded; i += 64) block(h, buf.data()+i);
  Sha256 result{};
  for (int i = 0; i < 32; ++i) result[i] = static_cast<std::uint8_t>(h[i/4] >> (24-8*(i%4)));
  return result;
}

Sha256 sha256_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("SHA256: 无法打开 " + path);
  return sha256(f);
}

std::string sha256_hex(const Sha256& h) {
  static const char* d = "0123456789abcdef";
  std::string s;
  s.reserve(64);
  for (const std::uint8_t b : h) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}
}  // namespace uavrgb
