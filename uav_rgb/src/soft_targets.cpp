#include "uavrgb/soft_targets.hpp"
#include "uavrgb/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace uavrgb {
namespace {
std::uint64_t le(const unsigned char* p, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= std::uint64_t(p[i]) << (8*i);
  return v;
}
void require(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(std::string("soft targets: ") + msg);
}
}  // namespace

std::vector<float> load_soft_targets(const TrainConfig& cfg, const UavrDataset& ds) {
  if (cfg.soft_targets.empty()) return {};
  // 与类别权重的互斥在 Trainer::run 里查最终生效的权重（resume 会从 checkpoint 带入，CLI 原文看不到）。
  require(cfg.label_smoothing <= 0.0f,
          "--soft-targets 与 --label-smoothing 互斥（蒸馏损失不看 ε，开了也不生效）");
  require(cfg.distill_alpha >= 0.0f && cfg.distill_alpha <= 1.0f,
          "--distill-alpha 必须在 [0,1]");
  std::ifstream f(cfg.soft_targets, std::ios::binary);
  unsigned char header[120]{};
  require(bool(f.read(reinterpret_cast<char*>(header), sizeof header)), "缺少 UAVSOFT1 头，请重新导出");
  require(std::memcmp(header, "UAVSOFT1", 8) == 0, "未知格式；旧裸数组须重新导出");
  require(le(header+8, 8) == static_cast<std::uint64_t>(ds.n) &&
          le(header+16, 4) == ds.num_classes, "N/C 与数据集不符");
  const auto flags = le(header+20, 4);
  require(flags <= 1, "未知 flags");
  const auto count = static_cast<std::uint64_t>(ds.n) * ds.num_classes;
  require(count <= (std::numeric_limits<std::size_t>::max()-120)/4, "大小溢出");
  require(std::filesystem::file_size(cfg.soft_targets) == 120 + count*4, "文件长度不符");
  // 训练器加载数据集时已算过整文件 SHA（ds.file_sha256）；全零表示调用方没要求算，自己补一遍。
  const auto data_hash = ds.file_sha256 != Sha256{} ? ds.file_sha256 : sha256_file(cfg.data_path);
  require(std::equal(data_hash.begin(), data_hash.end(), header+24), "UAVR SHA256 不匹配");
  const auto sidecar = std::filesystem::path(cfg.data_path).parent_path()/"instructions.jsonl";
  require(std::filesystem::exists(sidecar) == (flags == 1), "文本侧车存在状态不匹配");
  const auto text_hash = flags ? sha256_file(sidecar.string()) : Sha256{};
  require(std::equal(text_hash.begin(), text_hash.end(), header+56), "文本侧车 SHA256 不匹配");
  const auto payload_hash = sha256(f);
  require(std::equal(payload_hash.begin(), payload_hash.end(), header+88), "概率 payload SHA256 不匹配");
  f.clear();
  f.seekg(sizeof header);
  std::vector<float> q(static_cast<std::size_t>(count));
  require(bool(f.read(reinterpret_cast<char*>(q.data()), static_cast<std::streamsize>(count*4))),
          "读取概率失败");
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "UAVSOFT1 requires IEEE754 float32");
  const std::uint16_t endian = 1;
  require(*reinterpret_cast<const unsigned char*>(&endian) == 1, "仅支持小端 host");
  for (std::size_t i = 0; i < q.size(); i += ds.num_classes) {
    double sum = 0;
    for (std::size_t j = 0; j < ds.num_classes; ++j) {
      require(std::isfinite(q[i+j]) && q[i+j] >= 0.0f && q[i+j] <= 1.0f, "概率非有限或越界");
      sum += q[i+j];
    }
    require(std::abs(sum-1.0) < 1e-5, "概率行和不为 1");
  }
  return q;
}
}  // namespace uavrgb
