#pragma once
#include "uavrgb/sha256.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace uavrgb_test {
inline void write_soft(const std::string& path, const std::string& dataset,
                       const std::vector<float>& q) {
  const auto side = std::filesystem::path(dataset).parent_path()/"instructions.jsonl";
  const bool has_text = std::filesystem::exists(side);
  const auto data_hash = uavrgb::sha256_file(dataset);
  const auto text_hash = has_text ? uavrgb::sha256_file(side.string()) : uavrgb::Sha256{};
  const std::string payload(reinterpret_cast<const char*>(q.data()), q.size()*sizeof(float));
  std::istringstream stream(payload);
  const auto payload_hash = uavrgb::sha256(stream);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write("UAVSOFT1", 8);
  const auto put = [&](std::uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) f.put(static_cast<char>(v >> (8*i)));
  };
  put(q.size()/4, 8); put(4, 4); put(has_text ? 1 : 0, 4);
  for (const auto& hash : {data_hash, text_hash, payload_hash}) {
    f.write(reinterpret_cast<const char*>(hash.data()), hash.size());
  }
  f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  if (!f) throw std::runtime_error("write_soft failed");
}
}  // namespace uavrgb_test
