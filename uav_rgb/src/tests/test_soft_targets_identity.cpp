// Python 拼接器可调用此入口验证真正的 C++ 读端，无需启动 CUDA/训练。
#include "uavrgb/sha256.hpp"
#include "uavrgb/soft_targets.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

void vector_test(const std::string& data, const std::string& expected) {
  std::istringstream stream(data);
  const auto hash = uavrgb::sha256(stream);
  const char* digits = "0123456789abcdef";
  std::string got;
  for (auto byte : hash) { got += digits[byte >> 4]; got += digits[byte & 15]; }
  if (got != expected) throw std::runtime_error("SHA256 known vector mismatch: " + got);
}

int main(int argc, char** argv) {
  try {
    if (argc == 3) {
      uavrgb::TrainConfig cfg;
      cfg.data_path = argv[1];
      cfg.soft_targets = argv[2];
      auto ds = uavrgb::load_uavr(cfg.data_path);
      if (uavrgb::load_soft_targets(cfg, ds).empty()) throw std::runtime_error("empty payload");
    } else if (argc == 1) {
      vector_test("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
      vector_test("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
      vector_test("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
      vector_test(std::string(1000000, 'a'),
                  "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
      vector_test(std::string(65536, 'a'),
                  "bf718b6f653bebc184e1479f1935b8da974d701b893afcf49e701f3e2f9f9c5a");
      vector_test(std::string(65537, 'a'),
                  "008ffc88d3c96a9f307524eb361e47c5222a887fc45fa0c1fb8d429c5c23b430");
    } else {
      throw std::runtime_error("usage: test_soft_targets_identity [dataset.uavr soft.bin]");
    }
    std::cout << "PASS: soft targets identity\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
