// checkpoint 读写实现；格式见 include/uavrgb/checkpoint.hpp 的文件头注释。
#include "uavrgb/checkpoint.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace uavrgb {
namespace {

constexpr char kMagic[8] = {'U', 'A', 'V', 'R', 'C', 'K', '0', '1'};
constexpr std::uint32_t kVersion = 3;
constexpr std::uint32_t kMinVersion = 2;  // v2 没有 best_macro_f1，读回 -1

[[noreturn]] void bad(const std::string& msg) {
  throw std::invalid_argument("checkpoint: " + msg);
}

template <typename T>
void put(std::ofstream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

// 读定长标量；文件提前结束即报错。
template <typename T>
void get(std::ifstream& f, T& v, const char* what) {
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (f.gcount() != static_cast<std::streamsize>(sizeof(T))) bad(std::string("读取不足: ") + what);
}

void put_tensor(std::ofstream& f, const Tensor& t) {
  put<std::uint32_t>(f, static_cast<std::uint32_t>(t.ndim()));
  for (std::int64_t d : t.dims()) put<std::int64_t>(f, d);
  f.write(reinterpret_cast<const char*>(t.data()),
          static_cast<std::streamsize>(t.numel() * sizeof(float)));
}

// 读形状；与已有张量形状不一致时按 expect 报具体冲突（不覆盖、不重分配）。
void get_dims(std::ifstream& f, const std::string& name, std::vector<std::int64_t>& out_dims) {
  std::uint32_t nd = 0;
  get<std::uint32_t>(f, nd, "ndim");
  if (nd > 8) bad(name + ": ndim 异常 " + std::to_string(nd));
  out_dims.assign(nd, 0);
  for (std::uint32_t i = 0; i < nd; ++i) get<std::int64_t>(f, out_dims[i], "dims");
}

void get_floats(std::ifstream& f, Tensor& t, const std::string& name) {
  f.read(reinterpret_cast<char*>(t.data()),
         static_cast<std::streamsize>(t.numel() * sizeof(float)));
  if (f.gcount() != static_cast<std::streamsize>(t.numel() * sizeof(float))) {
    bad(name + ": 数据不足");
  }
}

void check_shape(const std::string& name, const Tensor& t, const std::vector<std::int64_t>& dims) {
  if (t.dims() != dims) bad(name + ": 形状与模型不符");
}

}  // namespace

TrainState make_train_state(const CnnModel& model, float lr) {
  TrainState st;
  for (const Tensor* p : cnn_params(model)) st.adam.push_back(adam_make(*p, lr));
  return st;
}

void save_checkpoint(const std::string& path, const CnnModel& model, const TrainState& st) {
  const std::vector<const Tensor*> ps = cnn_params(model);
  if (st.adam.size() != ps.size()) bad("Adam 状态数与参数数不一致");

  // 先写临时文件再 rename：中途失败不会留下半截 checkpoint 覆盖掉旧的。
  const std::string tmp = path + ".tmp";
  std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("checkpoint: 无法写 " + tmp);
  f.write(kMagic, sizeof kMagic);
  put<std::uint32_t>(f, kVersion);
  put<std::uint32_t>(f, static_cast<std::uint32_t>(ps.size()));
  put<std::int32_t>(f, model.in_h);
  put<std::int32_t>(f, model.in_w);
  put<std::int32_t>(f, model.c1.in);
  put<std::int32_t>(f, model.fc.out);
  put<std::uint32_t>(f, static_cast<std::uint32_t>(st.adam[0].t));
  put<std::uint32_t>(f, static_cast<std::uint32_t>(st.epoch));
  if (!st.class_weights.empty() &&
      st.class_weights.size() != static_cast<std::size_t>(model.fc.out)) {
    bad("class_weights 长度与 num_classes 不符");
  }
  put<std::uint32_t>(f, static_cast<std::uint32_t>(st.class_weights.size()));
  for (float w : st.class_weights) put<float>(f, w);
  put<float>(f, st.best_macro_f1);

  for (const Tensor* p : ps) put_tensor(f, *p);
  for (const Adam& a : st.adam) {
    f.write(reinterpret_cast<const char*>(a.m.data()),
            static_cast<std::streamsize>(a.m.numel() * sizeof(float)));
  }
  for (const Adam& a : st.adam) {
    f.write(reinterpret_cast<const char*>(a.v.data()),
            static_cast<std::streamsize>(a.v.numel() * sizeof(float)));
  }
  f.flush();
  if (!f) throw std::runtime_error("checkpoint: 写入失败 " + tmp);
  f.close();
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) throw std::runtime_error("checkpoint: rename 失败 " + path);
}

void load_checkpoint(const std::string& path, CnnModel& model, TrainState& st) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("checkpoint: 无法打开 " + path);
  char magic[8] = {};
  f.read(magic, sizeof magic);
  if (f.gcount() != static_cast<std::streamsize>(sizeof magic) ||
      std::memcmp(magic, kMagic, sizeof kMagic) != 0) {
    bad("magic 不符（不是 UAVRCK01）");
  }
  std::uint32_t version = 0, count = 0;
  get<std::uint32_t>(f, version, "version");
  if (version < kMinVersion || version > kVersion) {
    bad("version 不支持: " + std::to_string(version));
  }
  get<std::uint32_t>(f, count, "param_count");

  std::int32_t in_h = 0, in_w = 0, in_c = 0, ncls = 0;
  get<std::int32_t>(f, in_h, "in_h");
  get<std::int32_t>(f, in_w, "in_w");
  get<std::int32_t>(f, in_c, "in_c");
  get<std::int32_t>(f, ncls, "num_classes");
  if (in_h != model.in_h || in_w != model.in_w || in_c != model.c1.in ||
      ncls != model.fc.out) {
    bad("模型结构不符（ckpt " + std::to_string(in_c) + "x" + std::to_string(in_h) + "x" +
        std::to_string(in_w) + "->" + std::to_string(ncls) + "）");
  }

  const std::vector<Tensor*> ps = cnn_params(model);
  if (count != ps.size() || st.adam.size() != ps.size()) bad("参数个数不符");

  std::uint32_t t = 0;
  std::uint32_t epoch = 0;
  get<std::uint32_t>(f, t, "adam_t");
  get<std::uint32_t>(f, epoch, "epoch");
  st.epoch = static_cast<int>(epoch);

  std::uint32_t nw = 0;
  get<std::uint32_t>(f, nw, "class_weight_count");
  if (nw != 0 && nw != static_cast<std::uint32_t>(model.fc.out)) bad("class_weights 数量不符");
  st.class_weights.resize(nw);
  for (float& w : st.class_weights) {
    get<float>(f, w, "class_weight");
    if (!(w >= 0.0f) || !std::isfinite(w)) bad("class_weights 含非有限/负值");
  }
  st.best_macro_f1 = -1.0f;
  if (version >= 3) {
    get<float>(f, st.best_macro_f1, "best_macro_f1");
    if (!std::isfinite(st.best_macro_f1) || st.best_macro_f1 < -1.0f) {
      bad("best_macro_f1 非法");
    }
  }

  std::vector<std::int64_t> dims;
  for (std::size_t i = 0; i < ps.size(); ++i) {
    get_dims(f, "param", dims);
    check_shape("param[" + std::to_string(i) + "]", *ps[i], dims);
    get_floats(f, *ps[i], "param");
  }
  // m/v 落盘时不带形状头（顺序与参数一致），但本地张量必须与参数同形，否则 float 流错位。
  for (std::size_t i = 0; i < st.adam.size(); ++i) {
    check_shape("adam.m", st.adam[i].m, ps[i]->dims());
    get_floats(f, st.adam[i].m, "adam.m");
  }
  for (std::size_t i = 0; i < st.adam.size(); ++i) {
    check_shape("adam.v", st.adam[i].v, ps[i]->dims());
    get_floats(f, st.adam[i].v, "adam.v");
  }
  for (Adam& a : st.adam) a.t = static_cast<int>(t);
  if (f.peek() != std::char_traits<char>::eof()) bad("文件尾部有多余数据");
}

}  // namespace uavrgb
