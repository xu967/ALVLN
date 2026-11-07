// NativeTrainer 实现：分 batch → forward → softmax-CE → backward → 裁剪 → Adam → 指标。
#include "uavrgb/trainer.hpp"

#include "trainer_internal.hpp"
#include "uavrgb/class_weights.hpp"
#include "uavrgb/linear_cuda.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/sha256.hpp"
#include "uavrgb/softmax_cuda.hpp"
#include "uavrgb/soft_targets.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb {
namespace {

using Clock = std::chrono::steady_clock;

double since(const Clock::time_point& t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// 非有限值（nan/inf）不能写进 JSON，统一退化成 null；训练循环另有守卫会直接中止。
std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", v);
  return b;
}

std::string quote(const std::string& s) {
  std::string o = "\"";
  for (unsigned char ch : s) {
    if (ch == '"' || ch == '\\') {
      o += '\\';
      o += static_cast<char>(ch);
    } else if (ch == '\n') {
      o += "\\n";
    } else if (ch < 0x20) {
      char b[7];
      std::snprintf(b, sizeof b, "\\u%04x", static_cast<unsigned>(ch));
      o += b;
    } else {
      o += static_cast<char>(ch);
    }
  }
  return o + '"';
}

std::string hex_sha(const Sha256& h) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (unsigned char b : h) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}

std::string file_sha256_hex_impl(const std::string& path) { return hex_sha(sha256_file(path)); }

Sha256 sha_text(const std::string& s) {
  std::istringstream in(s);
  return sha256(in);
}

// 扁平 JSON 里取 "key": 后的标量（字符串去引号）；"null"/缺失返回空。
std::string json_scalar(const std::string& js, const std::string& key) {
  const std::string pat = "\"" + key + "\":";
  std::size_t p = js.find(pat);
  if (p == std::string::npos) return {};
  std::size_t b = p + pat.size();
  while (b < js.size() && (js[b] == ' ' || js[b] == '\t')) ++b;
  if (b < js.size() && js[b] == '"') {
    const std::size_t e = js.find('"', b + 1);
    return e == std::string::npos ? std::string{} : js.substr(b + 1, e - b - 1);
  }
  std::size_t e = b;
  while (e < js.size() && js[e] != ',' && js[e] != '}' && js[e] != '\n') ++e;
  std::string v = js.substr(b, e - b);
  return v == "null" ? std::string{} : v;
}

struct DatasetFacts {
  std::string split_scheme, window_from, hold_s;
};

// 从 .uavr 尾部 meta 取 config 需要的口径字段；读不到时留空（config 写 null）。
DatasetFacts read_dataset_facts(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  unsigned char h[48]{};
  f.read(reinterpret_cast<char*>(h), sizeof h);
  if (f.gcount() != static_cast<std::streamsize>(sizeof h)) return {};
  std::uint64_t off = 0;
  std::uint32_t len = 0;
  std::memcpy(&off, h + 36, 8);
  std::memcpy(&len, h + 44, 4);
  if (len > (64u << 20)) return {};  // meta 异常大：不猜，宁可不记录
  std::string meta(len, '\0');
  f.seekg(static_cast<std::streamoff>(off));
  f.read(meta.data(), static_cast<std::streamsize>(len));
  if (f.gcount() != static_cast<std::streamsize>(len)) return {};
  DatasetFacts d;
  d.split_scheme = json_scalar(meta, "split_scheme");
  if (d.split_scheme.empty()) d.split_scheme = json_scalar(meta, "split_policy");
  d.window_from = json_scalar(meta, "window_from");
  d.hold_s = json_scalar(meta, "hold_s");
  return d;
}

// join_soft_targets.py 把教师来源写在 <soft>.json；缺文件 = 旧软标签，来源不可审计。
struct SoftFacts {
  std::string teacher_ckpt_sha256, teacher_select_split;
};

SoftFacts read_soft_facts(const std::string& soft_path) {
  std::filesystem::path p(soft_path);
  p.replace_extension(".json");
  std::ifstream f(p);
  if (!f) return {};
  std::ostringstream os;
  os << f.rdbuf();
  const std::string js = os.str();
  return {json_scalar(js, "teacher_ckpt_sha256"), json_scalar(js, "teacher_select_split")};
}

// 运行目录附近的 git 提交（可选事实）。优先环境变量，便于离线/打包环境显式注入。
// worktree 的 .git 是一行 "gitdir: <main>/.git/worktrees/<name>" 的文件，HEAD 在那里，
// 而分支 ref 在 <gitdir>/commondir 指向的主 .git 下；两处都查，否则 worktree 里训出的
// config.json 会静默写成 source_git=""。packed-refs 不解析（返回空，与原行为一致）。
std::string detect_git_sha() {
  if (const char* e = std::getenv("UAVRGB_SOURCE_GIT")) return e;
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path p = fs::current_path(ec);
  for (int i = 0; i < 8 && !ec && !p.empty(); ++i, p = p.parent_path()) {
    fs::path gitdir = p / ".git";
    std::string line;
    if (fs::is_regular_file(gitdir, ec)) {  // worktree
      std::ifstream f(gitdir);
      if (!std::getline(f, line) || line.rfind("gitdir: ", 0) != 0) return {};
      gitdir = fs::path(line.substr(8));
      if (gitdir.is_relative()) gitdir = p / gitdir;
    }
    std::ifstream head(gitdir / "HEAD");
    if (!std::getline(head, line)) continue;
    if (line.rfind("ref: ", 0) != 0) return line;  // detached HEAD：直接是 sha
    const std::string ref = line.substr(5);
    fs::path common = gitdir;
    std::ifstream cf(gitdir / "commondir");  // 仅 worktree 有；内容相对 gitdir
    if (std::getline(cf, line) && !line.empty()) common = gitdir / line;
    for (const fs::path& base : {gitdir, common}) {
      std::ifstream rf(base / ref);
      if (std::getline(rf, line) && !line.empty()) return line;
    }
    return {};
  }
  return {};
}

CheckpointContract make_contract(const TrainConfig& cfg) {
  CheckpointContract c;
  c.dataset_sha256 = sha256_file(cfg.data_path);
  c.drop_collision = cfg.drop_collision;
  if (!cfg.soft_targets.empty()) c.soft_targets_sha256 = sha256_file(cfg.soft_targets);
  c.distill_alpha = cfg.distill_alpha;
  c.lr = cfg.lr;
  c.seed = cfg.seed;
  c.batch = cfg.batch;
  if (!cfg.split_scheme.empty()) c.split_scheme_sha256 = sha_text(cfg.split_scheme);
  return c;
}

bool is_zero(const Sha256& h) { return h == Sha256{}; }

// 数据集/超参/蒸馏来源不一致时默认拒绝续训；旧 checkpoint 无契约字段时只告警。
void check_resume_contract(const TrainConfig& cfg, const TrainState& st) {
  const CheckpointContract& c = st.contract;
  // 非 trainer 写出的 v5（如测试直接 save_checkpoint）字段全零：按未记录处理。
  if (!st.contract_present || (c.lr == 0.0f && c.batch == 0 && is_zero(c.dataset_sha256))) {
    std::fprintf(stderr,
                 "警告: resume checkpoint 无训练契约字段（旧格式），跳过数据/超参一致性校验\n");
    return;
  }
  std::vector<std::string> diff;
  if (!is_zero(c.dataset_sha256) && hex_sha(c.dataset_sha256) != cfg.dataset_sha256) {
    diff.push_back("dataset_sha256");
  }
  if (c.drop_collision != cfg.drop_collision) diff.push_back("drop_collision");
  const Sha256 soft = cfg.soft_targets.empty() ? Sha256{} : sha256_file(cfg.soft_targets);
  if (!(c.soft_targets_sha256 == soft)) diff.push_back("soft_targets_sha256");
  if (c.distill_alpha != cfg.distill_alpha) diff.push_back("distill_alpha");
  if (c.lr != cfg.lr) diff.push_back("lr");
  if (c.seed != cfg.seed) diff.push_back("seed");
  if (c.batch != cfg.batch) diff.push_back("batch");
  const Sha256 scheme = cfg.split_scheme.empty() ? Sha256{} : sha_text(cfg.split_scheme);
  if (!(c.split_scheme_sha256 == scheme)) diff.push_back("split_scheme");
  if (diff.empty()) return;
  std::string msg = "resume: checkpoint 训练契约与本次不一致:";
  for (const std::string& d : diff) msg += " " + d;
  if (!cfg.allow_resume_mismatch) {
    throw std::invalid_argument(msg + "（确需续训请显式加 --allow-resume-mismatch）");
  }
  std::fprintf(stderr, "警告: %s（已由 --allow-resume-mismatch 放行）\n", msg.c_str());
}

// resume 时把源 run 的 config.json 当作缺省 CLI；显式传入的仍以 CLI 为准。
void merge_resume_config(TrainConfig* cfg) {
  const std::filesystem::path rp(cfg->resume);
  const std::string dir = (rp.has_parent_path() ? rp.parent_path() : std::filesystem::path("."))
                              .string();
  RunConfigSnapshot snap;
  if (!read_run_config(dir, &snap)) {
    std::fprintf(stderr, "警告: resume 源目录 %s 无 config.json，未补全 CLI（旧 run）\n",
                 dir.c_str());
    return;
  }
  if (!cfg->lr_set && snap.has_lr) cfg->lr = snap.lr;
  if (!cfg->batch_set && snap.has_batch) cfg->batch = snap.batch;
  if (!cfg->seed_set && snap.has_seed) cfg->seed = snap.seed;
  if (!cfg->drop_collision_set && snap.has_drop_collision) cfg->drop_collision = snap.drop_collision;
  if (!cfg->soft_targets_set && snap.has_soft_targets) cfg->soft_targets = snap.soft_targets;
  if (!cfg->distill_alpha_set && snap.has_distill_alpha) cfg->distill_alpha = snap.distill_alpha;
  if (!cfg->text_dim_set && snap.has_text_dim) cfg->text_dim = snap.text_dim;
  if (!cfg->text_dim_set && snap.has_vision_only && snap.vision_only) cfg->vision_only = true;
  // 结构开关不继承就建不出同形模型，load_checkpoint 会按参数个数/形状拒绝；这里先接过来。
  if (!cfg->two_frame_set && snap.has_two_frame) cfg->two_frame = snap.two_frame;
  if (!cfg->prev_action_set && snap.has_prev_action) cfg->prev_action = snap.prev_action;
  if (!cfg->depth_aux_set && snap.has_depth_aux) cfg->depth_aux = snap.depth_aux;
  if (!cfg->aux_lambda_set && snap.has_aux_lambda) cfg->aux_lambda = snap.aux_lambda;
  if (!(std::isfinite(cfg->lr) && cfg->lr > 0.0f && cfg->lr <= 1.0f)) {
    throw std::invalid_argument("resume: 源 config 的 lr 非法（" + num(cfg->lr) + "）");
  }
  if (cfg->batch <= 0) throw std::invalid_argument("resume: 源 config 的 batch 非法");
}

// resume 的历史 best 必须能在新目录独立交付：同目录核对文件存在且状态一致；
// 跨目录先核对再复制，绝不能让新 run 继承一个分数却没有对应权重文件。
void carry_resume_best(const std::string& resume_path, const std::string& out_dir,
                       const CnnModel& like, std::uint32_t seed, float lr,
                       const TrainState& state) {
  namespace fs = std::filesystem;
  if (!(state.best_macro_f1 >= 0.0f)) return;  // 旧格式/从未评估过，无 best 可带
  const fs::path rp(resume_path);
  const fs::path src_dir = rp.has_parent_path() ? rp.parent_path() : fs::path(".");
  const fs::path src_best = src_dir / "ckpt_best.bin";
  const fs::path dst_best = fs::path(out_dir) / "ckpt_best.bin";
  if (!fs::exists(src_best)) {
    throw std::runtime_error("resume: checkpoint 记录 best_macro_f1=" +
                             num(state.best_macro_f1) + "，但 " + src_best.string() +
                             " 不存在；无法交付历史 best");
  }
  // 核对源 best 确实属于这次 resume：分数、类别权重都要与 checkpoint 记录一致。
  CnnModel m = make_cnn(like.in_h, like.in_w, seed, like.fc.out, like.text_dim, like.c1.in,
                        like.aux_classes, like.prev_dim);
  TrainState s = make_train_state(m, lr);
  load_checkpoint(src_best.string(), m, s);
  if (s.best_macro_f1 != state.best_macro_f1 || s.class_weights != state.class_weights) {
    throw std::runtime_error("resume: " + src_best.string() +
                             " 与 checkpoint 记录的历史 best 不一致（分数/类别权重对不上）");
  }
  std::error_code ec1;
  std::error_code ec2;
  const auto rc = fs::weakly_canonical(src_dir, ec1);
  const auto oc = fs::weakly_canonical(out_dir, ec2);
  if (!ec1 && !ec2 && rc == oc) return;  // 同目录：文件已经在目标位置
  if (fs::exists(dst_best)) {
    throw std::runtime_error("resume: 目标目录已有 " + dst_best.string() +
                             "；跨目录续训不会覆盖它");
  }
  fs::copy_file(src_best, dst_best);
}

std::uint64_t count_correct(const Tensor& logits, const std::vector<int>& labels) {
  const int n = static_cast<int>(labels.size());
  const int c = static_cast<int>(logits.dims()[1]);
  std::uint64_t ok = 0;
  for (int i = 0; i < n; ++i) {
    int best = 0;
    for (int j = 1; j < c; ++j) {
      if (logits[static_cast<std::size_t>(i) * c + j] >
          logits[static_cast<std::size_t>(i) * c + best]) {
        best = j;
      }
    }
    ok += (best == labels[static_cast<std::size_t>(i)]) ? 1 : 0;
  }
  return ok;
}

}  // namespace

std::string file_sha256_hex(const std::string& path) { return file_sha256_hex_impl(path); }

void append_test_eval_record(const std::string& out_dir, const std::string& origin,
                             const std::string& ckpt_path, const std::string& data_path,
                             const std::string& dataset_sha256, bool drop_collision,
                             std::size_t samples) {
  std::size_t count = 0;
  {
    std::ifstream in(out_dir + "/test_eval.jsonl");
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty()) ++count;
    }
  }
  const auto now = std::chrono::system_clock::now();
  const std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&tt, &tm);
  char utc[32] = {};
  std::strftime(utc, sizeof utc, "%Y-%m-%dT%H:%M:%SZ", &tm);
  const std::string run_id =
      std::string(utc) + "-" + std::to_string(now.time_since_epoch().count() % 1000000);
  std::ofstream f(out_dir + "/test_eval.jsonl", std::ios::app);
  if (!f) throw std::runtime_error("trainer: 无法追加写 test_eval.jsonl");
  f << "{\"schema\":\"uavrgb.test_eval.v1\",\"run_id\":" << quote(run_id)
    << ",\"utc\":" << quote(utc) << ",\"origin\":" << quote(origin)
    << ",\"ckpt\":" << quote(ckpt_path) << ",\"data\":" << quote(data_path)
    << ",\"dataset_sha256\":" << quote(dataset_sha256)
    << ",\"drop_collision\":" << (drop_collision ? "true" : "false")
    << ",\"samples\":" << samples << ",\"split\":\"test\",\"eval_count\":" << (count + 1)
    << "}\n";
  if (!f) throw std::runtime_error("trainer: 写 test_eval.jsonl 失败");
}

void Trainer::write_csv_header_if_needed() const {
  const std::string p = cfg_.out_dir + "/results.csv";
  std::error_code ec;
  const bool empty = !std::filesystem::exists(p, ec) || std::filesystem::file_size(p, ec) == 0;
  if (!empty) return;  // 续训直接追加，不重复写表头
  std::ofstream f(p, std::ios::app);
  if (!f) throw std::runtime_error("trainer: 无法写 " + p);
  f << kResultsCsvHeader << '\n';
  f.flush();
  if (!f) throw std::runtime_error("trainer: 写 results.csv 表头失败");
}

void Trainer::run() {
  if (!cfg_.resume.empty()) merge_resume_config(&cfg_);
  if (cfg_.epochs <= 0 || cfg_.batch <= 0) {
    throw std::invalid_argument("trainer: epochs/batch 必须为正");
  }
  UavrDataset ds = load_uavr(cfg_.data_path, LoadOptions{cfg_.drop_collision});
  if (ds.train_idx.empty()) throw std::runtime_error("trainer: 数据集没有 train split");
  std::filesystem::create_directories(cfg_.out_dir);

  // 事实字段：写进 config.json，也是 resume 契约的比较基准。
  cfg_.dataset_sha256 = file_sha256_hex_impl(cfg_.data_path);
  const DatasetFacts facts = read_dataset_facts(cfg_.data_path);
  cfg_.split_scheme = facts.split_scheme;
  cfg_.window_from = facts.window_from;
  cfg_.hold_s = facts.hold_s;
  cfg_.source_git = detect_git_sha();
  if (!cfg_.soft_targets.empty()) {
    cfg_.soft_targets_sha256 = file_sha256_hex_impl(cfg_.soft_targets);
    const SoftFacts sf = read_soft_facts(cfg_.soft_targets);
    cfg_.teacher_ckpt_sha256 = sf.teacher_ckpt_sha256;
    cfg_.teacher_select_split = sf.teacher_select_split;
    if (sf.teacher_ckpt_sha256.empty()) {
      std::fprintf(stderr, "警告: 软标签缺教师来源元信息（%s.json），蒸馏来源不可审计\n",
                   cfg_.soft_targets.c_str());
    }
    if (sf.teacher_select_split == "val") {
      const std::string msg =
          "教师用 val split 选模，与 student val 重叠，蒸馏指标可能偏乐观";
      if (cfg_.strict_distill) throw std::runtime_error(msg + "（--strict-distill）");
      std::fprintf(stderr, "警告: %s\n", msg.c_str());
    }
  }

  class_weights_ = resolve_class_weights(cfg_.class_weights, ds);
  class_weights_source_ = class_weights_source(cfg_.class_weights);
  soft_targets_ = load_soft_targets(cfg_, ds);

  if (cfg_.text_dim > 0 && ds.text_ids.empty()) {
    throw std::runtime_error(
        "trainer: --text-dim>0 但数据集没有 instructions.jsonl 侧车（用新版 preprocess.py 重打包）");
  }
  if (cfg_.text_dim == 0 && !ds.text_ids.empty() && !cfg_.vision_only) {
    throw std::runtime_error(
        "trainer: 数据集有 instructions.jsonl 但 --text-dim 缺省为 0，会静默训成纯视觉模型；"
        "要文本分支请传 --text-dim N，要纯视觉基线请显式传 --vision-only");
  }
  if ((cfg_.two_frame || cfg_.prev_action) && !ds.aux.present) {
    throw std::runtime_error(
        "trainer: --two-frame/--prev-action 需要 dataset.aux 侧车（用新版 preprocess.py 重打包）");
  }
  if (cfg_.depth_aux && !(ds.aux.present && ds.aux.has_depth)) {
    throw std::runtime_error(
        "trainer: --depth-aux 需要带深度分带目标的 dataset.aux（采集时 --save-depth 并重打包）；"
        "不要静默关掉辅助损失，去掉该参数即可只训动作头");
  }
  if (!(cfg_.aux_lambda >= 0.0f) || !std::isfinite(cfg_.aux_lambda)) {
    throw std::invalid_argument("trainer: --aux-lambda 必须为非负有限数");
  }
  model_ = build_model(ds);
  state_ = make_train_state(model_, cfg_.lr);
  if (!cfg_.resume.empty()) {
    load_checkpoint(cfg_.resume, model_, state_);
    check_resume_contract(cfg_, state_);
    class_weights_source_ = merge_resume_weights(cfg_.class_weights, class_weights_,
                                                 state_.class_weights);
  }
  state_.class_weights = class_weights_;
  // 本次运行的事实写入之后的所有 checkpoint（含 ckpt_best.bin）。
  state_.contract = make_contract(cfg_);
  state_.contract_present = true;
  cfg_.adam_beta1 = state_.adam[0].beta1;
  cfg_.adam_beta2 = state_.adam[0].beta2;
  cfg_.adam_eps = state_.adam[0].eps;
  if (!cfg_.resume.empty()) {
    carry_resume_best(cfg_.resume, cfg_.out_dir, model_, cfg_.seed, cfg_.lr, state_);
  }
  write_run_config(cfg_.out_dir, cfg_, class_weights_, class_weights_source_);
  dev_ = std::make_unique<DeviceCnn>(model_, cfg_.batch, &state_);

  std::vector<std::uint32_t> dev = ds.val_idx;
  const bool dev_is_train = dev.empty();
  if (dev_is_train) dev = ds.train_idx;

  const std::string metrics_path = cfg_.out_dir + "/metrics.jsonl";
  const std::string csv_path = cfg_.out_dir + "/results.csv";
  const std::string ckpt_path = cfg_.out_dir + "/ckpt.bin";
  const std::string best_ckpt_path = cfg_.out_dir + "/ckpt_best.bin";
  write_csv_header_if_needed();
  std::ofstream metrics(metrics_path, std::ios::app);
  std::ofstream csv(csv_path, std::ios::app);
  if (!metrics || !csv) throw std::runtime_error("trainer: 无法追加写 " + cfg_.out_dir);
  // 打开成功不代表后续写成功（磁盘满/quota/FUSE 错误）：每次 flush 后必须查流状态，
  // 否则训练照常 exit 0，结果文件却是空的。
  auto check_out = [](const std::ofstream& f, const char* name) {
    if (!f) throw std::runtime_error(std::string("trainer: 写 ") + name + " 失败（磁盘满/IO 错误）");
  };

  const std::size_t bs = static_cast<std::size_t>(cfg_.batch);
  const std::uint32_t per_epoch =
      static_cast<std::uint32_t>((ds.train_idx.size() + bs - 1) / bs);
  std::uint32_t step = static_cast<std::uint32_t>(train_step(state_));
  const std::uint32_t total_steps = step + per_epoch * static_cast<std::uint32_t>(cfg_.epochs);

  metrics << metrics_json(make_status(
                 "resume from step " + std::to_string(step) + ", " + std::to_string(ds.n) +
                 " samples (" + std::to_string(ds.train_idx.size()) + " train / " +
                 std::to_string(dev.size()) + (dev_is_train ? " dev=fallback-train)" : " dev)") +
                 ", params=" + std::to_string(num_params(model_))))
          << '\n';
  if (dev_is_train) {
    metrics << metrics_json(make_error(Severity::kWarn, "DEV_SPLIT_EMPTY", step,
                                       "数据集没有 val split，dev 指标回退到 train 集"))
            << '\n';
  }
  metrics.flush();
  check_out(metrics, "metrics.jsonl");

  std::mt19937 rng(cfg_.seed);
  std::vector<std::uint32_t> order = ds.train_idx;
  const Clock::time_point t0 = Clock::now();
  const int epoch_base = state_.epoch;  // 续训时 results.csv 接着编号
  const std::uint32_t start_step = step;  // eta 只按本次运行的步数算
  samples_seen_ = 0;

  for (int ep = 1; ep <= cfg_.epochs; ++ep) {
    std::shuffle(order.begin(), order.end(), rng);
    double ce_sum = 0.0, aux_ce_sum = 0.0;
    std::uint64_t seen = 0, aux_valid = 0, aux_correct = 0;
    for (std::size_t s = 0; s < order.size(); s += bs) {
      const std::size_t e = std::min(order.size(), s + bs);
      const std::vector<std::uint32_t> part = detail::slice(order, s, e);
      std::vector<int> y = detail::take_labels(ds, order, s, e);  // 增强翻转时会就地互换左右

      // 蒸馏批次：按全局样本下标 gather 教师分布，与 order/part 严格对齐。
      std::vector<float> q;
      if (!soft_targets_.empty()) {
        q.resize(part.size() * ds.num_classes);
        for (std::size_t i = 0; i < part.size(); ++i) {
          const float* src = soft_targets_.data() + static_cast<std::size_t>(part[i]) * ds.num_classes;
          std::copy(src, src + ds.num_classes, q.begin() + i * ds.num_classes);
        }
      }

      const Tensor logits = forward_batch(ds, part, &y);
      float loss = 0.0f;
      const Tensor dlogits = loss_and_grad(logits, y, q, &loss);
      if (!std::isfinite(loss)) {
        metrics << metrics_json(make_error(Severity::kFatal, "TRAIN_NONFINITE_METRIC", step + 1,
                                           "训练 loss 非有限值，中止"))
                << '\n';
        metrics.flush();
        throw std::runtime_error("trainer: loss 非有限值（step " + std::to_string(step + 1) + "）");
      }
      const float acc = static_cast<float>(count_correct(logits, y)) /
                        static_cast<float>(part.size());
      const float lr_now = step_lr(step + 1, total_steps);
      if (cfg_.depth_aux) {
        // 辅助损失的梯度留在显存，backward_step 把它叠到第三层特征上；loss 列仍只记动作 CE，
        // results.csv/metrics 的口径不变，深度项另记。
        const DeviceCnn::AuxStats a = aux_batch(ds, cfg_.aux_lambda);
        if (!std::isfinite(a.loss)) throw std::runtime_error("trainer: 深度辅助 loss 非有限值");
        aux_ce_sum += static_cast<double>(a.loss) * static_cast<double>(a.valid);
        aux_valid += a.valid;
        aux_correct += a.correct;
      }
      dev_->backward_step(dlogits, cfg_.max_grad_norm, state_, cfg_.weight_decay);

      ++step;
      const double elapsed = since(t0);
      const double done = static_cast<double>(step - start_step);
      const double eta = elapsed / done * static_cast<double>(total_steps - step);
      metrics << metrics_json(make_train_step(step, total_steps, loss, acc, lr_now, elapsed, eta))
              << '\n';
      metrics.flush();  // 监控台按追加偏移增量读，每步必须落盘
      check_out(metrics, "metrics.jsonl");

      // results.csv 的 train_ce 必须与 dev_ce 同口径（都是无权、无平滑的 CE），否则两列不可比。
      const float plain =
          (class_weights_.empty() && soft_targets_.empty() && cfg_.label_smoothing <= 0.0f)
              ? loss
              : softmax_ce_loss_cuda(logits, y);
      ce_sum += static_cast<double>(plain) * static_cast<double>(part.size());
      seen += part.size();
    }

    const double train_ce = ce_sum / static_cast<double>(seen);
    EvalStats dev_stats;
    evaluate(ds, dev, &dev_stats);
    // dev 非有限值必须中止：否则 CSV 写 "null"、宏 F1 仍有限，失效模型会被选成 best。
    if (!std::isfinite(dev_stats.ce) || !std::isfinite(dev_stats.acc) ||
        !std::isfinite(dev_stats.macro_f1) || !std::isfinite(dev_stats.balanced_acc)) {
      metrics << metrics_json(make_error(Severity::kFatal, "DEV_NONFINITE_METRIC", step + 1,
                                         "dev 指标非有限值（lr 过大或数值溢出），中止"))
              << '\n';
      metrics.flush();
      throw std::runtime_error("trainer: dev 指标非有限值（step " +
                               std::to_string(step + 1) + "）");
    }
    state_.epoch = epoch_base + ep;
    csv << state_.epoch << ',' << num(train_ce) << ',' << num(dev_stats.ce) << ','
        << num(dev_stats.acc) << '\n';
    csv.flush();
    check_out(csv, "results.csv");
    MetricsEvent vev = make_validation(step, static_cast<float>(dev_stats.ce),
                                       static_cast<float>(dev_stats.acc),
                                       dev_is_train ? Split::kTrain : Split::kVal);
    vev.macro_f1 = static_cast<float>(dev_stats.macro_f1);
    vev.balanced_accuracy = static_cast<float>(dev_stats.balanced_acc);
    vev.confusion.assign(dev_stats.confusion.begin(), dev_stats.confusion.end());
    metrics << metrics_json(vev) << '\n';
    // 类别不平衡下要看宏指标：原始 accuracy 被 75% 的 FORWARD 撑着。
    std::printf("epoch %d %s ce=%.4f acc=%.4f macro_f1=%.4f bal_acc=%.4f recall=[", ep,
                dev_is_train ? "train" : "dev  ", dev_stats.ce, dev_stats.acc,
                dev_stats.macro_f1, dev_stats.balanced_acc);
    for (std::size_t k = 0; k < dev_stats.recall.size(); ++k) {
      std::printf("%s%.3f", k ? " " : "", dev_stats.recall[k]);
    }
    std::printf("]");
    if (cfg_.depth_aux) {
      const double train_depth_ce = aux_valid ? aux_ce_sum / static_cast<double>(aux_valid) : 0.0;
      const double train_depth_acc =
          aux_valid ? static_cast<double>(aux_correct) / static_cast<double>(aux_valid) : 0.0;
      std::printf(" depth_ce=%.4f/%.4f depth_acc=%.3f/%.3f(train/dev)", train_depth_ce,
                  dev_stats.depth_ce, train_depth_acc, dev_stats.depth_acc);
      metrics << metrics_json(make_status(
                     "depth_aux epoch " + std::to_string(epoch_base + ep) + ": train_ce=" +
                     num(train_depth_ce) + " train_acc=" + num(train_depth_acc) +
                     " dev_ce=" + num(dev_stats.depth_ce) + " dev_acc=" + num(dev_stats.depth_acc) +
                     " lambda=" + num(cfg_.aux_lambda)))
              << '\n';
    }
    std::printf("\n");
    if (!class_weights_.empty()) {
      metrics << metrics_json(make_class_weights(static_cast<std::uint32_t>(state_.epoch), step,
                                                 class_weights_, class_weights_source_))
              << '\n';
    }
    metrics.flush();
    check_out(metrics, "metrics.jsonl");
    dev_->download(model_, &state_);  // 本轮两份 checkpoint 都写这份权重
    // 交付模型不能是最后一轮：dev 宏 F1 峰值常早于末轮（见 docs/results.md）。
    // dev 回退到 train 时不选模型，否则等于用训练集挑权重。
    if (!dev_is_train && dev_stats.macro_f1 > static_cast<double>(state_.best_macro_f1)) {
      state_.best_macro_f1 = static_cast<float>(dev_stats.macro_f1);
      save_checkpoint(best_ckpt_path, model_, state_);
      metrics << metrics_json(make_status("best checkpoint: epoch " +
                                          std::to_string(state_.epoch) + " macro_f1=" +
                                          num(dev_stats.macro_f1)))
              << '\n';
      metrics.flush();
      check_out(metrics, "metrics.jsonl");
    }
    // 必须在更新 best 之后写：否则续训从 ckpt.bin 读到的 best 比 ckpt_best.bin 落后一轮。
    save_checkpoint(ckpt_path, model_, state_);
    samples_seen_ += seen;
  }

  // test split 全程不参与训练和调参：只在最后评估一次，给出一个没被选择污染的数字。
  // 但必须用交付的那份权重：dev 宏 F1 峰值常早于末轮，评 model_ 会报出与
  // ckpt_best.bin 不符的数字（2026-11-03 审计）。
  if (!ds.test_idx.empty() && !dev_is_train && std::filesystem::exists(best_ckpt_path)) {
    TrainState best_state = state_;  // 只借用模型的权重布局校验，不回写训练状态
    load_checkpoint(best_ckpt_path, model_, best_state);
    dev_->upload(model_);
  }
  if (!ds.test_idx.empty()) {
    EvalStats test_stats;
    evaluate(ds, ds.test_idx, &test_stats);
    test_acc_ = test_stats.acc;
    MetricsEvent tev = make_validation(step, static_cast<float>(test_stats.ce),
                                       static_cast<float>(test_stats.acc), Split::kTest);
    tev.macro_f1 = static_cast<float>(test_stats.macro_f1);
    tev.balanced_accuracy = static_cast<float>(test_stats.balanced_acc);
    tev.confusion.assign(test_stats.confusion.begin(), test_stats.confusion.end());
    metrics << metrics_json(tev) << '\n';
    // 训练末尾的 test 评估照旧，但单独留痕（不改历史 metrics），便于审计「test 被看过几次」。
    append_test_eval_record(cfg_.out_dir, "train_inline",
                            std::filesystem::exists(best_ckpt_path) ? best_ckpt_path : ckpt_path,
                            cfg_.data_path, cfg_.dataset_sha256, cfg_.drop_collision,
                            ds.test_idx.size());
  }

  elapsed_sec_ = since(t0);
  metrics << metrics_json(make_status("done: " + std::to_string(samples_seen_) + " samples in " +
                                      num(elapsed_sec_) + " s, ckpt=" + ckpt_path))
          << '\n';
  metrics.flush();
  check_out(metrics, "metrics.jsonl");
}

}  // namespace uavrgb
