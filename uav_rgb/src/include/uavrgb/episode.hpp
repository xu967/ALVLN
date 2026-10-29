#pragma once
// EpisodeSchema：一条 episode 的字段。train/val/test 严格按场景/地图隔离，
// 禁止在同一地图随机拆帧造成泄漏（UBUNTU_BASE_GUIDE_V1 §1.2）。
#include <cstdint>
#include <string>

namespace uavrgb {

struct EpisodeStep {
  std::int64_t timestamp_ns = 0;
  std::uint8_t action_expert = 0;    // 教师量化动作（Action 枚举值 0..3）
  std::uint8_t action_executed = 0;  // 实际执行动作
  bool collision = false;
  // 图像帧单独存储（见 DatasetWriter），此处只记时间戳关联。
};

struct EpisodeMeta {
  std::int64_t episode_id = 0;
  std::int64_t scene_id = 0;  // 划分隔离的关键字段
  std::int64_t seed = 0;
  std::string version;
};

}  // namespace uavrgb
