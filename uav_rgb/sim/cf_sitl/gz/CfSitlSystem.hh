// CfSitlSystem.hh：gz-sim8 System 插件，把 E1 MINI 固件在环库（cf_sitl）接到 Gazebo 刚体上。
//
// 职责（每个 1 ms 物理步，在 PreUpdate 里）：
//   读 link 世界位姿/速度/角速度/加速度 -> cf_sitl_step_external -> 机体系 wrench 转世界系
//   -> Link::AddWorldWrench。Gazebo 负责积分、重力和接触；SITL 负责固件、电机、传感器噪声。
// 话题（默认名，SDF 可改）：
//   /e1_mini/action  gz.msgs.Double_V [action, until_s_sim]   订阅，回调只入队
//   /e1_mini/cmd     gz.msgs.StringMsg "reset <episode>" | "arm_takeoff <height>"   订阅
//   /e1_mini/state   gz.msgs.Double_V 200 Hz，字段索引表见 README.md             发布
// 时间：until_s_sim 是 Gazebo 仿真秒；插件在 reset 时记 gz 时间基准 base_s，
//   SITL 时间 = gz 时间 - base_s（头文件约定 SITL 时钟每次 reset 归零）。
#ifndef UAVRGB_CF_SITL_GZ_SYSTEM_HH
#define UAVRGB_CF_SITL_GZ_SYSTEM_HH

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/sim/Entity.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

#include "CfSitlState.hh"
#include "cf_sitl.h"

namespace uavrgb {

class CfSitlSystem : public gz::sim::System,
                     public gz::sim::ISystemConfigure,
                     public gz::sim::ISystemPreUpdate {
 public:
  void Configure(const gz::sim::Entity& entity, const std::shared_ptr<const sdf::Element>& sdf,
                 gz::sim::EntityComponentManager& ecm, gz::sim::EventManager& events) override;
  void PreUpdate(const gz::sim::UpdateInfo& info, gz::sim::EntityComponentManager& ecm) override;

 private:
  struct PendingAction {
    int action;
    double until_sim_s;
  };
  struct PendingCmd {
    std::string verb;
    double value;
  };

  void OnAction(const gz::msgs::Double_V& msg);
  void OnCmd(const gz::msgs::StringMsg& msg);
  // reset / arm_takeoff；reset 以当前 link 位姿作为 standalone 字段的出生点并重记时间基准。
  void DrainCmds(double sim_s, const gz::math::Pose3d& pose);
  void DrainActions(double t_sitl);
  bool ReadBody(const gz::sim::EntityComponentManager& ecm, double t_sitl,
                cf_sitl_body_in* in, gz::math::Pose3d* pose);
  void PublishState(double sim_s);
  void PlaceChaseCam(gz::sim::EntityComponentManager& ecm, const gz::math::Pose3d& body);
  static std::string ResolveParams(const std::string& path);

  gz::sim::Entity model_entity_{gz::sim::kNullEntity};
  gz::sim::Link link_;
  std::string link_name_{"base_link"};
  std::string params_path_;
  uint64_t seed_{0};
  bool init_ok_{false};
  bool dt_error_reported_{false};
  bool based_{false};        // 首次 PreUpdate 做一次隐式 reset（上电）
  double base_s_{0.0};
  uint64_t steps_{0};
  uint64_t late_actions_{0};
  uint64_t step_errors_{0};
  int state_every_{5};       // 1 kHz / 5 = 200 Hz

  // chase_cam：独立静态模型，只跟位置和偏航（不跟俯仰横滚），每 chase_every_ 步放一次。
  std::string chase_model_name_{"chase_cam"};
  gz::sim::Entity chase_entity_{gz::sim::kNullEntity};
  bool chase_looked_up_{false};
  gz::math::Vector3d chase_offset_{-1.0, 0.0, 0.5};
  double chase_pitch_{0.45};
  int chase_every_{20};      // 50 Hz；chase 相机本身只渲染 15 Hz

  gz::transport::Node node_;
  gz::transport::Node::Publisher state_pub_;
  std::mutex mu_;
  std::deque<PendingAction> actions_;
  std::deque<PendingCmd> cmds_;
};

}  // namespace uavrgb

#endif  // UAVRGB_CF_SITL_GZ_SYSTEM_HH
