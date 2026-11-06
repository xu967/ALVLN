// CfSitlSystem.cc：实现见 CfSitlSystem.hh 的说明。
#include "CfSitlSystem.hh"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>

#include <gz/common/Console.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>

namespace uavrgb {

namespace {
constexpr double kStepS = 0.001;   // 固件 1 kHz 控制环；cf_sitl_step_external 只接受这个步长
constexpr double kDtTol = 1e-9;

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}
}  // namespace

std::string CfSitlSystem::ResolveParams(const std::string& path) {
  namespace fs = std::filesystem;
  if (path.empty()) return path;
  if (fs::path(path).is_absolute()) return path;
  // 模板是内容寻址的，里面只能写仓库相对路径；运行时用 UAVRGB_REPO_ROOT（collect_parallel
  // 的 gz_env 会设）或当前目录补全。
  if (const char* root = std::getenv("UAVRGB_REPO_ROOT")) {
    fs::path p = fs::path(root) / path;
    if (fs::exists(p)) return p.string();
  }
  return fs::absolute(path).string();
}

void CfSitlSystem::Configure(const gz::sim::Entity& entity,
                             const std::shared_ptr<const sdf::Element>& sdf,
                             gz::sim::EntityComponentManager& ecm, gz::sim::EventManager&) {
  model_entity_ = entity;
  link_name_ = sdf->Get<std::string>("link_name", link_name_).first;
  params_path_ = ResolveParams(sdf->Get<std::string>("params", "").first);
  seed_ = sdf->Get<uint64_t>("seed", 0).first;
  const std::string action_topic = sdf->Get<std::string>("action_topic", "/e1_mini/action").first;
  const std::string state_topic = sdf->Get<std::string>("state_topic", "/e1_mini/state").first;
  const std::string cmd_topic = sdf->Get<std::string>("cmd_topic", "/e1_mini/cmd").first;
  chase_model_name_ = sdf->Get<std::string>("chase_model", chase_model_name_).first;
  chase_offset_ = sdf->Get<gz::math::Vector3d>("chase_offset", chase_offset_).first;
  chase_pitch_ = sdf->Get<double>("chase_pitch", chase_pitch_).first;
  const double state_hz = sdf->Get<double>("state_hz", 200.0).first;
  state_every_ = std::max(1, static_cast<int>(std::lround(1.0 / (kStepS * state_hz))));

  gz::sim::Model model(entity);
  const gz::sim::Entity link_entity = model.LinkByName(ecm, link_name_);
  if (link_entity == gz::sim::kNullEntity) {
    gzerr << "[CfSitlSystem] 模型 " << model.Name(ecm) << " 没有 link " << link_name_ << "\n";
    return;
  }
  link_ = gz::sim::Link(link_entity);
  link_.EnableVelocityChecks(ecm);
  link_.EnableAccelerationChecks(ecm);

  if (params_path_.empty()) {
    gzerr << "[CfSitlSystem] 缺 <params>（sim/dynamics/e1_mini.json 路径）\n";
    return;
  }
  if (cf_sitl_init(params_path_.c_str(), seed_) != 0) {
    gzerr << "[CfSitlSystem] cf_sitl_init 失败: " << params_path_ << "\n";
    return;
  }
  state_pub_ = node_.Advertise<gz::msgs::Double_V>(state_topic);
  if (!node_.Subscribe(action_topic, &CfSitlSystem::OnAction, this) ||
      !node_.Subscribe(cmd_topic, &CfSitlSystem::OnCmd, this)) {
    gzerr << "[CfSitlSystem] 订阅失败: " << action_topic << " / " << cmd_topic << "\n";
    return;
  }
  init_ok_ = true;
  gzmsg << "[CfSitlSystem] 就绪: link=" << link_name_ << " params=" << params_path_
        << " seed=" << seed_ << " action=" << action_topic << " cmd=" << cmd_topic
        << " state=" << state_topic << "@" << state_hz << "Hz\n";
}

void CfSitlSystem::OnAction(const gz::msgs::Double_V& msg) {
  if (msg.data_size() < 2) {
    gzwarn << "[CfSitlSystem] action 消息需要 [action, until_s_sim]，收到 "
           << msg.data_size() << " 个值，忽略\n";
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  actions_.push_back({static_cast<int>(std::lround(msg.data(0))), msg.data(1)});
}

void CfSitlSystem::OnCmd(const gz::msgs::StringMsg& msg) {
  std::istringstream ss(msg.data());
  PendingCmd cmd{"", 0.0};
  ss >> cmd.verb >> cmd.value;
  if (cmd.verb != "reset" && cmd.verb != "arm_takeoff") {
    gzwarn << "[CfSitlSystem] 未知命令 \"" << msg.data() << "\"（支持 reset <episode> / "
           << "arm_takeoff <height>）\n";
    return;
  }
  std::lock_guard<std::mutex> lock(mu_);
  cmds_.push_back(cmd);
}

void CfSitlSystem::DrainCmds(double sim_s, const gz::math::Pose3d& pose) {
  std::deque<PendingCmd> cmds;
  {
    std::lock_guard<std::mutex> lock(mu_);
    cmds.swap(cmds_);
  }
  for (const PendingCmd& c : cmds) {
    if (c.verb == "reset") {
      const double spawn[3] = {pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()};
      cf_sitl_reset(static_cast<uint64_t>(std::llround(c.value)), spawn, pose.Rot().Yaw());
      base_s_ = sim_s;
      late_actions_ = step_errors_ = 0;
      std::lock_guard<std::mutex> lock(mu_);
      actions_.clear();   // 上一集残留的动作不能带进新集
      gzmsg << "[CfSitlSystem] reset episode=" << std::llround(c.value)
            << " base_s=" << sim_s << "\n";
    } else {
      cf_sitl_arm_takeoff(c.value);
      gzmsg << "[CfSitlSystem] arm_takeoff height=" << c.value << "\n";
    }
  }
}

void CfSitlSystem::DrainActions(double t_sitl) {
  std::deque<PendingAction> acts;
  {
    std::lock_guard<std::mutex> lock(mu_);
    acts.swap(actions_);
  }
  for (const PendingAction& a : acts) {
    const double until_sitl = a.until_sim_s - base_s_;
    if (until_sitl <= t_sitl) {
      // 到达时窗口已过：照常传给 SITL（立即过期=悬停），只记数供调用方核对。
      ++late_actions_;
      gzwarn << "[CfSitlSystem] 动作 " << a.action << " 晚到 " << (t_sitl - until_sitl)
             << " s（until_sim=" << a.until_sim_s << "），累计 " << late_actions_ << "\n";
    }
    cf_sitl_set_action(a.action, until_sitl);
  }
}

bool CfSitlSystem::ReadBody(const gz::sim::EntityComponentManager& ecm, double t_sitl,
                            cf_sitl_body_in* in, gz::math::Pose3d* pose) {
  const auto p = link_.WorldPose(ecm);
  if (!p) return false;
  *pose = *p;
  // 速度/加速度组件在前几步可能还没被物理系统写入，按 0 处理（机体静止在地面上）。
  const gz::math::Vector3d vel = link_.WorldLinearVelocity(ecm).value_or(gz::math::Vector3d::Zero);
  const gz::math::Vector3d omega = link_.WorldAngularVelocity(ecm).value_or(gz::math::Vector3d::Zero);
  // WorldLinearAcceleration 是运动学加速度（静止=0、自由落体=-g），即头文件要的「不含重力」。
  const gz::math::Vector3d acc = link_.WorldLinearAcceleration(ecm).value_or(gz::math::Vector3d::Zero);
  in->t_s = t_sitl;
  for (int i = 0; i < 3; ++i) {
    in->pos[i] = p->Pos()[i];
    in->vel[i] = vel[i];
    in->omega[i] = omega[i];
    in->acc[i] = acc[i];
  }
  in->quat[0] = p->Rot().W();
  in->quat[1] = p->Rot().X();
  in->quat[2] = p->Rot().Y();
  in->quat[3] = p->Rot().Z();
  return true;
}

void CfSitlSystem::PublishState(double sim_s) {
  cf_sitl_state st{};
  cf_sitl_get_state(&st);
  // Double_V 没有 header：gz 仿真时间放在 data[32]，订阅方用它当时间戳。
  gz::msgs::Double_V msg;
  FlattenState(st, sim_s, late_actions_, step_errors_, &msg);
  state_pub_.Publish(msg);
}

void CfSitlSystem::PlaceChaseCam(gz::sim::EntityComponentManager& ecm,
                                 const gz::math::Pose3d& body) {
  if (!chase_looked_up_) {
    // 模型按 SDF 顺序创建，Configure 时 chase_cam 可能还不存在，首步再查。
    chase_looked_up_ = true;
    chase_entity_ = ecm.EntityByComponents(gz::sim::components::Name(chase_model_name_),
                                           gz::sim::components::Model());
    if (chase_entity_ == gz::sim::kNullEntity) {
      gzmsg << "[CfSitlSystem] 世界里没有模型 " << chase_model_name_ << "，不驱动追尾相机\n";
    }
  }
  if (chase_entity_ == gz::sim::kNullEntity) return;
  const double yaw = body.Rot().Yaw();
  const gz::math::Quaterniond yaw_only(0.0, 0.0, yaw);
  const gz::math::Pose3d pose(body.Pos() + yaw_only.RotateVector(chase_offset_),
                              gz::math::Quaterniond(0.0, chase_pitch_, yaw));
  gz::sim::Model(chase_entity_).SetWorldPoseCmd(ecm, pose);
}

void CfSitlSystem::PreUpdate(const gz::sim::UpdateInfo& info,
                             gz::sim::EntityComponentManager& ecm) {
  if (!init_ok_ || info.paused) return;   // 暂停时固件也不走：SITL 时钟与 gz 时钟同步停
  const double dt = Seconds(info.dt);
  if (std::fabs(dt - kStepS) > kDtTol) {
    if (!dt_error_reported_) {
      gzerr << "[CfSitlSystem] 物理步长 " << dt << " s != 0.001 s，拒绝施力；"
            << "请把世界 <max_step_size> 改成 0.001\n";
      dt_error_reported_ = true;
    }
    return;
  }
  const double sim_s = Seconds(info.simTime);
  cf_sitl_body_in in{};
  gz::math::Pose3d pose;
  if (!ReadBody(ecm, 0.0, &in, &pose)) return;
  if (!based_) {
    // 上电：首步隐式 reset，SITL 时钟从这一刻起算；collect 会再按 episode 显式 reset。
    based_ = true;
    const double spawn[3] = {pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()};
    cf_sitl_reset(0, spawn, pose.Rot().Yaw());
    base_s_ = sim_s;
  }
  DrainCmds(sim_s, pose);
  const double t_sitl = sim_s - base_s_;
  in.t_s = t_sitl;
  DrainActions(t_sitl);

  cf_sitl_wrench w{};
  const int rc = cf_sitl_step_external(&in, &w);
  if (rc != 0) {
    if (step_errors_++ == 0) {
      gzerr << "[CfSitlSystem] cf_sitl_step_external 返回 " << rc << "（t=" << t_sitl
            << "），本步不施力\n";
    }
  } else {
    // SITL 返回机体系 wrench（作用在 link 原点，不含重力）；AddWorldWrench 要世界系。
    const gz::math::Vector3d f_b(w.force[0], w.force[1], w.force[2]);
    const gz::math::Vector3d t_b(w.torque[0], w.torque[1], w.torque[2]);
    link_.AddWorldWrench(ecm, pose.Rot().RotateVector(f_b), pose.Rot().RotateVector(t_b));
  }
  ++steps_;
  if (steps_ % static_cast<uint64_t>(state_every_) == 0) PublishState(sim_s);
  if (steps_ % static_cast<uint64_t>(chase_every_) == 0) PlaceChaseCam(ecm, pose);
}

}  // namespace uavrgb

GZ_ADD_PLUGIN(uavrgb::CfSitlSystem, gz::sim::System, uavrgb::CfSitlSystem::ISystemConfigure,
              uavrgb::CfSitlSystem::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(uavrgb::CfSitlSystem, "uavrgb::CfSitlSystem")
