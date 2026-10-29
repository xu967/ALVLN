# V7 sim/ 实现清单（未实现）

按 `V7_RESEARCH_PLAN.md` 的 GZ0→GZ1→GZ2→GZ3→CF0→CF1 顺序增量写 C++：

- `worlds/`：官方 X3 基础 world 的本地固定版本引用；在此基础上组合 V6.1 的森林障碍。
- `models/`：挂在动态 UAV base_link 上的前视 RGB camera 及经过许可的机体/电机模型。
- `src/`：至多 6 个 .cpp，分别负责 WorldRunner、RgbBridge、MotorBridge、CFControlCore SIL、TaskGenerator、ExpertPlanner。
- `tests/`：电机旋向、pose/image 同时钟、camera 光轴、collision、reset、SIL 数学 golden。

严禁把此目录的 `.gitkeep` 当成可运行仿真证据。`sim_preflight.py` 仅负责官方 X3 smoke 启动与诊断。
