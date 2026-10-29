# TinyAerialVLN Final V8 / NavLab Studio

面向无人机视觉语言导航（Aerial VLN）的研究工程，包含本机监控界面、任务管理 API、训练/仿真证据链和部署脚手架。当前是可运行的**管理框架**，不是已训练完成的导航系统。

先看 [START_HERE.md](START_HERE.md)。

## 实现要点

- 同源本机 UI/API，默认仅监听 `127.0.0.1:8765`，不提供公网认证。
- 训练与仿真执行器、数据、权重、场景必须由管理员注册并通过预检；未认证资源一律阻断。
- 训练指标只读取真实 `results.csv`；解析异常不会杀死监控线程，进程结束后仍会落盘最终状态。
- 主动停止会持久化停止意图，无论进程正常退出还是被 SIGTERM 终止，最终状态都记录为 `stopped`。
- 活动任务下载审核包时，会短暂暂停托管进程组并读取全部证据文件；ZIP、SHA256 和大小来自同一份快照。
- 服务端独立校验训练/仿真参数，不能依赖或绕过浏览器校验。
- 前端保留离线演示，但真实数据、演示数据和缺失结果严格区分，不伪造训练或导航结果。

## 快速开始

```bash
./run.sh doctor
./run.sh test
./run.sh ui
```

其他命令：`./run.sh cpp-test`、`./run.sh scene --difficulty S1 --seed 42`。默认注册表为空，因此真实训练和仿真会被安全阻断。

## 当前验证

- 36 项 Python 测试通过。
- C++17 梯度检查通过。
- 确定性静态语义 SDF 场景生成可用。
- 尚未验证：Gazebo 实际飞行、UE4/AirSim 闭环、CUDA 训练、SG2002 NPU 和实体无人机。

## 许可

Proprietary，仅限授权人员内部研究使用，不授予对外复制、分发、商用或再许可权利。第三方组件继续适用各自许可证。详见 [LICENSE](LICENSE) 和 [docs/13_PROVENANCE_LICENSE.md](docs/13_PROVENANCE_LICENSE.md)。
