# 08｜CF-Drone 安全审计、控制权限与未来 ESP32 修改计划

> 用户明确同意**将来**可以修改 ESP32；但本 V8 严格冻结固件源码。`references/snapshots/CF-Drone.zip` 是未经修改的上游副本。本文件只做风险识别、仿真设计与未来隔离修改计划，不提供现场可直接控制电机的指令。

## 1. 源码事实与硬件不确定性

此前逐行审计（详见 `references/reviews/CF_DRONE_SOURCE_AUDIT.md`）说明：CF-Drone 是基于 ESP32 的微型四轴飞控，具有 IMU、姿态/角速度 PID、混控与部分遥控/通信功能。定高 `ALTHOLD` 缺少完整控制实现，`AUTO` 枚举不等于完整速度/位置闭环。姿态估计在某些默认路径中飞行时主要依赖陀螺积分；开环下降依赖未经充分标定的悬停推力常量。扩展板的 BMP388/QMC5883L/VL53L1X/PMW3901 引脚规划**不能证明**实物存在这些传感器，也不能证明对应驱动和闭环已经实现。

实际使用 ESP32 classic、S3 还是 C3 尚未由实体硬件确认。不同板的 UART 资源冲突：C3 的 RC 可能占用 Serial1，classic/S3 可能有可用 Serial1；必须先实测引脚与串口实例，不可照搬另一芯片的 GPIO。

## 2. 高风险权限与失效状态

- MAVLink 中可触达命令解释器的高权限路径缺少充分的鉴权/操作范围隔离；未来引入导航 UART 时，不能保留另一条可绕过安全仲裁的外部入口。
- 通过配置可能选到未实现飞行模式，产生不符合飞手预期的控制响应。未来需采用模式白名单与穷尽 switch/guard。
- 单个飞行控制帧未见完整 NaN/range 检查时，不应将其视为可在真实飞行器上安全接收的接口。
- `descend()` 是未测动力学的开环推力衰减，不应在文档/UI 中标记“自动定高、安全悬停或安全降落”。
- 供电瞬态、机体载荷、推重比、连续动作超时、相机延迟、Linux 失联与飞控人工接管都尚未以实物记录证明。

**安全原则**：ESP32 飞控保持唯一控制仲裁权；LicheeRV 视觉板只产生有界意图，不能下发任意电机 PWM；无 valid ACK 时不得把命令判执行成功。任务 STOP 不是急停电机、不是安全悬停命令。

## 3. 今后的接口合同（本轮仅设计）

```text
NAV_INTENT_V1
  version, command_seq, observation_seq, request_id
  command_type, bounded_values, coordinate_frame, units
  issued_monotonic_time_ns, ttl_ms, crc

NAV_ACK_V1
  command_seq, accepted/rejected, reason_code

NAV_EXEC_STATUS_V1
  command_seq, status(started/completed/aborted), observed_controller_mode
```

注意 UART 两端都需要独立本地时钟；不能简单比较两块板的未同步 wall-clock 来判 TTL。接收端必须使用本地接收时间、seq 和有效期判定；不能把 ACK（收到）当作 EXEC_DONE（完成）。

未来四组受限命令类型：短时 body velocity、有限 yaw-rate、REQUEST_STOP_POLICY、HEARTBEAT。**只有真实飞控证明具有速度/高度控制能力之后**才能开放相应 setpoint 类型。否则停在人工遥控/禁用导航输出的 HIL。

## 4. 实体安全准入门禁（在未来修改固件之前）

1. 识别实体芯片/PCB/电机数量与固件 commit，记录硬件/固件 hash。
2. 关闭或隔离任何未经授权的高权限外部控制渠道；更新默认凭据和无线接入方式，做代码级审查。
3. IMU 漂移/姿态、供电电压峰值、静推力曲线、满载推重比、电机发热与机体振动记录。
4. 断链时若没有经过验证的高度保持与受控降落，不允许声称可安全自动恢复；人工接管必须独立可用。
5. 先无桨台架：CRC 错误/重放/乱序/TTL 过期/非法值/握手丢失/模式切换/抢占/重启/低压，总共 10+ 注入项。
6. 未经监护、围栏、应急接管、空域许可、供电安全和定高验证，**不进行自主自由飞行**。

## 5. 后续修改分支的建议顺序

- PR-F0：只增加模式白名单、NaN/range 拒绝与保护现有高权限入口；不添加导航自主模式。
- PR-F1：只完成 UART 的解析/CRC/ACK/超时/日志（无桨）。
- PR-F2：只有硬件有传感器且实际控制能力可验证时，才补齐定高/速度状态估计和调节器；不向模型隐式提供传感器输入。
- PR-F3：SIL/HIL 与故障注入通过，最后才在受控测试环境启用有界 NavIntent。

以上不是工期保证。V8 没有对 ESP32 固件做任何写入或改动。复制/分发 CF-Drone/Flix 的派生实现前请核实 LICENSE、附加条款及上游来源关系。
