# 05｜ECPT 候选算法数学规格（V8；用于独立 AI 审查，不预先声明新颖）

## 0. 全称与边界

**ECPT = Evidence-Calibrated Progress and Termination**，证据校准的指令进度与终止决策；明确指向**同一个瓶颈证据状态**关联语言阶段、移动动作和任务 STOP。不要再分别写“证据校准记忆”、“进度证据机制”作为不同算法名。

**这是一个待证伪的设计候选，不是现有论文的已发表创新；下述维度可因部署测量调整，但修改必须同时更新等容量 baseline。**

## 1. 输入与输出

单个时间步合法在线输入：

- `V_t ∈ R^{B×P×D}`，轻量 CNN 提取 `P=84` 个 64 维视觉 patch (`D=64`)；
- `E ∈ R^{B×L×D}`，缓存的完整指令 token 表征，`L` 依真实文本分布确定；
- `a_{t-1} ∈ {0..K-1}`，上一时间步已发送且获得实际状态反馈的动作记录；
- `h^v_{t-1},h^p_{t-1} ∈ R^{B×64}`，视觉历史与语言阶段状态；
- `m^p_{t-1} ∈ R^{B×(L+1)}`，单调阶段分布，最后一位表示 DONE；
- `reset_mask∈{0,1}^B`。

输出：官方 Aerial 轨为 8 类动作 logits `l_t^a∈R^{B×8}` 和独立 `stop_logit∈R^B`；Gazebo 物理轨则经单独的 NavIntent head 和严格动作适配器（不能混用 Aerial 的 5m step）。辅助输出可包括指令进度分布、视觉证据可置信度、STOP 不确定度。

## 2. 前向流程（一个可实现初稿）

(1) 使用**上一时刻**的进度估计查询当前指令上下文，避免在同一时间步反复互相引用：

\[
  r_t = \sum_{j=1}^L \alpha^p_{t-1,j} E_j,\qquad
  \alpha^p_{t-1}=\operatorname{softmax}(m^p_{t-1}[1:L]).
\]

(2) 用当前语言上下文读取图像中的证据：

\[
 s_{t,i}=\frac{(W_qr_t)^\top(W_k V_{t,i})}{\sqrt D},\quad
 \alpha^v_{t}=\operatorname{softmax}_i(s_{t,i}),\quad
 z_t=\sum_i \alpha^v_{t,i} W_v V_{t,i}.
\]

(3) 用像素证据更新视觉历史和指令状态：

\[
 h^v_t=\operatorname{GRU}_v([z_t,r_t,\mathrm{Emb}(a_{t-1})],h^v_{t-1}),
\]
\[
 h^p_t=\operatorname{GRU}_p([h^v_t,r_t,z_t],h^p_{t-1}).
\]

(4) 计算语言进度分布及 DONE。引入相邻阶段的软单调限制，允许原阶段维持或向前滑动，但不强行增加固定阈值：

\[
 u_{t,j}= w_p^\top\tanh(W_hh^p_t+W_eE_j),\qquad
 p_t(j)=\operatorname{softmax}_j\big(u_{t,j}+\log M_t(j)\big).
\]

`M_t(j)` 是由上一时刻 `p_{t-1}` 生成的因果允许位置权重/掩码（不得包含未来 GT）；允许阶段停留、相邻前进、DONE 吸收态，具体变体需做消融。不能简单把教师 attention 当成进度真值。

(5) 在**同一证据状态**上计算动作与终止：

\[
 q_t=\sigma(w_q^\top[h^p_t,h^v_t,z_t,\max_i\alpha^v_{t,i}]+b_q),
\]
\[
 \ell^a_t=W_a[h^v_t,h^p_t,r_t],\qquad
 \ell^{stop}_t=w_s^\top[h^v_t,h^p_t,r_t,q_t,p_t(DONE)]+b_s.
\]

`q_t` 只是可训练的**置信特征候选**，并不因为用了 sigmoid 就天然是校准概率；必须用独立 held-out split 测 ECE、Brier、risk-coverage，必要时只在 calibration split 做 temperature scaling。STOP 动作应与飞行安全中断严格分离：任务成功 STOP 请求由导航 evaluator/controller 在确认后处理，不是突然切断电机。

## 3. 监督来源与丢失时的回退

| 监督 | 合法来源 | 不允许 |
|---|---|---|
| `L_action` | 同场景专家行为、官方训练轨迹或 DAgger 的学生状态专家重标注 | 用官方未来 GT 图像直接喂学生 |
| `L_stop` | 根据 episode 成功半径/是否在合规位置终止构造；必须将动作/终止标签分离 | 将 action=STOP 的先验频率当模型可靠性证据 |
| `L_progress` | 有审查记录的子句/地标通过事件、仿真中真实位置只作**训练标签** | 将教师 attention 或未来路径直接作为在线进度 |
| `L_evidence` | 由专家图像可见性标注、遮挡与路径可通行性弱标签 | 将 hidden geometry 精确监督成学生一定能“看见” |
| `L_KD` | 教师真实闭环优于学生且在 held-out 上校准后生成的动作/风险软分布 | 没检验教师就强制加大 KL 权重 |

建议初始损失：

\[
L=L_{BC}+\lambda_s L_{stop}+\lambda_pL_{progress}
+\lambda_c L_{calibration}+\lambda_{kd}L_{KD}.
\]

首个基线 `λ_p=λ_c=λ_kd=0`，随后单一变量开启；没有可靠 `weak_event_label` 时必须设 `λ_p=0` 而不是伪造子目标标签。没有可运行教师时 `λ_kd=0`。各权重必须在 train/开发集确定。

## 4. 参数/预算合同

视觉 backbone、文本 encoder、GRU、two heads 计划目标 `0.3M–1M`，具体参数量需由实际 C++ 结构自动统计而非文档虚报。候选 64 维 GRU 参数数量可按 `3*(Din*H + H*H + bias_terms)` 估算，文本 embedding `vocab_size × D`；上线前附逐层 MAC、激活 RAM、INT8 权重字节、CPU/RISC-V 时延实测表。**没有这些实测之前，不能宣称 ECPT 为 1MB 或 10Hz。**

## 5. 预注册的四个关键对照

1. **B0** 纯 CNN+文本均值+单 GRU+8 类 head（没有显式进度）。
2. **B1** B0 加独立 STOP head；控制参数数目，使 B0 也有等量容量但 STOP 一起训练。
3. **B2** B1 加 Self-Monitoring 式普通进度预测/无证据绑定；避免把老方法当新方法。
4. **B3** ECPT，使用同一证据状态关联进度与终止。若 B3 不超过 B1/B2，不能宣称该机制有效。

训练种子至少 3 个开发、5 个关键最终对照；同一批 episode+scene seed 配对。阈值、温度只在 train 内 calibration 或 val_seen 冻结后测试 `val_unseen`，**不可在 val_unseen 反复扫描阈值再挑最高分**。报告 STOP Early/Late、OSR-SR、ECE、Brier、loss、动作准确率、真实闭环 SR/NE/nDTW/SDTW、参数/MAC/CPU/NPU/P95。

## 6. 科研上的失败条件

- 如果同图换语言不改变输出，表明模型不利用指令；ECPT 研究无效，先修数据。
- 如果 B3 的提升完全来自额外参数而非证据耦合，创新不成立。
- 如果所有 staged progress 标签质量差，删除该辅助监督；不得用注意力图当真值。
- 如果关键改善仅是 STOP 阈值调得更好、等条件 B1 可复现，ECPT 主张失败。
- 如果任务仅在 Gazebo 的固定纹理/同一场景 seed 上有效，不能宣称开放世界自主飞行或实机鲁棒性。
- 如果 INT8 量化后严重损失闭环成功率，需报告负结果并降低模型复杂度，不能只展示 GPU FP32 结果。
