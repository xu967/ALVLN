# 数据集 vs 训练归因 + Git 基线

- 时间：2026-10-29 19:00
- 范围：只读分析（`/tmp/ds200`、`sim/worlds/uav_rgb.sdf`、`src/trainer.cpp`）；
  新增 `.gitignore`；建立本地 Git 仓库
- 结论：**数据集是主因，训练链路有效但有两个必须修的缺陷**；Git 已建库，未推送

## 做了什么

### 一、归因分析（写进 `docs/results.md` 19:00 条目）

- 用 train 的逐 step 众数在 test 上做 image-free 基线：**0.8300**，并把它拆成
  step 0–2（N=60，错 28，acc 0.533）与 step 3–19（N=340，错 40，acc 0.882）。
  模型 test 总共只错 10 个 → **先验错的地方模型基本都改对了**，证明它读的是图像。
- 审读 `src/trainer.cpp`，确认 T1「只保存最后一个 epoch」：`save_checkpoint` 在
  每个 epoch 末尾调用并覆盖，没有 best 跟踪。实测 dev acc 峰值 0.9784@ep27，
  交付的是 0.9757@ep30。
- 复算 T2 的量级：权重 `[745.25, 0.339, 1.827, 1.982]`，Σw=2980.3，
  单个 SAFE 样本占 25.0% 的 loss。
- 审读 `sim/worlds/uav_rgb.sdf`，发现三个固定彩色地标
  `landmark_red/green/blue`（1 m 立方体，固定在 (3,0)/(3,1.5)/(3,−1.5)），
  不在 `scene_gen` 的随机化区段内。
- 用 `sample_pillars` 复算 200 个 seed：**200/200 场景都有柱子几何上陷进地标内部**
  （每场景 3–4 根），因为 `validate_pillars` 只校验柱间缝隙和走廊边界。

### 二、Git 基线

- 新增 `.gitignore`：排除 `build/`（任意层级）、`.venv-tools/`、`__pycache__/`、
  `*.uavr` / `*.bin` / `*.ckpt`、`logs/runs/`。
- `git init -b main`，仓库根 = `uav_rgb/`（不是上层 `code/`，避免把
  `../idea/repos` 的 GPL 参考代码并进来）。
- 两次提交：`dd5c9cb` 基线（180 文件 / 24691 行 / 2.1 MB），
  `a70e17b` 收录另一会话在 18:35 追加的设备执行器集成验收记录。

## 没做什么

- **没有推送到远端。** 原因见「待确认」第一条：唯一可用的 deploy key 指向
  `git@github.com:xu967/ALVLN.git`，而那个仓库装的是规格文档包 + frontend + legacy
  快照，不是本工程的代码树。直接 push 会覆盖掉别人的东西。
- **没有修 T1（最优 checkpoint）。** 原因：改 `trainer.cpp` 的主循环属于代码改动，
  且当前数据即将作废，先记录再改更划算；下一格和新数据一起验证。
- **没有修 T2（类别权重退化）。** 原因同上；`class_weights.cpp` 目前只打警告，
  要不要在 `n_c==1` 时直接拒绝训练，属于行为变更，需要确认。
- **没有删掉三个固定地标。** 原因：地标在 z∈[0.5,1.5]，巡航高度 2.3 m，
  **不阻挡飞行**；删掉会改变所有已有数据的画面可比性。是否删除要连同「地标是不是
  给后续语言导航任务留的」一起决定。
- **没有重采数据、没有重训。** 原因：D1（episode 太短）没修之前重采只会复现同样结果。
- **没有跑 image-shuffle 对照实验。** 原因：需要另写一个打乱图像块的脚本（临时脚本用完即删），
  且结论已由「先验错误分解」给出，边际信息有限。
- **没有验证远端仓库的可见性（public/private）和许可证。** 原因：deploy key 只能读树，
  看不到仓库设置；这一条留给用户确认。
- **没有处理 `.gitconfig` 全局身份。** 原因：机器上原本没有全局 `user.name/email`，
  我只设了仓库级身份（见待确认第二条），没有动全局配置。

## 待确认

1. **推到哪个仓库？** 现状：
   - `ssh -T git@github.com` → `Hi xu967/ALVLN!`，说明这是一把 **deploy key**，
     只能访问 `xu967/ALVLN` 这一个仓库。
   - 该仓库现有 123 个文件：`docs/00–14_*.md`（规格包）、`frontend/`、`legacy/`、
     `references/`、`server/`，**没有 `uav_rgb` 的代码**。
   - 三个选项：(a) 把本工程作为子目录 `uav_rgb/` 推进去（非破坏性）；
     (b) 新建独立仓库并为它加一把 deploy key；(c) 先只保留本地。
2. **Git 身份**：仓库级设为 `xu967 <xu967@users.noreply.github.com>`，
   依据是公钥注释 `xu967-alvln-backup` 与 SSH 返回的账号名。若不对，改配置后
   可以 amend 第一次提交，不影响内容。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `ssh -T git@github.com` | `Hi xu967/ALVLN!` 认证成功 |
| `git ls-remote git@github.com:xu967/ALVLN.git` | `main` = `4502255…`，**与本工程无关** |
| `git ls-tree -r --name-only FETCH_HEAD` | 123 个文件，无 `uav_rgb/` 路径 |
| 提交前 `git diff --cached --numstat` | 180 文件、24691 行 |
| 提交前大文件扫描 | 最大 112 KB（`ui/evidence/monitor_control.png`），无 >1 MB 文件 |
| 提交前密钥扫描 | 无 `.env` / 私钥 / token 命中 |
| `git log --oneline` | `a70e17b` / `dd5c9cb` |
| `git status --porcelain` | 0 项未提交 |
| 归因分析的统计 | 全部在 `docs/results.md` 19:00 条目，可复跑 |

## 风险与遗留

- **本仓库存在并发写入的会话。** 18:35 有另一个进程（`claude` / `codex` 之一）
  往 `docs/worklog/2026-10-29-1734-device-state-agent.md` 追加了验收记录。
  Git 已建库后，并发改动会以未提交改动出现，**不要用 `git add -A` 盲提交**，
  要逐文件确认归属。
- **未推送 = 没有异地备份。** 本地 `.git` 和源码在同一块盘上，断电即一起丢。
  用户要的「方便回溯」目前只做到本地这一层。
- T1/T2 不修就采新数据，会让下一轮实验同时背两个已知污染源，届时仍然无法归因。
- D7（柱子陷在地标里）会让新数据继续带着无意义的视觉噪声；要不要先改场景再采，
  应和 episode 长度一起决定，避免采集两轮。
