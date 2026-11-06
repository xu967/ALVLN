# 2026-11-06 23:35 — 3DGS 几何/渲染修复（REVIEW #3/#4/#19、AUDIT #1/#2/#3/#4/#10/#12、发白发雾）

- 时间：2026-11-06 21:04 – 23:35（中间被宿主中断一次，文件未丢）
- 范围：`sim/3dgs/{collision,scene,renderer,collect_3dgs,eval_3dgs,test_collision,test_renderer,test_collect_eval}.py`、`tools/{splat_to_ply,test_splat_to_ply}.py`、本文。未动 `video.py`、`dynamics.py`、`sim/tools/`、`ui/`、CF-Drone。未 commit / push，未启动 Gazebo。
- 结论：清单上 9 个问题全部修完并有测试；`sim/3dgs` 44 个单测在 3dgs-venv（含 2 个真实 gsplat 用例）全绿，在 .venv-tools 44 跑 2 skip 0 error；`tools/test_splat_to_ply.py` 7 个用例（含真实渲染等价）全绿。Callinstrasse 4 个基线位姿 GPU 实渲：暗通道 0.514→0.401，饱和度 0.025→0.047，拉普拉斯方差 35.9→436.7，单帧 ~3.4 ms。产物在 `/root/rivermind-data/uav_rgb/tmp/agent-3dgs/`。

## 做了什么

### 1. 碰撞代理 v2（REVIEW #3 / AUDIT #2、#3）— `collision.py` 重写，`proxy_version=3dgs.occ.v2`
- 在**导航系**建 3D 体素占据：`m_nav = Rᵀ(m_world − t)`，`R_nav = Rᵀ R_g`（`nav_to_world` 非正交或行列式<0 直接报错）。
- 只用 `opacity >= opacity_min`（默认 0.3）且最大轴 1σ ≤ `max_scale_m`（默认跟渲染端同值 3.0 m）的高斯。
- 椭球→体素光栅化：轴对齐半径 `k·sqrt(diag(RS²Rᵀ))` 框出候选格，对格中心做马氏距离 ≤ k（`sigma_k` 默认 1.5）判定；最小半厚度 0.6 体素，薄墙不会从格子缝里漏掉。
- 机体包络：读 `sim/dynamics/e1_mini.json` 的 `collision_envelope`（文件尚不存在），回退 `/root/rivermind-data/uav_rgb/tmp/research-hw/e1_mini_params_draft.json`（半径 0.102 m、半高 0.028 m，已标 TODO）；`envelope: "x3"` 保留旧 0.3 m；也可直接给 `{radius_m, half_height_m}`。膨胀用固定圆柱结构元素（numpy，不依赖 scipy），格数 `round()`。
- `sweep(p0, p1)`：沿位移按 ≤ 半体素采样，返回 (是否碰, 首个碰撞点)；`collect_3dgs.run_episode` 每步改用 sweep，`actions.jsonl` 新增 `collision_point`。
- scene.json `collision` 段新键：`margin_m`、`opacity_min`、`sigma_k`、`envelope`、`max_scale_m`；v1 的 `inflate_m/drone_radius_m` 忽略并在 stderr 提示（现有 `callinstrasse/scene.json`、`bicycle_real.json` 仍带这两个键）。
- 测试 `test_collision.py` 16 个：薄墙（1 cm 厚）一步跨越必须报碰、只查终点会漏；不透明度 0.05 的幽灵墙不碰；`nav_to_world` 非单位旋转/平移下世界系与导航系判定一致；椭球范围按朝向变化；圆柱膨胀不是方形；e1/x3/显式包络解析；场景端到端。

### 2. `tools/splat_to_ply.py`（REVIEW #4）
- 只允许正交、行列式 +1 的旋转：`--up y`（绕 x +90°，默认）、`--up=-y`（COLMAP y-down，绕 x −90°；argparse 吃掉裸 `-y`，写 `--up=-y` 或 `--up ydown`）、`--up z`。
- 同一旋转同时作用到 means 和 quats（`q' = q_R ⊗ q`，wxyz；先核实 `.splat` rot 字节顺序 = PLY `rot_0..3` = wxyz，诊断里按 xyzw 解读时 lapvar 16.8→4.2），scales × scale。
- 删除 `--flip-height`（行列式 −1 镜像）和 `--no-rotate-y-up`：传了直接报错并提示用 `--up=-y` / `--up z`。
- 测试 `tools/test_splat_to_ply.py` 7 个：字节布局与四元数顺序；三种 up 都是正常旋转且 `-y = diag(1,−1,−1)·R_y`，旧 flip 的行列式是 −1；旋转矩阵↔四元数往返、Hamilton 积↔矩阵积；300 个随机高斯变换后协方差 `Σ' = s²RΣRᵀ`、中心 `μ' = sRμ + t`（并证明"只转中心不转四元数"会被查出来）；整文件往返；CLI 拒绝 `--flip-height`；`RUN_3DGS_REAL=1` 时 gsplat 渲染等价：变换前后同一物理相机 RGB/alpha 最大差 < 2e-3、深度按尺度缩放，而旧做法（四元数不转）平均差 > 1e-2。
- 用新工具重转 `bicycle-mini.splat`（`--up=-y --scale 2 --z-offset 0.516`）→ `3dgs-scenes/bicycle/bicycle_v2.{ply,json}`（567714 高斯；旧 `bicycle_real.*` 未动），与旧 `bicycle_real.ply` 对比（`tmp/agent-3dgs/bicycle_v2/bicycle_old_vs_v2_montage.png`、`bicycle_compare.json`；并行实例的同一结论见 `bicycle_old_vs_new_montage.png`、`bicycle_old_vs_new.json`）：新 means 与"旧 means 关于 y 镜像"最大差 1.2e-7（几何关系成立）；spawn 位姿 new vs old MAD 54.0，new vs fliplr(old) 18.7，残差来自旧文件四元数未旋转（中位各向异性比 12.4，拉长的高斯全被摆错方向，旧图呈左右对称的拖影"双车"）。新图是一辆清晰的自行车靠在长椅上，传动侧朝相机、车头在左，手性正确。

### 3. 渲染（REVIEW #19 + 发白发雾）— `scene.py` / `renderer.py`
- `read_gaussian_ply` 读 `f_rest`，返回 `sh [N,(L+1)²,3]` 和 `sh_degree`（缺/数量不对则 0）。
- `RENDER_DEFAULTS = {"sh_degree": 3, "max_scale_m": 3.0, "supersample": 2}`，scene.json `render` 段可覆盖；`GsplatRenderer` 按 `max_scale_m` 过滤、`sh_degree=min(想要, 文件)`、`supersample` 倍分辨率渲染后 avg_pool（深度按 alpha 加权），`stats()` 多了 `n_loaded/n_removed_by_scale/max_scale_m/supersample`；CLI 加 `--sh-degree/--max-scale/--supersample`。
- 参数扫描（4 个基线位姿平均，`tmp/agent-3dgs/sweep.json`）：

| 变体 | dark_ch | sat | lapvar | alpha | alpha<0.9 | 2 cm 前移闪烁 MAD |
|---|---|---|---|---|---|---|
| sh0 classic（旧） | 0.514 | 0.025 | 35.9 | 1.000 | 0 | 0.208 |
| sh3 classic | 0.489 | 0.032 | 68.4 | 1.000 | 0 | 0.312 |
| sh3 antialiased | 0.496 | 0.031 | 38.4 | 1.000 | 0 | 0.267 |
| sh3 ss2 | 0.491 | 0.031 | 83.1 | 1.000 | 0 | 0.306 |
| sh3 ss2 smax<10 | 0.439 | 0.050 | 279.5 | 0.990 | 0.033 | 0.383 |
| sh3 ss2 smax<5 | 0.439 | 0.049 | 279.4 | 0.979 | 0.079 | 0.387 |
| **sh3 ss2 smax<3（新默认）** | **0.401** | **0.047** | **436.7** | **0.964** | **0.110** | 0.483 |
| sh3 ss2 smax<2 | 0.386 | 0.054 | 576.6 | 0.921 | 0.172 | 0.576 |
| sh3 ss2 smax<1 | 0.381 | 0.065 | 841.3 | 0.821 | 0.311 | 0.814 |

  - `max_scale_m=3.0`：去掉 6226 个（0.30%）1σ>3 m 的"天空壳"，暗通道降 0.11、清晰度 ×12；再往下（<2）alpha<0.9 的洞超过 17%，背景色漏出来变成新的伪影。Callinstrasse 尺度分位：p50 0.17、p99 1.38、p99.9 9.2、最大 6508 m。
  - `supersample=2` 而不是 `rasterize_mode="antialiased"`：antialiased 的 3D 低通把 lapvar 从 68 压到 38（糊），ss2 保清晰度 83 且闪烁 MAD 相近，两者都 ~3 ms。
  - `sh_degree=3`：视角相关颜色回来，lapvar 36→68。
- 修复后 GPU 实渲（`before_after_montage.png`、`before_after.json`，`before_repro_f*.png`、`after_f*.png`）。先用新 renderer 以旧选项复现 `e0_base_f*.png`：4 帧逐像素一致（max_abs_diff=0；我用四舍五入位姿的那次 f11 差 1 级），证明旧路径没被改坏。新默认下 alpha<0.9 像素 0→11%（巨型高斯去掉后露出的空洞由背景色补）。

| 位姿 | before dark_ch | after | before sat | after | before lapvar | after | ms before→after |
|---|---|---|---|---|---|---|---|
| f01 (−13.92,0,2,0) | 0.511 | 0.408 | 0.025 | 0.042 | 36.0 | 392.7 | 3.3→3.6 |
| f04 (−12.61,0,2,0) | 0.511 | 0.386 | 0.025 | 0.043 | 34.4 | 433.7 | 2.8→3.1 |
| f07 (−10.71,0,2,0) | 0.516 | 0.396 | 0.024 | 0.046 | 38.7 | 491.9 | 7.2→3.5 |
| f11 (−8.00,0,2,0) | 0.518 | 0.414 | 0.026 | 0.059 | 34.4 | 428.4 | 2.7→3.4 |
| 平均 | 0.514 | 0.401 | 0.025 | 0.047 | 35.9 | 436.7 | 4.0→3.4 |

### 4. 数据身份（AUDIT #1）— `collect_3dgs.py`
- `scene_identity(scene)`：`geometry_gid = sha256("3dgs:" + {ply_sha256, nav_to_world})[:16]`，`scene_id = int(gid[:8],16)`（dataset.cpp 只认整数），`task_id = sha256(spawn/waypoint/reach_radius)`，`split = split_manifest.split_for_gid(gid, salt=geometry_version)`。
- `run_episode` 去掉 `scene_id` 参数，改收 `identity/split_policy`；不声明策略又给不一致的 split → `ValueError`。CLI `--split` 默认 None（按几何组哈希），显式且不同 → `split_policy="diag-manual-split"` 并警告，preprocess 需 `--allow-diagnostic-split`。manifest 写 `geometry_gid/scene_id/task_id/split_policy/split_salt`，`version="3dgs-pilot-2"`。
- 测试：同 PLY 不同 seed 同 gid/scene_id/split，换 waypoint 只变 task_id，换 PLY 换 gid；强制 split 标诊断；`tools/preprocess.py` 对同 gid 跨 split 的篡改 manifest 拒绝（真实打包 2 集 teacher 产物通过）。

### 5. 评测（AUDIT #4 / #10 / #12）— `eval_3dgs.py`
- #4：`run_episode` 抛异常、返回 `err`、或 `validity.physics_verified=false` 都进 `failed[]`，不进 SR 分母，stderr 打印 `FAILED episode=…`，退出码 1。测试通过 sitecustomize 让第 2 集抛异常：rc=1、`failed` 1 条、`per_episode` 只有 0 和 2。
- #10：`summarize_3dgs_episode` 走 `summarize_episode` 的 `stop_radius` 分支，`reach_radius_m` 来自 scene（缺省 `REACH_RADIUS`=1.5），`metric_version="3dgs.pilot.v2"`，`success_criterion="reach_waypoint_euclid_no_collision_v2"`。测试：目标前 0.5 m 与冲过 0.5 m 同判成功，老口径对同样两集给 False/True。
- #12：`--hold` 或 `--data`（.uavr 读 `read_uavr_meta`，或含 run.json/episode_manifest.json 的目录）至少给一个，都给必须一致，否则 `p.error`；不再回退 `scene.step_s`。报告写 `hold_source`。对 3 个真实 .uavr 验证读到 0.5/0.5/1.0。
- 评测产物 split 固定 test，`split_policy="diag-eval-fixed-test"`（split_manifest 的 diag* 约定，打包端默认拒绝）。
- 保留并核实了 D 的 #14 改动：fps=1/hold、.webm、`video_frames` 写入数、dyaw 度/弧度修复（AUDIT #13）及其测试，仍然通过。

### 6. 碰撞 旧 vs 新 对比（Callinstrasse 基线轨迹 12 步，`collision_compare_lateral.json` + 并行实例的 `collision_compare.json`）
- 旧 v1：切片 58030 点、2D 栅格 986×1374、建 0.1 s；新 v2：不透明度+尺度过滤后 725234/2062370 个高斯，z∈[1.8,2.2] 内 41364 个，占据格 31444，栅格 940×1523×2，建 2.7 s。
- 12 步直线轨迹两者都不报碰（new-only 0，old-only 0）；步 4–7 旧 clearance 1.27–1.70 m、新 2.0 m（地面上限）——旧的近点是不透明度 <0.3 的漂浮高斯，新的把它们筛掉了。
- 横向探针：道路右侧 2 m 处 (−13.9,−2) 与 (−12.6,−2) **新报碰、旧不报**（旧 clearance 0.32/0.28 m，新 0.03/0.08 m）：新代理把椭球范围光栅化，旧只打中心点再膨胀 1 格。
- 路尽头（沿 y=0 继续往 +x）：旧代理从 x=−2.5 起报占据，新代理从 x=+0.2 起。x≈−2.3 处只有 3 个不透明度 0.04 的漂浮高斯，旧代理把它们当墙（假阳性）；真正的建筑/树从 x≈0 开始，两者一致（0.3 m 差 = 各自膨胀差）。
- Callinstrasse 真实 GPU 评测冒烟（forward 策略 12 步，`--data` 取 hold）：`eval_callinstrasse_forward/eval.json`，v2 指标、身份字段、新代理摘要、12 帧 .webm fps=2，退出码 0。

## 没做什么 + 原因
- `tools/train_student_with_video.py` 的 3DGS 自动评测命令（第 171 行）没加 `--hold/--data`，现在会被 `eval_3dgs` 以"必须显式给 --hold"拒绝——这正是 #12 要的失败方式，但该文件不在我的改动范围。需要的补丁：在 `cmd = [...]` 后加 `if data: cmd += ["--data", str(data)]`（.uavr 里有 hold_s）。
- `docs/HANDOFF_3DGS_20261105.md` 仍写着 `--flip-height`（第 30/124/151 行）和旧成功判据（第 4 条已知坑）；文档不在范围内，未改。
- `sim/dynamics/e1_mini.json` 不存在，包络回退到草案 JSON（代码里 TODO）。
- `bicycle_v2.{ply,json}` 是新增文件，放在 `3dgs-scenes/bicycle/` 旁边，没有覆盖或删除 `bicycle_real.*`、`bicycle_flip*.ply`；是否替换/清理旧文件由你定。
- 现有 scene.json（callinstrasse、bicycle_real）的 v1 碰撞键没改，运行时忽略并提示。
- 学生策略（`--policy student --ckpt`）的真实评测没跑，只跑了 forward；学生链路代码没动。
- gsplat JIT 环境：`env.sh` 的 `CUDA_HOME=/usr/local/cuda-12.8` 与不 source 时的 `/usr/local/cuda` 是同一目录但字符串不同，`build.ninja` 变了就重编 3 分钟（今天重编了 2 次）。我用 `tmp/agent-3dgs/gpu_env.sh` 固定环境，没改 `env.sh`（不在范围）。

## 待确认
1. 是否用 `bicycle_v2` 替换 `bicycle_real`（旧文件是镜像且高斯朝向错，不该再用）。
2. `callinstrasse/scene.json` 的 `collision` 段是否改成 v2 键（例如 `{"z_slice":[1.8,2.2],"voxel_m":0.2,"margin_m":0.1,"envelope":"e1_mini"}`）；现在 `margin_m` 默认 0，只按机体半径 0.102 m 膨胀 1 格（0.2 m）。
3. `train_student_with_video.py` 的 `--data` 补丁由谁来加。
4. E1 MINI 草案包络值（0.102 / −0.02 / 0.028 m）是否就是最终值。

## 验证到哪一步
- `3dgs-venv`：`RUN_3DGS_REAL=1 python -m unittest discover -s sim/3dgs -p 'test_*.py'` → Ran 44, OK；`RUN_3DGS_REAL=1 python tools/test_splat_to_ply.py` → Ran 7, OK。
- `.venv-tools`：同一套 → Ran 44, OK (skipped=2)；splat 测试 Ran 7, OK (skipped=1)。
- GPU 实渲：`exp_after.py`（守护进程，log 末尾 DONE）、`exp_collision_compare.py`（DONE）、Callinstrasse forward 评测 rc=0。
- 未验证：学生 ckpt 闭环；Gazebo 侧任何东西。

## 风险与遗留
- 新代理占据比旧的"胖"（椭球范围而非中心点），贴墙飞的旧数据集重新评测可能多出碰撞；这是修 #3 的预期结果，不是回归。
- 旧 3DGS 数据集（`3dgs-pilot-20261105*`）manifest 的 `scene_id` 是 seed、没有 `geometry_gid`，与新采集混包时 preprocess 的"同 scene 跨 split"检查对不上同一 PLY，不要混包。
- `make_corridor` 的点密度/σ 提高了（为让 3D 占据连续），新生成的合成走廊 PLY 与旧的不同，gid 也不同。
- `summarize_episode` 的 `stop_radius` 分支只看 x,y 平面距离；3DGS 里 z 固定，暂无影响。
- `video.py` 用 VP80 tag 时 OpenCV 打印 "tag not supported" 警告但文件照样写出（12 帧可读回），属视频 agent 范围。
- 宿主中断后把本 agent 续成了两个实例（A/B），23:20 起并行做同一清单：代码改动全部在分叉前完成，之后两边只跑实验/测试，没有互相覆盖仓库文件；实验产物同名的（`exp_after.py`、`before_after.json`、`collision_compare.json`）以 A 的为准，数值与 B 一致，B 的横向探针另存 `collision_compare_lateral.json`。实例间协调记录在 `tmp/agent-3dgs/PROGRESS.md`。若出现第二份 worklog（`…-2340-…`），内容与本文重复，保留一份即可。
- 产物目录：`/root/rivermind-data/uav_rgb/tmp/agent-3dgs/`（sweep.json、sweep_f*_montage.png、before_after.json、before_after_montage.png、before_repro_f*.png、after_f*.png、collision_compare.json、collision_compare_lateral.json、collision_old.py、bicycle_v2/、bicycle_old_vs_new*、bicycle_v2_*.png、eval_callinstrasse_forward/、exp_*.py、run_all.sh、*.log、gpu_env.sh、PROGRESS.md）。

## 更新 2026-11-06 23:55（实例 A 追加，正文为实例 B 所写，未改动）

- 全量测试（`tmp/agent-3dgs/run_tests.sh` → `run_tests.log`，守护进程，23:40–23:45）逐文件：
  3dgs-venv + `RUN_3DGS_REAL=1`：`test_collision` 16、`test_renderer` 9（含 2 个真 gsplat）、`test_dynamics` 6、`test_collect_eval` 13、`tools/test_splat_to_ply` 7（含渲染等价）全 OK；
  `unittest discover -s sim/3dgs` 不开真实用例：44 OK / 2 skip。
  .venv-tools（无 torch/scipy）：16 / 9（2 skip）/ 6 / 13 / 7（1 skip）全 OK，0 error；`python -m sim.3dgs.collision` 自检 PASS。
- gsplat JIT 缓存：23:41 那次重编是 `run_tests.sh` 先 source `gpu_env.sh` 再 source `env.sh`，`CUDA_HOME` 被改回 `/usr/local/cuda-12.8` 触发的。
  现在 `tmp/agent-3dgs/gpu_env.sh` 固定 `CUDA_HOME=/usr/local/cuda-12.8`（与 `env.sh` 同一字符串；`/usr/local/cuda` 本就软链到它），
  当前缓存 `extensions-sm89-20261106/gsplat_cuda` 即按此值编的；之后 source 顺序 `env.sh` → `gpu_env.sh`，就不会再重编。
- 实例 A 的 GPU 作业全部按守护进程规范跑（`setsid nohup … & pid 文件`，结束写 `run_all.DONE` / `run_tests.DONE`）：
  `run_all.sh`（exp_after 15 s、exp_bicycle 10 s、exp_collision 7 s，JIT 已缓存，GPU 与用户 YOLO 训练共享时单帧 p50 3.2–3.6 ms）。
- 新 renderer 以旧选项复现 demo 帧：实例 A 的 `exp_after.py` 用 `e0_baseline.json` 里的原始位姿，4 帧 `max_abs_diff` 全为 0（B 正文里 f11 的 1 级差来自位姿取整）。
- `3dgs-scenes/bicycle/bicycle_v2.{ply,json}` 由实例 A 写入（新增文件，140.8 MB；`bicycle_v2.json` 的 `collision` 段已是 v2 键 `{"z_slice":[1.5,2.5],"voxel_m":0.1,"margin_m":0.1,"envelope":"e1_mini"}`）。
