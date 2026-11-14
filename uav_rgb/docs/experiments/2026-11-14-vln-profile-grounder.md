# VLN profile、grounder 与真实 UAVR smoke（2026-11-14）

## 目的

验证模型 profile 配置不会改变现有 `nano_proto_v6` 契约，同时让较大的
`base_torch_v1` 有一条独立的 Torch-only 实验入口；验证语言模块只做候选地标
选择，不把 route waypoint 当成语言输入。

## 环境与命令

- 环境：RTX 2080 Ti 主机；模型 smoke 使用 `/root/rivermind-data/uav_rgb/teacher-venv` 的 CPU，避免占用显卡上的其他服务。
- 数据：`/root/rivermind-data/uav_rgb/tmp/agent6-distill/trunc/ds/dataset.uavr`，532 MB，真实 UAVR。

```bash
/root/rivermind-data/uav_rgb/teacher-venv/bin/python -m unittest \
  proto.test_config proto.test_base_model proto.test_train -q
/root/rivermind-data/uav_rgb/3dgs-venv/bin/python -m unittest \
  sim.3dgs.test_vln sim.3dgs.test_guard sim.3dgs.test_renderer sim.3dgs.test_grounder -q
python3 -m unittest tools.test_audit_vln_grounder \
  tools.test_audit_route_candidates tools.test_audit_openfly_annotations -q
/root/rivermind-data/uav_rgb/teacher-venv/bin/python -m proto.train_base \
  --config config/vln/base_torch_v1.yaml \
  --out /root/rivermind-data/uav_rgb/tmp/base_torch_v1-smoke-verified-20261114 \
  --device cpu --max-steps 1
ctest --test-dir build/check --output-on-failure
```

## 实测结果

- `proto.test_config proto.test_base_model proto.test_train`：29 项通过，跳过 4 项。
- 3DGS 相关测试：77 项通过，跳过 2 项。
- 审计工具测试：6 项通过。
- CTest：34/34 通过。
- 真实 UAVR 的 `base_torch_v1` 单步训练：退出码 0，1 step；验证集 `574` 样本，
  输出 `/root/rivermind-data/uav_rgb/tmp/base_torch_v1-smoke-verified-20261114/`，
  `macro_f1=0.0`、`balanced_accuracy=0.0`、`accuracy=0.0`、逐类召回全 0；
  test 的 `macro_f1=0.000565`、`balanced_accuracy=0.25`、`accuracy=0.001131`，
  每类召回和混淆矩阵均已落盘。这是单步可运行性 smoke，不是模型效果结论。
- `nano_proto_v6` 配置入口也在同一份真实 UAVR 上运行 1 step 后按 `--max-steps` 正常退出，
  `resolved_config.json` 记录了 YAML 配置与数据签名；debug 模式按设计不评估、不保存 checkpoint。
- 端点词汇 grounder 的六场景诊断输出在
  `/root/rivermind-data/uav_rgb/tmp/vln-grounder-endpoint-diagnostic-20261114.json`：
  4429 条指令，1077 条被接受，830 条命中标注 `end`，总体命中率 18.74%。
  该结果明确带 `leakage_warning=true`，因为候选来自路线标注的 start/end，不能作为正式 G1 指标。

## 解释与限制

- `nano_proto_v6` 继续走原有 C++ 对拍训练器；配置默认拒绝语言、历史帧和 prev-action，避免“配置写了但模型没吃到”。
- `base_torch_v1` 只输出 `model.pt`，不导出 C++ checkpoint；多 epoch 按验证集 macro F1 保存最佳权重，test 只在最佳权重上报告。
- 当前 `scene.json` 没有独立物体 catalog 和坐标，grounder 没有接入 `run_vln` 或飞控；需要独立地图/感知候选表后才可做正式 G1 闭环。
- OpenFly 全量数据、权重、3DGS 资产没有下载；官方环境和本项目四动作/小步长动力学不兼容，见 `docs/OPENFLY_LOCAL_EVAL.md`。

## §10 B/C/D 执行增量（2026-11-14）

环境：RTX 2080 Ti，真实 GPU 探针；输出根 `/root/rivermind-data/uav_rgb/tmp/vln-plan-followup`。

- `source env.sh; "$UAV_RGB_PYTHON" -m unittest tools/test_scene_probe_quality.py sim/3dgs/test_grounder.py tools/test_audit_vln_grounder.py`：20 tests OK，rc=0。
- `/root/rivermind-data/uav_rgb/3dgs-venv/bin/python tools/scene_probe_quality.py --scene /root/rivermind-data/uav_rgb/3dgs-scenes/wanderland_1MlUucug/scene.json --route 0 --out /root/rivermind-data/uav_rgb/tmp/vln-plan-followup/wanderland-probe`：rc=0。路径12.81 m，22帧，低锐度0，lapvar最小287.9、中位632.0，最大空洞0.0161588542。仅诊断通过，不是30–40米长航线验收；阈值0.9尚未在验证集标定，不代表训练相机覆盖合格。
- `"$UAV_RGB_PYTHON" experiments/2026-11-11-teacher-clearance-3dgs/measure_disp.py --out /root/rivermind-data/uav_rgb/tmp/vln-plan-followup/clearance-recheck`：rc=0，历史240集9174步，验证半前进预测误差p90：P1 0.080m / P2 0.041m。属于旧hold=1.0离线复算，不是新hold=0.5实测。

未接正式G1闭环、未训练条件学生、未更改真实飞控；B/C/D均只完成增量而非整卡验收。

## 正式L1语言基线训练启动

命令：`setsid nohup bash experiments/2026-11-14-l1-language-formal/run.sh`。
环境：RTX2080Ti，C++/CUDA。数据 `l1-v4-cf/ds/dataset.uavr`，SHA256 `3a1478a265f98b4a8b38accb20cf8ee53833eeb9c6af4112c8b2c66b20b4c614`。3种子×20轮，沿用旧L1配方。
状态：运行中；首个进程实际占用GPU578MiB，尚无最终成绩。训练器报告旧数据缺split_scheme/split_salt，只验证scene_id隔离；只作为模板语言基线复训，不是3DGS开放VLN。
输出 `/root/rivermind-data/uav_rgb/exp-l1-language-formal-20261114`，逐轮数字读取results.csv；最终结果待完成后补录。

启动后数据复核：L1共16184帧/902集，train14496、val1688、test0（scene_id划分）。本轮仅训练/验证，不存在独立test成绩；正式泛化评测必须另用冻结测试集。

## 并发视觉对照与验证评测（2026-11-14）

- 并发命令：`experiments/2026-11-14-l1-language-formal/run_parallel.sh`，三个 seed 使用 `--vision-only`、20 epochs、batch 64。
- GPU：RTX 2080 Ti；三个进程同时运行，每进程约 572 MiB；`DONE=0`。
- 视觉对照最终一轮 results.csv：seed1234 acc 0.87095 / macro-F1 0.87530 / bal_acc 0.51481；seed1235 0.87041 / 0.87090 / 0.50770；seed1236 0.87232 / 0.86848 / 0.50652。
- 同一 val split 的 `--eval-only`：语言 seed1234/1235/1236 accuracy 0.9005/0.8987/0.9023，macro-F1 0.8767/0.8739/0.8749；视觉 seed1234/1235/1236 accuracy 0.5071/0.5071/0.5095，macro-F1 0.5083/0.5138/0.5150。语言分支在此模板数据上明显被使用，但这不是3DGS开放语言闭环成绩。
- SAGE 下载命令：`huggingface-cli download spatialverse/SAGE-3D_VLN_Data --repo-type dataset ...`；因当前环境连接 Hugging Face 超时中断，目录仍约4 KB，没有数据文件。
