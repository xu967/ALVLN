# 数据完整性修复：几何 split、位姿停更、教师全堵

- 时间：2026-10-29 22:00
- 范围：`sim/tools/split_manifest.py`、`sim/tools/collect.py`、`sim/tools/teacher.py`、
  `tools/preprocess.py`、`tools/check.sh`、对应测试与文档
- 结论：**完成**。审核中确认的 3 个 P1 数据问题已修：`-0.0` 几何分组跨 split、
  collect/preprocess 无几何门禁、位姿停更仍写成功行。全部有复现/回归，未启动 Gazebo 复采。

## 做了什么

1. **`-0.0` 归一**（`split_manifest.py`）
   - 复现：seed 337 的 pillar `y=-0.0`、3651 为 `0.0`；`"%.3f" % -0.0` = `-0.000`，
     两个物理相同的场景 gid 不同，split 分别为 test/train。
   - 修法：`_fmt(v)` 改为 `"%....f" % (v + 0.0)`，把 `-0.0` 归一成 `0.0`；
     `SPLIT_SALT` 从 `uavrgb.geom.v1` 升到 `v2`（按模块规则，几何语义变了就开新数据版本）。
   - 回归：`test_split_manifest.py` 新增 337/3651 同 gid 同 split 断言。
2. **几何 split 门禁**（`collect.py` + `preprocess.py`）
   - `collect.py` 带 `--scene-meta` 时计算 `geometry_gid`，校验手传 `--split`
     必须等于 `split_for_gid(gid)`，否则 argparse 报错退出；manifest 写 `geometry_gid`。
   - `preprocess.py` 读 `geometry_gid`，拒绝同一 gid 跨两个 split；
     `dataset.uavr` 的每 episode meta 追加该字段。旧数据/合成夹具无此字段时仍按 scene_id。
   - 回归：`tools/check.sh` 增加负例——把 ep1/ep3 的 manifest 改成同一 gid、
     不同 split，preprocess 必须非零退出；否则 check.sh 失败。
3. **位姿停更 fail-closed**（`collect.py`）
   - 复现（审核 agent 用 fake subs）：动作后位姿只有动作前那帧时，`sim_dt=0`、
     `dist=0`、`stuck=False`，仍写成功行且 `physics_verified=true`。
   - 修法：等 1 s 后仍没有 `pose_ts > action_start_ns` 的帧，`collect_one` 返回失败；
     `stuck` 额外要求 `sim_dt > 0`。
   - 回归：`test_collect.py` 新增 fake subs 用例（14 项通过）。
4. **教师全堵不再默认前进**（`teacher.py`）
   - 修法：所有方向都不可走、且 base 自身也 `ray_clear` 失败时返回 `SAFE_NEUTRAL`。
   - 回归：`teacher.py` 自测加一条 monkeypatch `ray_clear=False` 的用例（全部通过）。
5. 文档：`tools/DATASET_FORMAT.md` 增补 `geometry_gid`；`docs/EVAL_STANDARD.md`
   更新 split 口径与 salt v2。

## 没做什么

- **没有重新采数据/跑 Gazebo 复验位姿停更路径**。原因：审核期间已跑过一次 3 步冒烟；
  停更逻辑用 fake subs 覆盖，真实注入停更需要停插件或 SIGSTOP，成本高且会占软件 GL。
- **没有加深度/contact 停更阈值**。原因：历史实测 depth-RGB gap 曾到 168–308 ms，
  阈值需要真实多轮标定；现有 gap 已落盘，先记录为已知项。
- **没有把 `displacement_m`/`dyaw_deg` 移出 actions.jsonl**。原因：preprocess 不读它们，
  移动会改变落盘合同；记入风险。
- **没有 provenance 透传**（teacher/hold_s/scene_meta_sha256 进 dataset meta）。
  原因：与研究数据版本一起改更合适，单独改会增加格式变更面。
- **没有修 collect_parallel 的 SIGTERM 清理、同 `--out` TOCTOU**。原因：独立改造，
  需要进程级测试；已在审核报告里记录。

## 待确认

- salt 升到 v2 后，2026-10-29 之前采集的所有 episode（包括 `/root/rivermind-data`
  下的冒烟数据）split 标签与新策略不一致。是否重采，还是保留为“v1 历史数据”单独标注。
- 位姿停更现在会让该 episode 失败；这会降低真实采集成功率，需要一次 8 实例实采确认
  成功率/吞吐是否可接受。

## 验证到哪一步

| 命令 | 真实输出 |
| --- | --- |
| `python -m py_compile`（collect/preprocess/split_manifest/teacher/tests） | OK |
| `sim/tools/test_split_manifest.py` | 9 tests OK（含 337/3651） |
| `sim/tools/test_collect.py` | 14 项通过（含 pose 停更） |
| `sim/tools/teacher.py` | 全部通过（含全堵 base 用例） |
| 手工负例：同 gid 跨 train/val → preprocess | rc=1，`几何组 g_same 同时出现在 split=train 与 val`，不产 dataset |
| 手工正例：无 gid 的合成 episode → preprocess | rc=0，`dataset.uavr` 634630 B |
| 修改前复现 | 337 gid=`e8c96d…`→test；3651 gid=`cc45e8…`→train；几何相同 |

未验证：真实 `collect.py --scene-meta` 的 split 不匹配路径（用参数校验代码路径确认）；
`check.sh` 新增负例要等最终全量构建时跑。

## 风险与遗留

- **v1 数据全部失效**：salt v2 后不得与 v2 数据混用；已有数据需要重采或显式重标。
- `collect.py` 现在依赖 `split_manifest`（进而 `scene_gen`），三者都在 sim/tools，
  纯标准库；但单独复制 `collect.py` 到别的目录会 import 失败。
- `preprocess` 的 gid 门禁只在 manifest 有 `geometry_gid` 时生效；旧数据仍可能带
  未发现的跨 split 几何。
- 位姿停更从“静默成功”变为“整集中断”，成功率可能下降；这是正确性换取的代价。
