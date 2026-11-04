# sim/assets 资产目录

**当前仓库里的资产是离线演示占位物，不是真实 CC0 资产。**
`uavrgb_assets/meshes/*.obj` 与 `uavrgb_assets/materials/*.png` 由
`sim/tools/build_demo_assets.py` 程序化生成（固定随机种子），没有照片、扫描或第三方模型；
`campus_textures/*.png` 由 `sim/tools/make_campus_textures.py` 生成后复制到这里。
它们用于离线跑通渲染/碰撞/数据管线，不代表真实感或 sim-to-real 能力。

## 资源路径

运行时把本目录加入 `GZ_SIM_RESOURCE_PATH`：

```bash
export GZ_SIM_RESOURCE_PATH="$(pwd)/sim/assets${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
```

SDF 只写 `model://uavrgb_assets/...` 或 `model://campus_textures/...`，**不要写绝对路径**，
否则换机器/换目录后纹理和网格加载失败。`collect_parallel.py` 已自动设置该路径。

## catalog schema（`asset_catalog.json`）

- 顶层：`schema`（`uavrgb.assets.v1`）、`source`、`license`、`author`。
- `types.<kind>`：`mesh`、`dimensions_m`、可选 `albedo_map`/`visual_size_m`、
  `provenance{kind, license, source, author, sha256}`。
- `ground`：`albedo_map` + 同样的 `provenance`。
- `sha256` 是 `{相对本目录的路径: 哈希}`；`ground` 也允许单文件哈希字符串
  （兼容 `register_ground_texture.py`）。
- demo 条目的 `provenance.kind` 是 `demo`；用户导入的是 `imported`。
  `build_demo_assets.py` 只允许覆盖 demo 条目。

## 导入真实资产

先自行确认许可（本工具只接受 CC0），再用：

```bash
python3 sim/tools/import_asset.py \
  --type tree --mesh /path/to/tree.obj --dimensions 1.4 1.4 4.0 \
  --source https://example.org/asset --author "Author Name" --license CC0 \
  --albedo /path/to/tree_albedo.png

python3 sim/tools/import_asset.py --check   # 校验 schema/provenance/sha256
```

`import_asset.py` 不联网、不自动转换模型；OBJ 带外部 `mtllib` 会被拒绝。
真实资产要求：单位米、XY 居中、底面 z=0，视觉网格与碰撞盒分离。
