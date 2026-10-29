# 浙大 FAST-Lab 9 个开源仓库：工程配置研究

- 时间：2026-10-29 00:38（CST）
- 对象：`/root/claude-code-backup-20261028/code/idea/repos/` 下 ego-planner、ego-planner-swarm、EGO-Planner-v2、Elastic-Tracker、Fast-Drone-250、Fast-Racing、GCOPTER、Implicit-SDF-Planner、Swarm-Formation
- 方法：**只读**。读了顶层与代表性 `CMakeLists.txt`（264 个中的约 20 个）、`.clang-format`、`package.xml` 抽样、`launch/`、`config/`、`test/`、README、LICENSE。未编译、未运行、未修改任何外部文件。
- 对照对象：`uav_rgb/src`（C++17 + CUDA 12.8，`uavrgb_core` 单静态库，9 个 ctest 裸 main 测试，无配置/日志/CI/Doxygen）。

## 0. 结论速览

**一句话**：这 9 个仓库的算法很强，工程配置很弱且高度同质——9/9 是 ROS1 catkin，0/9 有 CI，0/9 有 `.clang-tidy`，0/9 用 `target_include_directories`，0/9 用 `CMAKE_CUDA_ARCHITECTURES`。能学的是"位置和形式"，不是"内容"。

**最值得做的 3 条**（详见 §2，均为原创写法，不抄任何代码）：

1. **Catch2 增量替换裸 main**（`src/tests/test_action.cpp` 起步）。本机已装 Catch2 3.4.0（`/usr/lib/cmake/Catch2/`），零联网；先把 1 个 40 行测试改掉，拿到断言级输出与逐 `TEST_CASE` 的 ctest 条目。工作量 45 min。
2. **在 `src/CMakeLists.txt` 打开 `-Wall -Wextra` 并设置 `CMAKE_EXPORT_COMPILE_COMMANDS ON`**。这是他们唯一的质量开关（每个包都写 `-O3 -Wall -g`），同时让 clangd/clang-tidy 可用。工作量 15 min（修警告另算）。
3. **训练配置落盘**（`src/trainer.cpp` 的 `run()`）。借鉴他们"launch/YAML 文件即实验记录"的思想，把生效的 `TrainConfig` + 数据维度 + 参数量写进 `<out_dir>/config.json`。工作量 30 min。

**.clang-format 结论**：**不照搬**。全语料库唯一一份 `.clang-format` 是随 mockamap 带进来的（`BasedOnStyle: Mozilla` + `Standard: Cpp03` + Allman 大括号 + 默认 80 列）。实测在我们 40 个源文件上产生 **2960 处** replacement；换成 Google 基底 + 100 列只有 **209 处**。正确做法是自写 8 行放 `src/.clang-format`（§2.4）。

**许可证结论**：7 个 GPL-3.0、1 个 MIT（GCOPTER）、1 个无许可证（Fast-Drone-250，README 明确"严禁商用与转载"）。**只学目录/构建/文档结构，一行代码都不抄**（§5）。

## 1. 9 个仓库工程配置对比

| 仓库 | 许可证 | 构建系统 | 测试框架 | 参数管理 | CI | 格式/静态检查 |
|---|---|---|---|---|---|---|
| ego-planner | GPL-3.0 | catkin（`src/CMakeLists.txt` → ROS kinetic toplevel；7 个 ROS 包：plan_env / bspline_opt / path_searching / traj_utils / plan_manage / uav_simulator） | 无（vendored `uav_utils` 里有 1 个 `catkin_add_gtest`） | ROS param server；值写在 `plan_manage/launch/advanced_param.xml`，C++ 用 `nh.param("fsm/…")` | 无 | 仅 vendored `mockamap/src/.clang-format`（Mozilla/Allman）；无 tidy |
| ego-planner-swarm | GPL-3.0 | catkin（kinetic toplevel） | `drone_detect/test/test_drone_detector.cpp` 存在，但 CMake 里 `catkin_add_gtest` 整段被注释；6 个包调 `roslint_cpp()` | 同上（launch XML + `nh.param`） | 无 | 同上 |
| EGO-Planner-v2 | GPL-3.0 | **4 个独立 catkin workspace**（main/formation/tracking/interlaced，各自 `src/CMakeLists.txt` → noetic），仓库顶层无 CMake | 无（同一份代码的 4 份副本） | launch XML | 无 | 同上（副本） |
| Elastic-Tracker | GPL-3.0 | catkin（**仓库内无顶层 `src/CMakeLists.txt`**，要求整仓放进 `ws/src` 再 `catkin_make`） | vendored `uav_utils` 1 个 gtest | launch XML + `<rosparam>` 加载相机/仿真 YAML | 无 | 无 |
| Fast-Drone-250 | **无 LICENSE**；README 写"严禁商用与转载，版权归 FASTLAB" | catkin（无顶层 CMake，放进 ws/src） | 无 | launch XML（`advanced_param.xml` / `advanced_param_exp.xml`）+ `nh.param` | 无 | 仅 mockamap 副本 |
| Fast-Racing | GPL-3.0 | catkin（`src/CMakeLists.txt` → melodic toplevel） | 仅第三方 rpclib 自带 tests；项目本身无 | launch XML + YAML | 无 | 仅第三方 rpclib 的 LLVM 风格 `.clang-format` |
| GCOPTER | **MIT** | catkin 包（`gcopter/` 与 `map_gen/`，放进 ws/src） | 无 | ROS param → `Config` 构造函数里逐项 `nh_priv.getParam()` | 无 | 仅 mockamap 副本 |
| Implicit-SDF-Planner | GPL-3.0 | catkin（noetic toplevel）+ `build.sh`（先 `make` 预编译的 `lmbm.so`，再 `catkin_make`）+ vendored libigl（MPL2） | 无 | **每个场景一份 YAML**（`plan_manager/config/config_*.yaml`），经 `<rosparam command="load">` 进参数服务器，再由 `utils/config.hpp` 的 `LOAD_PARAM` 宏 `getParam` | 无 | 仅 mockamap 副本 |
| Swarm-Formation | GPL-3.0 | catkin_tools（`.catkin_workspace`）+ 顶层 `Dockerfile`（ros:noetic）+ 3 行 `Makefile`（docker build/run） | vendored `uav_utils` 1 个 gtest | launch XML + YAML（`<rosparam>`） | 无 | 仅 mockamap 副本 |

**横切统计**

- 264 个 `CMakeLists.txt`、245 个 `package.xml`；`.github/`、GitLab CI：0。
- `.clang-tidy`：0；Doxygen：0（唯一 Doxyfile 在 rpclib 第三方里）。
- `target_include_directories`：0（他们自己的代码）；`target_link_libraries(... PUBLIC/PRIVATE)` 仅 3 处，全在 Implicit-SDF；`install(TARGETS)` 基本被注释。
- CUDA：全部用 legacy `find_package(CUDA)` + `CUDA_ADD_LIBRARY` + `CUDA_NVCC_FLAGS` 里手写 `-gencode`，写死 `sm_61`（Elastic）、`sm_61`/`sm_60`（Swarm）、`sm_75`（ego-planner）、`sm_60` 且默认关闭（Fast-Drone-250）。**没有一处用 `CMAKE_CUDA_ARCHITECTURES`，也没有 `-arch=native`**。
- 测试：活跃的 gtest 用例全语料库只有 **1 个源文件**（vendored `uav_utils/src/uav_utils_test.cpp`，在 4 个仓库里各复制一份）；ego-planner README 里说的 "Compiling tests passed on ubuntu 16.04/18.04/20.04" 指的是"能编译"，不是单元测试。
- 头文件：抽样 planner 目录 **include guard 26 个 vs `#pragma once` 3 个**；没有 PIMPL、没有前置声明模式；头文件直接 include ROS/PCL/Eigen 重依赖。

### 分题速答

1. **CMake 组织**：按 ROS 包拆 target——每个包 1 个库 + 1~2 个 node 可执行文件（如 `plan_env` 库 + `ego_planner_node`/`traj_server`），这是 catkin 的强制粒度，不是设计。`target_include_directories` 用法：**没有**；全是全局 `include_directories()` + `${catkin_LIBRARIES}` 一把梭。唯一像现代 CMake 的是 Implicit-SDF 为预编译 `lmbm.so` 写的 `add_library(lmbm SHARED IMPORTED)` 和 `PUBLIC` 链接。**我们的 `uavrgb_core` + 每测试一 target + `PUBLIC include` + `PRIVATE` 链接已经比他们规范，不要改回去。** CUDA arch 见 §3.2：保持我们 `project()` 前 `set(CMAKE_CUDA_ARCHITECTURES 75 ... FORCE)` 的写法。
2. **代码规范**：见 §2.4 与 §3.5。他们只有一份过时的 mockamap 配置，无 `.clang-tidy`。
3. **测试**：见 §2.2 与 §3.4。他们几乎没有测试文化，gtest 经 catkin 提供；本机没装 gtest，装了 Catch2 3.4.0。结论：换 Catch2，但只增量换。
4. **参数管理**：见 §2.3 与 §3.1。他们用 ROS 参数服务器（launch XML / YAML），优点是"配置文件即实验记录"，缺点是全局隐藏状态、`getParam` 返回值普遍不检查。我们的 `TrainConfig` + CLI 已更好；最小改进是把生效配置落盘，不是引入 YAML 框架。
5. **头文件组织**：见 §3.7。guard 为主、无 PIMPL、头文件偏重；我们 `#pragma once` + `include/uavrgb/` 前缀布局已经是对的。
6. **许可证**：见 §5。

## 2. 值得借鉴的（按价值排序）

### 2.1 打开编译警告 + 生成 compile_commands

- **他们的做法**：每个包 CMake 里都写 `set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")`；`.vscode/tasks.json` 里注释了 `-DCMAKE_EXPORT_COMPILE_COMMANDS=Yes`，`.vscode/c_cpp_properties.json` 指向 `build/compile_commands.json`。除此之外没有任何 lint 在跑。
- **我们怎么改**：`/root/claude-code-backup-20261028/code/uav_rgb/src/CMakeLists.txt`，在 `add_library(uavrgb_core ...)` 之前加：
  ```cmake
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wextra>")
  ```
  用生成器表达式是为了**不把 `-Wall` 传给 nvcc**（`.cu` 的 host 编译警告可后续用 `-Xcompiler` 单独处理）。不要抄他们"强制 Release"的做法——我们保留由 `-DCMAKE_BUILD_TYPE` 控制。
- **工作量**：改 3 行 10 min；跑一次构建看新警告 10 min；修警告数量未知（估计 0.5–2 h，单列，不算进"下周 1 小时清单"）。

### 2.2 Catch2 增量替换裸 main（不引入网络依赖）

- **他们的做法**：`catkin_add_gtest(...)`，依赖 ROS 自带的 gtest；活跃用例只有 vendored `uav_utils` 那 1 个。
- **本机现状**：`gtest` **未安装**（`/usr/include/gtest` 不存在）；`catch2 3.4.0-1build1` **已安装**，`/usr/lib/cmake/Catch2/Catch2Config.cmake`、`/usr/lib/libCatch2.a`、`/usr/lib/libCatch2Main.a` 齐全。`find_package(Catch2 3 REQUIRED)` 不需要联网。
- **我们怎么改**（只改 1 个文件 + CMake 4 行）：
  - `src/CMakeLists.txt`：加 `find_package(Catch2 3 REQUIRED)`；把 `test_action` 的链接改成 `target_link_libraries(test_action PRIVATE uavrgb_core Catch2::Catch2WithMain)`；加 `include(Catch)` 与 `catch_discover_tests(test_action)`。
  - `src/tests/test_action.cpp`：删掉 `main` 和 `check()` 样板，改成 `#include <catch2/catch_test_macros.hpp>` + `TEST_CASE("...", "[action]") { CHECK(...); }`（`Catch2WithMain` 自带 main）。
- **为什么值**：现在每个测试文件自带 ~25 行 `g_failures`/`check()` 样板，ctest 只看到 9 个"二进制级"结果；Catch2 给断言级失败信息、`--list-tests`/标签过滤，并让每个 `TEST_CASE` 成为独立 ctest 条目。
- **工作量**：首个 45 min；余下 8 个测试**不要一次性重写**（共约 1200 行，全改 4–6 h，收益不匹配）。规则：新测试一律 Catch2；老测试在下次因功能需要改动时顺手迁移。
- **风险**：Catch2 的 `CHECK` 对张量只能写布尔表达式，没有自定义 matcher；对现有"逐元素比 CPU/CUDA"的写法够用，但不会更漂亮。

### 2.3 把生效配置写进 run 目录（借"launch 文件即实验记录"）

- **他们的做法**：同一份 `advanced_param.xml` / `config_*.yaml` 既是运行参数又是实验记录，实验可复现靠"文件进了版本库"。GCOPTER 的 `Config` 构造函数、Implicit 的 `LOAD_PARAM(x) nh_priv.getParam(#x, x)` 都把参数集中在一个结构体里——这点是对的；但两者都**不检查 `getParam` 返回值**，缺 key 时静默保留 0，是反面教材。
- **我们怎么改**：`src/trainer.cpp` 的 `Trainer::run()`，在 `create_directories(cfg_.out_dir)` 之后写 `<out_dir>/config.json`：`epochs / batch / lr / max_grad_norm / seed / resume / data_path` + `ds.h / ds.w / ds.num_classes` + `num_params(model_)` + 时间戳。用 `std::ofstream` + 手写 JSON 字符串（项目里 `metrics_json` 已是这个风格），**不引 json 库**（虽然 jsoncpp 已装）。
- **工作量**：约 20–30 行，30 min；在 `test_trainer.cpp` 加一条"文件存在且含 `\"seed\":1234\"`"的断言 10 min。
- **为什么不做完整配置文件系统**：训练参数已由 `train_main.cpp` 的 CLI 覆盖，`TrainConfig` 有默认值，CLI 解析有异常处理，`trainer.cpp` 里有 `epochs<=0 / batch<=0` 的范围校验，比 ROS 参数服务器更好。真要文件配置时本机已有 `libyaml-cpp-dev 0.8.0`（无需联网），但为 8 个参数引 yaml-cpp 不划算；`key=value` 40 行解析即可。

### 2.4 自写 `.clang-format`（10 分钟，别抄他们的）

- **他们的配置全文**（`ego-planner/src/uav_simulator/mockamap/src/.clang-format`，9 个仓库里唯一的团队配置，且是随第三方 mockamap 带进来的）：
  ```yaml
  ---
  BasedOnStyle: Mozilla
  Standard: Cpp03
  AlignAfterOpenBracket: Align
  AlignConsecutiveAssignments: true
  AlignConsecutiveDeclarations: true
  AllowShortFunctionsOnASingleLine: false
  AllowShortIfStatementsOnASingleLine: false
  AllowShortLoopsOnASingleLine: false
  BreakBeforeBraces: Allman
  ...
  ```
- **实测差异**（clang-format 18.1.3，我们 `src/` 的 40 个源文件）：照搬这份 → **2960 处** replacement；`Google + ColumnLimit 100` → **209 处**（关掉 include 重排后 190 处，大头是 `.cu` 里超 100 列的行）。原因是 `Standard: Cpp03` 会误判 C++17 代码，Allman + 默认 80 列与现有 2 空格/行尾大括号/100 列风格完全相反。
- **我们怎么改**：新建 `src/.clang-format`（只作用于源码树，不碰仓库根的其它东西）：
  ```yaml
  BasedOnStyle: Google
  Language: Cpp
  Standard: c++17
  ColumnLimit: 100
  DerivePointerAlignment: false
  PointerAlignment: Left
  AllowShortFunctionsOnASingleLine: Inline
  AllowShortCaseLabelsOnASingleLine: true
  ```
  `PointerAlignment: Left` 匹配现有 `const Tensor& x`；后两条是为了贴近现有写法、减少无意义 diff。是否加 `SortIncludes: Never` 看团队：加了更省 diff，但 include 顺序会长期不统一。
- **工作量**：建文件 10 min；`clang-format --dry-run --Werror` 看差异 5 min；**不要**立刻全量 `-i`（180 处纯格式 diff 会淹没 code review），改为"改动文件时格式化"。总 15 min。

### 2.5 一键检查脚本（补他们没有的 CI）

- **他们的做法**：0/9 有 CI；只有 Swarm-Formation 有 Dockerfile（拉 ros:noetic 大镜像，不适合我们）。
- **我们怎么改**：新建 `tools/check.sh`（10 行）：
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  cmake -S "$ROOT/src" -B "$ROOT/src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ROOT/src/build"
  ctest --test-dir "$ROOT/src/build" --output-on-failure
  ```
- **工作量**：15 min。本地提交前跑，零网络、零服务。可选加 `clang-format --dry-run --Werror` 检查 `git diff` 涉及的文件（但仓库目前不是 git repo，见 §6）。

### 2.6 README 结构（9/9 都有，我们缺）

- **他们的共同骨架**：标题 + 论文/视频链接 → "Quick Start within 3 Minutes"（3~5 条命令）→ 致谢 → 许可证/维护声明。GCOPTER 多一个 "Updates" 变更记录；Swarm-Formation 有目录和 Docker 快速开始。
- **我们怎么改**：新建根 `README.md`，5 节：项目一句话 / 依赖（本机已装 CUDA 12.8、OpenCV 4.6、Eigen 3.4）/ 构建与 ctest（直接抄 `CLAUDE.md` 的命令，别抄他们的文字）/ 训练 CLI 示例 / 目录说明。不写论文致谢。
- **工作量**：30 min。优先级低于前 5 条。

## 3. 不值得借鉴的

### 3.1 catkin / ROS 参数服务器 / launch XML

我们用 CMake + Ninja，不引入 ROS。他们的参数管理本质是**全局可变状态**：参数名是字符串（`"fsm/waypoint0_x"`），拼错不报错；`getParam` 返回 bool 但 GCOPTER 和 Implicit-SDF 都不检查，缺 key 时静默 0；没有范围校验。我们的 `TrainConfig` + `std::stoi` 异常 + `epochs<=0` 校验已经更好。**不要为了"有配置文件"把参数搬进 XML/YAML。**

### 3.2 他们的 CUDA 写法

legacy `find_package(CUDA)` + `CUDA_ADD_LIBRARY` + 手写 `-gencode arch=compute_61,code=sm_61`，是 2015 年的写法；Elastic-Tracker 写死 sm_61、Swarm-Formation 写死 sm_60（在当前仓库 checkout 里直接编会跑不了或需要改文件）。我们 `project()` 之前 `set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING ... FORCE)` 是正确做法，**保持不动**；也确认他们没人用 `-arch=native`，不存在可抄的点。唯一可留意的细节是他们设 `CUDA_PROPAGATE_HOST_FLAGS OFF`，我们不传播 host flags，不需要。

### 3.3 他们的 CMake 风格

- 全局 `include_directories()` / `link_directories()`，`${catkin_LIBRARIES}` 一把梭，没有 target 级依赖。
- 强制 `set(CMAKE_BUILD_TYPE "Release")`，无法调试。
- 同一文件里 `ADD_COMPILE_OPTIONS(-std=c++11)` 又 `-std=c++14`（后者覆盖前者），且用全局 flags 而不是 `CMAKE_CXX_STANDARD`。
- `file(GLOB ...)` 收头文件（Implicit-SDF `planner_algorithm/CMakeLists.txt`）。
- 我们的 CMake 已用 `target_include_directories(... PUBLIC include)`、`target_link_libraries(... PRIVATE ...)`、`CMAKE_CXX_STANDARD 17`，保持。

### 3.4 他们的测试文化

全语料库 1 个活跃 gtest 源文件，还是 vendored 工具包；`drone_detect` 的测试 CMake 被注释；README 把"能编译"叫 test。**没有可抄的测试框架配置**。我们 9 个 ctest（`CLAUDE.md` 里写 7 个，已过期，见 §6）已经超过他们。gtest 本机没装、且经 catkin 提供，不要为了对齐他们去装 gtest。

### 3.5 mockamap 的 `.clang-format`

见 §2.4：Mozilla/Allman/Cpp03/80 列，对我们产生 2960 处改动。不抄，但"每仓库应有格式配置"这个**存在性**值得学。

### 3.6 无 CI、无 `.clang-tidy`、无 Doxygen

没有模板可抄。Doxygen 对我们这种 20 文件、头文件注释已用中文说明的项目是负担；`.clang-tidy` 现在加会产生大量噪音，先有 `compile_commands.json` 和 `-Wall` 再说。**不建议本周引入 clang-tidy 规则集。**

### 3.7 PIMPL / 前置声明 / include guard

- 抽样 26 个 include guard vs 3 个 `#pragma once`，他们是旧风格；我们全用 `#pragma once`，保持。
- 没找到 PIMPL、没找到前置声明优化；他们头文件直接 include ROS/PCL/Eigen 重依赖，这是他们编译慢的原因之一。我们的头文件最大 146 行（`event.hpp`），保持轻量，不要学"头文件里塞实现"。
- 我们 `include/uavrgb/<name>.hpp` 的包名前缀布局与他们的 `include/plan_env/grid_map.h` 等价，已经是对的。

### 3.8 Docker / 预编译二进制 / vendored 大库

- Swarm-Formation 的 Dockerfile 基于 `osrf/ros:noetic-desktop-full`，需要联网拉 GB 级镜像；不适用。
- Implicit-SDF 的 `lmbm.so` 预编译库、Fast-Drone-250 的 `modified_realsense2_camera.zip` 来源与许可不可验证；Elastic-Tracker 的 `DecompROS`、libigl 全量 vendor 进来；这些都不要学。我们的 `third_party/` 应保持"能说清来源和版本"。

## 4. 下周就能做（每条 <1 小时）

1. **`src/CMakeLists.txt` 加 `-Wall -Wextra` + `CMAKE_EXPORT_COMPILE_COMMANDS ON`**（15 min），构建一次，把新警告记成 todo，不当场全修。
2. **`test_action.cpp` 换成 Catch2**，CMake 加 `find_package(Catch2 3 REQUIRED)` / `Catch2::Catch2WithMain` / `catch_discover_tests`（45 min）。跑 `ctest -N` 确认测试数增加。
3. **`src/trainer.cpp` 写 `<out_dir>/config.json`**，含 TrainConfig + 数据维度 + 参数量（30 min），`test_trainer.cpp` 加一条断言（10 min）。
4. **新建 `src/.clang-format`**（§2.4 的 9 行），只跑 `--dry-run` 看差异，不批量改（15 min）。
5. **新建 `tools/check.sh`** 一键 configure/build/ctest（15 min），以后改完代码先跑它。

（可选第 6 条：根 `README.md`，30 min，优先级最低。）

## 5. 许可证影响

| 仓库 | 许可证 | 对借鉴的影响 |
|---|---|---|
| ego-planner / ego-planner-swarm / EGO-Planner-v2 / Elastic-Tracker / Fast-Racing / Implicit-SDF-Planner / Swarm-Formation | GPL-3.0 | **代码一行都不能进 uav_rgb**。GPL 的传染性对静态库链接同样适用；"只抄一个小函数/一个类"也不行。目录结构、target 划分、README 章节、参数组织思想属于思想/结构，不受版权保护，可以照做。 |
| GCOPTER | MIT | 法律上可以复制并保留版权声明；但它是轨迹优化（MINCO），我们不需要，仍然不抄。注意：GPL 仓库里个别文件带 MIT 头（如 `minco.hpp`、`lbfgs.hpp`），若将来真要用，从 MIT 上游取，不要从 GPL 仓库取。 |
| Fast-Drone-250 | 无 LICENSE；README 明确"严禁商用与转载，版权与最终解释权归 FASTLAB" | **默认全部权利保留**，比 GPL 更严。连"参考其参数默认值"都要谨慎；只看了结构和参数组织方式。 |

本报告中的所有 CMake/YAML/脚本片段均为按通用工程惯例原创，未复制任何仓库的代码或注释文本。

## 6. 明确没读的部分 / 不确定项

- 264 个 `CMakeLists.txt` 只精读了约 20 个（各仓库顶层、ego-planner 的 4 个 planner 包、带 CUDA 的 `local_sensing`、GCOPTER、Implicit-SDF 的 planner 包）；其余按文件名/关键字抽样。245 个 `package.xml` 未逐个读。
- 未读源码逻辑（算法、`.cpp` 实现），只读了与工程配置相关的部分：CMake、头文件开头、参数读取处、测试 CMake 段。未读 PDF、图片、视频、`purchase_list.xlsx`、固件、zip。
- **未编译、未运行任何东西**（按要求）；所有"实测"仅指 `clang-format --dry-run` 的 replacement 计数（只读，不写文件）。
- EGO-Planner-v2 的 4 个 workspace 只抽查了 `main_ws`，其余按"同一份代码的副本"处理，未逐文件比对差异。
- README 提到的 ROS2 分支不在当前 checkout 里，未验证。
- 第三方依赖（rpclib、AirLib、libigl、VINS-Fusion、realsense-ros）的内部构建细节未读，只记录其存在。
- **`uav_rgb` 当前不是 git 仓库**（`git status` 报 not a git repository）。不确定这是备份目录的有意状态还是遗漏；若是遗漏，版本控制本身比本报告任何一条建议都值得先做。
- `CLAUDE.md` 说"当前 7 项" ctest，但 `src/CMakeLists.txt` 实际有 9 个 `add_test`（多了 `test_trainer`、`test_overfit`），文档已过期，建议顺手改掉。
