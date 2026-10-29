# reader / run owner CPU 验证记录

- 时间：2026-10-29 17:54 CST
- 环境：Linux x86_64，GCC 13.3，C++17；`nvidia-smi -L` 检测到 RTX 2080 Ti，本实验仅使用 CPU。
- 产物目录：`/root/rivermind-data/uav_rgb/parallel_training/20261029-1734/cpu-reader-owner/`
- 数据：测试现场生成 UAVR，7张3×8×16图像，seed987；测试结束清理临时数据。

## 编译

在 `code/` 执行，两个命令都退出0，无警告输出：

```bash
source uav_rgb/env.sh
mkdir -p "$DATA_ROOT/uav_rgb/parallel_training/20261029-1734/cpu-reader-owner"
g++ -std=c++17 -O2 -Wall -Wextra -I uav_rgb/src/include \
  uav_rgb/src/tests/test_batch_reader.cpp uav_rgb/src/batch_reader.cpp \
  uav_rgb/src/dataset.cpp uav_rgb/src/tensor.cpp -Wl,--wrap=pread \
  -o "$DATA_ROOT/uav_rgb/parallel_training/20261029-1734/cpu-reader-owner/test_batch_reader"
g++ -std=c++17 -O2 -Wall -Wextra -I uav_rgb/src/include \
  uav_rgb/src/tests/test_run_owner.cpp uav_rgb/src/run_owner.cpp \
  -o "$DATA_ROOT/uav_rgb/parallel_training/20261029-1734/cpu-reader-owner/test_run_owner"
```

## 首次执行：暴露 stat 指纹边界

命令：

```bash
"$DATA_ROOT/uav_rgb/parallel_training/20261029-1734/cpu-reader-owner/test_batch_reader"
"$DATA_ROOT/uav_rgb/parallel_training/20261029-1734/cpu-reader-owner/test_run_owner"
```

实际输出，reader 退出1、owner 退出0：

```text
FAIL: 同长度原地修改拒绝
test_batch_reader: FAIL
test_run_owner: PASS
```

原因：生成文件后立即原地写入，size/mtime/ctime 在当前文件系统同一 tick 内可能相同。
这不是内容 hash 检查，不能承诺检出全部同长度修改。未改变生产检查逻辑或掩盖失败；
测试改成 pwrite 后显式 `utimensat` 推进 mtime 一秒，确定性验证元信息变化检测，未使用 sleep。
接口同步注明局限，正式 run manifest 仍需单独计算 dataset 内容 hash。

## 修订后执行

重新编译后 `test_batch_reader` 实际输出并退出0：

```text
test_batch_reader: PASS
```

reader 元数据随后改用 unique_ptr 持有，确保 noexcept move 不移动可能抛异常的 ifstream；
重新编译并执行相同测试，实际输出仍为 `test_batch_reader: PASS`，退出0。
owner 代码从首跑后未修改，未无理由重复运行。

已覆盖：生产 pread 的 EINTR、连续17字节短读、EOF和系统错误；乱序/重复样本逐 bit 对照；
目标容量/下标错误先校验；有效范围外哨兵；move；路径替换、截断、mtime变化；
单 writer 双进程竞争、canonical 别名、非空目录拒绝、异常/进程退出释放锁，以及实际 exec 的 CLOEXEC。

## 末次二进制 SHA256

```text
86f9599affcc6c2d0ca13a3a679e36f5c0abf7ebd10a3c2fe844fb34c085a8bf  test_batch_reader
4d25c0c13fc7d2a7eda09edde870e186ad0e7ceb1370c715883bd6eef22dae71  test_run_owner
```

未执行：GPU、性能测试、NFS和训练接线；本记录不表示完整并行训练方案已经验收。
