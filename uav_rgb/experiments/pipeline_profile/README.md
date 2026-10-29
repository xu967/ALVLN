# 整网训练步性能探针

目的：复用实际 CNN/读取器/CUDA 算子，分开测量读数据、前向、loss/grad、反向、裁剪、Adam。
使用无权 CE，先热身2步，再测指定步数。不是导航成绩或完整 trainer 吞吐：不含epoch验证、
checkpoint、accuracy计算、JSONL写盘和加权模式的额外无权指标计算。

## 构建与复跑

在 uav_rgb 执行：

```bash
source env.sh
cmake -S experiments/pipeline_profile -B build/pipeline-profile -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/pipeline-profile --target pipeline_profile --parallel 8
./build/pipeline-profile/pipeline_profile "$DATA_ROOT/uav_rgb/pipeline-profile-20261029/synthetic.uavr" 32 8
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$DATA_ROOT/uav_rgb/profile-new" \
  ./build/pipeline-profile/pipeline_profile \
  "$DATA_ROOT/uav_rgb/pipeline-profile-20261029/synthetic.uavr" 32 8
```

也可传任何符合当前读取器合同的 UAVR，train 样本数必须不小于 batch；使用新报告文件名，
避免覆盖旧证据。输出一行 JSON。当前阶段计时依赖现有算子同步返回，未来异步执行器需要
改用 CUDA event；不能沿用 host 分段计时解释异步 kernel 时长。

## 2026-10-29 本机结果

RTX2080Ti，CUDA12.8，FP32，120×160，CNN 100388参数，合成随机图像256张，标签0..3均衡。
数据只有train，性能探针不做训练集回看。各batch单独初始化相同模型seed、2步热身+4步计时。
不同batch的更新序列不相同，因此以下只比较吞吐，不比较loss或学习效果。

| batch | 单步ms | 样本/s | 读取ms | 前向ms | 反向ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.10 | 39.84 | 0.054 | 7.53 | 11.55 |
| 8 | 167.38 | 47.80 | 0.620 | 60.39 | 93.38 |
| 32 | 573.77 | 55.77 | 2.383 | 207.70 | 336.94 |
| 128 | 2161.91 | 59.21 | 11.282 | 781.69 | 1297.48 |

batch32→128只增加约6.2%吞吐，不能靠增大batch消除结构性瓶颈。
单轮短测且共享服务器存在抖动，不是置信区间或严格性能排行榜。

另一次Nsight捕获batch32，热身后8步共约4.045s，单步505.66ms、63.28样本/s。
它比前一轮快，不应解释为开启profiler带来加速；负载/内存分配抖动和采样步数不同。
下面所有API/时间线统计来自同一次8步捕获，避免跨运行相减。

| 每步项目 | 数量/体积 |
| --- | ---: |
| cudaMalloc / cudaFree | 各121次 |
| cudaMemcpy | 146次 |
| cudaDeviceSynchronize / kernel launch | 各47次 |
| H2D + D2H | 766375828 bytes，约766.4 MB |
| 输入float32图像大小（对照） | 7372800 bytes，约7.37 MB |

8步CUDA时间线：kernel区间并集814.79ms，copy区间并集1304.19ms，
kernel+copy+memset区间并集2119.95ms，首次GPU事件到末次GPU事件4033.85ms。
即**本进程被捕获的GPU kernel只覆盖时间窗约20.2%**，约47.4%的窗口没有本进程被捕获的
GPU kernel/copy/memset。这不是全设备空闲率，也不是SM占用率：其他进程、调度及追踪开销未排除。
CPU CUDA API耗时包含等待GPU的时间，不能再与kernel时间直接相加。

kernel时间中：dW占62.5%，db占28.3%，合计90.8%。Nsight实际网格确认：

| kernel | 网格block数 | 每次平均ms |
| --- | ---: | ---: |
| 第一层卷积dW | 2 | 47.74 |
| 第二层卷积dW | 18 | 10.73 |
| 第三层卷积dW | 72 | 5.18 |
| 三层db合并统计 | 每层1 | 9.62 |

当前GPU有68个SM；第一层dW将N×H×W放在单线程循环中，batch增加不扩大网格。
db每通道一个线程也只有1block。即使消除CPU空隙，仍需改归约并行粒度。

## 改造顺序与验收

1. 保留host Tensor路径作参考；新增设备buffer唯一所有权、设备view、执行stream及
   按最大batch预分配的workspace。权重、Adam m/v、激活和梯度常驻设备，reshape仅改view。
   先一个compute stream保证依赖，稳态每步不再cudaMalloc/Free，不再逐层H2D/D2H。
2. 将dW/db的N×H×W分块并行归约，partial sums再归约；处理FC点积和梯度norm单地址
   原子热点。新归约改变求和顺序，要对比CPU参考、梯度和优化器更新，并测完整训练步收益。
3. reader改独立文件句柄或pread/mmap；两个有界pinned buffer，CPU准备下一batch，
   copy stream通过event与compute stream交接。当前热缓存读取占比低，预取应排在前两项后。
4. loss/grad/accuracy融合、平铺参数批量Adam、必要的kernel融合/Graph最后实测决定。

多线程不能并发更新同一model/Adam；裁剪须等待全部梯度；checkpoint须在optimizer完成后
制作独立一致快照。不要只删同步：临时显存生命周期和执行错误检查必须一起改。

## 证据与限制

数据、各batch原始JSON、source/binary SHA256、Nsight report/SQLite/CSV和区间统计：
`/root/rivermind-data/uav_rgb/pipeline-profile-20261029/`。
训练核心未被本探针修改；关联trainer当前还有其他会话改动，捕获以已链接二进制及源码hash为准。
未进行resident重写、kernel优化、GPU硬件occupancy采集、实际数据集冷盘测试或多进程竞争测试。
