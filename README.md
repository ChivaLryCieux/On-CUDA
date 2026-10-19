# CUDA 学习项目：矩阵乘法与卷积优化

在 **RTX 2080 SUPER（图灵 SM 7.5，48 SM，8GB）+ CUDA 13.4 + VS2026 + Nsight 2026.3** 上，
按「写对 → 改快 → 解释为什么快」的循环，把 CUDA 优化的主干走一遍。

**所有 kernel 都经过 CPU 参考实现校验**，基准表里的 `relL2 err` 列就是正确性证明。
本项目实际调试中修掉了 7 个真实 bug，全部记录在下面「踩坑记录」里。

---

## 目录结构

```
CMakeLists.txt                 构建配置（已固定 sm_75）
src/common/
  cuda_utils.h                 CUDA_CHECK、GpuTimer（事件计时）、确定性随机填充
  reference.h / .cu            CPU 参考实现 + 误差比较
src/matmul/
  matmul.h / .cu               v0 naive / v1 tiled / v2 reg-tile / v3 WMMA
  bench_matmul.cu              基准 + 正确性校验 + cuBLAS 参照
  debug_mm.cu                  调试器：小规模跑单个实现，打印错误元素分布
src/conv2d/
  conv.h / .cu                 v0 naive / v1 smem halo / v2 im2col+GEMM / v3 深度可分离
  bench_conv.cu                基准 + 正确性校验
  debug_conv.cu                调试器：隔离 im2col / GEMM / bias
scripts/
  build.ps1                    CMake + Ninja 构建（常规终端用）
  build_direct.ps1             直接调 nvcc 构建（沙箱内用，见下）
  ncu_profile.ps1              Nsight Compute 单 kernel 剖析
  nsys_profile.ps1             Nsight Systems 时间线剖析
smoke_test.cu                  工具链冒烟测试（已验证通过）
```

---

## 构建与运行

```powershell
# 方式 A：直接 nvcc（在受限沙箱里也能用，产出到 bin\）
powershell -ExecutionPolicy Bypass -File scripts\build_direct.ps1

# 方式 B：CMake + Ninja（需要能正常创建子进程管道的终端）
powershell -ExecutionPolicy Bypass -File scripts\build.ps1

# 方式 C：VS2026 直接 File > Open > CMake... 选本目录
#   好处：可以在 device 代码里下断点，单步调试 kernel

# 运行
.\bin\bench_matmul.exe                          # 默认 1024x1024x1024
.\bin\bench_matmul.exe 2048 2048 2048
.\bin\bench_conv.exe                            # C=64 224x224 OC=64 3x3
.\bin\bench_conv.exe 3 512 512 8 7 3 2          # 首层风格：7x7 stride 2
.\bin\bench_conv.exe 256 56 56 256 3 1 1        # 深层风格：256 通道
.\bin\bench_conv.exe --noverify ...             # 跳过慢速 CPU 校验

# 调试器（定位数值错误用，比盯着代码看快得多）
.\bin\debug_mm.exe    64 64 64 regtile
.\bin\debug_conv.exe
```

**每次改动后跑这个**（回归门禁）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1            # 全量：构建+跑分+校验
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -SkipBuild # 只跑分
```

它会回答两个问题：**每个 kernel 还正确吗**（relL2 对 CPU 参考）、**有没有变慢**
（GFLOP/s 对基线）。正确性失败是硬错误（exit 1），性能回退只警告——
GPU 频率和主机负载会让数字漂移，回退 10% 先重跑一次再信。

基线在 [scripts\verify.ps1](scripts/verify.ps1) 顶部的 `$baselines` 表里。
**每次你真的把某个 kernel 调快了，就去更新那张表**，否则门禁只会一直喊狼来了。

---

## 实测结果

### 矩阵乘法（1024×1024×1024，FP32）

| 实现 | ms | GFLOP/s | 加速比 | relL2 |
|---|---|---|---|---|
| v0 naive | 2.76 | 778 | 1.0x | 6.3e-08 |
| v1 tiled 16×16 | 1.72 | 1248 | 1.6x | 6.3e-08 |
| v1 tiled 32×32 | 1.65 | 1303 | 1.7x | 6.3e-08 |
| v2 reg-tile 64×64 | 0.62 | **3463** | 4.5x | 6.3e-08 |
| v3 wmma fp16 | 0.66 | 3231 | 4.2x | 2.2e-05 |
| cuBLAS（库参照） | 0.22 | 9891 | 12.7x | 4.7e-07 |

参考基线：FP32 峰值 ≈ 11.2 TFLOP/s，HBM ≈ 496 GB/s。

**v3 没跑过 v2，这是真实现象，不是 bug。** 两个原因：累加器依赖链
（每个 warp 的 4 次 `mma_sync` 全累加进同一个 fragment，无法并行）
和 32KB shared memory 限制了 occupancy。标准解法是用多个 accumulator
fragment 打断依赖链——留作练习。

### 卷积（C=64 224×224 OC=64 3×3，3.699 GFLOP）

| 实现 | ms | GFLOP/s | 加速比 | relL2 |
|---|---|---|---|---|
| v0 naive direct | 8.85 | 418 | 1.00x | 6.6e-08 |
| v1 shared-memory | 8.01 | 462 | 1.11x | 6.6e-08 |
| v2 im2col + GEMM | 1.48 | **2493** | 5.96x | 8.6e-08 |
| v3 深度可分离 | 0.30 | 12164 | — | 7.4e-08 |

v3 是**不同算子**（0.469 GFLOP，FLOPs 少 7.9 倍），所以单独按「等 FLOPs 折算」比较：
折算基线 1.12 ms，实测 0.30 ms，等效 3.7x。

### 两个反直觉的实测结论

**1. shared memory 版本几乎没用（1.11x）。** 因为 `grid.z = N*OC`，
每个输出通道都把同一块输入重读一遍——shared memory 只在单个 block 内复用，
跨 oc 的复用完全没吃到。真正的解法是 im2col，或者让一个 block 算多个 oc。

**2. im2col 不是永远更快。** 换成 7×7 stride 2 的首层形状（0.154 GFLOP）后：

| 实现 | ms | GFLOP/s | 相对 naive |
|---|---|---|---|
| v0 naive | 0.224 | 689 | 1.00x |
| v1 smem | 0.249 | 618 | 0.90x |
| v2 im2col | 0.527 | 293 | **0.42x** |

im2col 要把 (C·KH·KW)×(OH·OW) 的矩阵**物化**到显存，小问题上这笔写入和回读
开销直接吞掉 GEMM 的收益。工程上的教训：**选算法要看问题规模，不是看哪个"更先进"**。

---

## 各级优化的原理

| 版本 | 核心思想 | 关键指标变化 |
|---|---|---|
| v0 naive | 一线程一输出，零复用 | 全局访存量 ∝ FLOPs，永久 memory-bound |
| v1 tiled | 块内 A/B tile 进 shared memory | 每个全局元素复用 TILE 次 |
| v2 reg-tile | 每线程算 4×4 微 tile，累加器驻留寄存器 | 每线程算术强度 1 → 8，最大的一跳 |
| v3 WMMA | FP16 输入 + Tensor Core，FP32 累加 | Tensor Core 利用率 0 → 拉满 |

图灵 Tensor Core 只支持 FP16/INT8，**没有 TF32 和 BF16**，所以 v3 是这块卡
唯一的张量核心路线。FP16 带来的精度损失直接从 relL2 列看出来（2.2e-05 vs 6.3e-08）。

---

## Nsight 工作流

两个工具分工不同，别混用：

- **Nsight Compute（`ncu`）— 单 kernel 显微镜**：SM 利用率、访存吞吐、occupancy、
  Speed Of Light、Roofline。回答「这个 kernel 为什么慢」。
- **Nsight Systems（`nsys`）— 系统级时间线**：kernel 启动间隙、memcpy 与计算
  是否重叠、GPU 有没有闲着等 CPU。回答「流水线为什么慢」。

```powershell
# 看 v2 reg-tile 的内存吞吐 vs 算力
powershell -ExecutionPolicy Bypass -File scripts\ncu_profile.ps1 `
    -Exe ..\bin\bench_matmul.exe -Kernel RegTile

# 看整条时间线上 kernel 之间有没有气泡
powershell -ExecutionPolicy Bypass -File scripts\nsys_profile.ps1 `
    -Exe ..\bin\bench_conv.exe
```

> **Windows 上 ncu 必须以管理员身份运行**，否则报
> `Launching the target application failed`——那是权限问题，不是代码问题。
> 本项目在非提权会话下实测无法跑 ncu，所以上面的命令请在管理员终端里执行。

**读报告的顺序**：先看 Roofline 图确认是算力受限还是带宽受限（方向错了怎么优化
都没用），再看 Achieved Occupancy 和 Memory Throughput，最后才看具体指标。

**练习节奏**：每改一版 → 跑 bench 看 GFLOP/s → `ncu` 看哪个指标动了 → 解释变化
→ 再改。把每次的指标抄下来，你会得到一张自己的优化曲线。

---

## 踩坑记录（本项目实际修掉的 7 个）

**1. 中文注释导致编译失败。** 中文 Windows 代码页 936 下 `cl` 按 GBK 读 UTF-8
源文件，字节错位会把第一行注释撕成多行，导致后续全部行号偏移、`<<<>>>` 语法解析
失败。本项目所有 `.cu/.h` 一律纯 ASCII 注释。

**2. shared memory tile 装载时行/列索引写反。** `As` 声明为 `[BK][BM]` 即 `[k][m]`，
装载时 `r = idx/BM` 是 k 索引、`c = idx%BM` 是 m 索引，但取值写成了
`A[(blockRow + r) * K + (k0 + c)]`——把 r 当行号、c 当列号，等于把 A 的 tile 转置了。
这个 bug 极难肉眼发现：结果不是乱码，而是**系统性偏小**（relL2 = 0.086，约 1/√K 收缩），
因为转置后仍有部分乘积侥幸正确。定位方式：用小规模跑 `debug_mm`，看错误是否随 K 按 1/√K 缩放。

**3. 修 bug 时把合并访存修丢了。** 改成 `A[(blockRow + c) * K + (k0 + r)]` 后数值对了，
但连续线程读的是跨行跳 K 的地址，性能从 3704 掉到 2321 GFLOP/s。正解是改变
线程→元素的映射（`idx = m*BK + k`）而不是改索引含义，让连续线程仍然读连续内存。

**4. WMMA fragment 布局不匹配。** `load_matrix_sync` 要求 A 在 shared 里是 **M×K 行主序**，
我存成了 `[K][M]`，fragment 读到的就是乱的。另外把 shared 的 k 跨度从 16 加到 64
（`WSK=64`）才能让全局加载达到 128 字节合并，而不是 32 字节碎片。

**5. `too many resources requested for launch`。** WMMA 的加载循环 16 路展开后
寄存器涨到 140，512 线程就要 71680 > 每 SM 65536 的上限，启动直接失败。
**危险的是失败后 dOut 里残留的是上一个 kernel 的结果**，看起来"又快又对"。
去掉 `#pragma unroll` 后降到 42 寄存器。用 `nvcc -Xptxas -v` 可以直接看每个
kernel 的寄存器和 shared memory 用量。

**6. 基准测试没检查启动错误。** 因此第 5 条被掩盖了。现在每个 `bench` 在 warmup 后
都调用 `CUDA_CHECK(cudaGetLastError())`——**这是 CUDA 基准测试的必备项**。

**7. 拿错了参考实现。** v3 深度可分离是不同算子，却被拿去和标准卷积的 CPU 参考比，
得到 relL2 = 0.185 的假错误。修正后 7.4e-08。

另外两条工程环境问题：

- **nvcc 不会自己找 cl.exe**，必须加载 vcvars64；VS 捆绑的 Ninja 也不在 PATH 上。
- **在 `cmd /c` 链里用 `set "PATH=...;%PATH%"` 会被 cmd 的引号规则破坏**，
  静默清空 PATH。要在 PowerShell 侧设 `$env:PATH` 让 cmd 继承。

---

## 接下来可以做的实验

- [ ] v3 用多个 accumulator fragment 打断 `mma_sync` 依赖链，看能否超过 v2
- [ ] v2 的 Bs 读取有 2-way bank conflict，改线程→数据映射消掉它
- [ ] v2 加双缓冲，用 `cuda::memcpy_async`（`cp.async`）让访存与计算重叠
- [ ] 让 conv 的一个 block 算多个 oc，吃掉跨 oc 的输入复用（应该能大幅超过 1.11x）
- [ ] 给 conv 加多 stream，用 `nsys` 看 depthwise 和 pointwise 能否重叠
- [ ] 用 CUDA Graph 合并 conv 的多次 kernel 启动，消掉启动开销
- [ ] 把 bench 的计时改成 nsys 的 kernel duration，和事件计时交叉验证
- [ ] 试 `bench_conv 512 28 28 512 3 1 1`，看 v1/v2 的拐点在哪里
