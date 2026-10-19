# 实验记录

每次调优后追加一条。格式：**改了什么 → 指标怎么变 → 为什么**。
只记有数据支撑的结论，不记猜测。

---

## E0 — 基线建立（2026-10-19）

**环境**：RTX 2080 SUPER（图灵 SM 7.5，48 SM，8GB），CUDA 13.4.92，VS2026 MSVC 14.51，
驱动 591.86。参考基线：FP32 峰值 ≈ 11.2 TFLOP/s，HBM ≈ 496 GB/s。

**矩阵乘法 1024³（FP32）**：

| 实现 | ms | GFLOP/s | relL2 |
|---|---|---|---|
| v0 naive | 2.79 | 770 | 6.3e-08 |
| v1 tiled 16×16 | 1.74 | 1233 | 6.3e-08 |
| v1 tiled 32×32 | 1.65 | 1303 | 6.3e-08 |
| v2 reg-tile 64×64 | 0.67 | 3215 | 6.3e-08 |
| v3 wmma fp16 | 0.69 | 3135 | 2.2e-05 |
| cuBLAS | 0.24 | 9090 | 4.7e-07 |

**卷积 C=64 224² OC=64 3×3（3.699 GFLOP）**：

| 实现 | ms | GFLOP/s | relL2 |
|---|---|---|---|
| v0 naive | 8.51 | 435 | 6.6e-08 |
| v1 shared-memory | 8.40 | 440 | 6.6e-08 |
| v2 im2col + GEMM | 1.50 | 2461 | 8.6e-08 |
| v3 depthwise-separable | 0.32 | 11572 | 7.4e-08 |

v3 是不同算子（0.469 GFLOP，FLOPs 少 7.9 倍），按等 FLOPs 折算基线 1.08 ms，等效 3.4x。

---

## E1 — v2 reg-tile：shared memory tile 装载索引写反

**现象**：relL2 = 8.57e-02，4096 个元素全错。

**原因**：`As` 声明为 `[BK][BM]` 即 `[k][m]`，装载时 `r = idx/BM` 是 k 索引、`c = idx%BM`
是 m 索引，但取值写成 `A[(blockRow + r) * K + (k0 + c)]`——把 r 当行号、c 当列号，
等于把 A 的 tile 转置了。

**为什么难发现**：结果不是乱码，而是**系统性偏小**（约 1/√K 收缩），因为转置后仍有
部分乘积侥幸正确。定位方式：小规模跑 `debug_mm`，看误差是否随 K 按 1/√K 缩放。

**教训**：shared memory tile 的"逻辑索引"和"装载索引"必须分别命名，混用必错。

---

## E2 — 修 E1 时把合并访存修丢了

**现象**：数值正确了，但 v2 从 3704 掉到 2321 GFLOP/s。

**原因**：改成 `A[(blockRow + c) * K + (k0 + r)]` 后，连续线程读的是跨行跳 K 的地址。

**解法**：不改索引含义，改**线程→元素的映射**：`idx = m*BK + k`（m 主序），
让连续线程读连续的 k。修复后 3215 GFLOP/s（较最初 3704 的低点仍有差距，见待办）。

**教训**：正确性 bug 和性能 bug 要分开修。一次只改一件事，否则无法归因。

---

## E3 — v3 WMMA：fragment 布局不匹配

**现象**：relL2 = 7.61e-02（E1/E2 修完后仍然错）。

**原因**：`load_matrix_sync` 要求 A 在 shared 里是 **M×K 行主序**，代码存成了 `[K][M]`。
同时把 shared 的 k 跨度从 16 加到 64（`WSK=64`），让全局加载达到 128 字节合并而非 32 字节碎片。

**教训**：WMMA 的 fragment 布局是硬件约定，不是"能跑就行"。声明 shared 数组时就要
按 fragment 的视角倒推布局。

---

## E4 — v3 WMMA：寄存器超限导致启动静默失败

**现象**：报告 0.0024 ms / 905652 GFLOP/s，而 relL2 与 v2 完全相同。

**原因**：加载循环 16 路 `#pragma unroll` 把寄存器推到 140，512 线程需要 71680 >
每 SM 65536 上限，`too many resources requested for launch`。失败后 `dC` 里残留的是
上一个 kernel（v2）的结果，所以看起来"又快又对"。

**诊断**：`nvcc -Xptxas -v` 直接看每个 kernel 的寄存器和 shared memory 用量。
去掉 `#pragma unroll` 后 140 → 42 寄存器。

**教训**：**基准测试必须在 warmup 后调用 `cudaGetLastError()`**。否则失败的启动
会伪装成极快的正确结果。这是 CUDA benchmark 的必备项，不是可选项。

---

## E5 — 卷积 v3：拿错了参考实现

**现象**：relL2 = 8.47e-01（一度 1.85e-01）。

**原因**：`bench` 硬编码拿 `hRef`（标准卷积的 CPU 参考）去比，而 depthwise-separable
是**不同算子**、不同权重，本应有自己的参考 `hRef2`。

**修复**：`bench` 改为接收 `const float* ref` 参数。修复后 7.4e-08。

**教训**：验证不同算子时，参考实现必须和被测算子同源。

---

## E6 — 陈旧二进制导致假错误

**现象**：v2 relL2 = 1.44e-01，但 `debug_conv` 单独跑同一函数是 8.58e-08。

**原因**：`bench_conv.exe` 链接的是修复前的 `matmul.cu`，而 `bench_matmul.exe` 已重建。
改了共享源文件后只重建了部分目标。

**教训**：多目标项目里，改共享 `.cu` 后必须全量重建，或让构建系统正确处理依赖。

---

## E7 — 卷积 v1：把 filter taps 放进 shared memory

**改动**：原来每个线程在 `(kh, kw)` 循环里从全局读 filter，即每线程 C×KH×KW 次全局加载。
改为 cooperative load 到 shared memory 的 `fw[KH*KW]` 区域。

**结果**：462 GFLOP/s，仅 1.11x（改前 1.07x）。

**结论（重要）**：收益极小，因为**真正瓶颈不是 filter 加载**，而是 `grid.z = N*OC`
导致每块输入被重读 OC=64 次——shared memory 只在 block 内复用，跨 oc 的复用完全没吃到。

**下一步**：让一个 block 算多个 oc（预期能大幅超过 1.11x），或直接用 im2col。

---

## E8 — im2col 不是永远更快（形状敏感性）

**C=64 224² OC=64 3×3（3.699 GFLOP）**：v2 2461 GFLOP/s，5.7x。

**换成 7×7 stride 2 首层形状（C=3 512² OC=8，0.154 GFLOP）**：

| 实现 | ms | GFLOP/s | 相对 naive |
|---|---|---|---|
| v0 naive | 0.224 | 689 | 1.00x |
| v1 smem | 0.249 | 618 | 0.90x |
| v2 im2col | 0.527 | 293 | **0.42x** |

**原因**：im2col 要把 (C·KH·KW)×(OH·OW) 的矩阵**物化**到显存，小问题上这笔写入
和回读开销直接吞掉 GEMM 收益。

**结论**：选算法要看问题规模，不是看哪个"更先进"。拐点值得用 `bench_conv` 扫一遍。

---

## 待办实验

- [ ] v3 用多个 accumulator fragment 打断 `mma_sync` 依赖链，看能否超过 v2
- [ ] v2 消掉 `Bs` 的 2-way bank conflict
- [ ] v2 加双缓冲 + `cp.async`，让访存与计算重叠
- [ ] conv 一个 block 算多个 oc（针对 E7 的结论）
- [ ] conv 的 depthwise + pointwise 用多 stream 重叠，`nsys` 验证
- [ ] CUDA Graph 合并 conv 的 kernel 启动开销
- [ ] 用 `bench_conv 512 28 28 512 3 1 1` 找 v1/v2 的拐点
- [ ] 用 `nsys` 的事件计时和 bench 的 CUDA event 计时交叉验证
