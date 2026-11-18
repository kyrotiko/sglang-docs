# DeepSeek-V4.1 HiCache L2（Host KV Offload）独立设计方案

## 需求

在 Ascend A5 上让 DeepSeek-V4.1 支持 HiCache 特性：启用 **L2（host DRAM）** 层级，把 device 上被淘汰的 KV 卸载到 host，后续请求命中 host 前缀时再回载到 device，避免重复 prefill。

- 层级：L1 = device HBM，L2 = 本机 host DRAM；**不做 L3**（不配 `--hicache-storage-backend`），不做跨实例/跨节点共享。
- 目标收益：共享前缀、多轮问答、长上下文场景下 device 容量不足时，host 命中替代重算，降低 TTFT、提高有效缓存容量。
- 非目标（首批）：L3 存储、PD 分离、buffer_only 暂存模式、跨实例共享、HiSparse 与 HiCache 组合。

## 基线与交付物

| 项目 | 内容 |
| --- | --- |
| 主仓 | `sglang`（fork） |
| 开发分支 | `dsv41-a5` @ `9898f7ce9a`（合并 `randgun/dsv41-a5@c790194814` 后） |
| 本方案工作分支 | `dsv41-a5-hicache-l2`，worktree `D:/workspace/SGLang/sglang-dsv41-a5-hicache-l2` |
| 文档仓 | `sglang-docs`，`main` 分支 |
| 分析日期 | 2026-11-18 |

本文档只做**独立代码分析**：结论全部来自上述基线的代码、提交历史和 `sgl-kernel-npu` 参考仓文档，不引用其他分析文档的结论。**设计分析阶段没有 A5 实机运行**，所有涉及设备行为的判断都标记为待验证，并给出验证方法；后续 A5 算子探测结果见下节。

## 实现状态（2026-11-18）

- 代码分支：`dsv41-a5-hicache-l2`，commit `b376ac9e25`（低压缩比页几何/预留/断言）与 `4ddf6b5246`（V4 paged pools auto-size）。
- CPU：`test_dsv4_hicache_l2.py`、`test_dsv4_compressed_pools.py`、`test_hicache_auto_size.py` 共 48 passed。
- A5：`a5_kv_exchange_probe.py` 的 `page_size=128`、`layers=2` AIV D2H/H2D 已通过；sglang 侧 HiCache E2E 仍待跑。

## 结论摘要

1. **HiCache L1/L2 主链路和 DSV4 的 host 池装配在这条分支上已经存在。**
   - `--enable-hierarchical-cache` 及 ratio / write policy / mem layout / io backend 参数齐全：`python/sglang/srt/arg_groups/fields/memory.py:100-174`。
   - NPU 上会自动解析 `kernel_ascend` IO、DSV4 的 `page_first` host 布局和 `SGLANG_HICACHE_HOST_MEM_BACKEND=acl`：`python/sglang/srt/hardware_backend/npu/utils.py:181-227`。
   - DSV4 的 HiCache host 池由 `build_deepseek_v4_hicache_stack` 统一构建，包含 SWA、C4/C128（V4）以及 **C1/C2 + indexer K/scale（V4.1 低压缩比）**：`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:600-690, 788-1100`。
   - A5 的 AIV 分片 KV 传输算子声明支持 arch22（A3）与 arch35（A5）：`sgl-kernel-npu/docs/api/en/operators/mem_cache/kv_transfer.md`。

2. **但不能直接宣告 DSV4.1 L2 可用。** 代码级有多处未闭环（详见 [缺口分析](analysis/gap_analysis.md)）：
   - **G1（高优先级，NPU 4-D 路径）**：低压缩比 host 池的页几何未闭环。NPU 4-D KV 的 `item_bytes` 按 `shape[1] * element_size` 计算，得到槽数 `S` 而不是 `S*dim*itemsize`；indexer K/scale 被整理成 2-D `[full_pages, bytes]` 行，而 Ascend 传输路径按 4-D PA_ND 推导 `page_slots`，ratio-1 的 full page 还需要把 `I=2` 个 native index page 合成 `S=128` 槽。GPU 的 2-D 页行是正确的，现有 CPU 测试也锁定了该契约，修复不能把 GPU 路径一起改成 4-D。修复方向是 NPU indexer 规范化为 full-page 4-D 视图、`item_bytes` 取 full-page 字节、传输 `page_size=S`。
   - **G2**：`reserve_dsv4_paged_host_memory` 没有把 C1/C2 KV 与 indexer 的 host 块计入预留；ACL 后端下 `ensure_capacity` 是空操作所以默认不触发，但 memfabric 后端会漏预留。
   - **G3**：DSV4 明确不支持 `--hicache-size`（`hybrid_pool_assembler.py:551-555`），只能按 ratio 定容。
   - **G10（已修复，待 A5 验证）**：`_estimate_hicache_bytes` 现在支持 V4.1 的 paged pools，只写 `--enable-hierarchical-cache` 时会走 auto-size；带 C4 持久状态或 V4 draft 的模型仍要求显式 `--hicache-ratio`。首批仍建议显式写 ratio，便于容量可预测。
   - 已有 `test_dsv4_compressed_pools.py:553-606` 覆盖 CPU/2-D 低压缩比装配；缺 NPU 4-D 单测、indexer 传输调用测试和 A5 端到端用例。

3. **首批推荐配置（L2-only）**：

   ```bash
   --enable-hierarchical-cache
   --hicache-ratio 2.0          # 首批建议显式给；不写时走 auto-size（G10 已修）
   --hicache-write-policy write_through
   # 其余由 NPU 侧自动解析：page_first / kernel_ascend / acl host 内存后端
   ```

   不要同时给 `--hicache-storage-backend`、`--hicache-size`、`--hicache-host-memory-mode buffer_only` 或 `--disable-radix-cache`。

4. **首批交付边界**：A5 单机、聚合推理、page_size 128、C1/C2 低压缩比模型、无 L3、DCP=1、暂不含投机解码（DSpark 单列验证）、PP=1；TP / DP attention 逐项验证。完整矩阵见 [设计](design/l2_host_offload.md#兼容性矩阵) 与 [验收](validation/acceptance_plan.md)。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [代码现状与调用链](analysis/code_map.md) | 参数解析、HiRadixTree、host 池装配、L2 传输、A5 行格式与算子的证据清单 |
| [缺口与风险分析](analysis/gap_analysis.md) | G1–G9，按优先级给出证据、影响和验证/修复方向 |
| [L2 Host Offload 设计](design/l2_host_offload.md) | 启用方式、池布局、容量公式、数据面时序、正确性不变量、兼容矩阵 |
| [代码改动清单](design/code_change_plan.md) | 逐文件/函数的改动建议与提交顺序 |
| [验收与验证计划](validation/acceptance_plan.md) | CPU 契约测试、A5 单卡功能/正确性/性能/压力测试与通过标准 |
| [A5 验证脚本](validation/scripts/README.md) | 可直接上 A5 跑的算子 full-page 探测、HiCache E2E、G10 默认 ratio 复现 |
| [A5 算子探测结果（2026-11-18）](validation/a5_operator_probe_2026-11-18.md) | `page_size=128` + `layers=2` 的 AIV D2H/H2D 已通过，full-page 传输假设成立 |
| [交互式教学页](../../learn/sglang-hicache-kv-offload-viz.html) | 动画演示 L1/L2 生命周期、写策略、DSV4 多池几何与 A5 AIV 传输 |
| [SWA / C1·C2 / indexer K·scale 动画讲解](../../learn/dsv41-kv-structure-viz.html) | 12 token 小例子：窗口、压缩槽、indexer 打分与 top-k 动画 |
| [full page 拆解动画](../../learn/dsv41-fullpage-transfer-viz.html) | 一页如何按 SWA / C1·C2 / indexer 拆成多次 D2H、H2D 传输 |

## 术语

| 术语 | 含义 |
| --- | --- |
| L1 / L2 | device KV 池 / host DRAM KV 池 |
| FULL / SWA | 逻辑页锚点（仅元数据） / 滑窗 KV 池 |
| C1 / C2 | V4.1 的压缩比 1 / 2 压缩池（`compress_ratios` 为 1、2 的层共享） |
| indexer K / scale | V4.1 两级 indexer 的 K 与 e8m0 scale，用于 sparse attention top-k |
| AIV / `kernel_ascend` | NPU 上由 `kv_exchange` 承载的 host↔device 分片 KV 传输 |
| page_first | host 池按 `[page, layer, bytes]` 组织的布局 |
