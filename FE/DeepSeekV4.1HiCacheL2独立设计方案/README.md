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
| 分析基线 | `dsv41-a5` @ `9898f7ce9a`（合并 `randgun/dsv41-a5@c790194814` 后） |
| 交付基线 | `origin/main` @ `a662e1efbb`（main 已合入 dsv41-a5 代码，本方案 rebase 到该基线） |
| 本方案工作分支 | `dsv41-a5-hicache-l2` @ `b7a1bcf25d`，worktree `D:/workspace/SGLang/sglang-dsv41-a5-hicache-l2` |
| PR | [sgl-project/sglang#53204](https://github.com/sgl-project/sglang/pull/53204) |
| 文档仓 | `sglang-docs`，`main` 分支 |
| 分析日期 | 2026-11-18（2026-11-25 按 main 基线更新） |

本文档只做**独立代码分析**：结论全部来自分析基线的代码、提交历史和 `sgl-kernel-npu` 参考仓文档，不引用其他分析文档的结论。**设计分析阶段没有 A5 实机运行**，所有涉及设备行为的判断都标记为待验证，并给出验证方法；后续 A5 算子探测与 E2E 结果见下节。

## 实现状态（2026-11-26）

- 代码分支：`dsv41-a5-hicache-l2`，基于 `origin/main` @ `a662e1efbb`，5 个提交：
  - `5c32e0ee0a`：NPU 4-D 低压缩比页几何、C1/C2 预留、draft SWA 行宽断言；
  - `23056a7e24`：V4 paged pools auto-size；
  - `7040d0b4f9`：sidecar draft 与测试补齐；
  - `23de16b280`：draft SWA 行宽不一致时回退到 `DRAFT_SWA` sidecar（DSpark + V4.1 target）；
  - `b7a1bcf25d`：把 `torch` import 移到模块级，修复 ruff F821。
- CPU：`test_dsv4_hicache_l2.py`、`test_dsv4_compressed_pools.py`、`test_hicache_auto_size.py`、`test_hicache_mtp_state.py` 共 66 passed。
- A5：
  - `a5_kv_exchange_probe.py` 的 `page_size=128`、`layers=2` AIV D2H/H2D 已通过；
  - DP8 target-only E2E 已出现 host 命中 768 token，7 个池的 D2H/H2D 页 CRC 全部一致；
  - DSpark + HiCache 配方的实机启动与推理功能验证通过（2026-11-25）；
  - A5 冒烟在 `23de16b280` 上完成（2026-11-26）；PR head `b7a1bcf25d` 之后只把 `torch` import 移到模块级。`draft_swa` 池 77824 B 行、`swa` 池 69632 B 行，8 个 rank 一致，device 命中 128 token。
  - 详见 [E2E 结果](validation/a5_hicache_e2e_result_2026-11-20.md)。

## 结论摘要

1. **HiCache L1/L2 主链路和 DSV4 的 host 池装配在这条分支上已经存在。**
   - `--enable-hierarchical-cache` 及 ratio / write policy / mem layout / io backend 参数齐全：`python/sglang/srt/arg_groups/fields/memory.py:100-174`。
   - NPU 上会自动解析 `kernel_ascend` IO、DSV4 的 `page_first` host 布局和 `SGLANG_HICACHE_HOST_MEM_BACKEND=acl`：`python/sglang/srt/hardware_backend/npu/utils.py:181-227`。
   - DSV4 的 HiCache host 池由 `build_deepseek_v4_hicache_stack` 统一构建，包含 SWA、C4/C128（V4）以及 **C1/C2 + indexer K/scale（V4.1 低压缩比）**：`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:600-690, 788-1100`。
   - A5 的 AIV 分片 KV 传输算子声明支持 arch22（A3）与 arch35（A5）：`sgl-kernel-npu/docs/api/en/operators/mem_cache/kv_transfer.md`。

2. **代码级缺口已修复并完成 A5 验证；剩余限制见 [缺口分析](analysis/gap_analysis.md)。**
   - **G1（已修复）**：NPU 4-D 低压缩比 host 池的页几何已按 full page 闭环——NPU indexer 规范化为 `[full_pages, S, 1, dim]`，`item_bytes` 取 full-page 字节，传输 `page_size=S`；GPU 2-D 路径保持不变。A5 上 7 个池的 D2H/H2D 页 CRC 已逐池一致。
   - **G2（已修复）**：`reserve_dsv4_paged_host_memory` 已把 C1/C2 KV 与 indexer K/scale 的 host 块计入预留，indexer 按 `num_host_pages * I` 计。ACL 后端已实机验证；memfabric 后端仍不在首批范围。
   - **G3（未变）**：DSV4 明确不支持 `--hicache-size`（`hybrid_pool_assembler.py:551-555`），只能按 ratio 定容。
   - **G6（已修复）**：DSpark draft 是 V4 的 608 B/token，V4.1 target 是 544 B/token，二者不能 packed 到同一 SWA host 池。现在计划阶段比较两侧 SWA 行宽，不一致时回退为 `SIDECAR`，由独立的 `DRAFT_SWA` host pool（608 B 行）承载 draft。
   - **G10（已修复）**：`_estimate_hicache_bytes` 支持 V4.1 的 paged pools，只写 `--enable-hierarchical-cache` 时会走 auto-size；带 C4 持久状态或 V4 draft 的模型仍要求显式 `--hicache-ratio`。首批仍建议显式写 ratio，便于容量可预测。
   - 测试：NPU 4-D 装配、indexer 全页传输、预留、auto-size、draft 行宽回退均有 CPU 单测；A5 端到端使用 `validation/scripts/` 下的脚本，尚未注册为 NPU CI 用例。

3. **首批推荐配置（L2-only）**：

   ```bash
   --enable-hierarchical-cache
   --hicache-ratio 2.0          # 首批建议显式给；不写时走 auto-size（G10 已修）
   --hicache-write-policy write_through
   # 其余由 NPU 侧自动解析：page_first / kernel_ascend / acl host 内存后端
   ```

   不要同时给 `--hicache-storage-backend`、`--hicache-size`、`--hicache-host-memory-mode buffer_only` 或 `--disable-radix-cache`。

4. **首批交付边界**：A5 单机、聚合推理、page_size 128、C1/C2 低压缩比模型、无 L3、DCP=1、PP=1；DSpark 已按独立 `DRAFT_SWA` sidecar 支持并完成功能验证；TP / DP attention 逐项验证。完整矩阵见 [设计](design/l2_host_offload.md#兼容性矩阵) 与 [验收](validation/acceptance_plan.md)。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [代码现状与调用链](analysis/code_map.md) | 参数解析、HiRadixTree、host 池装配、L2 传输、A5 行格式与算子的证据清单 |
| [缺口与风险分析](analysis/gap_analysis.md) | G1–G10，按优先级给出证据、影响和验证/修复状态 |
| [L2 Host Offload 设计](design/l2_host_offload.md) | 启用方式、池布局、容量公式、数据面时序、正确性不变量、兼容矩阵 |
| [代码改动清单](design/code_change_plan.md) | 逐文件/函数的改动建议与提交顺序 |
| [验收与验证计划](validation/acceptance_plan.md) | CPU 契约测试、A5 单卡功能/正确性/性能/压力测试与通过标准 |
| [A5 验证脚本](validation/scripts/README.md) | 可直接上 A5 跑的算子 full-page 探测、HiCache E2E、G10 默认 ratio 复现 |
| [A5 算子探测结果（2026-11-18）](validation/a5_operator_probe_2026-11-18.md) | `page_size=128` + `layers=2` 的 AIV D2H/H2D 已通过，full-page 传输假设成立 |
| [A5 HiCache L2 E2E 结果（2026-11-20 / 2026-11-25）](validation/a5_hicache_e2e_result_2026-11-20.md) | DP8 下 host 命中 768 token，7 个池的 D2H/H2D 页 CRC 全部一致；DSpark + HiCache 功能验证通过 |
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
