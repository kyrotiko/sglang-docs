# L2 Host Offload 设计

## 1. 目标与非目标

**目标**

1. A5 上的 DeepSeek-V4.1 可通过 `--enable-hierarchical-cache` 启用 HiCache。
2. L2（host DRAM）作为 device KV 的淘汰缓存：device 淘汰的完整页保留在 host，后续请求命中 host 前缀时回载，避免重新 prefill。
3. 覆盖 V4.1 的全部 KV 数据：SWA 窗口、C1/C2 压缩 KV、两级 indexer 的 K 与 scale。
4. 行为与原计算路径字节级一致；host 池容量可预测、可观测、可回收。

**非目标（首批）**

- L3 存储后端、跨实例/跨节点共享。
- `buffer_only` 暂存模式（它需要 L3）。
- PD 分离、`--hicache-size` 按 GB 定容、HiSparse 组合。
- 投机解码（DSpark）的严格精度/接受率 A/B。功能已按独立 `DRAFT_SWA` sidecar 支持并完成 A5 功能验证，见 [G6](../analysis/gap_analysis.md#g6中dspark-draft-与-target-的-swa-行宽兼容性)。

## 2. 启用方式与参数解析

### 2.1 用户配置

```bash
# 最小配置：NPU 侧自动解析 kernel_ascend / page_first / acl
--enable-hierarchical-cache --hicache-ratio 2.0

# 建议显式写出关键项
--enable-hierarchical-cache
--hicache-ratio 2.0
--hicache-write-policy write_through
--hicache-mem-layout page_first
--hicache-io-backend kernel_ascend
# env: SGLANG_HICACHE_HOST_MEM_BACKEND=acl（DSV4 默认已是 acl）
```

必须保持默认 radix cache（不能加 `--disable-radix-cache`，二者互斥，`arg_groups/kv_cache_hook.py:392-396`）。

### 2.2 参数解析规则

| 参数 | 解析规则 | 依据 |
| --- | --- | --- |
| `enable_hierarchical_cache` | 显式开关，默认关 | `fields/memory.py:100-103` |
| `hicache_host_memory_mode` | 固定 `cache`；`buffer_only` 需要 L3，首批拒绝 | `hicache_hook.py:340-389` |
| `hicache_ratio` | 未设置时 hook 声明 2.0，再由 auto-size 按 host 预算收缩；V4.1 paged pools 已支持（G10）。首批仍建议显式写 ratio，便于容量可预测 | `hicache_hook.py:105-131`；`fields/memory.py:119-122` |
| `hicache_size` | 首批不支持，启动期报错 | `hybrid_pool_assembler.py:551-555` |
| `hicache_write_policy` | 默认 `write_through`；首批只开放 `write_through` / `write_back` 二选一验证 | `fields/memory.py:123-129` |
| `hicache_io_backend` | NPU 上强制 `kernel_ascend` | `npu/utils.py:191-195` |
| `hicache_mem_layout` | DSV4 固定 `page_first`；用户显式 `layer_first` 时保留 | `npu/utils.py:196-210` |
| host 内存后端 | `SGLANG_HICACHE_HOST_MEM_BACKEND` 默认 `acl`；`memfabric` 待 G2 修复 | `npu/utils.py:200-201`；`npu_host_memory.py:45-67` |

### 2.3 明确拒绝的组合

- `--hicache-storage-backend`：首批 L2-only 不配。
- 只写 `--enable-hierarchical-cache`、不写 `--hicache-ratio`：V4.1 paged pools 已支持 auto-size（G10 已修）；带 C4 持久状态或 V4 draft 时仍要求显式 ratio。
- `--hicache-size > 0`：DSV4 不支持。
- `--hicache-host-memory-mode buffer_only`：需要 L3。
- `--disable-radix-cache`：互斥。
- `--enable-hisparse` + DCP>1：既有代码已拒绝（`hicache_hook.py:156-160`）；V4.1 首批不启用 HiSparse。
- EAGLE/EAGLE3 + unified memory + HiCache：既有断言拒绝（`kv_cache_hook.py:648-666`）。

## 3. 架构与池布局

```
L1 device KV
  ├─ SWA window pool           PF8 行 (mxfp8_bf16)
  ├─ C1 compressed KV          FP4 行 (mxfp4_bf16), 128 槽/full page
  ├─ C2 compressed KV          FP4 行, 64 槽/full page
  ├─ C1/C2 indexer K + scale   独立 PA_ND buffer, 每 index page 64 槽
  └─ request-scoped compressor state（不卸载）

L2 host pools（每 rank 私有）
  ├─ LogicalHostPool                   FULL 页锚点，只记账
  ├─ DeepSeekV4PagedHostPool(SWA)      [pages, layers, item_bytes]
  ├─ DeepSeekV4PagedHostPool(C1/C2)    同上
  └─ DeepSeekV4PagedHostPool(C1/C2 indexer K / scale)

HybridCacheController
  ├─ prefetch: host → device（逐层 H2D，完成事件）
  ├─ backup:   device → host（整页 D2H）
  └─ evict:    host LRU；device 淘汰由 write policy 触发备份
```

### 3.1 池的职责

| 池 | 是否持有字节 | 说明 |
| --- | --- | --- |
| `LogicalHostPool` (PoolName.KV) | 否 | host 槽位记账；所有 sidecar 复用它的 host 索引 |
| `DeepSeekV4PagedHostPool` (SWA) | 是 | 窗口 KV；仅当 draft 与 target 的 SWA 行宽一致时才把 draft 层打包进同一个池 |
| `DeepSeekV4PagedHostPool` (`DRAFT_SWA`) | 是 | DSpark draft 的独立 SWA 池；行宽与 target 不同时使用（如 V4 draft 608 B vs V4.1 target 544 B） |
| `DeepSeekV4PagedHostPool` (C1/C2) | 是 | 压缩 KV；每个 full page 的 host 行持有 `page_size // ratio` 个槽 |
| `DeepSeekV4PagedHostPool` (C1/C2 indexer K/scale) | 是 | 见 §3.2 的几何设计 |
| `DeepSeekV4StateHostPool` | — | V4 的 ratio-4 持久状态；V4.1 不使用 |

### 3.2 低压缩比 indexer 的页几何（G1 的设计）

记号：

- `P = 128`：逻辑 page size（token）。
- `ratio ∈ {1, 2}`：压缩比。
- `S = P // ratio`：一个 full page 覆盖的压缩槽数（C1: 128，C2: 64）。
- `K = DSV41_INDEX_PAGE_SIZE`：indexer 设备页槽数（A5 当前为 64；`deepseek_v4_memory_pool.py:542-552`）。
- `I = S // K`：一个 full page 覆盖的 index page 数；**必须整除**，否则启动期报错。C1 `I=2`，C2 `I=1`。

设备/主机表示：

| 数据 | NPU device | GPU device | host 页（每层） |
| --- | --- | --- | --- |
| KV | `[dev_pages, S, 1, dim]` | `[dev_pages, page_bytes]` | `[host_pages, 1, S, 1, dim]` |
| indexer K | `[npu_pages, K, 1, dim]` → 规范化为 `[full_pages, S, 1, dim]` | `[full_pages, page_bytes*I]` | `[host_pages, 1, S, 1, dim]` |
| indexer scale | `[npu_pages, K, 1, d]` → 规范化为 `[full_pages, S, 1, d]` | `[full_pages, page_bytes*I]` | `[host_pages, 1, S, 1, d]` |

设计要点：

1. **host 行仍是一个 full page**：`item_bytes = S * row_bytes = I * native_page_bytes`；不把 `I` 个 native page 当成 host 页。
2. **NPU indexer 在装配时规范化**：`full_pages = npu_pages // I`，`buffer[: full_pages*I].reshape(full_pages, S, *trailing)`；丢掉尾部 padding。这样 `device_buffers[0].shape[1]` 就是 `S`，`_host_page_view` / `_to_page_slots` / `kernel_ascend` 都不需要新参数。
3. **GPU 2-D 保持不变**：GPU 的 indexer 行本来就是 `[full_pages, page_bytes*I]`，直接保持 2-D；不能为了 NPU 把 GPU 也改成 4-D。
4. **传输以 full page 为单位**：`page_size=S`，`device_indices = p*S + arange(S)`，host 同理。C1 indexer 一次传 128 槽（两个 native page 的连续内容），C2 一次传 64 槽。
5. **不要扁平化 `page_first` host 行**：`self.kv_buffer[:, l, :]` 的页间 stride 是 `layers * item_bytes`，`num_host_pages*I` 个 native page 不连续；host 视图只能是 `[host_pages, 1, S, 1, dim]`，页间 gap 由算子允许。
6. KV 池（SWA/C1/C2）同样保持 `page_size=S`；C2 的 `slot_page_size=P=128` 与每页槽数 `S=64` 不是一回事，不能混用。

该设计把“host 页 = full page”的记账粒度保持不变，避免改动 sidecar 的 host 索引绑定与命中/淘汰逻辑；只修正装配时的 buffer 视图和 `item_bytes`。indexer K 与 scale 的行宽不同，必须分别用各自的 `buffer_info.page_bytes` 计算 `item_bytes`，不能共用一个 `index_item_bytes`。

## 4. 容量与定容

### 4.1 公式

每个 host 池：

```
host_pages      = int(device_pages * ratio)      # 与 _deepseek_v4_num_host_pages 一致
host_bytes_pool = host_pages * item_bytes * layers_pool
```

其中 `item_bytes = page_slots * row_bytes`，`page_slots` 是该池一个 full page 的槽数。总 host 字节：

```
H ≈ ratio * Σ(device KV bytes of all mirrored pools)
```

`LogicalHostPool` 不占数据字节；V4.1 不镜像持久状态；DSpark draft 使用独立 `DRAFT_SWA` pool 并需要显式 `--hicache-ratio`，不纳入 target 锚点池的 auto-size。

### 4.2 每 token 字节（以 nope 448 + rope 64、index_head_dim 128 为例）

| 池 | 行宽/每 full token | 依据 |
| --- | --- | --- |
| SWA | 544 B / 窗口槽 | `V41_SWA_FP8`：512 + 2×16 个 scale，32 字节对齐 |
| C1 KV | 320 B / full token | `V41_CMP_FP4`：256 packed + 2×32 scale，32 对齐 |
| C2 KV | 160 B / full token（320 B × 64/128） | 同上，槽数减半 |
| C1 indexer | 64 B K + 4 B scale | `head_dim//2` + `head_dim//32` |
| C2 indexer | 32 B K + 2 B scale（半槽） | 同上 |

按上述参数，**每个 full token 每层**的压缩侧池约 582 B；SWA 另按窗口槽数计 544 B/槽。启动日志会打印每个 host 池的 `Allocating ... GB`（`memory_pool_host.py:354-363`），验收时以日志为准核对公式。

### 4.3 定容建议

- 首批只支持 `--hicache-ratio`；ratio 2.0 表示每个镜像池的 host 页数是 device 页数的 2 倍。未显式写 ratio 时 auto-size 已覆盖 V4.1 paged pools（G4/G10），但首批仍建议写死 ratio。
- 多 TP/DP rank 同机部署时，host 总量是各 rank 之和；需要按 cgroup/物理内存设置上限并人工核算。
- `--hicache-host-memory-fraction` 已覆盖 C1/C2/indexer（G4）；带 C4 持久状态或 V4 draft 时 auto-size 会拒绝，需要显式 ratio。

## 5. 数据面设计

### 5.1 生命周期

1. **匹配**：请求进入 radix tree，先匹配 L1，再匹配 L2；命中长度按 full page 向下取整。组件为 `{FULL, SWA}`；sidecar 按 `PoolSidecar` 规则检查——SWA 为 `TRAILING_PAGES`，C1/C2 与 indexer 为 `ALL_PAGES`（`hybrid_pool_assembler.py:2097-2131`）。
2. **回载**：为 host 命中的页分配 device 槽（必要时淘汰 device 页）；控制器按池、按层提交 H2D。每层完成时记录事件；模型在读取该层 KV 前 `wait_layer_transfer(layer_id)`（`deepseek_v4_memory_pool.py:2041-2043`）。
3. **Prefill**：只对未命中部分计算；新 KV 写入 device。
4. **备份**：`write_through` 在 prefill 完成后立即把新页备份到 host；`write_back` 在 device 淘汰时备份。备份以 full page 为单位，sidecar 复用源页的 host/device 索引（`pool_host/group.py:275-330`）。
5. **淘汰**：device 淘汰时，若 host 已有副本则直接释放 device；host 满时按 LRU 回收 host 页，并同步移除 radix 节点的 host 元数据。

### 5.2 传输路径

| 方向 | 池 | 调用 | 粒度 |
| --- | --- | --- | --- |
| D2H 备份 | SWA / C1 / C2 | `transfer_kv_all_layer_kv_split[_lf_pf]` | full page，all-layer |
| D2H 备份 | indexer K / scale | 同上（5-D full-page 视图，`page_size=S`） | full page |
| H2D 回载 | SWA / C1 / C2 | `transfer_kv_per_layer_kv_split[_pf_lf]` | 逐层 |
| H2D 回载 | indexer K / scale | 同上（`page_size=S`） | 逐层 |

- V4.1 的 sidecar 全部 `page_aligned_only=True`（`hybrid_pool_assembler.py:683`）：只做整页传输，不做 token 粒度尾部搬运。尾部不足一页的 KV 不进入 host，属预期行为。
- AIV 传输要求 host 内存已注册到 device VA 空间；ACL 后端在分配时用 `aclrtHostRegister` 注册（`npu_host_memory.py:241-276`），memfabric 在 `ensure_capacity` 时预留（`npu_host_memory.py:150-240`）。
- AIV 传输路径**不支持 NPU graph capture**（`sgl-kernel-npu` kv_transfer 文档），因此 L2 传输只在调度/普通流上发生，不进入 decode graph。

### 5.3 时序与并发

- H2D 在控制器 stream 上逐层提交，完成事件挂在 `LayerLoadingEvent`；注意力读取前等待，保证 layer 级流水。
- D2H 备份在写 stream 上提交；host 页在完成前不得被 host 淘汰或复用。
- 当前 `DeepSeekV4PagedHostPool` 只有 `free_slots` + `release_slots` 延迟释放（`memory_pool_host.py:479-502`），**没有 per-page 引用计数**。在途页的保护必须由控制器层的 backup/load 完成事件与树节点锁保证；如果后续要引入引用计数，需要单独设计并补并发测试，不能把 `release_slots` 当成 refcount。
- 请求被中止或 retract 时，已分配的 host/device 槽走统一回收路径；不允许悬挂引用。中止发生在 H2D 途中时，必须先等完成事件或使该页失效，不能直接放回 free list。

## 6. 正确性不变量

1. **整页**：只有完整的 full page 才写入 host 或从 host 命中；尾部不完整页不缓存。
2. **sidecar 一致**：一个前缀只有在 FULL 页及所有 `ALL_PAGES` sidecar（C1/C2 KV、indexer K/scale）都完整时才允许命中；SWA 只要求 trailing 页。
3. **层序**：第 `l` 层的 KV/indexer 回载完成后，注意力才能读第 `l` 层（`wait_layer_transfer`）。
4. **字节等价**：传输算子不做量化/反量化，host 页是 device 页的字节副本（`kv_transfer.md` 明确 payload unchanged）。
5. **页地址**：压缩槽 = `full_loc // ratio`；一个 full page 覆盖 `S = P//ratio` 个槽；indexer 的设备 buffer 规范化为 `S` 槽/页，`I = S//K` 必须整除；传输 `page_size=S`。
6. **只读快照**：host 页在被引用期间内容不变；写回只发生在重新计算的新页上。
7. **私有性**：L2 为本 rank/实例私有，不做跨 rank/实例共享。
8. **draft 隔离**：draft 层只有在行宽与 target 一致时才允许挂到 target host 池；否则拒绝启动（G6 修复项）。
9. **一致性**：回载后的 KV + indexer + 压缩状态组合必须与冷启动重算等价（G8 验证项）。

## 7. 压缩状态与 indexer 的专门处理

- **indexer**：V4.1 的 sparse attention 依赖 indexer K/scale 选择 top-k。回载必须同时恢复 K 与 scale；二者是独立 host 池，按 `ALL_PAGES` 命中，缺一不可。
- **压缩状态**：V4.1 ratio 1/2 的状态是请求级、按 ring 组织，不进 HiCache；回载前缀后从页边界重新预热。现有路径在请求释放时调用 `clear_request_scoped_state`（`hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py:148`），该函数清空 ratio-2 pending-pair state（`mem_cache/deepseek_v4_memory_pool.py:2112-2126`）；新分配 req row 从已清理/零初始状态开始。设计必须确认“host 命中后的请求”走同一条分配路径，并在 128-token 的偶数 full-page 边界验证没有 pending pair；验收需要证明“命中 + 重建”与“冷启动”的首 token logits/top-k 及 greedy 输出一致（G8），不能只比 32-token 文本。
- **C4/C128**：V4.1 没有这两个池；如果模型同时包含 ratio 4/128 层，走现有 V4 路径（C4 state 注册、C128 sidecar），不在本方案首批范围。

## 8. 兼容性矩阵

| 维度 | 首批 | 后续 |
| --- | --- | --- |
| 硬件 | Ascend A5（arch35） | A3 的 V4 路径不受影响 |
| 模型 | DSV4.1-Flash，compress ratios 1/2，无 C128 | 混合 ratio 模型 |
| 并行 | TP；DP attention 单独验证 | PP > 1、DCP > 1 |
| 部署 | 聚合推理 | PD 分离 |
| 投机 | DSpark 已支持（`DRAFT_SWA` sidecar） | 严格精度/接受率 A/B |
| 布局 | `page_first`（`layer_first` 保留为诊断选项） | — |
| host 内存后端 | `acl` | `memfabric`（G2 修复后） |
| 存储 | 无 L3 | L3（另立需求） |

## 9. 失败处理

| 场景 | 期望行为 |
| --- | --- |
| 启动时 host 内存不足 | 启动失败，打印各池请求字节与可用字节（现有 `Not enough host memory for V4 paged pool`） |
| 运行时 host 分配失败 | 该页不备份，radix 节点保留 device 副本；记录 warning，不影响正确性 |
| host 回载失败 | 请求必须回退为重新计算，不得使用部分回载的 KV/indexer。现有控制器对 AIV 返回错误/未完成事件的行为**待确认**；实现前必须做失败注入测试，并把结果写入设计 |
| host 池满 | LRU 淘汰；被引用的页跳过；D2H 在途页不得回收 |
| host 后端未注册 / 算子缺失 | 启动期报错并提示改用其他 backend 或关闭 HiCache（`pool_host/_kvcacheio.py:64-91` 的模式） |

## 10. 可观测性

- 命中分层：`sglang:prefill_effective_tokens_total{mode="device_hit"|"host_hit"}`。
- host 池占用：`sglang:hicache_host_used_tokens` / `sglang:hicache_host_total_tokens`。注意现有 used/total 只统计锚点池（`scheduler_components/metrics_reporter.py:1220-1221`），不覆盖 C1/C2/indexer。
- 启动日志：每个 host 池的页数、`item_bytes`、layout、总 GB。
- **首批必须新增**：每池的 `item_bytes`、页数、backup/load 次数与字节数、耗时分布。否则 G1/G2 的几何问题无法在 E2E 中定位，也无法解释 host 命中率。

## 11. 风险

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| G1 NPU 4-D 几何 | 结果错误或传输崩溃 | full-page 规范化 + 单测 + A5 冒烟；GPU 2-D 路径保持回归 |
| host 容量核算偏差 | OOM / 利用率低 | 以启动日志核对公式；多 rank 手工核算 |
| AIV 传输吞吐 | TTFT 回载慢 | page_first + all-layer 备份；测 ratio/写策略 A/B |
| DSpark 行宽 | 静默错误 | 计划阶段比较行宽；不一致回退 `DRAFT_SWA` sidecar + 装配层断言；残留严格精度 A/B |
| 压缩状态重建 | 长前缀数值漂移 | G8 专项验证 |
