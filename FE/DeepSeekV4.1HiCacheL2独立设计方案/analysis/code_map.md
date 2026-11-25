# 代码现状与调用链

基线：`sglang@dsv41-a5`（`9898f7ce9a`）。行号均指该基线。

## 1. 端到端链路

```
server_args 解析
  └─ hicache_hook / kv_cache_hook / npu utils  ── 参数合法性与 NPU 默认值
       └─ UnifiedRadixCache.init_hicache
            └─ HybridStack 策略选择 (_DeepSeekV4Strategy)
                 └─ build_deepseek_v4_hicache_stack
                      ├─ LogicalHostPool           FULL 逻辑页锚点
                      ├─ DeepSeekV4PagedHostPool   SWA / C1 / C2 / indexer K / indexer scale
                      ├─ DeepSeekV4StateHostPool   C4/C128 持久压缩状态（V4；V4.1 无）
                      └─ HybridCacheController     L2 prefetch / backup / eviction
                           └─ L2TransferEngine     kernel_ascend (AIV kv_exchange) / kernel / direct
```

## 2. 参数与默认值

| 行为 | 位置 |
| --- | --- |
| `--enable-hierarchical-cache`（别名 `--enable-hicache`） | `python/sglang/srt/arg_groups/fields/memory.py:100-103` |
| `--hicache-host-memory-mode` 默认 `cache` | `fields/memory.py:104-110` |
| `--hicache-ratio` 默认 2.0（cache 模式） | `fields/memory.py:111-114`；`hicache_hook.py:105-131` |
| `--hicache-write-policy` 默认 `write_through` | `fields/memory.py:123-129` |
| `--hicache-io-backend` 默认 `kernel`，NPU 上重写为 `kernel_ascend` | `fields/memory.py:130-136`；`hardware_backend/npu/utils.py:191-195` |
| `--hicache-mem-layout` 默认 `page_first`，DSV4 上保持/重写为 `page_first` 或用户显式 `layer_first` | `fields/memory.py:137-151`；`npu/utils.py:196-210` |
| host 内存后端默认 `acl`（DSV4） | `npu/utils.py:196-201`；`pool_host/npu_host_memory.py:45-67` |
| `--hicache-size` | DSV4 启动期直接拒绝，见 `gap_analysis.md` G3 |
| buffer_only 需要 L3 | `hicache_hook.py:340-389` |
| HiCache 与 `--disable-radix-cache` 互斥 | `arg_groups/kv_cache_hook.py:392-396` |
| DCP>1 的 HiCache 约束 | `hicache_hook.py:134-172`；仅 MLA 模型，V4 系列走 MLA 判定 |

`is_deepseek_v4()` 以 `architectures` 判定（`configs/model_config.py:447-454`）；V4.1 的 `architectures=["DeepseekV41ForCausalLM"]` 在配置归一化时被改写为 `DeepseekV4ForCausalLM`（`configs/deepseek_v41.py:23-45`），因此 NPU 的 DSV4 专用默认值对 V4.1 生效。

## 3. DSV4.1 的 KV 池结构

V4.1 与 V4 的关键差异：`compress_ratios` 为 1/2（低压缩比），没有 ratio-128（C128）池；压缩槽位是 `full_loc // ratio` 的纯函数（`mem_cache/kv_cache_configurator.py` 中 A5 引入的分支注释）。

| 池 | device 类 | 行格式 / 几何 | host 池 |
| --- | --- | --- | --- |
| SWA 窗口 | `DSV41NPUSingleKVPool` (`hardware_backend/npu/dsv4/dsv4_memory_pool.py:109`) | `V41_SWA_FP8`，mxfp8_bf16，group 32，512 维 → 544 B/token | `DeepSeekV4PagedHostPool(PoolName.SWA)` |
| C1 压缩 KV | `DSV41NPUSingleKVPool` | `V41_CMP_FP4`，mxfp4_bf16，group 16，512 维 → 320 B/token；每 full page 128 槽 | `DeepSeekV4PagedHostPool(PoolName.DEEPSEEK_V4_C1)` |
| C2 压缩 KV | `DSV41NPUSingleKVPool` | 同上；每 full page 64 槽 | `DeepSeekV4PagedHostPool(PoolName.DEEPSEEK_V4_C2)` |
| indexer K | `DSV41NPUIndexerPool` (`dsv4_memory_pool.py:362`) | packed fp4，`head_dim//2` B + e8m0 `head_dim//32` B；A5 的 native page = `DSV41_INDEX_PAGE_SIZE` = 64（CUDA DeepGEMM 路径可能为 128，不在 A5 首批范围） | `DeepSeekV4PagedHostPool(PoolName.DEEPSEEK_V4_C1/C2_INDEXER)` |
| indexer scale | 同上 | 独立 PA_ND buffer | `DeepSeekV4PagedHostPool(..._INDEXER_SCALE)` |
| 压缩状态 | `NPUCompressStatePool` | V4.1 ratio 1/2 为请求级、无持久 ring | 不注册（见 §3.1） |

行格式定义：`hardware_backend/npu/dsv4/dsv4_kv_layout.py:18-108`。池构建：`mem_cache/deepseek_v4_memory_pool.py:1728-1832`；NPU 覆写：`dsv4_memory_pool.py:387-624`。

### 3.1 device pool 描述（HiCache 的输入）

`DeepSeekV4TokenToKVPool.get_device_pool_infos()`（`deepseek_v4_memory_pool.py:1517-1591`）只导出**真实持有数据的池**：

- `PoolName.SWA`：SWA buffer；
- 每个 ratio 的压缩 KV（C1/C2/C4/C128）、indexer key/scale；
- 仅 ratio 4 导出 `C4_STATE` / `C4_INDEXER_STATE`（`1555-1569`）。V4.1 的 ratio-2 请求级状态不导出，依靠页边界重建（`_dsv4_low_ratio_entries` 的 docstring，`hybrid_pool_assembler.py:667`）。

`DSV4NPUTokenToKVPool.get_contiguous_buf_infos()`（`dsv4_memory_pool.py:626-677`）进一步说明 host 页 id 的对齐规则：压缩池按 `page_size // ratio` 行成页，indexer 池按 `DSV41_INDEX_PAGE_SIZE` 成页，一个 full page 对应 `slots_per_full_page / index_page_size` 个 index page。

## 4. Host 池装配

入口：`_DeepSeekV4Strategy`（`hybrid_pool_assembler.py:2031-2153`）→ `build_deepseek_v4_hicache_stack`（`788-1100`）。

| 池 | 装配位置 | 说明 |
| --- | --- | --- |
| FULL 锚点 | `LogicalHostPool`（`868-880`） | 只做 host 槽位记账，不持有字节；所有 sidecar 复用它的 host 索引 |
| SWA | `882-911` | 行宽一致时把 MTP draft 的 SWA 层追加进同一 host 池；行宽不一致时走 `DRAFT_SWA` sidecar（`build_swa_draft_pools`） |
| C4 / C128 | `913-1071` | V4 路径；V4.1 无 C128 |
| C1 / C2 + indexer | `600-690`, `1073-1077` | V4.1 低压缩比路径；`page_aligned_only=True`（只做整页传输） |
| 持久状态 | `979-1015` | ratio 4（V4） |
| rope | `967-977`, `1061-1071` | 仅 unified fp8 KV |
| extra host regions | `693-754` | 供子类扩展；V4.1 当前为空 |

容量：`_deepseek_v4_num_host_pages`（`538-566`）按 `host_pages = int(device_pages * ratio)` 计算 FULL/SWA；C128 由独立 allocator 页数计算（V4）；V4.1 的 C1/C2 复用 `num_host_pages`。

`PoolEntry` / `HostPoolGroup`（`pool_host/group.py:21-36`）负责把 sidecar 的 host 索引绑定到源池：`indices_from_pool` 的 sidecar 不单独分配，直接复用锚点或 SWA 的 `host_indices` / `device_indices`（`group.py:275-330`）。`_DeepSeekV4Strategy.build` 为每个 host 池生成 `SidecarPoolSpec`：SWA 为 `TRAILING_PAGES`，其余为 `ALL_PAGES`（`hybrid_pool_assembler.py:2097-2131`）。

## 5. L2 数据面

### 5.1 Host 池实现

`DeepSeekV4PagedHostPool`（`memory_pool_host.py:242-737`）：

- 布局：`layer_first` 为 `[layers, pages, item_bytes]`；`page_first` 为 `[pages, layers, item_bytes]`；统一 `uint8` 页行。
- `item_bytes`：一个 host 页（对应一个 full logical page）的字节数。
- `alloc/free`：按 `slot_page_size`（= 逻辑 page size，128）取整；`release_slots` 延迟合并（`479-502`）。
- `backup_from_device_all_layer`：整页 D2H；`kernel_ascend` 分支见 `585-623`。
- `load_to_device_per_layer`：逐层 H2D；`kernel_ascend` 分支见 `695-732`。
- `_host_page_view`：为 Ascend split 算子构造 5-D 视图，`page_shape = (device_buffer.shape[1], 1, device_buffer.shape[-1])`（`402-418`）。`page_first` 下 `self.kv_buffer[:, l, :]` 的页间 stride 是 `layers * item_bytes`，因此不能把多个 host 页扁平化成一个连续的 native-page 维。
- `_to_page_slots`：把一个逻辑页展开为 `page_slots` 个槽（`430-436`）；NPU 低压缩比 indexer 必须先把 4-D buffer 规范化为 full-page `[full_pages, S, 1, dim]`，否则这里会按 `K` 而不是 `S` 展开。

### 5.2 控制器与传输

`HybridCacheController`（`hybrid_cache/hybrid_cache_controller.py`）：

- L2 传输描述由 `_l2_transfers` 生成（`1071-1117`），每个池一个 `L2Transfer`；
- `L2TransferEngine`（`mem_cache/l2_transfer.py:83-159`）按 `prepare_transfer_indices` 做池内索引解析后，分发到 host 池的 `backup/load` 方法；
- 逐层 H2D 完成事件通过 `LayerLoadingEvent` / `LayerDoneCounter` 记录（`managers/cache_controller.py:81-126`），模型的 KV 访问器在读取每层前调用 `wait_layer_transfer(layer_id)`（`deepseek_v4_memory_pool.py:2041-2043`），因此注意力不会读到未完成回载的层。

### 5.3 NPU IO

| 项 | 位置 |
| --- | --- |
| `kernel_ascend` 传输算子 | `memory_pool_host.py:585-623, 695-732`，调用 `transfer_kv_{all,per}_layer_kv_split[_lf_pf/_pf_lf]` |
| 算子导入 | `memory_pool_host.py:65-71`（来自 `sgl_kernel_npu.operators.mem_cache`） |
| host 内存后端 | `pool_host/npu_host_memory.py`：`acl`（`aclrtHostRegister`）或 `memfabric`（Offload 池） |
| AIV 算子硬件支持 | `sgl-kernel-npu/docs/api/en/operators/mem_cache/kv_transfer.md`：`ascendc_aiv` 在 arch22 / arch35 均支持；A5 需 `--arch-version 3510` 构建 |
| 图捕获 | 文档明确 AIV 传输路径**不支持 NPU graph capture** |

## 6. 元数据与命中

- `UnifiedRadixCache.init_hicache` 在 `enable_hierarchical_cache` 时挂载 host 池与控制器（`mem_cache/registry.py:218-229`）。
- V4.1 的树组件为 `{FULL, SWA}`：只有 `compress_ratios` 含 128 时才追加 C128 组件（`registry.py:196-210`）。
- host 命中长度在调度侧以 `req.host_hit_length` / `req.window_host_hit_lengths` 传播（`managers/schedule_policy.py:1739-1820`）；命中粒度是整页（128 token），尾部不足一页不缓存。
- 命中分层的观测指标：`sglang:prefill_effective_tokens_total{mode="device_hit"|"host_hit"|"storage_hit"}`（`observability/metrics_collector.py:1091-1103`）；host 池占用：`sglang:hicache_host_used_tokens` / `hicache_host_total_tokens`（`metrics_collector.py:818-828`）。

## 7. 现有测试覆盖

| 测试 | 覆盖 | 未覆盖 |
| --- | --- | --- |
| `test/registered/npu/basic_function/HiCache/test_npu_dsv4_hicache_l2_pool.py` | DSV4 C4/C4-indexer host 池装配（mock） | V4.1 低压缩比池 |
| `test/registered/unit/mem_cache/test_dsv4_hicache_l2.py` | 页视图、state 绑定、Ascend split 调用（`page_first` / `layer_first`）、ACL 容量声明 | 低压缩比 indexer 的 2-D 行 |
| `test/registered/unit/mem_cache/test_npu_hicache_sparse_dispatch.py` | 独立 K/scale layer 槽位的 P2P 分发、acl/memfabric 分配 | `kernel_ascend` 整页几何 |
| `test/registered/npu/basic_function/HiCache/test_npu_hierarchical_cache*.py` | Qwen3 / DeepSeek-R1 的 NPU HiCache E2E | DSV4 / V4.1 |

结论：**V4.1 低压缩比 host 池的装配和整页 AIV 传输没有测试覆盖，也没有 A5 端到端证据。**
