# 缺口与风险分析

基线：`sglang@dsv41-a5`（`9898f7ce9a`）。以下结论均为**代码级分析**，G1/G2/G6/G10 需要 A5 实机确认后才能定性为缺陷或已验证路径。修复状态已在 2026-11-25 更新；交付分支为 `dsv41-a5-hicache-l2` @ `b2e36bb605`（基于 `origin/main`）。

## G1（高）NPU PA_ND 低压缩比 host 池的页几何未闭环

V4.1 的低压缩比 host 池由 `_dsv4_low_ratio_regions` / `_dsv4_low_ratio_entries` 单独构建，没有复用高压缩比路径按 `DSV4PagedBufferInfo.page_bytes` 取页宽的写法。

**范围**：问题只出现在 NPU 的 4-D PA_ND buffer（`hardware_backend/npu/dsv4/dsv4_memory_pool.py:141-156`）。GPU 的 KV/indexer 是 2-D `[pages, page_bytes]` 页行（`mem_cache/deepseek_v4_memory_pool.py:339-343, 666-674`），`shape[1]` 本来就是 full-page 字节数，因此 GPU 路径是正确的；现有 CPU 测试也锁定了这个 2-D 契约（`test/registered/unit/mem_cache/test_dsv4_compressed_pools.py:553-606`）。

### G1a：NPU 4-D KV 的 `item_bytes` 少乘行宽

```python
# hybrid_pool_assembler.py:675-679
DeepSeekV4PagedHostPool(
    device_buffers=buffers,
    item_bytes=buffers[0].shape[1] * buffers[0].element_size(),
    ...
)
```

NPU 低压缩比 KV buffer 是 4-D PA_ND `[pages, S, 1, dim]`，`shape[1]` 是每页槽数 `S`，`element_size()` 是 1（uint8/fp8）；于是 `item_bytes = S` 字节，而一个 full page 实际是 `S * dim * itemsize` 字节。GPU 的 2-D 页行没有这个问题。

影响：NPU 上 host 池按错误页宽分配/拷贝，KV 备份与回载会越界或读错数据。

### G1b：indexer 的 2-D 页行与 Ascend 5-D 视图不一致

1. `_dsv4_low_ratio_regions` 把 C1/C2 的 indexer key/scale buffer 统一整理成 **2-D** `[full_pages, bytes]` 页行：

   ```python
   # hybrid_pool_assembler.py:647-656
   for buffer in index.buffer_info.buffers:
       full_pages = buffer.shape[0] // index_pages_per_full_page
       rows.append(
           buffer[: full_pages * index_pages_per_full_page]
           .view(torch.uint8)
           .reshape(full_pages, -1)
       )
   ```

2. 这些 2-D 张量被直接传给 `DeepSeekV4PagedHostPool(device_buffers=buffers, ...)`（`hybrid_pool_assembler.py:672-688`）。

3. `DeepSeekV4PagedHostPool._host_page_view` 假定 device buffer 是 4-D PA_ND（`[pages, page_slots, 1, dim]`）：

   ```python
   # memory_pool_host.py:402-418
   page_shape = (device_buffer.shape[1], 1, device_buffer.shape[-1])
   ```

   对 2-D `[full_pages, bytes]` 会得到 `(bytes, 1, bytes)` 的 page 形状，`.view(...)` 的元素数不匹配。

4. `kernel_ascend` 分支用 `page_slots = int(device_buffers[0].shape[1])`（`memory_pool_host.py:590, 697`）并把一个逻辑页展开成 `page_slots` 个槽（`_to_page_slots`，`memory_pool_host.py:430-436`）。对 2-D 行，`shape[1]` 是“整页字节数”，不是槽数。

5. 2-D 页行本身对 GPU 是正确表示；不能把 GPU 路径也改成 4-D。修复必须按 buffer 维度分派：NPU 4-D 保留 PA_ND，GPU 2-D 保持现状。

### G1c：ratio-1 的 `I=2` 需要 full-page 规范化

`S = page_size // ratio`，`K = DSV41_INDEX_PAGE_SIZE`，`I = S // K`：

- C1：`S=128, K=64, I=2`；
- C2：`S=64, K=64, I=1`。

当前 2-D 行已经把 `I` 个 native index page 折成一个 full-page 行，所以 `item_bytes` 本身是对的；问题在 NPU 侧没有对应的 4-D full-page 视图。**不能**简单把 4-D native buffer `[npu_pages, K, 1, dim]` 直接交给 host 池：`_host_page_view` 会按 `K` 槽建页，一个 full page 需要 `S` 槽，ratio-1 会漏掉一半 index page。

正确做法是把 NPU indexer buffer 规范化为 **full-page 4-D 视图** `[full_pages, S, 1, dim]`（`full_pages = npu_pages // I`，丢弃尾部 padding），host 行仍然是一个 full page；传输时 `page_size=S`，不再把 `I` 个 native page 扁平化到 host 行。

### 影响

- **KV（NPU）**：host 页宽只有实际值的 `1/dim`（`S*itemsize` vs `S*dim*itemsize`），host 池按错误页宽分配；备份/回载时越界或覆盖相邻页。
- **indexer（NPU）**：D2H 备份时可能直接抛 view/参数错误；即使不抛，也会把字节数当槽数，读写越界或错位。H2D 回载后 indexer K/scale 与 KV 不一致，**sparse attention 的 top-k 选择错误**，表现为前缀命中后输出漂移，而不是崩溃，问题更隐蔽。
- **GPU**：2-D 路径不受影响；任何“统一改成 4-D”的修复都会破坏现有 2-D 契约和测试。

### 为什么现有测试没发现

- `test_dsv4_compressed_pools.py:553-606` 已经覆盖低压缩比 host 池的 CPU/2-D 契约，并断言 `item_bytes == native.page_bytes * I` 与 2-D full-page 行；它不能覆盖 NPU 4-D。
- `test_dsv4_hicache_l2.py` 的 Ascend 用例使用 4-D fake buffer，但不是低压缩比路径。
- `test_npu_dsv4_hicache_l2_pool.py` 只覆盖 C4/C4-indexer（4-D，32 槽/页）。
- 没有 NPU 低压缩比 host 池测试，也没有 A5 E2E。

### 验证方法

1. CPU 单测（NPU 4-D KV）：构造 `[pages, S, 1, dim]` fake KV buffer，断言 `item_bytes == buffers[0][0].numel() * element_size`，并等于 `region.page_bytes`。
2. CPU 单测（NPU 4-D indexer）：构造 `[npu_pages, K, 1, dim]` fake indexer buffer，走 `_dsv4_low_ratio_regions` + `DeepSeekV4PagedHostPool`，断言：
   - 规范化后 device buffer 是 `[full_pages, S, 1, dim]`；
   - `_host_page_view` 形状 `[pages, 1, S, 1, dim]`（`page_first`）；
   - `_to_page_slots` 返回 `S` 个槽；
   - 传输算子收到 `page_size=S`。
3. 回归：现有 GPU 2-D 测试保持通过，断言 2-D 页行与 `item_bytes` 不变。
4. A5 冒烟：启动 V4.1 + HiCache，打印每个 PoolEntry 的 `item_bytes`、`device_buffers[0].shape`、`_host_page_view` 形状；执行一次含前缀命中的请求，检查是否报错、输出是否与无 HiCache 一致。

### 修复方向

采用 **full-page 传输**，不引入新的 host 池几何参数：

- **KV（G1a）**：`item_bytes = buffers[0][0].numel() * buffers[0].element_size()`（对 4-D 和 2-D 都正确），不再用 `shape[1] * element_size`。
- **indexer（G1b/G1c）**：NPU 4-D buffer 规范化为 `[full_pages, S, 1, dim]`；GPU 2-D 行保持不变。host 行仍按 full page 分配，`page_size=S`，索引展开 `S` 个槽。
- 这样 `_host_page_view` / `_to_page_slots` / `kernel_ascend` 的现有逻辑都能工作，不需要 `native_page_size` / `native_pages_per_full_page` 参数，也不会触碰 `page_first` 的跨页 stride。

## G2（中）host 内存预留没有覆盖 C1/C2

### 证据

`reserve_dsv4_paged_host_memory` 只统计 SWA、C4、C128、C4 state（`pool_host/npu_host_memory.py:329-393`），没有 C1/C2 以及它们的 indexer。

### 影响

- 默认 DSV4 使用 `SGLANG_HICACHE_HOST_MEM_BACKEND=acl`（`npu/utils.py:196-201`），ACL 后端继承的 `ensure_capacity` 是空操作（`npu_host_memory.py:81`），所以默认路径不受影响。
- 使用 `memfabric` 时，第一次 `ensure_capacity` 固定预留大小；漏掉 C1/C2 会导致后续分配失败或越界。当前只能通过“不使用 memfabric”规避。

### 验证/修复

- 单测断言 `reserve_dsv4_paged_host_memory` 的 `paged_blocks` 包含低压缩比池（按 full page 数 × page bytes × 层数，indexer 再乘 `I`）。
- 或在 v1 明确：DSV4.1 L2 只支持 ACL 后端，memfabric 留待 G2 修复后开放。

## G3（低，需产品确认）DSV4 不支持 `--hicache-size`

`_deepseek_v4_num_host_pages` 显式抛错：

```python
# hybrid_pool_assembler.py:551-555
if get_memory().hicache_size > 0:
    raise ValueError(
        "DeepSeek V4 HiCache currently does not support --hicache-size; "
        "use --hicache-ratio instead."
    )
```

首批接受“按 ratio 定容”。如果产品要求按 GB 配置，需要实现各池（FULL/SWA/C1/C2/indexer）的字节预算拆分，属于独立需求。

## G4（已修复，CPU 验证）自动定容不覆盖侧池

`hicache_auto_size._estimate_hicache_bytes` 原先只统计 `_SIZEABLE_POOLS` 或 layer-router 子池的设备字节，不统计 C1/C2/indexer 侧池。

已增加 `_dsv4_pool_bytes`：遍历 `DeepSeekV4TokenToKVPool.get_device_pool_infos()`，按每个 `DSV4PagedBufferInfo` 的 `page_bytes * buffer.shape[0]` 统计 SWA、C1/C2 KV 和 indexer K/scale。C4 持久状态（`DSV4CheckpointBufferInfo`）和 V4 draft 仍显式要求用户给 `--hicache-ratio`。CPU 用例：`test_hicache_auto_size.py::test_dsv4_paged_pools_are_sized_for_auto`。

## G5（中）测试覆盖缺口

- 已有 `test_dsv4_compressed_pools.py:553-606` 的 CPU/2-D 低压缩比装配测试，覆盖 `item_bytes` 与 2-D full-page 行。
- 已增加 NPU 4-D fake buffer 测试：`test_dsv4_hicache_l2.py::test_npu_pa_nd_rows_use_full_page_bytes`，覆盖 KV/indexer/scale 的 full-page 字节与 4-D 规范化。
- 仍无 A5 上 DSV4.1 HiCache 的端到端用例；现有 NPU HiCache E2E 是 Qwen3 / DeepSeek-R1（`test/registered/npu/basic_function/HiCache/`）。
- 无“强制 L2 命中”的验证：普通重复请求可能命中 L1，不能证明 host 路径。

对策见 [验收计划](../validation/acceptance_plan.md)：必须构造 device 淘汰后再命中的场景，并用 `sglang:prefill_effective_tokens_total{mode="host_hit"}` 证明 L2 命中。

## G6（中）DSpark draft 与 target 的 SWA 行宽兼容性

- DSpark/MTP 的 draft SWA 层会被追加到 target 的 SWA host 池（`hybrid_pool_assembler.py:811-838`），共享 `item_bytes`。
- V4.1 target 的 SWA 行是 `V41_SWA_FP8`（544 B/token，`dsv4_kv_layout.py:69-97`）；draft 被 pin 到 V4 行格式（提交 `3cc67a7544`）。
- SWA 装配路径原先**没有逐层检查** draft 与 target 的 `page_bytes`；只有 `_extra_host_region_entries` 对 extra region 做了检查（`hybrid_pool_assembler.py:725-729`）。若行宽不同，`_host_page_view` 会按 target 宽度解释 draft buffer。
- 已增加 `_check_dsv4_draft_swa_row_bytes`：用 `buffer[0].numel() * element_size()`（full-page bytes）比较 target 的 `page_bytes`，4-D PA_ND 上不会把槽数 `S` 与 `S*544` 误比。

但仅靠断言不够：当前 main 把 DSpark compressed-KV draft 判成 `PACKED`，draft 不会进入 `sidecar_draft_device_pools`，断言会在启动时直接失败。修复分两层：

- `base_spec_worker._packed_swa_rows_match_target()` 在计划阶段比较 target 与 draft 的 SWA `page_bytes`；
- 不一致时 pack 资格置空，`mtp_draft_device_pools` 保持空，计划降级为 `SIDECAR`；
- `build_swa_draft_pools()` 建独立的 `DRAFT_SWA` host pool（608 B 行），target SWA 保持 544 B 行。

2026-11-25 已在 A5 上完成 DSpark + HiCache 的启动与推理功能验证；greedy 输出/接受率/精度的严格 A/B 仍建议补测。

## G7（中）并行/部署组合未定义

HiCache 的既有约束（`hicache_hook.py:134-172`、`kv_cache_hook.py:640-675`）与 V4.1 A5 的实际部署组合没有形成矩阵：

| 组合 | 当前状态 | 建议 |
| --- | --- | --- |
| TP | MLA KV 按 rank 复制；host 池每 rank 私有 | 首批验证 |
| DP attention | 每 DP rank 独立调度与 host 池 | 首批验证（注意不要复用 A5 测试里的 `--disable-radix-cache`） |
| PP > 1 | host 池按 stage 本地层构建 | 延后，单独验收 |
| PD 分离 | prefill/decode 各自的 host 池与备份策略不同 | 延后 |
| DCP > 1 | 代码仅允许 MLA + 特定投机，且 V4.1 DCP 本身有独立约束 | 首批 DCP=1 |
| 投机（DSpark） | 已按独立 `DRAFT_SWA` sidecar 支持；A5 功能验证通过 | 补 A/B 精度对比 |

## G8（中）V4.1 压缩状态在命中回载后的重建语义

- V4.1 的 ratio 1/2 压缩状态是请求级（`NPUCompressStatePool`，`request_scoped=True`），不进入 HiCache；`_dsv4_low_ratio_entries` 的注释也写明“ratio-2 request state is rebuilt at page boundaries”（`hybrid_pool_assembler.py:667`）。
- 需要验证：host 前缀回载后，从页边界重新预热压缩器，得到的压缩槽与“不启用 HiCache、从头计算”一致。否则长前缀命中会在压缩层引入偏差。
- 设计必须明确 reset 点（清空 request-scoped state 的调用位置）并证明 128-token 的偶数 full-page 边界没有 pending pair；仅靠注释和 32-token 文本对比不足以证明。
- 验证方式：同一 prompt 在“HiCache 命中”与“冷启动”下的首 token logits/top-k 对比，以及 greedy 输出逐 token 对比。

## G9（低）容量与可观测

- 容量公式与节点级预算未文档化（见 [设计](../design/l2_host_offload.md#容量与定容)）。多 TP/DP rank 共享一台机器时，host 池总量是各 rank 之和，需要按 cgroup/物理内存做上限。
- `--hicache-host-memory-fraction` 的默认 0.8 与 ratio 默认值的关系容易误用；auto-size 已支持 V4.1 paged pools，但首批仍建议显式写 `--hicache-ratio` 以固定容量。
- 现有 `sglang:hicache_host_used_tokens` 只统计锚点池（`scheduler_components/metrics_reporter.py:1220-1221`），不统计 C1/C2/indexer 字节。首批必须补每池的 `item_bytes`、页数、backup/load 次数与字节数，否则无法定位 G1/G2。

## G10（已修复，待 A5 验证）只开 `--enable-hierarchical-cache` 的默认启用路径

### 原问题

- `hicache_host_memory_fraction` 默认 `0.8`（`fields/memory.py:119-122`）。
- `handle_hicache_ratio_default` 用 `resolving_view(server_args)` 读取 `hicache_ratio`；当用户没有显式设置时，它在函数末尾才 `declare_resolution(..., hicache_ratio=2.0)`（`hicache_hook.py:123-131`）。同一次读取里 `cfg.hicache_ratio` 仍是 `None`，所以第 128 行的 `if cfg.hicache_ratio is not None or cfg.hicache_size > 0` 不会把 fraction 置空。
- `auto_size_hicache` 看到 fraction 非空就会运行（`hicache_auto_size.py:110-112`），而 `_estimate_hicache_bytes` 原先对 `DeepSeekV4TokenToKVPool` 不支持（V4 pool 不在 `_SIZEABLE_POOLS`，也没有 `layer_router`），启动失败。

### 已实现修复

- `_estimate_hicache_bytes` 增加 V4 分支，调用 `_dsv4_pool_bytes` 统计 SWA、C1/C2 KV、indexer K/scale 的设备字节。
- 带 C4 持久状态（`DSV4CheckpointBufferInfo`）或 V4 draft 的模型仍抛明确错误，要求显式 `--hicache-ratio`。
- 单测：`test_hicache_auto_size.py::test_dsv4_paged_pools_are_sized_for_auto`。

### 待验证

- A5 上只给 `--enable-hierarchical-cache` 启动，确认 auto-size 选出的 ratio、host 池总量和 v4.1 侧池一致；显式 ratio 的路径仍是首批推荐配置。

## 优先级汇总

| ID | 优先级 | 内容 | 阻断首批？ |
| --- | --- | --- | --- |
| G1 | 高 | NPU 4-D 低压缩比 host 池几何：KV `item_bytes` + indexer full-page 视图 | 已修复；A5 7 池页 CRC 通过 |
| G2 | 中 | host 内存预留漏 C1/C2（memfabric） | 代码已补；ACL 已验证，memfabric 待验证 |
| G5 | 中 | 已有 NPU 4-D 单测；A5 E2E 用脚本验证 | 否（未注册 CI） |
| G8 | 中 | 压缩状态页边界重建正确性 | 清理/边界不变量已验证；数值等价待定 |
| G10 | 中 | 只开 HiCache 不开 ratio 的默认路径失败 | 代码已修，auto-size 路径待 A5 |
| G6 | 中 | DSpark draft 行宽 | 已修复；A5 功能验证通过 |
| G7 | 中 | 并行组合矩阵 | 否（限定首批组合） |
| G3 | 低 | `--hicache-size` 不支持 | 否（用 ratio） |
| G4 | 低 | 自动定容不覆盖侧池 | 否（首批显式 ratio） |
| G9 | 低 | 容量文档与观测接入 | 否 |
