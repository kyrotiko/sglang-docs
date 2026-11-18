# 代码改动清单

主仓 `sglang`，分支 `dsv41-a5-hicache-l2`（工作区 `D:/workspace/SGLang/sglang-dsv41-a5-hicache-l2`）。**本文档只给方案，代码尚未实施。**

## 0. 提交拆分

| 顺序 | 提交 | 内容 | 依赖 |
| --- | --- | --- | --- |
| 1 | `dsv41 hicache: normalize NPU low-ratio host pages to full-page rows` | 修正 NPU 4-D KV 的 `item_bytes`，把 NPU indexer 规范化为 full-page 4-D 视图；GPU 2-D 保持不变 | 无 |
| 2 | `dsv41 hicache: cover low-ratio pools in host memory reservation` | 补齐 C1/C2 + indexer 的预留 | 1 |
| 3 | `dsv41 hicache: add low-ratio host pool and transfer unit tests` | CPU 契约测试 | 1, 2 |
| 4 | `dsv41 hicache: guard draft SWA row width in packed host pools` | DSpark 行宽断言 | 1 |
| 5 | `dsv41 hicache: size V4 paged pools in auto-size` | G4/G10；覆盖 C1/C2/indexer；C4 state/draft 显式拒绝 | 2 |
| 6 | `npu tests: add DSV4.1 HiCache L2 end-to-end` | 注册 NPU CI | 1–3 |

## 1. 提交 1：低压缩比 host 页几何

**核心原则：host 页仍然是一个 full page，传输也以 full page 为单位。**

- KV 的 host 页：`S = page_size // ratio` 个槽，`item_bytes = S * row_bytes`。
- indexer 的 host 页：同样是一个 full page，`item_bytes = S * row_bytes = I * native_page_bytes`。
- 不新增 `slots_per_full_page` / `native_page_size` / `native_pages_per_full_page` 参数，也不把 `page_first` 的 host 行跨页扁平化。

原因：`page_first` 的 host 层视图是 `self.kv_buffer[:, l, :]`，相邻 host 页之间隔着其他 layer 的字节；`num_host_pages * I` 个 native page 不是连续内存，`.view()` 会直接失败。把传输页设成 full page 后，host 视图是 `[pages, 1, S, 1, dim]`，页与页之间的 gap 是算子明确允许的（`sgl-kernel-npu/docs/api/en/operators/mem_cache/kv_transfer.md`：“gaps between pages in Host layer views are allowed”）。

### 1.1 `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`

`_dsv4_low_ratio_regions`（`600-657`）改为按维度分派；KV buffer 原样保留，indexer buffer 通过 `_dsv4_full_page_index_row` 归一化为 full-page 行：

```python
def _dsv4_full_page_index_row(buffer, *, full_pages, index_pages_per_full_page,
                              slots_per_full_page):
    page = buffer[: full_pages * index_pages_per_full_page]
    if page.ndim == 4:
        # NPU PA_ND: [full_pages*I, K, 1, dim] -> [full_pages, S, 1, dim]
        return page.reshape(full_pages, slots_per_full_page, *page.shape[2:])
    # GPU 2-D: [full_pages*I, native_page_bytes] -> [full_pages, full_page_bytes]
    return page.view(torch.uint8).reshape(full_pages, -1)
```

要点：

- `full_pages = buffer.shape[0] // I`，丢弃尾部 padding；`I = slots_per_page // key.page_slots`，已有断言 `slots_per_page % key.page_slots == 0`。
- `page.ndim == 4` 只会在 NPU PA_ND 出现；GPU 2-D 行继续走原来的 `view(uint8).reshape(full_pages, -1)`，不要无条件改成 4-D。
- region 仍返回 4 元组，不携带 `item_bytes`，避免改 `linker_pool_assembler` 的解包；host 页字节由 entries 从 buffer 本身计算。

`_dsv4_low_ratio_entries`（`660-690`）不再用 `shape[1] * element_size`：

```python
def _dsv4_full_page_bytes(buffers):
    return buffers[0][0].numel() * buffers[0].element_size()

for name, device_pool, buffers, layer_mapping in \
        _dsv4_low_ratio_regions(kvcache, infos, page_size):
    entries.append(
        build_pool_entry(
            name=name,
            host_pool=DeepSeekV4PagedHostPool(
                pool_name=str(name),
                device_buffers=buffers,
                item_bytes=_dsv4_full_page_bytes(buffers),
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout=get_memory().hicache_mem_layout,
                allocator_type=_get_allocator_type(),
                page_aligned_only=True,
            ),
            device_pool=device_pool,
            layer_mapping=layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max,
        )
    )
```

`_dsv4_full_page_bytes` 对 2-D `[pages, page_bytes]` 和 4-D `[pages, slots, 1, dim]` 都返回 full-page 字节；indexer K/scale 各自从自己的 buffer 计算，不共用 `index_item_bytes`。

### 1.2 `memory_pool_host.py`：不改 API

`DeepSeekV4PagedHostPool` 的现有逻辑在规范化后即可工作：

- `_host_page_view` 用 `device_buffer.shape[1]` 作为每页槽数；KV 的 `shape[1]=S`，NPU indexer 规范化后也是 `S`。
- `_to_page_slots` 用同一个 `shape[1]` 展开；一个 full page 展开 `S` 个槽。
- `kernel_ascend` 的 `page_size` 自动变成 `S`；C1 indexer `S=128`，C2 indexer `S=64`，KV 同样是 `S`。
- host 视图：`page_first` 为 `[pages, 1, S, 1, dim]`（页间 gap 合法），`layer_first` 为 `[1, pages, S, 1, dim]`。

不需要新增构造参数，也不需要在 `_host_page_view` 里做 `num_host_pages*I` 的扁平化。`kernel_ascend` 分支里的局部变量已从 `native_page_size` 改名为 `page_slots`，`_to_native_page_indices` 已改名为 `_to_page_slots`；这是纯命名清理，不改行为。

### 1.3 测试

新增到 `test/registered/unit/mem_cache/test_dsv4_hicache_l2.py`：

1. **GPU 2-D 回归**：现有 `test_dsv4_compressed_pools.py:553-606` 必须保持通过；`_dsv4_low_ratio_regions` 对 2-D indexer 仍返回 `[full_pages, full_page_bytes]`，`item_bytes` 不变。
2. **NPU 4-D KV**：构造 `[pages, S, 1, dim]` fake KV buffer，断言 `item_bytes == S * dim * itemsize`，而不是 `S`。
3. **NPU 4-D indexer**：构造 `[npu_pages, K, 1, dim]` fake indexer/scale buffer，断言：
   - `_dsv4_low_ratio_regions` 返回的 device buffer 是 `[full_pages, S, 1, dim]`；
   - `item_bytes == I * key.page_bytes`（scale 用自己的 `page_bytes`）；
   - `_host_page_view` 形状为 `[pages, 1, S, 1, dim]`（`page_first`）；
   - `_to_page_slots` 返回 `S` 个槽。
4. **传输调用**：mock `transfer_kv_all_layer_kv_split(_lf_pf)` 与 `transfer_kv_per_layer_kv_split(_pf_lf)`，断言 `page_size == S`，`device_k.shape == (1, full_pages, S, 1, dim)`，索引覆盖 `S` 个连续槽。
5. 覆盖 ratio 1 与 ratio 2（C1 indexer `I=2, S=128`；C2 indexer `I=1, S=64`）。

## 2. 提交 2：host 内存预留覆盖 C1/C2

`python/sglang/srt/mem_cache/pool_host/npu_host_memory.py:329-393`：

在 `reserve_dsv4_paged_host_memory` 中追加低压缩比池。**不需要新增 `page_size` 参数**：用 KV region 的 `page_slots` 就是 full page 的槽数 `S`。

```python
for ratio, kv_name, index_names in (
    (1, PoolName.DEEPSEEK_V4_C1,
     (PoolName.DEEPSEEK_V4_C1_INDEXER, PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE)),
    (2, PoolName.DEEPSEEK_V4_C2,
     (PoolName.DEEPSEEK_V4_C2_INDEXER, PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE)),
):
    kv = infos.get(kv_name)
    if kv is None or not kv.layer_ids:
        continue
    add_paged(kv_name, num_host_pages)
    slots_per_full_page = kv.buffer_info.page_slots
    for name in index_names:
        info = infos.get(name)
        if info is None:
            continue
        native_slots = info.buffer_info.page_slots
        assert slots_per_full_page % native_slots == 0
        add_paged(name, num_host_pages * (slots_per_full_page // native_slots))
```

注意 `add_paged` 的语义是 `pages * region.page_bytes`；indexer 的 `region.page_bytes` 是单个 native index page 的字节数，因此必须乘 `I`。KV 的 `page_bytes` 已经是 full page 字节数，只乘 `num_host_pages`。

补充单测：`test_dsv4_hicache_l2.py::test_dsv4_acl_capacity_declares_paged_pool_bytes` 扩展为 v4.1 侧池，断言总字节包含 C1/C2 与 indexer。

## 3. 提交 3：单元测试补齐

- `_dsv4_low_ratio_regions` / `_dsv4_low_ratio_entries` 的 NPU 4-D 装配测试。
- indexer full-page 传输几何测试（见 §1.3）。
- GPU 2-D 回归测试（现有测试保持通过）。
- 预留测试（见 §2）。

## 4. 提交 4：draft SWA 行宽断言

`hybrid_pool_assembler.py:811-838`（`mtp_swa_device_buffers` 收集处）：

```python
for buffer in draft_swa.buffer_info.buffers:
    full_page_bytes = buffer[0].numel() * buffer.element_size()
    if full_page_bytes != swa_info.page_bytes:
        raise ValueError(
            "packed DSV4 draft SWA rows do not match the target's "
            f"{swa_info.page_bytes} B pages: {buffer.shape=}"
        )
```

4-D PA_ND 上 `buffer.shape[1] * element_size()` 只会得到槽数 `S`，不能用它和 `S*544` 比较；`buffer[0].numel() * element_size()` 对 4-D 和 2-D 都是 full-page 字节数。

只在启用投机时触发；首批即使不启用，也应加上显式拒绝，避免静默错读。

## 5. 提交 5：自动定容支持 V4 paged pools

`hicache_auto_size._estimate_hicache_bytes` 增加 V4 分支：

- `_dsv4_pool_bytes` 遍历 `get_device_pool_infos()`，统计 SWA、C1/C2 KV、indexer K/scale 的 `page_bytes * pages`。
- `DSV4CheckpointBufferInfo`（C4 持久状态）和 V4 draft 池显式拒绝 auto-size，要求用户给 `--hicache-ratio`；首批不做这两类组合。
- CPU 用例：`test_hicache_auto_size.py::test_dsv4_paged_pools_are_sized_for_auto`。

首批仍建议显式写 ratio 以固定容量；auto-size 是默认启用路径的兜底。

## 6. 提交 6：A5 端到端用例

位置建议：`test/registered/npu/basic_function/HiCache/test_npu_dsv41_hicache_l2.py`，`register_npu_ci` 选择与 DSV4.1 A5 匹配的 suite。

用例至少覆盖：

1. 冷启动 + 重复长前缀（≥ 3 页）→ 第二次 `cached_tokens > 0`；
2. 构造 device 淘汰（唯一前缀数 > device 页数、但 < `ratio × device_pages`）后重复首个前缀 → `sglang:prefill_effective_tokens_total{mode="host_hit"}` 增加（证明走 L2，而非 L1）；
3. 短 prompt（< 1 页）不命中；
4. 不同前缀不误命中；
5. greedy 输出与关闭 HiCache 的基线逐 token 一致；
6. 启动日志中的 host 池 GB 与容量公式一致；每个池打印 `item_bytes`、页数、`backup/load` 字节数；
7. 只给 `--enable-hierarchical-cache` 不给 ratio 时，auto-size 能选出一个可用 ratio；显式 ratio 路径作为对照。

详细步骤与通过标准见 [验收计划](../validation/acceptance_plan.md)。

## 7. 不做/另立需求

- `--hicache-size` 的字节预算拆分（G3）。
- memfabric 后端的完整支持（G2 修完后开放）。
- L3 存储、PD 分离、DSpark + HiCache 的正式支持（G6/G7）。
