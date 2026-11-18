# 代码改动清单

主仓 `sglang`，分支 `dsv41-a5-hicache-l2`（工作区 `D:/workspace/SGLang/sglang-dsv41-a5-hicache-l2`）。**本文档只给方案，代码尚未实施。**

## 0. 提交拆分

| 顺序 | 提交 | 内容 | 依赖 |
| --- | --- | --- | --- |
| 1 | `dsv41 hicache: normalize NPU low-ratio host pages to full-page rows` | 修正 NPU 4-D KV 的 `item_bytes`，把 NPU indexer 规范化为 full-page 4-D 视图；GPU 2-D 保持不变 | 无 |
| 2 | `dsv41 hicache: cover low-ratio pools in host memory reservation` | 补齐 C1/C2 + indexer 的预留 | 1 |
| 3 | `dsv41 hicache: add low-ratio host pool and transfer unit tests` | CPU 契约测试 | 1, 2 |
| 4 | `dsv41 hicache: guard draft SWA row width in packed host pools` | DSpark 行宽断言 | 1 |
| 5 | `dsv41 hicache: extend auto-size estimate to V4 sidecars`（可选） | G4/G10；首批用显式 ratio 规避 | 2 |
| 6 | `npu tests: add DSV4.1 HiCache L2 end-to-end` | 注册 NPU CI | 1–3 |

## 1. 提交 1：低压缩比 host 页几何

**核心原则：host 页仍然是一个 full page，传输也以 full page 为单位。**

- KV 的 host 页：`S = page_size // ratio` 个槽，`item_bytes = S * row_bytes`。
- indexer 的 host 页：同样是一个 full page，`item_bytes = S * row_bytes = I * native_page_bytes`。
- 不新增 `slots_per_full_page` / `native_page_size` / `native_pages_per_full_page` 参数，也不把 `page_first` 的 host 行跨页扁平化。

原因：`page_first` 的 host 层视图是 `self.kv_buffer[:, l, :]`，相邻 host 页之间隔着其他 layer 的字节；`num_host_pages * I` 个 native page 不是连续内存，`.view()` 会直接失败。把传输页设成 full page 后，host 视图是 `[pages, 1, S, 1, dim]`，页与页之间的 gap 是算子明确允许的（`sgl-kernel-npu/docs/api/en/operators/mem_cache/kv_transfer.md`：“gaps between pages in Host layer views are allowed”）。

### 1.1 `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`

`_dsv4_low_ratio_regions`（`600-657`）改为按维度分派，并返回每个 region 的 `item_bytes`：

```python
# KV: NPU 4-D PA_ND / GPU 2-D page row，均保持原样。
item_bytes = kv.buffer_info.page_bytes
regions.append(
    (names[0], kvcache.kv_pools[ratio], list(kv.buffer_info.buffers),
     layer_mapping, item_bytes)
)

for name in names[1:]:
    index = infos.get(name)
    if index is None:
        continue
    rows = []
    for buffer in index.buffer_info.buffers:
        full_pages = buffer.shape[0] // index_pages_per_full_page
        page = buffer[: full_pages * index_pages_per_full_page]
        if page.ndim == 4:
            # NPU PA_ND: [full_pages*I, K, 1, dim] -> [full_pages, S, 1, dim]
            page = page.reshape(
                full_pages, slots_per_page, *page.shape[2:]
            )
        else:
            # GPU 2-D: [full_pages*I, native_page_bytes] -> [full_pages, full_page_bytes]
            page = page.view(torch.uint8).reshape(full_pages, -1)
        rows.append(page)
    item_bytes = index_pages_per_full_page * index.buffer_info.page_bytes
    regions.append((name, kvcache.index_pools[ratio], rows, layer_mapping, item_bytes))
```

要点：

- `full_pages = buffer.shape[0] // I`，丢弃尾部 padding；`I = slots_per_page // key.page_slots`，已有断言 `slots_per_page % key.page_slots == 0`。
- `page.ndim == 4` 只会在 NPU PA_ND 出现；GPU 2-D 行继续走原来的 `view(uint8).reshape(full_pages, -1)`，不要无条件改成 4-D。
- 每个 region 用自己的 `buffer_info.page_bytes` 计算 `item_bytes`；indexer K 与 scale 的行宽不同，不能复用一个 `index_item_bytes`。

`_dsv4_low_ratio_entries`（`660-690`）不再用 `shape[1] * element_size`：

```python
for name, device_pool, buffers, layer_mapping, item_bytes in \
        _dsv4_low_ratio_regions(kvcache, infos, page_size):
    assert item_bytes == buffers[0][0].numel() * buffers[0].element_size(), (
        f"{name}: full-page item_bytes mismatch: {item_bytes} vs "
        f"{buffers[0][0].numel() * buffers[0].element_size()}"
    )
    entries.append(
        build_pool_entry(
            name=name,
            host_pool=DeepSeekV4PagedHostPool(
                pool_name=str(name),
                device_buffers=buffers,
                item_bytes=item_bytes,
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

### 1.2 `memory_pool_host.py`：不改 API

`DeepSeekV4PagedHostPool` 的现有逻辑在规范化后即可工作：

- `_host_page_view` 用 `device_buffer.shape[1]` 作为每页槽数；KV 的 `shape[1]=S`，NPU indexer 规范化后也是 `S`。
- `_to_native_page_indices` 用同一个 `shape[1]` 展开；一个 full page 展开 `S` 个槽。
- `kernel_ascend` 的 `page_size` 自动变成 `S`；C1 indexer `S=128`，C2 indexer `S=64`，KV 同样是 `S`。
- host 视图：`page_first` 为 `[pages, 1, S, 1, dim]`（页间 gap 合法），`layer_first` 为 `[1, pages, S, 1, dim]`。

不需要新增构造参数，也不需要在 `_host_page_view` 里做 `num_host_pages*I` 的扁平化。顺手把 `kernel_ascend` 分支里的局部变量 `native_page_size` 改名为 `page_slots`（值仍是 `shape[1]`），避免在 full-page 传输语义下继续使用“native page”这个名字；`_to_native_page_indices` 同步改名为 `_to_page_slots`。这是纯命名清理，不改行为。

### 1.3 测试

新增到 `test/registered/unit/mem_cache/test_dsv4_hicache_l2.py`：

1. **GPU 2-D 回归**：现有 `test_dsv4_compressed_pools.py:553-606` 必须保持通过；`_dsv4_low_ratio_regions` 对 2-D indexer 仍返回 `[full_pages, full_page_bytes]`，`item_bytes` 不变。
2. **NPU 4-D KV**：构造 `[pages, S, 1, dim]` fake KV buffer，断言 `item_bytes == S * dim * itemsize`，而不是 `S`。
3. **NPU 4-D indexer**：构造 `[npu_pages, K, 1, dim]` fake indexer/scale buffer，断言：
   - `_dsv4_low_ratio_regions` 返回的 device buffer 是 `[full_pages, S, 1, dim]`；
   - `item_bytes == I * key.page_bytes`（scale 用自己的 `page_bytes`）；
   - `_host_page_view` 形状为 `[pages, 1, S, 1, dim]`（`page_first`）；
   - `_to_native_page_indices` 返回 `S` 个槽。
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

## 5. 提交 5（可选）：自动定容 / 默认 ratio

G10 的规避方案是首批**必须显式给 `--hicache-ratio`**；只开 `--enable-hierarchical-cache` 当前会在 auto-size 阶段失败。若要把默认路径做成可用，二选一：

- 扩展 `_estimate_hicache_bytes`，支持 `DeepSeekV4TokenToKVPool` 及其 C1/C2/indexer 侧池（与 G4 一起做）。
- 在 `handle_hicache_ratio_default` 填入默认 ratio 的同时把 `hicache_host_memory_fraction` 置空；需要评估对其他模型的默认行为影响。

在这两项完成前，提交 5 可以不做；验收命令必须显式写 ratio。

## 6. 提交 6：A5 端到端用例

位置建议：`test/registered/npu/basic_function/HiCache/test_npu_dsv41_hicache_l2.py`，`register_npu_ci` 选择与 DSV4.1 A5 匹配的 suite。

用例至少覆盖：

1. 冷启动 + 重复长前缀（≥ 3 页）→ 第二次 `cached_tokens > 0`；
2. 构造 device 淘汰（唯一前缀数 > device 页数、但 < `ratio × device_pages`）后重复首个前缀 → `sglang:prefill_effective_tokens_total{mode="host_hit"}` 增加（证明走 L2，而非 L1）；
3. 短 prompt（< 1 页）不命中；
4. 不同前缀不误命中；
5. greedy 输出与关闭 HiCache 的基线逐 token 一致；
6. 启动日志中的 host 池 GB 与容量公式一致；每个池打印 `item_bytes`、页数、`backup/load` 字节数；
7. 只给 `--enable-hierarchical-cache` 不给 ratio 时，得到明确错误或按 G10 修复成功启动。

详细步骤与通过标准见 [验收计划](../validation/acceptance_plan.md)。

## 7. 不做/另立需求

- `--hicache-size` 的字节预算拆分（G3）。
- memfabric 后端的完整支持（G2 修完后开放）。
- L3 存储、PD 分离、DSpark + HiCache 的正式支持（G6/G7）。
