# PR #53204 代码说明

本文按文件、按 hunk 说明 `dsv41-a5-hicache-l2` 相对 `origin/main` 的全部代码改动，以及每一处为什么这么改。对应 PR：

- <https://github.com/sgl-project/sglang/pull/53204>
- 分支：`dsv41-a5-hicache-l2` @ `b7a1bcf25d`
- 基线：`origin/main` @ `a662e1efbb`
- 规模：9 个文件，+569 / -50

## 0. 先理解数据模型

后面所有改动都围绕这几个量：

| 名称 | 含义 |
| --- | --- |
| `page_size` | 逻辑 full page 的 token 数，V4.1 为 128 |
| `S` | 一个 full page 在某个压缩池里的槽数，`S = page_size // ratio` |
| `K` | indexer native page 的槽数，当前实现为 64 |
| `I` | 一个 full page 包含的 native index page 数，`I = S / K` |
| `row_bytes` | 一个槽的字节数（KV 或 indexer K/scale 各自不同） |
| `item_bytes` | host 池一个 full page 的字节数 |

V4.1 的实际取值：

| 池 | ratio | `S` | indexer `K` | `I` | `item_bytes` |
| --- | --- | --- | --- | --- | --- |
| SWA | — | 128 | — | — | 128 × 544 = 69632 |
| C1 KV | 1 | 128 | — | — | 128 × 320 = 40960 |
| C1 indexer K | 1 | 128 | 64 | 2 | 128 × 64 = 8192 |
| C1 indexer scale | 1 | 128 | 64 | 2 | 128 × 4 = 512 |
| C2 KV | 2 | 64 | — | — | 64 × 320 = 20480 |
| C2 indexer K | 2 | 64 | 64 | 1 | 64 × 64 = 4096 |
| C2 indexer scale | 2 | 64 | 64 | 1 | 64 × 4 = 256 |
| DSpark draft SWA | — | 128 | — | — | 128 × 608 = 77824 |

两个硬件差异是问题的根源：

1. **GPU 的 indexer buffer 是 2-D 行**：`[pages, native_page_bytes]`；**NPU 是 4-D PA_ND**：`[pages, K, 1, dim]`。原实现无脑做 `.view(torch.uint8).reshape(full_pages, -1)`，在 4-D 上会把 PA_ND 页当成连续字节行，形状和 Ascend 传输路径对不上。
2. **NPU 上 `buffer.shape[1]` 是槽数 `S`，不是字节数**。原 `item_bytes = buffers[0].shape[1] * element_size()` 在 4-D 上只得到 `S` 个元素的大小，而不是 `S * row_bytes`。

另一个独立问题是 DSpark：draft 是 V4 架构，SWA 行宽 608 B/token；V4.1 target 是 544 B/token。两者不能共用 target 的 SWA host 页。

---

## 1. `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`

### 1.1 模块级 `import torch`

```python
+import torch
```

新增的 `_dsv4_full_page_index_row`、`_dsv4_full_page_bytes` 和 `_check_dsv4_draft_swa_row_bytes` 都在签名/注解里用了 `torch.Tensor`。模块有 `from __future__ import annotations`，运行时不会求值注解，但 ruff 的 `F821` 会报 `Undefined name torch`。

原文件只在两个函数内部 `import torch`；把 import 提到模块级后，这两处局部 import 也删掉了（见 1.5）。这是纯规范修复，不改变运行时行为。

### 1.2 新增 `_dsv4_full_page_index_row`

```python
+def _dsv4_full_page_index_row(
+    buffer: torch.Tensor,
+    *,
+    full_pages: int,
+    index_pages_per_full_page: int,
+    slots_per_full_page: int,
+) -> torch.Tensor:
+    page = buffer[: full_pages * index_pages_per_full_page]
+    if page.ndim == 4:
+        return page.reshape(full_pages, slots_per_full_page, *page.shape[2:])
+    return page.view(torch.uint8).reshape(full_pages, -1)
```

逐行说明：

- `page = buffer[: full_pages * I]`：设备池可能按 full page 向上取整分配，尾部有多余的 native page；截掉尾部，保证行数正好是 `full_pages * I`。
- `page.ndim == 4`：只有 NPU PA_ND 走到这里，形状是 `[full_pages*I, K, 1, dim]`。
- `reshape(full_pages, S, *page.shape[2:])`：把 `I` 个 `K` 槽折叠成一个 full page 的 `S` 槽，结果形状 `[full_pages, S, 1, dim]`。`page.shape[2:]` 保留 `1, dim`，避免把 PA_ND 的尾维拍平；这样 Ascend 传输仍然看到 5-D 视图（`unsqueeze(0)` 后）。
- 2-D 分支保持原来的 `view(torch.uint8).reshape(full_pages, -1)`：GPU 的 indexer buffer 本来就是字节行，不能改成 4-D；这也是“不能无条件按 4-D 处理”的原因。

为什么单独抽函数：这段几何换算现在同时被 `_dsv4_low_ratio_regions` 和测试使用，抽出来后可以独立做 CPU 单测，不必启动 NPU。

### 1.3 新增 `_dsv4_full_page_bytes`

```python
+def _dsv4_full_page_bytes(buffers: list[torch.Tensor]) -> int:
+    return buffers[0][0].numel() * buffers[0].element_size()
```

- `buffers[0]`：同一 region 的所有 layer 共享相同的行宽，取第一个即可。
- `[0]`：取第一页。
- `numel() * element_size()`：一页所有元素的总字节数。对 2-D `[pages, bytes]` 和 4-D `[pages, S, 1, dim]` 都成立。
- 不能用 `shape[1] * element_size()`：4-D 上 `shape[1]` 是槽数 `S`，不是 `S * row_bytes`。

### 1.4 新增 `_check_dsv4_draft_swa_row_bytes`

```python
+def _check_dsv4_draft_swa_row_bytes(
+    draft_buffers: list[torch.Tensor], target_page_bytes: int
+) -> None:
+    for buffer in draft_buffers:
+        draft_page_bytes = buffer[0].numel() * buffer.element_size()
+        if draft_page_bytes != target_page_bytes:
+            raise ValueError(
+                "packed DSV4 draft SWA rows do not match the target's "
+                f"{target_page_bytes} B pages: {tuple(buffer.shape)}"
+            )
```

- 只有在“draft 被 packed 进 target SWA host 池”时才会调用。packed 的语义是 draft 层直接复用 target 的 host 行，因此 draft 的 full-page 字节必须等于 target 的 `item_bytes`。
- 用 `buffer[0].numel() * element_size()` 而不是 `shape[1] * element_size()`，理由同上：4-D PA_ND 上后者是槽数。
- 报错信息带 `target_page_bytes` 和实际 `shape`，启动时能直接看出是哪一侧宽度不一致。

注意：这个断言只是最后一道防线。DSpark 的 draft 在当前 main 上会被 `_can_pack_hicache_mtp()` 判成 `PACKED`，如果只加这个断言，启动会直接失败而不是回退。真正的选择逻辑在 `base_spec_worker.py`（第 5 节）。

### 1.5 `_dsv4_low_ratio_regions`：改用 helper

```python
-                    buffer[: full_pages * index_pages_per_full_page]
-                    .view(torch.uint8)
-                    .reshape(full_pages, -1)
+                    _dsv4_full_page_index_row(
+                        buffer,
+                        full_pages=full_pages,
+                        index_pages_per_full_page=index_pages_per_full_page,
+                        slots_per_full_page=slots_per_page,
+                    )
```

- 原来的内联 2-D 逻辑对 GPU 正确、对 NPU 4-D 错误；换成按 `ndim` 分派的 helper。
- `slots_per_page` 是该 ratio 的 `S`（来自 KV region 的 `page_slots`），传给 helper 作为折叠后的槽数。
- 同时删除了函数体内的 `import torch`，因为模块级已经 import。

### 1.6 `_dsv4_low_ratio_entries`：item_bytes 改为 full-page 字节

```python
-                    item_bytes=buffers[0].shape[1] * buffers[0].element_size(),
+                    item_bytes=_dsv4_full_page_bytes(buffers),
```

- 旧公式在 2-D 上返回一个 **native page** 的字节数（`I=1` 时恰好等于 full page，`I=2` 时少一半），在 NPU 4-D 上只得到“`S` 个元素的大小”；新公式对 2-D/4-D 都返回一个 full page 的字节数。
- 每个 region 独立计算：indexer K 和 scale 的行宽不同，不能共用一个 `index_item_bytes`。

### 1.7 `build_deepseek_v4_hicache_stack`：sidecar 跳过 + packed 守卫

```python
+        sidecar_draft_ids = {id(pool) for pool in params.sidecar_draft_device_pools}
         for pool in params.mtp_draft_device_pools:
+            if id(pool) in sidecar_draft_ids:
+                continue
             ...
             draft_swa = _dsv4_pool_infos(pool).get(PoolName.SWA)
             if draft_swa is None:
                 raise ValueError("Packed DSV4 draft requires paged SWA storage")
+            _check_dsv4_draft_swa_row_bytes(
+                draft_swa.buffer_info.buffers, swa_info.page_bytes
+            )
             mtp_swa_device_buffers.extend(draft_swa.buffer_info.buffers)
```

- `sidecar_draft_ids`：sidecar draft 会由 `build_hicache_draft_sidecars()` 单独建 `DRAFT_SWA` host pool；如果它同时出现在 `mtp_draft_device_pools` 里，不能再拼进 target SWA 池，否则同一份 draft KV 会被备份两次，且行宽可能不匹配。
- 用 `id(pool)` 而不是 `pool in ...`：池对象没有定义有意义的相等语义，`id()` 判断的是“同一对象”。
- `_check_dsv4_draft_swa_row_bytes`：对真正进入 packed 路径的 draft 做最终校验；宽度不一致就 fail loud，避免静默错读。
- 在当前 main 上，sidecar 模式会把 `mtp_draft_device_pools` 置空，所以 `sidecar_draft_ids` 通常是空的；保留这个 skip 是为了兼容同时填充两个列表的调用方，并有 `test_sidecar_draft_is_not_packed_into_target_swa` 锁定该行为。

---

## 2. `python/sglang/srt/mem_cache/memory_pool_host.py`

这个文件**只有重命名**，没有行为变化。

```python
-    def _to_native_page_indices(
-        self, indices: torch.Tensor, native_page_size: int
+    def _to_page_slots(
+        self, indices: torch.Tensor, slots_per_page: int
     ) -> torch.Tensor:
-        """Expand logical page indices into a device buffer's native slots."""
+        """Expand logical page indices into one page's token slots."""
         rows = self._to_page_indices(indices)
-        offsets = torch.arange(native_page_size, dtype=rows.dtype, device=rows.device)
-        return (rows[:, None] * native_page_size + offsets).reshape(-1)
+        offsets = torch.arange(slots_per_page, dtype=rows.dtype, device=rows.device)
+        return (rows[:, None] * slots_per_page + offsets).reshape(-1)
```

为什么改名：

- 经过第 1 节的 full-page 规范化后，`self.device_buffers[0].shape[1]` 已经是**一个 full page 的槽数 `S`**，不是 indexer 的 native `K`。
- 继续叫 `native_page_size` 会误导维护者，以为这里有“native page”和“full page”两套坐标。实际传输单位就是 full page，`page_size=S`。
- `_to_page_slots` 的语义仍然是“把一个逻辑页展开成页内所有 token 槽”，名字与 `slot_page_size` 对齐。

D2H/H2D 两个分支里的局部变量同步改名：

```python
-            native_page_size = int(self.device_buffers[0].shape[1])
+            page_slots = int(self.device_buffers[0].shape[1])
-            native_device_indices = ...
+            page_device_indices = self._to_page_slots(device_indices, page_slots)
...
-                        page_size=native_page_size,
+                        page_size=page_slots,
```

`transfer_kv_all_layer_kv_split*` / `transfer_kv_per_layer_kv_split*` 的 `page_size` 参数仍然传 `S`，所以 AIV 算子每次搬一个 full page；host 页之间的 gap 仍由 `page_first` 视图地址自然表达，和原设计一致。

---

## 3. `python/sglang/srt/mem_cache/pool_host/npu_host_memory.py`

### 3.1 预留覆盖 C1/C2 KV 与 indexer

```python
+    for kv_name, index_names in (
+        (PoolName.DEEPSEEK_V4_C1,
+         (PoolName.DEEPSEEK_V4_C1_INDEXER,
+          PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE)),
+        (PoolName.DEEPSEEK_V4_C2,
+         (PoolName.DEEPSEEK_V4_C2_INDEXER,
+          PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE)),
+    ):
+        kv = infos.get(kv_name)
+        if kv is None or not kv.layer_ids:
+            continue
+        add_paged(kv_name, num_host_pages)
+        slots_per_full_page = kv.buffer_info.page_slots
+        for name in index_names:
+            index = infos.get(name)
+            if index is None:
+                continue
+            native_slots = index.buffer_info.page_slots
+            if slots_per_full_page % native_slots:
+                raise ValueError(...)
+            add_paged(name, num_host_pages * (slots_per_full_page // native_slots))
```

逐行说明：

- `kv = infos.get(kv_name)`：模型可能没有 ratio 1 或 ratio 2 的层；没有就跳过。
- `add_paged(kv_name, num_host_pages)`：KV region 的 `page_bytes` 已经是 full-page 字节数，host 有 `num_host_pages` 个 full page，直接按页数预留。
- `slots_per_full_page = kv.buffer_info.page_slots`：从 KV region 取 `S`，不重复从 `page_size // ratio` 推算，避免两处公式漂移。
- `native_slots = index.buffer_info.page_slots`：indexer region 的 `page_bytes` 是**一个 native page** 的字节数，`page_slots` 是 `K`。
- `slots_per_full_page % native_slots`：要求 `S` 能被 `K` 整除（V4.1 是 128/64 或 64/64）。不能整除说明布局假设不成立，直接报错，不猜测。
- `add_paged(name, num_host_pages * I)`：indexer 的一个 host full page 覆盖 `I` 个 native page；`add_paged` 内部按 `pages * region.page_bytes` 计算，所以这里必须乘 `I`，否则只预留了 1/I 的字节。
- `add_paged` 对 `layer_first` 按 layer 分块、对 page_first 按整池分块，和实际 allocator 调用一致；这里不需要关心 layout。

为什么放在 SWA 预留之后、C4 之前：保持“先 SWA、再低压缩比、再 C4/C128”的顺序，和池的构建顺序一致，便于日志对照。

---

## 4. `python/sglang/srt/mem_cache/hicache_auto_size.py`

### 4.1 新增 import

```python
+from sglang.srt.mem_cache.deepseek_v4_buffer_info import (
+    DSV4CheckpointBufferInfo,
+    DSV4CompressedBufferInfo,
+    DSV4IndexerBufferInfo,
+    DSV4PagedBufferInfo,
+)
```

`_dsv4_pool_bytes` 需要按 region 类型分派，所以显式导入四个 buffer info 类型。

### 4.2 新增 `_paged_info_bytes`

```python
+def _paged_info_bytes(region: DSV4PagedBufferInfo) -> int:
+    return region.page_bytes * sum(buffer.shape[0] for buffer in region.buffers)
```

- `region.page_bytes`：一个 page 的字节数。对 KV 是 full page；对 indexer 是 native page。
- `sum(buffer.shape[0])`：所有 layer 的页数之和。
- 用 `page_bytes * 页数` 而不是直接 `buffer.numel() * itemsize`：前者和预留、host 池的几何使用同一个 `page_bytes`，2-D/4-D 两种 layout 不需要分别处理，也不依赖 buffer 的尾维是否被拍平。

### 4.3 新增 `_dsv4_pool_bytes`

```python
+def _dsv4_pool_bytes(pool: DeepSeekV4TokenToKVPool) -> int:
+    total = 0
+    for info in pool.get_device_pool_infos():
+        region = info.buffer_info
+        if isinstance(region, DSV4PagedBufferInfo):
+            total += _paged_info_bytes(region)
+        elif isinstance(region, DSV4CompressedBufferInfo):
+            total += _paged_info_bytes(region.kv)
+            if region.rope is not None:
+                total += _paged_info_bytes(region.rope)
+        elif isinstance(region, DSV4IndexerBufferInfo):
+            total += _paged_info_bytes(region.key)
+            if region.scale is not None:
+                total += _paged_info_bytes(region.scale)
+        elif isinstance(region, DSV4CheckpointBufferInfo):
+            raise ValueError(...)
+        else:
+            raise ValueError(...)
+    return total
```

逐类说明：

- `DSV4PagedBufferInfo`：SWA、C1/C2 KV 等普通分页池。
- `DSV4CompressedBufferInfo`：压缩池由 `kv` 和可选 `rope` 两部分组成，两者都要计入 host 镜像。
- `DSV4IndexerBufferInfo`：indexer 由 `key` 和可选 `scale` 组成，行宽不同，分别计数。
- `DSV4CheckpointBufferInfo`：C4 的持久 compressor 状态。这部分不按 ratio 镜像，auto-size 无法给出正确比例，直接要求显式 `--hicache-ratio`。
- 未知 region：说明出现了新的池类型，auto-size 不该猜；带类型名报错，方便扩展。

为什么不用 `_pool_bytes`：V4 pool 不在 `_SIZEABLE_POOLS` 里，也没有通用 `layer_router` 视图；必须按 `get_device_pool_infos()` 的 region 语义统计。

### 4.4 `_estimate_hicache_bytes` 增加 V4 分支

```python
     pool = params.token_to_kv_pool_allocator.get_kvcache()
-    if not (isinstance(pool, _SIZEABLE_POOLS) or _sub_pool_views(pool)):
-        raise ValueError(...)
-    total = _pool_bytes(pool)
+    if isinstance(pool, DeepSeekV4TokenToKVPool):
+        if params.mtp_draft_device_pools or (
+            draft_plan is not None and draft_plan.mode == "sidecar"
+        ):
+            raise ValueError(
+                "HiCache auto-sizing with DeepSeek V4 draft pools is not "
+                "supported; set --hicache-ratio explicitly."
+            )
+        total = _dsv4_pool_bytes(pool)
+    else:
+        if not (isinstance(pool, _SIZEABLE_POOLS) or _sub_pool_views(pool)):
+            raise ValueError(...)
+        total = _pool_bytes(pool)
     if params.req_to_token_pool.holds_state_slots:
         total += _leaf_bytes(params.token_to_kv_pool_allocator.state.pool)
+    if isinstance(pool, DeepSeekV4TokenToKVPool):
+        return total
```

- V4 分支放在通用分支之前：V4 pool 不满足 `_SIZEABLE_POOLS`，必须先识别。
- `mtp_draft_device_pools` 或 `sidecar` 计划存在时直接拒绝：V4 draft 的 host 镜像不是简单地等于 `_dsv4_pool_bytes`（sidecar 跟随 target SWA 槽，行宽还可能不同），auto-size 的公式不成立。要求显式 ratio，避免算小导致 OOM 或算大浪费内存。
- `total = _dsv4_pool_bytes(pool)`：统计所有 paged region。
- `holds_state_slots` 之后的状态字节：V4 的 recurrent state 也按 ratio 镜像，保留通用逻辑。
- V4 提前 `return total`：draft 已经在上面拒绝，后面的 `_draft_bytes` 只适用于非 V4 路径；提前返回避免把 V4 pool 当成有 `layer_router` 的普通池处理。

行为结果：

- 只开 `--enable-hierarchical-cache`：无 draft 的 V4.1 可以 auto-size。
- DSpark + HiCache：必须显式 `--hicache-ratio`，否则启动期给出明确报错。
- C4 持久状态：同样要求显式 ratio。

---

## 5. `python/sglang/srt/speculative/base_spec_worker.py`

### 5.1 logger

```python
+import logging
...
+logger = logging.getLogger(__name__)
```

新增的 fallback 需要在启动日志里说明“为什么没走 PACKED”。这是预期支持路径，不是 warning，用 `logger.info`。

### 5.2 新增 `_packed_swa_rows_match_target`

```python
+def _packed_swa_rows_match_target(target_pool, draft_pools) -> bool:
+    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
+
+    if not isinstance(target_pool, DeepSeekV4TokenToKVPool):
+        return True
+    target_swa = target_pool.swa_kv_pool
+    if target_swa is None:
+        return True
+    target_row_bytes = target_swa.paged_buffer_info().page_bytes
+    for draft in draft_pools:
+        if not isinstance(draft, DeepSeekV4TokenToKVPool):
+            continue
+        draft_swa = draft.swa_kv_pool
+        if draft_swa is None:
+            continue
+        if draft_swa.paged_buffer_info().page_bytes != target_row_bytes:
+            return False
+    return True
```

逐行说明：

- 局部 import：避免 `base_spec_worker` 在模块加载时把整条 V4 memory pool 依赖拉进来；该函数只在 HiCache 初始化时调用，成本可接受。
- 非 V4 target 直接返回 `True`：不改变其他模型/其他 draft 的 packed 行为。
- `target_swa is None`：target 没有 SWA 池，谈不上行宽不匹配；交给后续路径处理。
- `target_row_bytes`：target SWA 的 `paged_buffer_info().page_bytes` 是一个 page 的字节数，V4.1 为 544 × 128 = 69632。
- 非 V4 draft `continue`：只检查 V4 draft；其他 draft 的兼容性由既有校验负责。
- `draft_swa is None`：draft 没有 SWA 池，无需比较。
- 任意一个 V4 draft 的行宽不同就返回 `False`：packed 会让所有 draft 层共享 target 的 host `item_bytes`，只要有一个不匹配就不安全。

这里取的是 `draft.swa_kv_pool`。当前 V4 draft 的构造路径（`_build_dsv4_kv_pool`）不走 `follows_target_swa_ids`，所以 SWA 一定是子池；如果以后把 draft 改成 follows 模式，这里需要和 `build_swa_draft_pools()` 一样判断 `follows_target_swa_ids`。

### 5.3 `_build_hicache_draft_plan` 的 pack 决策

```python
-        if _can_pack_hicache_mtp(spec_algorithm, draft_runners):
+        pack_eligible = _can_pack_hicache_mtp(spec_algorithm, draft_runners)
+        if pack_eligible and not _packed_swa_rows_match_target(
+            target_model_runner.token_to_kv_pool, draft_pools
+        ):
+            logger.info(
+                "HiCache draft SWA rows differ from the target's; using a "
+                "DRAFT_SWA sidecar instead of the target SWA host pool."
+            )
+            pack_eligible = False
+        if pack_eligible:
             _check_hicache_draft_dcp_layout(HiCacheDraftMode.PACKED, draft_pools)
```

- `_can_pack_hicache_mtp` 仍然决定“算法上能不能 pack”（NextN、DSpark compressed KV 等）。
- 新增的几何检查决定“物理上行不行”：行宽不一致时把 `pack_eligible` 置空。
- `logger.info` 的内容是 A5 上定位问题的关键：看到这行就说明走了 sidecar，而不是 packed。
- 不抛异常：DSpark + V4.1 是受支持的组合，sidecar 是正确路径；抛异常会让用户无法启动。
- `_check_hicache_draft_dcp_layout(PACKED, ...)` 只在真正 packed 时执行，避免对 sidecar 计划做错校验。

随后代码自然落到**已有的** sidecar 分支（这部分不是本 PR 新增的行）：

```python
        sidecar_pools = draft_pools[:1]
        _check_hicache_draft_dcp_layout(HiCacheDraftMode.SIDECAR, sidecar_pools)
        return HiCacheDraftPlan(
            mode=HiCacheDraftMode.SIDECAR,
            device_pools=sidecar_pools,
        )
```

`target_model_runner.mtp_draft_device_pools` 在函数开头已被置空，所以 packed 循环不会看到这个 draft；`maybe_register_hicache_draft()` 之后调用 `build_hicache_draft_sidecars()`：

- 识别 `DeepSeekV4TokenToKVPool`；
- 走 `build_swa_draft_pools()`；
- 用 draft 自己的 `paged_buffer_info().page_bytes` 建 `DRAFT_SWA` host pool（608 × 128 = 77824）；
- target SWA 池保持 544 × 128 = 69632。

---

## 6. 测试改动

### 6.1 `test/registered/unit/mem_cache/test_dsv4_hicache_l2.py`

- 新增 import：`DSV4PagedBufferInfo`（构造 4-D fake buffer）、`DeepSeekV4TokenToKVPool`（调用 `clear_request_scoped_state`）、`DevicePoolInfo`（拼 `_dsv4_low_ratio_entries` 的输入）。

**`test_low_ratio_pools_reserve_indexer_native_pages`**

- 构造 C1/C2 KV、indexer K、indexer scale 的 fake `page_bytes/page_slots`。
- 期望字节数显式展开：每层 C1 KV 4 页 × 40960、C1 indexer 8 页 × 4096、C1 scale 8 页 × 256，再乘 2 层；C2 每层 4 页。
- 锁定“indexer 预留要乘 `I`”这个结论；如果以后有人把 `add_paged(name, num_host_pages)` 写成不乘 `I`，这个用例会红。

**`TestDSV4LowRatioHostPages.test_npu_pa_nd_rows_use_full_page_bytes`**

- 用 `[pages, K, 1, dim]` 的 fake buffer 构造 7 个 region。
- 断言每个池的 `item_bytes` 是 full-page 字节（40960/8192/512/20480/4096/256）。
- 断言 C1 indexer 的设备 buffer 形状变成 `(2, 128, 1, 64)`，C2 是 `(2, 64, 1, 64)`。
- 锁定“NPU 4-D 不能走 2-D 拍平”和“indexer `I=2` 折叠”两个契约。

**`test_ratio2_compressor_state_is_cleared_at_page_boundary`**

- 构造 ratio-2、request-scoped 的 `CompressStatePool`，先把 ring 填成 1.0。
- 调用 `clear_request_scoped_state`，断言 ring 的 kv 为 0、score 为 `-inf`。
- `seq_len=768` 是偶数页边界，`transfer_indices` 必须为空；`769` 才返回 pending pair `[1]`。
- 锁定 G8 的清理/边界不变量：host 命中恢复到 768 时不会继承上一个请求的半对状态。

**`test_draft_swa_row_bytes_must_match_target`**

- `544` 宽度、`128*544` target：通过。
- `545` 宽度：抛 `ValueError`，消息包含 `do not match`。
- 锁定 packed 路径的最后一道守卫。

### 6.2 `test/registered/unit/mem_cache/test_dsv4_compressed_pools.py`

- 既有 packed 测试的 `SimpleNamespace` 补 `sidecar_draft_device_pools=()`，因为 `build_deepseek_v4_hicache_stack` 现在会读这个字段。
- 新增 `test_sidecar_draft_is_not_packed_into_target_swa`：
  - 同一个 draft 同时放进 `mtp_draft_device_pools` 和 `sidecar_draft_device_pools`；
  - 断言 SWA host 池只有 target 的 3 个 buffer，没有 draft 的 2 个；
  - 锁定“sidecar 不并入 target SWA”的 skip 逻辑。

### 6.3 `test/registered/unit/mem_cache/test_hicache_auto_size.py`

- 新增 import：`DSV4PagedBufferInfo`、`DeepSeekV4TokenToKVPool`、`DevicePoolInfo`、`PoolName`，用于构造 V4 pool 的 `get_device_pool_infos()` 返回值。

`test_dsv4_paged_pools_are_sized_for_auto`：

- 用 7 个 `DevicePoolInfo` 模拟 SWA、C1/C2 KV、indexer K/scale。
- 期望值是各池 `pages * page_slots * row_bytes` 的和。
- 锁定 V4 分支确实遍历 `get_device_pool_infos()` 并按 region 类型统计；如果漏掉 indexer 或 scale，用例会红。

### 6.4 `test/registered/unit/mem_cache/test_hicache_mtp_state.py`

- `_dsv4_pool_stub` 内部局部 import `DeepSeekV4TokenToKVPool`，用 `__new__` 绕过 `__init__` 构造最小池，只提供 `swa_kv_pool.paged_buffer_info().page_bytes` 和 `layer_router.part_view()`。
- `test_dspark_draft_swa_rows_decide_the_hicache_layout` 参数化两条路径：
  - target 544 / draft 544：`PACKED`，`mtp_draft_device_pools == (draft_pool,)`；
  - target 544 / draft 608：`SIDECAR`，`plan.device_pools == (draft_pool,)`，且 `mtp_draft_device_pools == ()`。
- 锁定“行宽决定 packed/sidecar”以及 sidecar 时 packed 列表必须为空。

---

## 7. 行为矩阵

| 场景 | 改动前 | 改动后 |
| --- | --- | --- |
| GPU 2-D indexer host 池 | 正确 | 不变 |
| NPU 4-D indexer host 池 | `item_bytes` 少乘行宽、形状错误 | full-page 字节 + `[pages, S, 1, dim]` |
| C1/C2 + indexer 预留（memfabric） | 漏预留 | 按 `num_host_pages * I` 预留 |
| 只开 `--enable-hierarchical-cache`，无 draft | 启动失败 | auto-size 正常 |
| V4 C4 持久状态 / V4 draft + auto-size | 启动失败 | 明确报错，要求显式 ratio |
| DSpark V4 draft（608） + V4.1 target（544） | packed，启动断言失败 | `SIDECAR`，独立 `DRAFT_SWA` 池 |
| draft 与 target 行宽一致 | packed | 仍 packed |

---

## 8. A5 证据与已知限制

已在 A5 上验证：

- target-only DP8：host 命中 768 token，7 个池 D2H/H2D 页 CRC 全部一致；
- 814-token prompt：cold prefill 15.456 s → host hit 0.355 s；
- DSpark + HiCache：8 个 rank 都打印 sidecar 日志，`draft_swa` 77824 B、`swa` 69632 B，重复 200-token prompt 命中 128 device token；
- DSpark 冒烟在 `23de16b280` 上完成；PR head `b7a1bcf25d` 只把 `torch` import 移到模块级。

已知限制：

- `hicache_host_used_tokens` 仍只统计锚点池，不覆盖 sidecar/回载字节；
- `a5_hicache_page_roundtrip.py` 的 `c1_indexer` 假阴性是测试脚本视角问题，生产 E2E CRC 已通过；
- G8 的数值等价未直接用文本证明（模型输出不可重复），目前验证的是清理/页边界不变量；
- memfabric 后端未纳入首批验证。

## 9. 提交对应关系

| commit | 内容 |
| --- | --- |
| `5c32e0ee0a` | NPU 4-D 页几何、C1/C2 预留、draft 行宽断言、`_to_page_slots` 命名 |
| `23056a7e24` | V4 paged pools auto-size |
| `7040d0b4f9` | sidecar 池不并入 target SWA、compressor 页边界测试 |
| `23de16b280` | 计划阶段行宽比较，DSpark 回退 `DRAFT_SWA` |
| `b7a1bcf25d` | 模块级 `torch` import（ruff F821） |
