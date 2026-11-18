# DeepSeek V4.1 NPU Prefill Graph 设计

基线：`sglang@dsv41-a5@9898f7ce9a`。本文不改变现有 breakable NPU graph 的 runner/segment 生命周期，只定义 V4.1 的捕获契约、固定状态和使能条件。

本目录已合并原参考方案。设计结论以本文和 [implementation.md](implementation.md) 为准；原参考方案的 A5 单算子证据保留在 `validation/`，两套方案的分歧见第 10 节。

## 1. 问题定义

### 1.1 目标

对满足容量和配置约束的 V4.1 target prefill batch：

1. 选中已捕获的 token bucket；
2. 把本轮的 query/KV 长度、prefix、页表、压缩 loc、索引页表、window/compressed 长度和 dispatch metadata 刷新到**固定地址**；
3. replay 已捕获的 transformer body；
4. 图内完成 ratio-1/2 的压缩写回、索引 K 写入、QLI/QSLI top-k、candidate 发布与消费、MQSMLA attention；
5. 图外的 Engram、LM head/logits tail、DSpark 注入按现有 eager 路径执行。

目标是减少每层、每个小算子的 Python/launch 开销，而不是改变模型计算行数或缓存语义。

### 1.2 不变量

以下任一不成立都必须回退 eager 或启动期拒绝，不能“带病 replay”：

- 图读到的每个 tensor 在 capture 后地址不变，replay 前内容已刷新；
- 本轮真实 token/prefix/页表/cache loc 能由固定 buffer 完整表达；
- 死行不会写到真实请求的 KV、index、compressor state 或 candidate；
- 同一 DP 组的所有 rank 对 replay/eager 的选择一致；
- 共享同一个 collective 的 rank 使用同一通信布局。

### 1.3 为什么不能只打开开关

现有 V4 容器在 capture 时只分配 ratio 4/128 的字段，并把 `kernel_metadata` 换成四个 V4 key。V4.1 的 attention/source 会读：

- `dsv41_win_sparse_indices`、`dsv41_win_topk_length`
- `dsv41_cmp_topk_length_c1/c2`
- `c1_loc/c2_loc`、`positions_cmp_padding_c1/c2`
- `c1_index_page_table/c2_index_page_table`
- `dsv4_cycle_state_block_table`
- `win_metadata`、`c1a_metadata`、`c2a_metadata`、`c*_qli_*`、candidate/QSLI metadata
- 每层 indexer 产出的 topk 和 candidate blocks

这些字段若未进入固定容器，capture 时记录的是 eager 临时 tensor 的地址；replay 既不刷新也不重建，结果要么 KeyError，要么静默读旧数据。

## 2. 设计原则

### P1 代际独占，不做通用变体注册

一个 backend 实例按模型代际只命中 V4 或 V4.1 其中一套契约。实现上是共享生命周期基类 `_Dsv4PrefillGraphMetadataBase` 加两个完整协议类 `Dsv4PrefillGraphMetadata`（ratio 4/128）和 `Dsv41PrefillGraphMetadata`（ratio 1/2）；backend 用模型代际 `_dsv4_is_v41_model` 显式选择类，而不是用 low-ratio 布尔值隐式判断。基类只放两代语义完全相同的操作：共享序列标量的容量分配、capture/replay 入口、逐层 topk staging、kernel metadata 原地拷贝和公共绑定。V4.1 不借用 V4 的 c4/c128 字段，V4 对象也不携带 V4.1 字段。不引入 dataclass、通用 owner、variant registry 或第二套 runner。

### P2 producer 与 consumer 原子捕获

同一层的 source 输出（topk、candidate、index K、compressed latent）和消费它的 attention/QSLI 必须同处一个 captured segment。这样：

- 张量依赖由计算图自然表达，不依赖 Python 属性在 replay 重放；
- 不需要把 `T_cap x index_topk` 的输出逐层复制进固定 buffer；
- 不需要为“source eager、consumer captured”单独设计一套地址桥接和失效协议。

如果某个 source 算子不可捕获，**整套 V4.1 prefill 契约关闭**，source 与 attention 都走现有 `bcg_` eager break。V4 的 `low_ratio_prefill_graph=True` 在 CUDA 上就是这个语义；NPU 只需要在契约完整时把该属性置 True。

### P3 容量形状执行

固定 buffer 一律按 bucket 容量 `T_cap` 分配。replay 时：

- 真实行数由 `actual_seq_lengths_q_pa[-1]` 和 `live_mask` 表达；
- 真实行放在固定 buffer 的前缀；
- 尾部行按字段语义填 0 或 `-1`；
- 所有消费这些 buffer 的算子继续以 `T_cap` 行执行，但以 `cu/seqused/length` 界定有效工作。

这与 A5 单算子探针已执行的模式一致：固定 T 的输入张量，`cu=[0, live]`，死行 length=0 / loc=-1，多轮 replay exact。V4.1 把这套模式从探针搬进真实 metadata 容器。

### P4 metadata 计算与存储分离

V4.1 的 dispatch metadata（MQSMLA/QLI/QSLI）由 SDK 的 `*_metadata_dsl` 生成，形状固定为 `[1024] int32`（A5 探针记录）。这些值依赖本轮的 `cu`、`seqused_k`、`cmp_residual_k`、topk length 和 positions，必须每轮重建。

设计上把“按显式张量算 metadata”抽成纯 helper：

```
_dsv41_attention_row_buffers(*, positions, live_mask, capacity, window, topk, ratios)
    -> win_indices, win_lengths, cmp_lengths_by_ratio

_dsv41_attention_dispatch_metadata(*, cu_seqlens_q, win_lengths, cmp_lengths, ...)
    -> dict[str, Tensor]

_dsv41_index_dispatch_metadata(*, cu_seqlens_q, kv_lens, positions, batch_size, ...)
    -> dict[str, Tensor]
```

eager 路径、capture seed、replay refresh 三处调用同一套 helper。这样 replay 不依赖 `init_forward_metadata(live)` 在 live 形状下产生的副作用，也避免“live 形状的 metadata 复制进容量 buffer”造成的尾部语义不一致。

### P5 只原地刷新

capture 后：

- 不重新分配被图读到的 tensor；
- 不把 `ForwardMetadata` 字段重绑到新对象；
- 不把 `kernel_metadata` 字典换键或换 tensor；
- 所有越界、缺键、形状变化都抛出明确错误。

这与基类生命周期的既有契约一致（`dsv4_prefill_graph.py`）。V4.1 额外把 key 集合和 metadata numel 也纳入校验。

### P6 逐层输出绑定

`c{ratio}_topk_indices` 是**每层不同**的输出。共享的 `ForwardMetadata` 属性在 capture 时会依次被每层覆盖，只保留最后一层的对象。V4 已用 `stage_topk(layer_id, tensor)` 解决（`ascend_dsv4_backend.py:1191-1196`）。V4.1 必须对每个低比例层的 topk 做同样的 `topk_by_layer[layer_id]` 绑定；attention 读取时优先取该层对象。

candidate blocks 只有一个 publisher，消费者共享同一个对象，不需要逐层 map；但它的生命周期同样由 graph pool 保证。

### P7 失败显式、回退整体

能力判定只有两种结果：

- **契约完整且配置支持**：V4.1 source + attention 全部入图；
- **否则**：`use_captured_forward_metadata_for_breakable_cuda_graph=False`、`low_ratio_prefill_graph=False`，source 与 attention 一起走 eager break。

不允许“metadata 分配成功即宣布支持”，也不允许按层选择不同模式。

## 3. 总体架构

```
capture 阶段（每个 token bucket 一次）
  capture_prepare(bucket)
    -> dummy ForwardBatch: B=1, req_pool_indices=[0], positions=0, out_cache_loc=0
  init_forward_metadata_for_breakable_cuda_graph_capture(batch)
    -> eager init_forward_metadata(batch)          # 生成 seed
    -> backend 按代际选择 Dsv41PrefillGraphMetadata.allocate(seed, T_cap, B_cap=1)
         - 分配容量 buffer
         - 把 seed 值按尾部规则填入
         - 绑定到 capture metadata
  capture_one:
    -> warmup 2 次（完整 eager，写入保留槽 0 / bank 0 / dummy page）
    -> torch.npu.graph capture 整个 transformer body
         段内顺序：q/kv 准备 -> SWA 写 -> compressor -> index K
                   -> QLI/QSLI/candidate -> MQSMLA -> o_proj -> MoE
         Engram hash/lookup 是 eager break；LM head 在 eager tail

replay 阶段（每次命中 bucket）
  prepare_forward_metadata_for_breakable_cuda_graph_replay(capture_metadata, live_batch)
    -> init_forward_metadata(live_batch)            # 得到 live 值
    -> state.refresh(backend, live, live_batch)     # 原地刷新固定 buffer
         - 校验 B/T/context/mode
         - 刷新 cu/seqused/start_pos
         - 构造 live_mask
         - 刷新 positions/locs/win lengths/cmp lengths
         - 刷新页表 + index 页表
         - 刷新 swa_loc / cycle bank table
         - 按容量 buffer 重建 dispatch metadata
         - 校验地址不变
    -> self.forward_metadata = capture_metadata
  backend.replay(shape_key, static_batch)
    -> 图内所有 source/attention/MoE
    -> 段间 eager break: Engram hash/lookup
    -> 外层 eager tail: LM head/logits/logprob/DSpark 注入
```

关键点：replay 的 metadata 刷新发生在进入 `enable_breakable_cuda_graph()` 之前，属于普通 eager Python，可以做 host 读取（如 `cu.cpu().tolist()`）、shape 检查、必要时的小规模 device op，但**不能**重绑被图引用的 tensor。

## 4. 路由与能力闸门

### 4.1 代际判定

在 `DeepseekV4AscendAttnBackend.__init__` 中已有：

- `_dsv4_has_low_ratio`：ratio 1/2；
- `_dsv4_has_c4 / _dsv4_has_c128`：ratio 4/128。

增加启动期断言：两者互斥。V4.1 的 `_supports_dsv4_prefill_graph_capture()` 先走 V4.1 专用判定：

```
if self._dsv4_has_low_ratio:
    if not is_npu_arch35(): raise
    if parallel.attn_cp_size != 1 or parallel.attn_dcp_size != 1: raise
    if parallel.pp_size != 1: raise
    if self.is_draft_worker: return False
    if not self._v41_candidate_spec_valid(): raise
    return True
# V4 原路径
```

能力判定本身就是闸门：未实现的配置返回 False，V4.1 整体留在 eager break；实现完整后不再保留额外的全局布尔开关。**不引入用户可见开关**；配置不支持时回退 eager，不新增环境变量。需要整张 prefill graph 回退时使用现有 CLI。

### 4.2 使能后果

`use_captured_forward_metadata_for_breakable_cuda_graph` 同时控制：

1. runner 是否调用 capture/replay metadata hooks；
2. attention 走裸函数（图内）还是 `bcg_` eager break（`deepseek_v4.py:2970-2974`）；
3. `prefill_cuda_graph_max_batch_size` 是否被设置为 1。

因此它必须在 `__init__` 一次确定，不能在 capture 后改变。

当它为 True 时，V4.1 mixin 的 `low_ratio_prefill_graph` 也置 True，使 `_low_ratio_sources_break()` 返回 False，source 留在图内：

```
low_ratio_prefill_graph = use_captured_forward_metadata and model_is_v41
```

两者必须成对设置。只改一个会形成半捕获，违反 P2。

### 4.3 准入

增加 V4.1 的 `can_run_prefill_cuda_graph(forward_batch)` 与 `vote_prefill_cuda_graph(schedule_batch)`：

| 条件 | 处理 |
| --- | --- |
| MIXED | 本地与 DP vote 一致拒绝 |
| `_original_forward_mode` 为 decode 的转换批次 | 拒绝；V4.1 的 1-token EXTEND 拓扑未验证，保持 eager |
| `batch_size > 1` | 由 runner 的 `_captured_attn_metadata_max_bs=1` 拒绝 |
| token 行数 > bucket | runner 拒绝 |
| `max(seq_lens) > backend.max_context_len` | 拒绝（固定页表容量） |
| CP/DCP/PP/TBO | 启动期已拒绝 |
| DP 组内某 rank 无 token | `global_num_tokens_cpu` 在各 rank 一致，现有 `_has_inactive_dp_rank` 让**整组** eager；首版依赖该行为，不实现 idle replay |
| draft prefill | 启动期为 draft worker 时契约关闭 |
| 多模态 / input_embeds / LoRA | runner 现有准入拒绝 |

MIXED 必须显式拒绝：runner 会把 MIXED 的 `pcg_forward_mode` 映射成 EXTEND 来 replay（`prefill_cuda_graph_runner.py:2061-2071`），而 V4.1 的 extend 与 decode 使用不同的 source/attention 语义；把 decode 行塞进 EXTEND 图会静默算错。

> `--cuda-graph-prefill-max-context` 不实现。`supports_prefill_cuda_graph_max_context_size` 保持 False，页表容量使用请求池实际行宽（见 §5.1）；该值对 128k 上下文也只有几 KB 级页表，不值得引入新配置。准入语义仍按 `backend.max_context_len` 判断。

## 5. 固定 metadata 契约

### 5.1 容量

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `T_cap` | 当前 capture bucket 的 token 数 | runner 的 `capture_num_tokens` 单项 |
| `B_cap` | `MAX_PREFILL_GRAPH_REQS = 1` | 首版 |
| `P_cap` | `backend.req_to_token.shape[1]` | 请求池行宽，等于 `model context_len + extra`；不是 `max_context_len` |
| `P_cols` | `ceil(P_cap / page_size)` | full 页表列数 |
| `C1_cap` | `T_cap` | ratio 1 是逐 token 行 |
| `C2_cap` | `min(T_cap, T_cap // 2 + B_cap)` | 与 fused compressor 输出公式一致 |
| `IDX_cols(ratio)` | `P_cols * ((page_size // ratio) // DSV41_INDEX_PAGE_SIZE)` | index 页表扩展 |

`P_cap` 必须取请求池的实际行宽。eager 的 `block_tables` 来自 `req_to_token[req_pool_indices, ::page_size]`（`ascend_backend.py:708-716`），其列数由整个池行宽决定，而不是 live seq_len；用 `model context_len` 估算会少算池的 extra 行并在拷贝时报溢出。

`page_size // ratio` 必须能被 `DSV41_INDEX_PAGE_SIZE` 整除；否则启动期拒绝。

### 5.2 字段表

所有 tensor 在 `allocate()` 中一次分配，之后只做 `copy_`/`fill_`。dtype 以算子实际 ABI 为准；下表的 dtype 是当前 eager 路径实际产生的类型。

| 字段 | 形状 / dtype | replay 刷新来源 | 尾部规则 |
| --- | --- | --- | --- |
| `actual_q` | `[1]` int32 | `live.actual_seq_lengths_q` | 无 |
| `actual_q_pa` | `[B_cap+1]` int32 | `live.actual_seq_lengths_q_pa` | 无（B=1 时恰为 2 项） |
| `actual_kv` | `[1]` int32 | `live.actual_seq_lengths_kv` | 无 |
| `seqused` | `[1]` int32 | `live.seqused` | 无 |
| `start_pos` | `[1]` int32 | `live.start_pos`（extend prefix） | 无 |
| `swa_loc` | `[T_cap]` int64 | `translate_loc_from_full_to_swa(live.out_cache_loc)` | 0 |
| `live_mask` | `[T_cap]` bool | `arange(T_cap) < actual_q_pa[-1]` | false |
| `positions_token` | `[T_cap]` int64 | `live.positions` | 0 |
| `positions_cmp_padding_c1` | `[C1_cap]` int64 | ratio-1 行公式 | 0 |
| `loc_c1` | `[C1_cap]` int32 | ratio-1 行公式 | `-1` |
| `positions_cmp_padding_c2` | `[C2_cap]` int64 | ratio-2 稳定 compact | 0 |
| `loc_c2` | `[C2_cap]` int32 | ratio-2 稳定 compact | `-1` |
| `dsv41_win_sparse_indices` | `[T_cap, 1, W]` int32 | position + live_mask | `-1` |
| `dsv41_win_topk_length` | `[T_cap, 1]` int32 | `min(pos+1, W)`，死行 0 | 0 |
| `dsv41_cmp_topk_length_c1/c2` | `[T_cap, 1]` int32 | `min((pos+1)//r, index_topk)`，死行 0 | 0 |
| `full_block_table` | `[1, P_cols]` int32 | `live.block_tables`（full 页表） | 0 |
| `swa_page_table` | `[1, P_cols]` int32 | `live.swa_page_table`；capture 时若与 `full_block_table` 是同一对象则共享同一固定 buffer | 0 |
| `block_tables` | 同 `full_block_table` | 别名，不复制 | — |
| `c1_page_table` / `c2_page_table` | 同 `full_block_table` | 别名，不复制 | — |
| `c1_index_page_table` / `c2_index_page_table` | `[1, IDX_cols(r)]` int32 | 由 `full_block_table` 展开 | 0 |
| `dsv4_cycle_state_block_table` | `[1]` int32 | `live.req_pool_indices` | 0（B=1 无尾） |
| `kernel_metadata[key]` | `[max(1024, seed.numel())]` int32 | 每轮重建 | 不适用，整块覆盖 |
| `topk_by_layer[layer_id]` | `[T_cap, index_topk]` int32（图内产出） | 图内 source 写入 | — |
| `candidate_blocks` | 图内产出 | 图内 source 写入 | — |

`block_tables` 与 `c{ratio}_page_table` 在图内必须指向同一份 full 固定 buffer。`swa_page_table` 在非 hybrid-SWA 模型上与它共享同一 buffer；若 `live.swa_page_table` 与 `live.block_tables` 不是同一对象，则单独分配一份固定 buffer，不能把 SWA 映射写进 full 表。capture 后不允许出现“某个是别名、某个是 live 新张量”的混合状态。

### 5.3 `_bind` 后的关键属性

```
metadata.actual_seq_lengths_q / _q_pa / _kv = fixed
metadata.seqused / start_pos = fixed
metadata.swa_loc = fixed
metadata.block_tables = full_block_table
metadata.swa_page_table = swa_page_table
metadata.c1_page_table = metadata.block_tables
metadata.c2_page_table = metadata.block_tables
metadata.c1_index_page_table / c2_index_page_table = fixed
metadata.positions_cmp_padding_c1 / c2 = fixed
metadata.c1_loc / c2_loc = fixed
metadata.dsv41_win_sparse_indices / win_topk_length = fixed
metadata.dsv41_cmp_topk_length_c1 / c2 = fixed
metadata.dsv41_live_mask_buf = fixed
metadata.dsv4_cycle_state_block_table = fixed
metadata.dsv4_explicit_state_block_tables = {}
metadata.dsv4_max_input_capacity = T_cap
metadata.kernel_metadata = fixed dict
metadata.dsv4_prefill_graph_state = state
metadata.c1_topk_indices = None   # 由图内 source 产出；读取走 topk_by_layer
metadata.c2_topk_indices = None
metadata.dsv41_candidate_blocks = None
```

不绑定 `ori_sparse_indices/ori_topk_length`：V4.1 的 MQSMLA 用显式 window 索引，不用 V4 的 ori 侧参数。

### 5.4 kernel metadata key 集合

不在 metadata 容器里硬编码 key 名。`allocate()` 从 capture seed 的 `kernel_metadata` 取 key 集合，为每个 key 分配至少 1024 个 int32。当前 V4.1 的 key 来自：

- `_dsv41_attention_kernel_metadata`：`win_metadata`、`c1a_metadata`、`c2a_metadata`
- `_dsv41_index_kernel_metadata`：`c1a_qli_metadata`、`c2a_qli_metadata`
- candidate source：`c{r}a_qli_candidate_metadata`
- QSLI consumer：`c{r}a_qsli_metadata`

replay 刷新后必须与 capture 的 key 集合完全一致；缺键或 numel 变化直接报错。这与 V4 的“缺键补零”不同：V4.1 的 dispatch table 长度为 0 或内容全零是否合法没有设备证据，不能静默写零。

## 6. Replay 刷新算法

### 6.1 伪代码

```
def refresh(state, backend, live_fm, fb):
    # 1. 准入与形状
    assert fb.batch_size <= state.max_reqs
    assert fb.forward_mode.is_extend_without_speculative()   # 首版；MIXED 已在上游拒绝
    assert not fb.can_run_tbo
    t_live = int(fb.positions.numel())
    assert t_live <= state.T_cap
    assert max(seq_lens) <= state.P_cap

    # 2. 标量与 cu
    copy_cu(state.actual_q_pa, live_fm.actual_seq_lengths_q_pa)
    copy_1d(state.actual_q,   live_fm.actual_seq_lengths_q)
    copy_1d(state.actual_kv,  live_fm.actual_seq_lengths_kv)
    copy_1d(state.seqused,    live_fm.seqused)
    copy_1d(state.start_pos,  live_fm.start_pos)

    # 3. live mask
    state.live_mask.copy_(arange(T_cap) < state.actual_q_pa[-1])

    # 4. token 行与 ratio 行
    copy_padded(state.positions_token, fb.positions[:t_live], 0)
    out_loc = padded(fb.out_cache_loc[:t_live], T_cap, 0)
    state.swa_loc.copy_(translate_full_to_swa(out_loc))
    for r in (1, 2):
        build_ratio_rows(state, r, state.positions_token, out_loc, state.live_mask)

    # 5. window / compressed 长度与索引
    build_win_rows(state, positions=state.positions_token, mask=state.live_mask)
    for r in (1, 2):
        build_cmp_lengths(state, r, state.positions_token, state.live_mask)

    # 6. 页表
    copy_table(state.full_block_table, live_fm.block_tables)
    if state.swa_page_table is not state.full_block_table:
        copy_table(state.swa_page_table, live_fm.swa_page_table)
    # c{1,2}_page_table 是 full_block_table 的别名，不复制
    for r in (1, 2):
        copy_table(state.c{r}_index_page_table,
                   expand_index_page_table(state.full_block_table, r))

    # 7. cycle bank
    copy_1d(state.dsv4_cycle_state_block_table, fb.req_pool_indices[:1])

    # 8. dispatch metadata（用容量 buffer + live cu 重建）
    dispatch = backend._dsv41_attention_dispatch_metadata(
        cu_seqlens_q=state.actual_q_pa,
        win_lengths=state.win_topk_length,
        cmp_lengths={r: state.cmp_topk_length[r] for r in (1, 2)},
    ) | backend._dsv41_index_dispatch_metadata(
        cu_seqlens_q=state.actual_q_pa,
        kv_lens=state.actual_kv,
        positions=state.positions_token,
        batch_size=state.max_reqs,
        candidate_spec=backend._v41_candidate_spec(),
    )
    assert set(dispatch) == set(state.kernel_metadata)
    for k, v in dispatch.items():
        copy_flat(state.kernel_metadata[k], v)

    # 9. 地址自检（debug）
    state.assert_bound_addresses_unchanged()
```

### 6.2 ratio 行公式

复用现有 `_dsv41_ratio_row_values` 的语义：

```
keep = (positions % ratio == ratio - 1) & (out_loc >= page_size) & live_mask
pos_values = positions if ratio == 1 else (positions // ratio) * ratio
slot_values = out_loc if ratio == 1 else out_loc // ratio
```

- **ratio 1**：不 compact，逐 token 行；`pos_dst = where(keep, pos_values, 0)`，`loc_dst = where(keep, slot_values, -1)`。
- **ratio 2**：稳定 compact 到 `[C2_cap]`；`_stable_compact_1d(dst, values, keep, fill=0 or -1)`；先清零再 scatter，尾部填 `-1`。

必须保留 ratio 1 的行序。V4.1 的 ratio 1 没有 fused compressor，行号就是 token 号；compact 会破坏 epilog 和 attention 的 loc 对应关系。

### 6.3 live mask

- `actual_q_pa[-1]` 是本 rank 本轮真实 token 行数（`extend_seq_lens_cpu` 之和）。
- DP 场景下，DP padding 行在真实行之后；固定 buffer 的尾部和 mask 一致。
- 若某 rank 本轮 0 token，现有 runner 的 `_has_inactive_dp_rank` 会让整组 eager；首版不依赖 `_fill_idle` 正确性。
- `_fill_idle` 仍需实现为防御路径：cu=`[0,1]`、seqused=0、kv=1、所有 loc=`-1`、所有长度=0、cycle bank=0、mask=false；device 探针单独记录，但不作为首版验收门槛。

### 6.4 window / compressed 行

- `win_valid = clamp(pos+1, max=W)`；死行置 0；索引 `window_start + j`，`j >= valid` 的位置填 `-1`。零长度会让算子跳过该侧；A5 探针已在死行全 `-1` + 零长度下验证 exact。idle 防御路径额外把每行首索引置 0。
- `cmp_len(r) = min((pos+1)//r, index_topk)`；死行置 0。
- 这些 buffer 的容量固定为 `T_cap`；尾部由刷新函数显式清零，不能依赖上一次 replay 的内容。

### 6.5 页表与 index 页表

- full 页表只保留一份固定 storage；`block_tables`、`c{1,2}_page_table` 指向它。
- `swa_page_table` 在非 hybrid-SWA 模型上复用同一 storage；否则单独一份。是否共享由 capture 时 `live.swa_page_table is live.block_tables` 决定，并写入 state。
- 每轮先清空固定表，再拷贝 live 的 `rows x cols`；`cols > capacity` 直接报错。
- index 页表由固定 full 表展开，整块覆盖；`expand_index_page_table` 的输出列数必须 ≤ 容量。
- 尾部填 0（页 0 是保留 dummy page）。不能沿用 V4 的“0 是 skip”假设，也不能沿用 V4 decode 的 `-1` 高水位逻辑；V4.1 的 writer/skip 语义是 `-1`。

### 6.6 dispatch metadata 重建

必须使用**容量 buffer** 作为输入：

- `cu_seqlens_q = state.actual_q_pa`（live 值，容量形状）
- `win_lengths = state.dsv41_win_topk_length`（T_cap 行，死行 0）
- `cmp_lengths = state.dsv41_cmp_topk_length_c{1,2}`（T_cap 行，死行 0）
- `kv_lens = state.actual_kv`（live）
- `positions = state.positions_token`（T_cap 行，死行 0）
- candidate 可见块数由容量 positions + ratio/block 公式计算；死行 clamp 到 1 是合法下界。

这与 A5 `metadata` 探针相同：metadata 输入是完整 T 行和 live cu。这样同一个 bucket 内不同 live T 得到相同形状 `[1024]`、不同内容的 schedule；固定 buffer 只做整块覆盖。

首版 `init_forward_metadata(live)` 仍会为 live 形状构建一次 dispatch metadata，refresh 再用容量 buffer 构建一次，属于可接受的小开销。若 profile 显示其占比明显，可在 replay hook 期间让 `_compute_kernel_metadata` 跳过 V4.1 dispatch 构建，只保留容量版本；这属于后续优化，不改变契约。

### 6.7 逐层 topk

在 `_low_ratio_index_topk` 末尾：

```
if fm.dsv4_prefill_graph_state is not None:
    topk = graph_state.stage_topk(layer.layer_id, topk)
else:
    fm.c{ratio}_topk_indices = topk
```

`_forward_low_ratio_attention` 读取时：

```
topk = graph_state.topk_by_layer[layer.layer_id] if graph_state else fm.c{ratio}_topk_indices
```

原因：capture 时每层都会覆盖共享属性，若 replay 的 attention Python body 重新执行（或 attention 位于 eager break），共享属性只剩最后一层，必须按 layer_id 取回本层对象。V4 已有同款机制，V4.1 只是把 key 从 c4 换成 c1/c2。

### 6.8 地址自检

调试模式（`--debug-cuda-graph` 或模块级 debug flag）下，`allocate()` 记录所有绑定字段的 `data_ptr()`；`refresh()` 完成后比对。任何绑定字段被替换即抛错。生产模式只做 shape/numel/key 校验，不做额外同步。

## 7. 数据流与原子性

### 7.1 每层顺序（图内）

```
MQALayer:
  q/kv 投影 + norm + RoPE
  attn_backend.store_cache(swa_k)                 # 读 state.swa_loc
  if ratio in (1,2):
      compressor:
          ratio1: project -> norm -> RoPE -> set_compress_buffer(loc_c1)
          ratio2: ds41.compressor(state bank table) -> norm -> RoPE -> set_compress_buffer(loc_c2)
      index K:
          indexer_prologue_k(storage_mode=0, cache_index=loc_c{r})
          if candidate source: indexer_prologue_k(storage_mode=1, QSLI cache)
      QLI / QSLI:
          q = indexer wq_b + RoPE + mx quant
          topk = quant_lightning_indexer / quant_sparse_lightning_indexer
          candidate source: publish blocks
          stage_topk(layer_id, topk)
  MQSMLA:
      window side: win indices + lengths
      cmp side: c{r}_topk_indices + cmp_topk_length + c{r}_page_table
  o_proj
  MoE（ascend_ops allgather / 其他已验证配置）
```

所有 Python 分支（`ratio`、`owns_k`、`is_candidate_source`、`uses_candidates`、`use_fused_gate`）都由模型配置决定，capture 时固定；replay 不重算。

首次使用时才分配的辅助对象（`_dsv41_qsli_cache`、`_dsv41_zero_sinks`、RoPE 表）必须由 capture 前的 warmup 建立；capture 内若发现缺失，应抛错而不是在捕获区新建（新建会改变后续 replay 的地址假设）。

### 7.2 为什么不允许半捕获

若 source 保持 eager、attention 入图：

1. **topk 桥接代价**：QLI/QSLI 没有 `out=` 参数，输出由算子分配。要让图内 attention 读到本轮 topk，只能 `dst[:t].copy_(topk)`。`T_cap` 取 2048 时，单层 `2048 x index_topk x 4B`；`index_topk=2048` 就是 16 MB/层/轮，几十层不可接受。
2. **candidate 链断裂**：QSLI 消费者读的是 source 发布的块；source eager 时对象每轮新建，图内 captured 消费者仍绑定旧地址，需要额外的固定输出桥接和失效协议。
3. **路由成对性**：`_low_ratio_sources_break` 与 attention 的 `bcg_`/裸函数选择是两个独立布尔；只改一个会产生“source eager、consumer captured”或反过来的组合，验证矩阵翻倍。
4. **语义分散**：compressor 写回、index K 写回、topk、candidate 的 producer/consumer 本就在同一层相邻；拆开只会让 Python 属性赋值成为跨图契约的一部分。

因此本文把“整层低比例数据流”作为最小捕获单元。

### 7.3 与 V4 的差异

| 维度 | V4 | V4.1 |
| --- | --- | --- |
| 比例 | 0/4/128 | 0/1/2 |
| 压缩 | c4/c128 allocator bundle + 显式 state table | ratio1 逐 token 投影；ratio2 fused compressor + cycle bank |
| state 表 | `dsv4_explicit_state_block_tables` | 不使用；`dsv4_cycle_state_block_table` |
| 索引 | C4 indexer（QLI 读 c4 页） | QLI/QSLI + candidate blocks + index 页表扩展 |
| attention | `_forward_swa` / `_forward_compressed` | `_forward_low_ratio_attention`（window + c1/c2） |
| window 元数据 | ori 侧参数 + c1a | 显式 `dsv41_win_sparse_indices/length` |
| 压缩长度 | 由位置/padding 推导 | 显式 `dsv41_cmp_topk_length_c{1,2}` |
| topk | c4_topk_indices | 每层 c1/c2 topk + candidate |
| 行宽 | c4/c128 页 | 544B SWA / 320B compressed |

这些差异使得 V4 容器的字段不能“顺手复用”；只能共享生命周期和尾部规则。

## 8. 状态隔离

### 8.1 请求槽 0

`ReqToTokenPool` 分配 `size + 1` 行，`free_slots = range(1, size+1)`（`memory_pool.py:414-424`）。capture dummy 使用 `req_pool_indices=[0]`，真实请求永远不会拿到槽 0。

因此：

- compressed/state 的请求 bank 0 属于槽 0；
- ratio-2 pair state pool 每个请求 slot 一个 bank（`dsv4_memory_pool.py:546-568`），bank 0 只被 capture/warmup 使用；
- replay 的 `dsv4_cycle_state_block_table` 刷新为 live `req_pool_indices`，切到真实 bank。

### 8.2 写回 sentinel

- `c{1,2}_loc` 的无效行 = `-1`；writer/epilog/`indexer_prologue_k` 跳过 `-1`。A5 探针已覆盖 `-1`、全 skip、NaN 和 guard rows。
- 压缩页 0 保留；`out_loc >= page_size` 过滤保证 capture 的零值不会落到 live 压缩页。
- index 页 0 保留；capture 的 dummy block table 全 0，index K 写入被 `-1` 跳过，QLI 只读页 0。

### 8.3 capture/warmup 副作用

`BreakableCudaGraphBackend.capture_one` 在真正 capture 前会完整 eager 跑两次（`breakable_cuda_graph_backend.py:137-145`），且 BCG 不调用 `post_warmup_hook`（`prefill_cuda_graph_runner.py:1921-1938`）。因此 capture 阶段会真实执行：

- compressor：写 bank 0 的 state；
- index K：loc=-1，全部跳过；
- compressed KV：loc=-1，全部跳过；
- SWA K：`swa_loc` 为 capture 的 dummy 值，落到保留槽/页；
- attention/MoE：计算到 dummy 输出。

这些写必须只落在保留槽 0 / bank 0 / page 0。实现后加一条设备断言/测试：capture 前后对比一个真实请求 bank 与若干 live 页的内容，必须完全不变。若某 SDK 版本无法保证，再引入 capture-only bank 或快照恢复；**不要预置**这套机制。

### 8.4 跨 chunk 的 ratio-2 状态

- `start_pos` = 本轮 chunk 的 prefix；fused compressor 用它定位 bank 内 ring 偏移。
- 上一 chunk 的 pending pair 留在同一请求 bank；本轮 replay 继续消费。
- capture 的 bank 0 状态与真实请求无关，不需要在每次 replay 前清理。
- 首版验收包含：奇数/偶数 prefix、跨 chunk、恰好新建块/没有新块、bank 复用、下一 chunk decode。

## 9. 外围组件

### 9.1 MoE

沿用已迁入实现：

- `BreakableNPUGraphBackend` 在 capture scope 发布固定 `ascend_allgather_row_sizes=[bucket]*world_size`；
- `ascend_ops.dispatch` 用 `is_in_prefill_graph_scope()` 选择 prefill AllGather，不依赖被清掉的 extend 标记；
- construction 期拒绝 ascend_ops AllToAllV。

首版只声明已验证的 ascend_ops allgather 配置。其他 MoE backend 若要走 V4.1 契约，必须单独通过同一套整模验收；设计上不为其新增开关。

### 9.2 Engram

保持 eager：

- hash：`bcg_deepseek_v4_engram_hash_ids`，写入 `new_hash_ids` 的固定输出；
- lookup：`_bcg_engram_embed` 是 eager break，写 `out`；
- history commit 在 break 内完成，或由 scheduler staged rows 提供。

它们把结果写成图内 consumer 的固定输入；不要求 hash/lookup 入图。首版验证同一 shape 不同 token/history、请求槽复用和 verify 不重复提交。

### 9.3 mHC

A5 路径是单次 premix：

- `_npu_combine_premix` 折叠上一子层 pre、生成本层 `(pre, post, comb)`；
- `post` 消费同一 sublayer 的 stats；
- 无多流。

图内只要保持 premix → attention/MLP → post 的顺序，stats 张量由 graph pool 持有，replay 自然正确。Engram 改写 residual 后不能复用旧 stats；这是模型层顺序，不需要额外 metadata。首版整模验证。

### 9.4 DSpark

- target prefill 的 aux hidden 由 `AuxHiddenStatePacker` 在 body 内采集；它属于 captured output 的一部分。
- 图外 eager tail 把 hidden/token indices 交给 DSpark worker，再注入 draft KV。
- draft prefill 仍是 eager（draft worker 契约关闭）。
- 首版验收：target prefill 输出、有效行、token indices、draft KV 注入、下一轮生成。

### 9.5 LM head / logits / logprob

BCG 只捕获 transformer body；LM head、logits processor、prompt logprob 走 eager tail。V4.1 不需要改变这部分。

## 10. 与参考方案的关键分歧

| # | 参考方案 | 本文 | 理由 |
| --- | --- | --- | --- |
| D1 | 先让 low-ratio 返回 False，作为过渡基线 | 只作为 PR-0 的安全闸门，不作为交付形态 | 过渡基线本身没有 attention 图覆盖；交付必须原子捕获 |
| D2 | source 可暂留 eager，用固定输出 bridge | 不允许半捕获；source 与 attention 同进同出 | topk 桥接的带宽/内存代价，以及 candidate/topk 生命周期复杂度 |
| D3 | 扩展 `Dsv4PrefillGraphMetadata`，共享 allocate/fill/refresh/bind | 采用共享生命周期，但拆成基类 + 两个协议类；动态 kernel key、纯 helper 重构 | 避免两个代际的布局/sentinel 混在一个容器里，也避免 replay 依赖 live metadata 副作用 |
| D4 | 同 bucket 动态长度时 shape 必须匹配并原地更新 | 进一步要求 dispatch metadata 用容量 buffer 重建 | live 形状的 metadata 尾部语义不明确；容量形状与探针一致 |
| D5 | 优先复用槽 0 | 采用，并补源码依据与 device 断言 | `free_slots=range(1,size+1)`、pair state 每 slot 一 bank |
| D6 | 首批固定实际 DP idle 策略 | 明确首版靠现有 inactive-rank 回退，不实现 idle replay | compressor/MQSMLA 的 idle ABI 未验证 |
| D7 | 验收先 CPU，再单算子，再 metadata | 先组合图探针（一层完整 source+attention），再容器闭环 | 单算子 exact 不能证明同一 segment 的跨算子组合 |
| D8 | 未特别处理 MIXED | 显式拒绝 MIXED | runner 会把 MIXED 映射成 EXTEND replay，语义错误 |

## 11. 风险与设备未知项

按风险从高到低：

| 风险 | 触发条件 | 观察/缓解 |
| --- | --- | --- |
| R1 组合图不可捕获 | 某个 source/attention op 在 aclgraph 下回退或 host sync | 一层组合探针；失败则整个 V4.1 契约关闭，不做半捕获 |
| R2 容量形状 + live cu 的 schedule 语义 | metadata 或算子假设 `T == cu[-1]` | 同 bucket 不同 live T、死行零长度、真实容器刷新；失败则重新评估 bucket 粒度 |
| R3 graph pool 内存 | `T_cap x index_topk` 的 topk/candidate 中间张量被图池持有 | 记录每个 bucket 的 capture memory；必要时限制 bucket 数或最大 bucket |
| R4 bank/slot 隔离 | capture/warmup 写入真实 bank 或页 | capture 前后状态对比；必要时引入 capture-only bank（仅在有证据后） |
| R5 ratio-2 跨 chunk | 奇数 prefix、跨 chunk pair、ring wrap | 真实 allocator + 真实 metadata 的多轮 graph/eager 对照 |
| R6 candidate/QSLI | 候选数变化、边界块、top-k sentinel | QLI→QSLI→attention 同段探针；真实模型候选层 |
| R7 mHC/Engram 顺序 | residual 改写后复用旧 stats、history 重复提交 | 整模多轮；保留 eager break |
| R8 DSpark | aux hidden 行数/索引错误、draft 注入错位 | target prefill 输出 + 下一轮生成 |
| R9 DP/TP 布局 | 各 rank bucket 或路由不一致；DP padding 后本地 token 宽度超过 bucket | DP vote 成对；固定 AllGather 行数；`refresh` 对 `positions.numel() > T_cap` 报错，`can_run`/vote 提前拒绝；多 rank 测试 |
| R10 SDK 漂移 | 新 SDK 改 metadata 形状/ABI | 版本固定；变化后重跑组合探针，不凭 Python hash 推广 |
| R11 独立精度问题 | CPU FP32 参考与算子差异 | 保留单独议题；不放宽阈值，不用 graph/eager exact 覆盖 |

## 12. 完成判定

同时满足才宣布某配置支持：

1. 路由到 `BreakableNPUGraphBackend`，prefill graph 命中/回退日志可复核；
2. 同 bucket 不同 live T、prefix、连续 chunk、页表变化的多轮 replay 与 eager 数值/缓存一致；
3. 图内实际执行 compressor、index K、QLI/QSLI、candidate、MQSMLA、MoE；段间 break 清单与设计一致；
4. 真实 allocator 下 capture/warmup 不污染 live bank/页；bank 复用和下一 chunk 正确；
5. Engram/mHC/DSpark 的 eager 边界与整模输出正确；
6. 不支持的配置（MIXED、B>1、CP/DCP/PP/TBO、draft、多模态等）有明确、各 rank 一致的拒绝或 eager 回退；
7. 给出该配置的 prefill CPU 下发开销、TTFT/吞吐、graph hit/fallback、bucket padding、capture memory 数据；
8. 独立精度议题单独记录，不混入 graph 正确性结论。

验收操作细节见 [implementation.md](implementation.md)。
