# BreakableNPUGraphBackend

让 Ascend NPU 的 prefill 入图：沿用 CUDA 的 breakable graph 框架，每段用一个
`torch.npu.NPUGraph`，吃 host 侧输入的算子（attention、compressor、indexer）在段之间
eager 执行。行号会漂移，以内容为准。

## 状态

| 项 | 值 |
| --- | --- |
| 开关 | `--cuda-graph-backend-prefill breakable`（NPU 上只接受 `breakable`，其他值报 `ValueError`） |
| bucket | `--cuda-graph-bs-prefill`，**必须 >= 128** |
| GLM-5.2（DSA） | 重构后直接支持，已测，无需模型侧适配 |
| DSV4 Flash | 经 §5 的适配后支持，已跑通 capture 与在线请求（DP16/TP16/EP16） |
| 提交 | `d08d2b52b0` P1 接线 · `1c095d6948` DSV4 断点 · `85860762c4` rope 口径 |

已知限制：

- **bucket 必须 >= 128。** 小于压缩粒度 128 的 bucket 会被 sparse-attn metadata AICPU
  算子拒绝（`execute kernel param invalid`）。capture batch 对该算子而言与一个真实的
  N token 单请求 prefill 无法区分，所以 eager 下的超短 prompt 多半同样不可用；这是
  DSV4-on-NPU 的既有限制，不是入图引入的。未做自动过滤。
- decode 不走这条路（已有全图后端）；EAGLE / draft prefill、段间 dedup 不支持。
- 其他普通 MLA 模型未验证。
- `docs/docs/advanced_features/breakable_cuda_graph.mdx:169` 的平台兼容表仍写 NPU
  不支持，待更新。
- 精度尚未用 sgl-eval 对 breakable 与 disabled 做 gsm8k 对比。

## 1. 为什么 breakable 适合 NPU

NPU 的注意力算子吃 **host 侧的 IntArray** 序列长度，捕获时会固化进图。CUDA 上可变数据
放在显存格子里，重放前 `copy_()` 就能换值；NPU 要换这个 host 值只能走
`graph.update(cpu_update_input=...)`。

| 维度 | CUDA | NPU |
|---|---|---|
| 可变数据 | 显存格子，重放前 `copy_()` | 注意力 KV 长度是 host 侧 IntArray，捕获时固化 |
| 换值通道 | `copy_()` | `graph.update(cpu_update_input=...)` |
| breakable 下 | 不变 | 注意力在断点处 eager 执行，每次重放现读现传，**不需要 `update`** |

这是整个方案的立足点：把吃 host 输入的算子摘到段之间 eager 执行，就绕开了
`graph.update` 那套机制及其限制（§4.1）。

## 2. 机制

设备相关的只有三处：`_device_graph_cls()` 决定每段用什么图类、`_is_stream_capturing`
的便携查询分支、`resolve_prefill_backend` 的 NPU 分支。`BreakableCudaGraphBackend`、
`BreakableCUDAGraphCapture` 和 `PrefillCudaGraphRunner` 都没有为 NPU 改动，
`BreakableNPUGraphBackend` 不覆盖任何方法。

```mermaid
flowchart LR
    S1["段1（NPUGraph）"] -->|"断点：eager"| S2["段2（NPUGraph）"]
    S2 -->|"断点：eager"| S3["段N（NPUGraph）"]
```

**录制期不执行断点函数体**，只把 `functools.partial(run, *args)` 记进 `_break_fns`。
重放时 runner 先在图外为 live batch 重建 metadata，然后"段 replay + 断点 eager 执行"
交替。所以断点里的逻辑每次重放都重跑、读的是当次 metadata——这是 replay 正确性的全部
依据，也意味着断点里不能依赖任何 capture 期算出的 host 值。

## 3. 真机已验证的事实

2026-10 在 Ascend 真机验证，spike 脚本在本目录。

| 结论 | 证据 | 影响 |
|---|---|---|
| `NPUGraph` 有裸 `capture_begin/capture_end`，签名与 CUDA 一致 | `inspect.getsource` | 父类段捕获代码原样可用，不需要捕获子类或工厂方法 |
| 同 stream + 同 pool 连捕两段合法，地址稳定，换输入重放 `err=0` | [v2](v2_spike_two_segments.py) | 共享池分段成立 |
| 段图存活期间，弱引用跨重放可用 | [v4](v4_weak_ref_spike.py) | 默认弱引用，不加开关 |
| 空段可以捕获和重放 | v2 脚本末尾 | 不需要合并连续断点 |
| 裸捕获在三种在途场景下地址稳定、`err=0`（显式 sync / 另一流有 kernel / 捕获流本身有 kernel） | [v8](v8_raw_capture_spike.py) | 不需要每段同步，见 §4.2 |
| `is_current_stream_capturing` / `Stream.wait_stream` 均存在 | 真机 introspect | `_is_stream_capturing` 并入便携分支即可 |

v4 的负对照（删光段图后读弱引用）不作为安全保证，判定依据是"段图存活时"的正向用例。

## 4. 三个坑

### 4.1 不要对段图调 `graph.update()`

`auto_dispatch_capture` 的唯一作用是让 `graph.update(cpu_update_input=...)` 可用，且只
对 5 类注册过的注意力算子生效（`torch_npu/npu/_npugraph_handlers/`）。breakable 下注意力
在断点 eager 执行，段内没有这些算子，`graph_dispatch_records` 为空 → `update()` 无用，
且会因长度校验报错。将来若把注意力圈回段内，必须改用
`torch.npu.graph(..., auto_dispatch_capture=True)` 并重新验证，代价是最大 workspace
预分配、`.out` 重载、record 数对齐。

### 4.2 裸捕获省掉了每段的整设备同步

不走 `torch.npu.graph` 上下文，就没有它每段的 `synchronize()` + `empty_cache()`。v8
验证可以省，收益是 61 层 ≈ 122 次整设备同步消失。

### 4.3 池 token 在图全销毁后不可复用

`graph_pool_handle()` 的 token，在所有引用它的图销毁后**不能再用**：`use_count` 计的是
活图数（`NPUCachingAllocator.cpp`，递减在 `NPUGraph` 析构）。降到 0 后池被标记 freeable
但 token 仍在表里，再用会撞 `create_or_incref_pool: use_count > 0`，进程随之不可恢复。
BCG 的图常驻 `_graphs`，当前不触发；将来做图卸载 / 重捕时必须重新取 token。

## 5. DSV4 Flash 适配

### 5.1 一层 attention 落在哪里

| 步骤 | 执行方式 |
| --- | --- |
| wq_a / wkv / norm / RoPE | 录进 segment，读静态 `positions` slot |
| SWA `store_cache` | 录进 segment，读静态 `out_cache_loc` slot |
| c4 indexer（含自带 compressor） | **eager 断点** |
| 主 compressor（c4 / c128 层） | **eager 断点** |
| sparse attention | eager 断点（`bcg_deepseek_v4_attention_with_output`，本来就有） |

NPU 上 `is_unified_kv_triton()` 恒为 False（只在 HIP 为 True），mega attention 只在 CUDA
启用，所以 prefill attention 一定走最后一行的断点。

### 5.2 为什么 indexer 和 compressor 必须是断点

它们依赖 host 侧算出的 metadata，本来就进不了图。录进 segment 有三类后果，前两类**不
报错，只是算错**：

- **读到已释放的显存。** replay 时 runner 为 live batch 重建整套 metadata，capture 期的
  旧张量随之释放，而它们分配在图 pool 之外。`forward_compress` 读
  `positions_cmp_padding_c*` / `start_pos` / `seqused` / `actual_seq_lengths_q_pa` /
  state block table / `c*_loc`，lightning indexer 读 `actual_seq_lengths_q/kv` /
  `c4_page_table` / `kernel_metadata["li_quant_metadata"]`。
- **capture 期的 host 值被固化。** 如 `dsv4_max_input_capacity` 来自 `cu.cpu().tolist()`，
  决定 state block table 形状；还有 `cmp_kv[: loc.numel()]` 的截断长度和
  `if loc is not None` 这类 Python 分支。
- **c4 层拿不到 topk。** indexer 把结果赋给 `forward_metadata.c4_topk_indices`，这是
  Python 属性赋值，只在 capture 时执行一次，且赋给 capture 期的 fm 对象。

CUDA 侧有同样先例：ratio 1/2 的 indexer 和 compressor 由
`bcg_deepseek_v4_low_ratio_sources` 作为断点执行，注释写的是 "The compressor and prefill
indexer sync with the host, like the attention."

### 5.3 三个口径 bug

断点把"live 行数"和"bucket 行数"两套口径同时带进同一个函数，三个 bug 都出在这里。

**(1) capture batch 没有分配 bundle。** `capture_prepare` 不设 `out_cache_loc_dsv4`
（它只由调度器给真实 batch 分配 KV 时写入），于是 `c4_loc` / `c128_loc` 为 `None`，
`forward_compress` 不拦，最终 `set_index_k_scale` 的 `loc.view` 崩。修法是**只在 capture
期间**补全 0 占位 loc，把压缩写入导到 padding slot 0：

```python
if bundle_loc is None and get_flags().dp.capturing_prefill_graph:
    bundle_loc = torch.zeros(
        sum(blocks.numel() for blocks in ratio_lists[ratio]),
        dtype=torch.int32, device=device,
    )
```

- 块数直接用同一函数里刚算好的 `ratio_lists`，与 `positions_cmp_padding_c*` 天然一致。
- 条件用 `capturing_prefill_graph`，**不能用 `is_in_breakable_cuda_graph()`**——后者
  replay 时也为真，万一 replay 缺 bundle 会把真实数据悄悄写进 slot 0。该标志挂在 `dp`
  下但不论是否开 DP 都会设置，且在 `warmup()` 之后才置位。
- slot 0 不会分给真实请求：DSV4 在 NPU 上 `page_size` 固定 128；
  `NPUPagedTokenToKVPoolAllocator` 继承的 `clear()` 让空闲页从 1 开始，注释写明 slot 0
  留给 padding 写入；c4 loc 由 `full_loc // 4` 推出，full page 0 的 slot 0~127 不分配，
  对应 c4 slot 0~31 也不会分出去。

**(2) 断点内要把输入切到 live 行。** `static_forward_batch` 的 `positions` 按 bucket
补齐，而 indexer 的 `_compute_q_npu` 要按 live 行算 RoPE。做法是对 batch 浅拷贝再切片，
不改 replay batch 本身，后面的 attention 断点仍看到 bucket 长度的 positions：

```python
num_tokens = forward_batch.global_num_token_non_padded_cpu
live_batch = copy.copy(forward_batch)
live_batch.positions = forward_batch.positions[:num_tokens]
backend.forward_c4_indexer_eager(
    x=x[:num_tokens], q_lora=q_lora[:num_tokens], forward_batch=live_batch, ...
)
```

indexer 写结果用的是后端上的 `forward_metadata`、不写 batch，所以浅拷贝不丢结果；RoPE
的 per-forward 缓存按 positions 对象匹配，切片后会按真实 positions 重算。

**(3) compress metadata 的 token 数要取自 cu。** `forward_compress` 用
`positions_cmp_padding_c{ratio}` 建 rope 表，其长度原来按 `t = positions.shape[0]` 算，
而同一函数里的 `cu` / `seqused` / `start_pos` 和算子收到的 `x` 全是 live 的。算子自己从
`cu_seqlens` 推 tokenSize，于是对不上：

```
[compressor] ropeSin shape dim 0, should be equal to
min(tokenSize, tokenSize/cmpRatio+batchSize): 17, but got 33
```

64 token 的请求跑在 128 bucket 里：算子按 64 期望 17 行，rope 表按 128 建了 33 行。
eager 下 `positions.shape[0] == cu[-1]` 恒成立，所以从未暴露。修法是
`t = int(cu_cpu[bs])`；`positions` 在那个函数里除了取 `t` 别无用途，已删。

> 通用形态：**live 口径与 bucket 口径同时出现在一个函数里时，要逐个确认每个派生量跟的
> 是哪一套。** 下一个候选是断点里的 `global_num_token_non_padded_cpu`——它必须恒等于
> `cu[-1]`，目前只在单请求场景核对过，DP 下未在设备上验证。

### 5.4 为什么不照搬 CUDA 的 captured-metadata 契约

CUDA 的 DSV4 后端打开 `use_captured_forward_metadata_for_breakable_cuda_graph`：capture
时建一份地址稳定的 metadata，replay 时原地刷新。这套契约的前提是**段内有 kernel 读这份
metadata**。NPU 的 compressor、lightning indexer 和 sparse attention 都依赖 host 侧的值
（`kernel_metadata` 由 host 算子生成，state block table 宽度来自 host 长度），本来就进
不了图。照搬只是把 metadata 改写成可原地刷新的 buffer，工作量与 decode graph 的
`_init_dsv4_graph_metadata` / `_refresh_graph_*` 相当（数百行），换不来任何算子进图。只有
将来 compressor 有了纯 device 侧接口、能进图时，这套契约才值得做。

## 6. 未覆盖的风险

- **`SGLANG_NPU_USE_MULTI_STREAM=1`。** 打开后 `enable_multi_stream` 在 prefill BCG 下
  为真，attention 走 `_forward_prepare_multi_stream_npu`：先用 `wait_stream` 把 KV 和 Q
  分到侧流。BCG 的 hook 能跟踪这次分流，断点结束 segment 时会自动 join；indexer 自带的
  多流准备现在只在断点里 eager 执行。代码上能走通，**未在设备上验证**。官方部署文档只在
  decode 节点设置这个开关。
- **attn-TP 序列切分。** 断点按 `global_num_token_non_padded_cpu` 切行，与已有 attention
  断点一致，前提是 `attn_tp_size == 1`（DP16 部署满足）。
- **CP 与 MoE。** DSV4 不在 `PREFILL_CP_BCG_SUPPORTED_COMBINATIONS` 里，CP 的 BCG capture
  不会为它打开；CP batch 是否会去 replay 非 CP 的图没有分析。这些路径里若有 host 同步，
  capture 会直接报错，不会悄悄算错。
- **段粒度。** 每层 1~2 个断点，61 层 ≈ 122 次段 launch；indexer 与主 compressor 断点之间
  没有被录制的算子，会产生空 segment。空 segment 可录可放（§3），但每次重放的开销需要
  profiling 确认。必要时改粗断点（每 N 层），或在模型层合成一个断点，仿照
  `bcg_deepseek_v4_low_ratio_sources`，代价是要改共享的模型文件。
