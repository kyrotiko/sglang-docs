# DSV4 Flash 在 NPU breakable prefill graph 下的适配

已跑通 capture 与在线请求（DP16/TP16/EP16）。相关提交：`1c095d6948` 断点与占位 loc、
`85860762c4` rope 口径。行号会漂移，以内容为准。

bucket 必须 >= 128，原因见 [README](README.md) 的已知限制。

## 1. 一层 DSV4 attention 落在哪里

| 步骤 | 位置 | 执行方式 |
| --- | --- | --- |
| wq_a / wkv / norm / RoPE | `deepseek_v4.py` NPU 分支 | 录进 segment，读静态 `positions` slot |
| SWA `store_cache` | `ascend_dsv4_backend.py` | 录进 segment，读静态 `out_cache_loc` slot |
| c4 indexer（含自带 compressor） | `forward_c4_indexer` | **eager 断点** |
| 主 compressor（c4 / c128 层） | `forward_core_compressor` | **eager 断点** |
| sparse attention | `bcg_deepseek_v4_attention_with_output` | eager 断点（本来就有） |

NPU 上 `is_unified_kv_triton()` 恒为 False（只在 HIP 为 True），mega attention 只在
CUDA 启用，所以 prefill attention 一定走最后一行的断点。

## 2. 为什么 indexer 和 compressor 必须是断点

它们依赖 host 侧算出的 metadata，本来就进不了图。录进 segment 会有三类后果，其中
前两类**不报错，只是算错**：

- **读到已释放的显存。** replay 时 runner 为 live batch 重新调
  `init_forward_metadata`，生成一整套新 metadata，capture 期的旧张量随之释放。
  `forward_compress` 读 `positions_cmp_padding_c*` / `start_pos` / `seqused` /
  `actual_seq_lengths_q_pa` / state block table / `c*_loc`，lightning indexer 读
  `actual_seq_lengths_q/kv` / `c4_page_table` / `kernel_metadata["li_quant_metadata"]`，
  这些都分配在图 pool 之外。
- **capture 期的 host 值被固化。** 例如 `dsv4_max_input_capacity` 来自
  `cu.cpu().tolist()`，决定 state block table 的形状；还有 `cmp_kv[: loc.numel()]`
  的截断长度和 `if loc is not None` 这类 Python 分支。
- **c4 层拿不到 topk。** indexer 把结果赋给 `self.forward_metadata.c4_topk_indices`，
  这是 Python 属性赋值，只在 capture 时执行一次，且赋给 capture 期的 fm 对象。

CUDA 侧有同样的先例：ratio 1/2 的 indexer 和 compressor 由
`bcg_deepseek_v4_low_ratio_sources` 作为断点执行，注释写的是 "The compressor and
prefill indexer sync with the host, like the attention."

断点保证 replay 正确的依据：capture pass 不执行函数体，只记 partial；replay 直接调
这个 partial，读的是 `_prepare_forward_metadata_for_replay` 刚为 live batch 建好的
metadata，走的就是已验证过的 eager prefill 路径。断点的输入张量是上一段在图 pool 里
产出的，以弱引用保存、由 pool use-count 保活，每次 replay 由该段在同一地址重写。
断点之间按顺序 eager 执行，所以 attention 读到的是本次的 `c4_topk_indices`。

## 3. 三个必须守住的口径

断点把"live 行数"和"bucket 行数"这两套口径同时带进了同一个函数，三个 bug 都出在这里。

### 3.1 capture batch 没有分配 bundle

`capture_prepare` 构造的 `ForwardBatch` 不设 `out_cache_loc_dsv4`（它只由调度器给
真实 batch 分配 KV 时写入），而 `_build_npu_compress_metadata_prefill` 在 bundle 为
`None` 时把 `c4_loc` / `c128_loc` 置 `None`，`forward_compress` 又不拦 `None`，最终
`set_index_k_scale` 的 `loc.view` 崩。

修法：**只在 capture 期间**补全 0 占位 loc，把压缩写入导到 padding slot 0。

```python
if bundle_loc is None and get_flags().dp.capturing_prefill_graph:
    bundle_loc = torch.zeros(
        sum(blocks.numel() for blocks in ratio_lists[ratio]),
        dtype=torch.int32, device=device,
    )
```

- **块数**直接用同一函数里刚算好的 `ratio_lists`，与 `positions_cmp_padding_c*`
  天然一致，不另写公式。
- **条件用 `capturing_prefill_graph`，不能用 `is_in_breakable_cuda_graph()`。** 后者
  在 replay 时也为 True；万一 replay 时缺 bundle，占位 loc 会把真实请求的数据悄悄写进
  slot 0。用 capture 标志能保证 replay 缺 bundle 时照样报错。该标志挂在 `dp` 下，但
  `PrefillCudaGraphRunner._capture` 不论是否开 DP 都会设置，且在 `warmup()` 之后才
  置位，不影响 capture 前的 kernel warmup。
- **slot 0 不会分给真实请求**：DSV4 在 NPU 上 `page_size` 固定 128；full KV 和 c128
  的分配器 `NPUPagedTokenToKVPoolAllocator` 继承的 `clear()` 让空闲页从 1 开始，
  注释写明 slot 0 留给 padding 写入；c4 loc 由 `full_loc // 4` 推出，full page 0 的
  slot 0~127 不分配，对应 c4 slot 0~31 也就不会分给真实请求。

### 3.2 断点内要把输入切到 live 行

`static_forward_batch` 的 `positions` 按 bucket 补齐，而 indexer 的 `_compute_q_npu`
要按 live 行算 RoPE。断点里对 batch 做**浅拷贝**再切片，不改 replay batch 本身，后面的
attention 断点看到的仍是 bucket 长度的 positions：

```python
num_tokens = forward_batch.global_num_token_non_padded_cpu
live_batch = copy.copy(forward_batch)
live_batch.positions = forward_batch.positions[:num_tokens]
backend.forward_c4_indexer_eager(
    x=x[:num_tokens], q_lora=q_lora[:num_tokens], forward_batch=live_batch, ...
)
```

indexer 写结果用的是后端上的 `forward_metadata`、不写 batch，所以浅拷贝不丢结果；
RoPE 的 per-forward 缓存按 positions 对象匹配，切片后会按真实 positions 重算。

### 3.3 compress metadata 的 token 数要取自 cu，不能取自 positions

`forward_compress` 用 `fm.positions_cmp_padding_c{ratio}` 建 rope 表，而这个数组的
长度原来按 `t = positions.shape[0]` 算。同一函数里的 `cu` / `seqused` / `start_pos`
和算子收到的 `x` 全是 live 的，只有 `t` 取自 bucket 长度的 positions。算子自己从
`cu_seqlens` 推 tokenSize，于是对不上：

```
[compressor] ropeSin shape dim 0, should be equal to
min(tokenSize, tokenSize/cmpRatio+batchSize): 17, but got 33
```

64 token 的请求跑在 128 bucket 里：算子按 64 期望 17 行，rope 表按 128 建了 33 行。
eager 下 `positions.shape[0] == cu[-1]`，两者永远相等，所以从未暴露。

修法是 `t = int(cu_cpu[bs])`；`positions` 在那个函数里除了取 `t` 没有别的用途，已删。

> 这类 bug 的通用形态：**凡是 live 口径与 bucket 口径同时出现在一个函数里，就要确认
> 每个派生量跟的是哪一套。** 下一个候选是断点里的 `global_num_token_non_padded_cpu`
> ——它必须恒等于 `cu[-1]`，目前只在单请求场景核对过，DP 下未在设备上验证。

## 4. 为什么不照搬 CUDA 的 captured-metadata 契约

CUDA 的 DSV4 后端打开 `use_captured_forward_metadata_for_breakable_cuda_graph`：
capture 时建一份地址稳定的 metadata，replay 时原地刷新。这套契约的前提是**段内有
kernel 读这份 metadata**。

NPU 的 compressor、lightning indexer 和 sparse attention 都依赖 host 侧的值
（`kernel_metadata` 由 host 算子生成，state block table 的宽度来自 host 长度），本来
就进不了图。照搬只是把 metadata 改写成可原地刷新的 buffer，工作量与 decode graph 的
`_init_dsv4_graph_metadata` / `_refresh_graph_*` 相当（数百行），换不来任何算子进图。
只有将来 compressor 有了纯 device 侧接口、能进图时，这套契约才值得做。

## 5. 未覆盖的风险

- **`SGLANG_NPU_USE_MULTI_STREAM=1`。** 打开后 `enable_multi_stream` 在 prefill BCG
  下为真，attention 走 `_forward_prepare_multi_stream_npu`：先用 `wait_stream` 把 KV
  和 Q 分到侧流。BCG 的 hook 能跟踪这次分流，断点结束 segment 时会自动 join；indexer
  自带的多流准备现在只在断点里 eager 执行。代码上能走通，**未在设备上验证**。官方部署
  文档只在 decode 节点设置这个开关。
- **attn-TP 序列切分。** 断点按 `global_num_token_non_padded_cpu` 切行，与已有
  attention 断点一致，前提是 `attn_tp_size == 1`（DP16 部署满足）。
- **CP 与 MoE。** DSV4 不在 `PREFILL_CP_BCG_SUPPORTED_COMBINATIONS` 里，CP 的 BCG
  capture 不会为它打开；CP batch 是否会去 replay 非 CP 的图没有分析。这些路径里若有
  host 同步，capture 会直接报错，不会悄悄算错。
- **精度。** 尚未用 sgl-eval 对 breakable 与 disabled 做过 gsm8k 对比。
