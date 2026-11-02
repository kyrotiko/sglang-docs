# SGLang 批次形状字段速查

`seq_lens`、`num_tokens`、`cu_seqlens`、`out_cache_loc` 这组名字长得都差不多，但单位不同：
有的按**请求**计，有的按**token**计，有的是**KV 槽位号**。混用不会报错，只会算错。

定义取自 `python/sglang/srt/model_executor/forward_batch_info.py` 的字段注释与
`managers/schedule_batch.py` 的构造处。行号会漂移，以内容为准。

## 1. 三种单位

| 单位 | 含义 | 典型字段 |
| --- | --- | --- |
| **请求数**（bs） | 这一批有几个请求 | `batch_size`、`req_pool_indices` |
| **token 数** | 这次 forward 要算几行 | `extend_num_tokens`、`positions`、`input_ids` |
| **槽位 / 页** | KV cache 里的物理位置 | `out_cache_loc`、`page_size` |

仓库的注释规范里专门有一条：名字带不动的单位要写注释（tokens vs reqs vs pages vs slots）。

## 2. 按请求给的向量（长度 = bs）

这三个是核心，理解它们的关系就理解了一半：

| 字段 | 含义 |
| --- | --- |
| `seq_lens` | 这个请求的 **KV 总长度** = 已缓存的 + 这次新算的 |
| `extend_prefix_lens` | 其中**已经在 KV cache 里**的部分（prefix cache 命中的长度） |
| `extend_seq_lens` | 其中**这次要新算**的部分 |

```
seq_lens  =  extend_prefix_lens  +  extend_seq_lens
   ↑ KV 有多长                        ↑ query 有多少行
   （attention 读多少历史）             （这次算多少 token）
```

这个区分要紧，因为 attention 的两个维度就是它们：KV 长度决定读多少历史，query 行数决定算
多少新 token。

| 场景 | `seq_lens` | `extend_prefix_lens` | `extend_seq_lens` |
| --- | --- | --- | --- |
| 全新请求，1000 token prompt | 1000 | 0 | 1000 |
| prompt 前 900 token 命中 prefix cache | 1000 | 900 | 100 |
| chunked prefill 第 2 块（每块 512） | 1024 | 512 | 512 |

**decode 不走这组字段。** `extend_*` 全部挂在 `# For extend` 分组下、类型是 `Optional`，
decode 时为 `None`；此时 `seq_lens` 直接表示 KV 总长度，每个请求贡献 1 行 query。
所以 decode 下 token 数 == 请求数，这也是 decode 能按 bs 分桶、prefill 不能的原因（见 §7）。

## 3. 标量

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `batch_size` / `bs` | 请求 | 请求个数 |
| `extend_num_tokens` | token | 这次 forward 的总行数，构造处就是 `sum(len(ids) for ids in input_ids)` |
| `seq_lens_sum` | token | `sum(seq_lens)`，字段注释写的是 "The sum of all sequence lengths"，用来估 KV 占用 |

## 4. 偏移与索引向量

最容易混的一组。注意前两个的**长度差 1**：

| 字段 | 长度 | 形态 | 用途 |
| --- | --- | --- | --- |
| `extend_start_loc` | bs | `[0, 1000, 1100, …]`，**不含总和** | 每个请求在扁平 token 数组里的起始下标 |
| `cu_seqlens`（NPU 上是 `actual_seq_lengths_q_pa`） | **bs + 1** | `[0, 1000, 1100, …, 总数]`，**首 0 尾总数** | 前缀和 |
| `out_cache_loc` | token | 槽位号 | 这次每个 token 的 KV 写到哪个槽 |
| `req_pool_indices` | bs | 请求槽号 | 每个请求在 `req_to_token` 表里占第几行 |

因为 `cu_seqlens` 多了一头一尾，所以：

```python
cu[i + 1] - cu[i]     # 第 i 个请求的长度
cu[bs]                # 前 bs 个请求的 token 总数（即全部）
```

## 5. 两张间接表

| 表 | 形状 | 含义 |
| --- | --- | --- |
| `req_to_token` | `[请求槽数 + 1, max_context_len]` int32 | 请求槽 → 它每个 token 的 KV 槽位号 |
| KV pool | 按 `page_size` 分页 | 真正存 K/V 的地方 |

取一个请求的 KV 位置是两跳：`req_pool_indices[i]` 拿到行号，再查 `req_to_token` 这一行。

这两处都**保留了 index 0 给 cuda graph 的 padding**，是同一个设计模式出现两次：

```python
# ReqToTokenPool：+1 padding row at index 0: cuda-graph padded batches default
# req_pool_indices to 0, so dummy reads/writes land here harmlessly.
self.free_slots = list(range(1, self._alloc_size))

# 分页 KV 分配器：The padded slot 0 is used for writing dummy outputs from padded tokens.
self.free_pages = torch.arange(1, self.num_pages + 1, ...)
```

真实请求永远拿不到第 0 行 / 第 0 页，所以图捕获用的假 batch（`req_pool_indices = arange(bs)`，
从 0 开始）读写的正好是这两处 padding 位，不会破坏真实数据。

## 6. padding 相关的三个 token 数

开了 prefill cuda graph 之后，同一次 forward 里**同时存在三个 token 数**：

| 数 | 含义 | 例（bucket 128、实际 64 token） |
| --- | --- | --- |
| bucket 大小 | 图录下的形状，也是 `positions` / `input_ids` 的长度 | 128 |
| `global_num_token_non_padded_cpu` | 真实（非 padding）token 数 | 64 |
| `cu[bs]` | 前缀和末项 = 真实 token 数 | 64 |

后两个**必须恒等**，但来源完全不同：一个来自 DP 侧的计数，一个来自各请求长度的前缀和。
eager 下三者都是 64，分歧永远不会暴露——这正是为什么这类 bug 只在入图后才出现。

还有 **GLOBAL / LOCAL** 一层，字段注释明确写了 "meaning never changes once set"：

| 字段 | 作用域 |
| --- | --- |
| `global_num_token_non_padded` | GPU 标量，整个 attn-TP 组**切分前**的真实 token 数 |
| `global_num_token_non_padded_cpu` | 同上，host int |
| `num_token_non_padded` | GPU 标量，本 rank **切分后**自己那份（由 GLOBAL 派生） |

## 7. 一个命名陷阱：`--cuda-graph-bs-prefill` 里的 bs 不是请求数

| 参数 | 名字里是 bs | 实际语义 |
| --- | --- | --- |
| `--cuda-graph-bs-decode 1 2 4 8` | 是 | **请求数**（decode 下 token 数 == 请求数） |
| `--cuda-graph-bs-prefill 128 256 … 2048` | 是 | **token 数** |

启动日志会按实际语义打印出来：

```
Capture target prefill CUDA graph begin. backend=breakable,
num_tokens=[128, 256, 512, 1024, 2048]
```

prefill 按 token 数分桶，是因为它的计算量由 token 总数决定而不是请求个数——**一个 2048
token 的请求和 8 个 256 token 的请求，对图来说是同一个形状**（都是 2048 行）。decode 下
每个请求固定 1 行，两者才等价。
