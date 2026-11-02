# 把其他 MLA / DSA 模型接进 NPU breakable prefill

**未实现。** 目前只有 DSV4 Flash 跑通，见
[dsv4_flash_prefill_graph.md](dsv4_flash_prefill_graph.md)。本文是后续适配的计划。

目标是普通 TP prefill。CP / DCP 应明确报不支持；DP、多流、MTP 需另外验证。

## 1. 要改的入口

`handle_attention_ascend()` 当前的选择：

| 模型 | 普通 prefill | decode / verify |
| --- | --- | --- |
| 普通 MLA | `MHA_NPU` → `forward_mha_prepare_npu()` | `MLA_NPU` |
| DSA | `DSA_NPU` → `forward_dsa_prepare_npu()` | `DSA_NPU` |

所以要改的是 **`forward_mha_prepare_npu()`**，不是 `forward_mla_core_npu()`——给后者
加装饰器补不齐这条路径。两条 core 都经 `RadixAttention` 调用 attention，已有的
`_eager_attention()` 断点会读当次 batch、裁剪 Q/K/RoPE/top-k、写入固定输出；
`AscendAttnBackend.forward_sparse()` 的主 KV 写入也在这个断点内。

| 文件 | 改动 |
| --- | --- |
| `hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | `forward_mha_prepare_npu()`：Yarn 分支的 `npu_kv_rmsnorm_rope_cache()` 放进新断点，调用前分配固定形状的 `k_pe` / `kv_a` 输出；非 Yarn 分支的 `set_kv_buffer()` 也经断点执行 |
| `layers/attention/dsa/dsa_npu_indexer.py` | `forward_npu()` 主体拆出 `_forward_npu_eager()`；新入口在 BCG prefill 下分配 `[bucket_tokens, index_topk]` 的 int32 输出，经断点执行原计算并写回。其他模式走原实现 |

`PrefillCudaGraphRunner` 已有的 backend 准备钩子和每批 metadata 初始化可直接复用。
不需要新建 runner、改 torch_npu，或对段图调 `graph.update()`（见
[backend design §4.1](breakable_npu_graph_backend_design.md)）。

## 2. capture batch 的缓存位置

DSV4 的经验：**不要**为 capture 单独分配真实 dummy KV 再在 session 结束时回收。
capture batch 没有分配 bundle，直接把这一期的写入导到 padding slot 0 就够了，条件用
`capturing_prefill_graph`（不是 `is_in_breakable_cuda_graph()`，后者 replay 时也为真）。
依据和 slot 0 的安全性论证见
[dsv4_flash_prefill_graph.md §3.1](dsv4_flash_prefill_graph.md)。

## 3. DSA 断点的形状

```python
# forward_npu() 的 BCG prefill 分支；q_lora 此时按全局 bucket 布局
out = torch.empty(
    (q_lora.shape[0], self.index_topk), dtype=torch.int32, device=q_lora.device
)
bcg_npu_dsa_indexer(indexer=self, x=x, q_lora=q_lora, positions=positions,
                    out=out, layer_id=layer_id, ...)
return out


@eager_on_graph
def bcg_npu_dsa_indexer(*, indexer, x, q_lora, positions, out, layer_id, ...) -> None:
    batch = get_forward_batch()          # 每次取当次请求
    n = batch.global_num_token_non_padded_cpu
    out.fill_(-1)
    if n == 0:
        return
    result = indexer._forward_npu_eager(..., real_num_tokens=n)
    out[:n].copy_(result)
    # 断点返回 None；调用方持有并传递 out
```

**`real_num_tokens` 不等于把所有输入先切成 `[:n]`。** 启用 AG-after-QLoRA 时 `x` 可能
是 TP 分片而 `q_lora` 已经聚合。要先保留 bucket 形状完成投影和原有 all-gather，再在
写 index K / scale、执行 top-k 之前裁剪聚合后的 Q/K/weights 和缓存位置；positions 的
计算布局也须与聚合后的行数一致。

这正是 DSV4 踩过的那类口径问题，参考
[dsv4_flash_prefill_graph.md §3.3](dsv4_flash_prefill_graph.md)：live 口径与 bucket
口径同时出现在一个函数里时，逐个确认每个派生量跟的是哪一套。

CUDA 的 `bcg_dsa_indexer_prefill_split()` 显式只接受 CUDA/ROCm，内部还有 CUDA 量化与
metadata 接口，不能在 NPU 上直接调用。

## 4. MLA 的融合缓存写入

Yarn 分支的融合算子既写 KV，又返回后续 `kv_b_proj()` 要用的 `k_pe` 和 `kv_a`。新断点
应继续调用原融合算子，把真实行结果复制到调用前分配的固定输出、补零剩余行。不能只把
算子移出图而不给后续段提供输出，也不应为拆图换成另一套 norm / RoPE 算法。

新断点只用于 BCG prefill。`forward_mla_prepare_npu()` / MLAPO 的 decode 路径沿用已有
机制，无需全局关闭 MLAPO。

## 5. 怎么判断已经支持

同一权重、相同量化和 KV 布局，对比 eager 与 BCG：

1. 同一个 1024-token 桶：`1×1024` → `2×512` → 不等长多请求，连续 replay。
2. 1000 个真实 token 补到 1024：attention 输出 padding 为零，top-k padding 为 -1，
   padding 不写真实 KV。
3. 第二次 replay 更换请求槽、物理缓存地址和 prefix 长度，确认没有用到 capture 期数据。
4. prefill 后继续 decode，检查 logits 和缓存结果；DSA 要覆盖 top-k 共享层，以及序列
   长度跨过 `index_topk` 的情况。
5. TP 多卡下覆盖 AG-after-QLoRA 开关；确认真实发生了 segment replay，之后再测性能。
