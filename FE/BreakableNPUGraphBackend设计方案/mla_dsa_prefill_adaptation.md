# NPU breakable prefill：MLA / DSA 最小改动

代码实现位于 `support_breakable_graph@e16facc2ba`，包含 DSV4、MLA/DSA 普通 prefill 适配，尚未进行整模型 NPU 验证。

实现：复用现有 attention 断点，MLA 补 prepare 中的 KV 写入，DSA 补 indexer 的 eager 断点及固定输出。首版目标是普通 TP prefill；CP/DCP 配置会明确报不支持，DP、多流、MTP 需另外验证。

## 1. 实际走哪条路

[`handle_attention_ascend()`](../../../sglang/python/sglang/srt/models/deepseek_common/attention_backend_handler.py) 当前选择：

| 模型 | 普通 prefill | decode / verify 等模式 |
| --- | --- | --- |
| 普通 MLA 模型 | `MHA_NPU` → `forward_mha_prepare_npu()` | `MLA_NPU` |
| DSA 模型 | `DSA_NPU` → `forward_dsa_prepare_npu()` | `DSA_NPU` |

因此支持 MLA 模型的 prefill，首先改 `forward_mha_prepare_npu()`。直接给 `forward_mla_core_npu()` 加装饰器不会补齐这条路径。

两条 core 都经 `RadixAttention` 调用 attention，已有 `_eager_attention()`：读取当前 batch、裁剪 Q/K/RoPE/top-k、写入固定输出。`AscendAttnBackend.forward_sparse()` 的主 KV 写入也在这个 attention 断点内。

## 2. 具体改什么代码

以下文件路径均相对 `sglang/python/sglang/srt/`。

| 文件 | 具体修改 |
| --- | --- |
| `hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | 改 `forward_mha_prepare_npu()`：Yarn 分支的 `npu_kv_rmsnorm_rope_cache()` 放进新 eager 断点，调用前分配固定形状的 `k_pe` / `kv_a` 输出；非 Yarn 分支的 `set_kv_buffer()` 也经 eager 断点执行。断点读取当前缓存位置，只写真实 token，输出 padding 清零。 |
| `layers/attention/dsa/dsa_npu_indexer.py` | 原 `forward_npu()` 主体拆成 `_forward_npu_eager()`；新入口在普通 BCG prefill 下分配 `[bucket_tokens, index_topk]` 的 int32 输出，经 eager 断点执行原计算并写回。其他模式继续走原实现。 |
| `hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py` | 扩展 `prepare_capture_batch()`：普通 MLA/DSA 也建立有效 dummy 请求和 KV 映射。现有实现只处理 DSV4，其他模型直接返回。 |
| `hardware_backend/npu/graph_runner/mla_prefill_capture.py`（新增） | 用实际 allocator 的 `alloc_extend()` 分配普通 KV，page_size=1 使用 `alloc()`；写请求表并填静态 `out_cache_loc`，最大桶共用，capture session 结束或失败后回收。普通 MLA/DSA 不使用 `DSV4OutCacheLoc`、C4/C128 分配。 |
| `test/registered/unit/npu/graph_runner/test_mla_dsa_prefill_graph.py`（新增） | 覆盖当前 batch、固定输出、padding、量化分支、桶形状 TP 聚合、dummy 分配与回收。NPU 算子使用 mock，真实 NPU 再比 logits 与后续 decode。 |

`PrefillCudaGraphRunner` 已有 backend 准备钩子和每批 metadata 初始化，可直接复用。首版不要求新建 runner、改 torch_npu 或对段图调用 `graph.update()`。

## 3. DSA 断点的接口示意

下面是核心接口示意，省略原有投影与算子参数；完整实现见代码。

```python
# forward_npu() 的 BCG prefill 分支；q_lora 此时按全局 bucket 布局。
out = torch.empty(
    (q_lora.shape[0], self.index_topk),
    dtype=torch.int32, device=q_lora.device,
)
bcg_npu_dsa_indexer(self, x, q_lora, positions, out, layer_id,
                    input_on_attn_tp_slices, dynamic_scale)
return out

@eager_on_graph
def bcg_npu_dsa_indexer(indexer, x, q_lora, positions, out, layer_id,
                       input_on_attn_tp_slices, dynamic_scale):
    batch = get_forward_batch()       # 每次获取当前请求
    n = batch.global_num_token_non_padded_cpu
    out.fill_(-1)
    if n == 0:
        return
    result = indexer._forward_npu_eager(
        x, q_lora, positions, batch, layer_id,
        input_on_attn_tp_slices, dynamic_scale,
        real_num_tokens=n,
    )
    out[:n].copy_(result)
    # 断点返回 None；调用方持有并传递 out。
```

`real_num_tokens` 不是把所有输入先统一切成 `[:n]`：启用 AG-after-QLoRA 时，x 可能是 TP 分片，而 q_lora 已经聚合。先保留桶形状完成投影和原有 all-gather，再在写 index K / scale、执行 top-k 前裁剪聚合后的 Q/K/weights 和缓存位置。positions 的计算布局也须与聚合后的行数一致。

现有 CUDA 的 `bcg_dsa_indexer_prefill_split()` 显式只接受 CUDA/ROCm，内部还有 CUDA 量化与 metadata 接口，不能在 NPU 上直接调用。

## 4. MLA 的融合缓存写入

Yarn 分支的融合算子既写 KV，又返回后续 `kv_b_proj()` 要用的 `k_pe` 和 `kv_a`。新断点应继续调用原融合算子，把真实行结果复制到调用前分配的固定输出，补零剩余行；不能只把算子移出图而不给后续图提供输出，也不应为拆图随意换成另一套 norm / RoPE 算法。

新增断点只用于普通 BCG prefill。`forward_mla_prepare_npu()` / MLAPO 的 decode 路径沿用已有机制；首版无需全局关闭 MLAPO。

## 5. 怎么判断已经支持

同一权重、相同量化和 KV 布局，对比 eager 与 BCG：

1. 同一个 1024-token 桶：`1×1024` → `2×512` → 不等长多请求，连续 replay。
2. 1000 个真实 token 补到 1024：attention 输出 padding 为零，top-k padding 为 -1，padding 不写真实 KV。
3. 第二次 replay 更换请求槽、物理缓存地址和 prefix 长度，验证没有使用 capture 时的数据。
4. prefill 后继续 decode，检查 logits 和缓存结果；DSA 覆盖 top-k 共享层及序列长度跨过 `index_topk` 的情况。
5. TP 多卡下覆盖 AG-after-QLoRA 开关；捕图成功、失败均回收 dummy 分配。确认真实 segment replay，之后再测性能。

验证记录：本轮 MLA/DSA 9 项隔离 CPU 测试通过；连同已有 DSV4 与 capture session 回收检查，共 18 项通过。11 个 Python 文件语法检查、Ruff 与 `git diff --check` 通过。完整应用导入在本机 Windows 受 Unix `resource` 模块缺失限制；尚未启动 NPU 实例做捕图、logits 或性能验证。
