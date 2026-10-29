# BreakableNPUGraphBackend：DSV4.1 Flash 代码修改稿

基于 `sglang/main@2a6c2045705b`。首版验证 **DSV4.1 Flash**，先使用 TP-only，关闭 DSpark 和模型多流。

做法：复用现有 BCG，把每段换成 NPUGraph；**KV 写入、compressor/indexer、Engram lookup 在图外执行，Q/K 投影、WO、MLP 等计算留在图内。**

**[查看具体代码 diff](breakable_npu_graph_backend_main.patch)**：修改 5 个现有文件，新增 1 个文件。它是代码预览，尚未应用到主仓，也不是已经跑通模型的完整补丁。

## 改哪几个文件

以下路径都位于 `sglang/python/sglang/srt/`。

| 文件 | 具体修改 |
| --- | --- |
| [breakable_cuda_graph.py](../../../sglang/python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py) | 新增 `_device_graph_cls()`，NPU 返回 `torch.npu.NPUGraph`；stream capture 查询增加 NPU 分支 |
| [runner_backend/utils.py](../../../sglang/python/sglang/srt/model_executor/runner_backend/utils.py) | `resolve_prefill_backend()` 遇到 NPU 时构造 `BreakableNPUGraphBackend` |
| `hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py`（新增） | 继承 BCG backend；只增加首轮配置检查和 NPU 私有池 session，复用父类 capture/replay |
| [models/deepseek_v4.py](../../../sglang/python/sglang/srt/models/deepseek_v4.py) | 新增两个返回 `None` 的断点：`bcg_dsv4_npu_store_cache`、`bcg_dsv4_npu_sources`；替换 NPU prepare 中对应的直接调用 |
| [prefill_cuda_graph_runner.py](../../../sglang/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py) | `load_batch()` 把本批 `out_cache_loc_dsv4` 传给 replay 的 batch |
| [layers/engram.py](../../../sglang/python/sglang/srt/layers/engram.py) | NPU BCG prefill 的 lookup 也使用已有 `_bcg_engram_embed`，不再只对 scheduler-staged lookup 断图 |

不新增 policy 模块，不复制 prefill runner，也不给返回 Tensor 的 core 直接加装饰器。已有 `bcg_deepseek_v4_attention_with_output` 和 Engram hash 断点继续复用。

## 关键代码是什么

### 1. 每段使用 NPU 图

```python
def _device_graph_cls():
    if is_npu():
        return torch.npu.NPUGraph
    if _is_xpu:
        return torch.xpu.XPUGraph
    return torch.cuda.CUDAGraph

# _begin_new_segment() 原来写死 CUDA/XPU，这里改一行：
graph_cls = _device_graph_cls()
```

### 2. KV 写入改成图外执行

```python
@eager_on_graph
def bcg_dsv4_npu_store_cache(layer_id, kv) -> None:
    batch, n = _npu_bcg_live_batch()
    if n:
        get_attn_backend().store_cache(
            layer_id=layer_id, swa_k=kv[:n], forward_batch=batch
        )
```

`_npu_bcg_live_batch()` 在 diff 中有完整定义：读取当前 replay 的 batch，浅拷贝并裁掉 padding，不保存 capture 时的旧 batch。

在 `MQALayer._forward_prepare()` 的 NPU 分支中，将直接 `store_cache(...)` 改为调用这个断点。compressor/indexer 同理放进 `bcg_dsv4_npu_sources(...)`，完整函数和替换位置都在 diff 中。

### 3. 本批缓存位置传进 replay

```python
# PrefillCudaGraphRunner.load_batch()，已有 Engram 字段传递的旁边增加：
static_forward_batch.out_cache_loc_dsv4 = forward_batch.out_cache_loc_dsv4
```

这些断点在 capture 时只登记，在 replay 时才执行，因此不会把当前请求的缓存位置和 metadata 固定到图中。

## 还缺哪两块

这两块必须补齐，才能说 DSV4.1 Flash 真正可用；上面的 diff 尚未实现它们。

| 位置 | 现在的问题 | 要补的代码 |
| --- | --- | --- |
| [ascend_dsv4_backend.py](../../../sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py) | `forward()` 只接受 ratio 0/4/128，也没有 `forward_low_ratio_sources()` | 核对实际 Flash checkpoint；若使用 ratio 1/2，补对应 NPU attention、source/reuse 和缓存支持。不能仅把允许值改成 0/1/2/4/128 |
| `PrefillCudaGraphRunner.capture_prepare()` | 通用 dummy batch 没有有效的 DSV4 缓存 bundle，compressor warmup 会用到它 | 用 NPU DSV4 allocator 分配 dummy 请求/KV，填 `out_cache_loc_dsv4` 和请求映射；保留分配到图销毁，清理时释放。不能填 `None` 冒充完成 |

这两块没有现成的一行适配，本稿不以未定义的 helper 或假实现掩盖缺口。若目标模型的 NPU eager 尚未通过，先补第一块；图捕获前必须补第二块。

### 当前分支的捕图修复（2026-10-29）

`support_breakable_graph@e16facc2ba` 上已补代码：`DSV4PrefillCapture` 用现有 allocator 分配最大桶的 FULL/SWA/C128 KV，建立请求表，供各桶 warmup 复用；`load_batch()` 传递真实 bundle。

生命周期调整：SWA 写入和 indexer/compressor 使用 eager 断点，从当前 batch 读取位置，因此 dummy 请求行和物理 KV 页只保留到整个 capture session 结束，成功和失败均释放，不占用服务期间的缓存容量。dummy batch 中的 Tensor 引用由图保留。这取代上表中“分配保留到图销毁”的初步安排。

首版仍按普通 TP prefill 验证；CPU 回归覆盖分配、页表、回收及断点读取当前请求，尚未进行整模型 NPU 验证。

## 怎么验证

1. 同一份 DSV4.1 Flash 权重，先确认 NPU eager 能完成 prefill 和后续 decode。
2. 完成上述缺口后，开启 breakable，比对同桶不同请求数、不同长度、padding 的 logits，以及下一步 decode；同时确认实际执行了图。
3. 正确后再测延迟、吞吐和图内存。

main 补丁预览仅检查了 `git apply --check`；当前分支的代码与 CPU 回归记录见 [MLA/DSA 实现说明](mla_dsa_prefill_adaptation.md)，尚未进行整模型 NPU 验证。原设计稿保留。
