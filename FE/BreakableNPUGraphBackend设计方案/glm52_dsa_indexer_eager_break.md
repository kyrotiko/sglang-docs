# GLM-5.2：NPU DSA indexer 的 breakable prefill 断点

基于 `sglang@5257b6d5b4`。实测现象：只有实际命中 prefill graph replay 才挂起；启用功能但请求回落 eager 时正常。`error1.log` 中 DP6/TP12、13 停在 attention 参数 `.to()` 的流同步，其余 rank 停在 DeepEP dispatch。该日志尚不能确定具体阻塞的设备任务。

本次只补 DSA indexer 的 eager 边界，用于验证图内 indexer 的 metadata、缓存写入与多流捕获是否参与触发，不宣称已经确认挂起根因。

实现提交：`sglang@6ceb3d9fa7`，位于 `support_breakable_graph_back` 分支，包含运行时代码和 7 项 CPU 回归。

真机反馈：该提交可以完成启动和捕图，但几个请求后仍然挂起、NPU 100%，尚未解决原问题。后续按[诊断日志采集说明](glm52_prefill_graph_debug.md)追踪实际 replay。

[运行时代码单文件补丁](glm52_dsa_indexer_eager_break.patch) 可在基于上述提交、尚未包含此改动的服务器源码根目录用 `git apply /path/to/glm52_dsa_indexer_eager_break.patch` 应用。当前工作区主仓已应用，勿重复应用；应用后需重启服务重新捕图。

## 修改范围

只修改 `sglang/python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py`，保留现有 attention 和 MoE 断点；普通 eager、decode、target verify 和 draft extend 仍调用原计算主体。

1. 原 `forward_npu()` 主体提取为 `_forward_npu_eager()`。
2. BCG 普通 prefill 入口在捕获池内分配 `[q_lora.shape[0], index_topk]` 的 int32 输出。
3. `bcg_npu_dsa_indexer()` 使用 `@eager_on_graph`，每次执行通过 `get_forward_batch()` 读取当前 batch；不把 capture batch 作为断点参数保留下来。
4. 断点填充输出为 `-1`；空闲 DP 组不写 index KV、不执行 top-k。真实行的结果复制到固定输出，函数返回 `None`。
5. 为原主体增加可选的 `real_num_tokens`。保留投影、RoPE 和 attn-TP all-gather 的桶形状，完成原有多流 join 和 gather 后裁剪 Q/K/weights；缓存位置也只取真实行，量化 K 和 scale 的写入均不包含 padding。

`global_num_token_non_padded_cpu` 表示一个 attn-TP 组的真实 token 数。空闲分支以整个 attn-TP 组为单位跳过 indexer，后续 MoE/EP 通信仍由已有图段和断点执行。

## 边界与限制

- 本补丁针对普通 prefill（含 DP attention + attn-TP），CP/DCP 使用不同的行布局，明确拒绝其 DSA BCG 路径，避免把普通布局的裁剪规则套进去。
- 不在 `forward_npu()` 入口统一裁剪输入：AG-after-QLoRA 开启时，`x` 可能是 TP 分片而 `q_lora` 已经聚合。
- 不复用仅支持 CUDA/ROCm 的 `bcg_dsa_indexer_prefill_split()`，不对返回 Tensor 的原函数直接加 `@eager_on_graph`。
- 不增加设备同步或调用 `graph.update()`。
- 这不是 dummy KV allocation 或整个 NPU prefill graph 兼容性的完整修复。

## 验证

CPU 回归使用真实 Python 计算主体与模拟 NPU 算子，检查捕获时不运行 indexer、同一断点跨请求读取新页表/长度/缓存位置、固定输出地址、padding/idle、量化 K/scale、AG-after-QLoRA 的桶形状聚合，以及非 BCG 路径。

本地验证：新增 `test_npu_dsa_indexer_break.py` 的 7 项隔离 CPU 回归通过，使用真实 `eager_on_graph` 和 `BreakableCUDAGraph.replay()`，模拟设备算子与段执行。Ruff 检查及格式、Python 语法检查、`git diff --check` 均通过；导出的补丁已通过反向应用检查。Windows 普通包导入受 Unix `resource` 模块缺失限制，测试启动时仅跳过 `sglang.__init__` 的包初始化；尚未验证真实 NPUGraph 捕获、NPU 算子或分布式并发。

真机验收：沿用原 GLM-5.2 压测配置和负载，确认实际 graph 命中；先对比 eager 的 top-k、logits 和后续 decode，再持续压测。需覆盖同桶不同请求数/前缀/物理 KV 页、DP 空闲组、AG-after-QLoRA、多流和 MTP 组合。若仍挂起，按 rank、batch、layer、segment 追踪新断点前后的设备进展，不能仅凭 watchdog 栈归因。
