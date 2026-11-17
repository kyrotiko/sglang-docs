# DeepSeek V4.1 Prefill Graph 验收计划

开发基线：`sglang@9898f7ce9a`；2026-11-17。[合入 randgun 后](../analysis/randgun_merge_20261117.md)的 CPU 回归 281 项通过。旧基线设备证据见 [run1](a5-20261116-run1/findings.md) / [run2](a5-20261116-run2/findings.md)；[run3](a5-20261117-run3/findings.md)在 950DT 和实际内容已变化的 SDK 中验证了 compressor、公开 writer 和 mHC premix。revision 4 只修正 writer→attention 的脚本 dtype 视图，待三组重跑；独立精度和整模/多卡仍待验收。

## 1. 实验环境记录

每份真机结果必须记录：模型和 checkpoint revision、主仓 SHA、A5 型号和卡数、CANN/driver/torch/torch_npu/sgl-kernel-npu 版本及 DSL 算子包来源、TP/EP/DP/CP/DCP/PP、量化、KV dtype、Engram 模式、DSpark 参数、prefill buckets、request/context 上限和完整启动命令。

运行、部署、复现或 benchmark 前先读模型 cookbook：`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`。本轮已读取；其推荐矩阵覆盖 NVIDIA/AMD，配方来自 `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`，不能直接充当 V4.1 A5 的推荐部署；NPU 的 V4 Flash 页面也不能替代 V4.1 配方。执行前还需核对本分支实际 A5 启动配置，并记录相对 cookbook 的硬件/参数差异。`test/` 配置只能作为明确标注的 CI 复现配置。

## 2. 执行优先级与 CPU 契约验证

先验证闸门和现有 source/attention/Engram break 的候选闭环，再做决定 ABI 的小探针，随后联合验证固定 metadata 与计算数据流。完整矩阵分配到对应实施阶段，不要求 metadata 编码前先验证全部算子和全部部署组合。首版固定 B=1 和实际 DP idle 策略；B>1、复杂并行和 DP idle 非 eager 的组合后置。

这些测试不能证明 NPU 算子可入图，只防止固定状态协议和路由规则回归。优先加入已有相应子系统测试文件；只有独立 fixture、依赖或 CI 契约确有需要时才新建文件。

| 场景 | 应防住的错误 |
| --- | --- |
| NPU prefill backend 解析与非法配置拒绝 | 把 NPU 请求交给 CUDA graph，或把不支持配置默认为另一执行路径 |
| V4-only 闸门及 source/attention 路由配对 | V4.1 错用 V4 metadata；source eager 而 consumer 捕获了旧输出 |
| metadata 生命周期及原地刷新 | 字段重新绑定后图继续读旧 storage；新增字段未加入 replay 更新 |
| 保留请求槽与 cycle-bank 绑定 | 真实 allocator 分配槽 0；replay 忘记把 capture bank 0 换成本次 request-bank |
| 同 bucket 不同真实长度、较长后接较短 | 未清理尾部 cu/page table/loc/candidate，串读前一个 batch |
| request/context/candidate 容量边界 | admission 允许越界，再由 copy 截断数据，或调度/执行判断不一致 |
| source/consumer 复用和失效 | 第二组 source 覆盖第一组；index source 刷新与 KV source 混淆 |
| graph scope 的 MoE dispatch | prefill capture 被当成 decode MC2；AllGather 走 host 计数；AllToAllV 未被拒绝 |
| cache loc / Engram history / hidden-state 的 replay 传递 | live batch 的新字段没有进入 static batch 或 eager tail |

遵守仓库 unit-test admission：每个 case 必须对应派生性质、关键生命周期 bookkeeping 或真实 bug 回归。真实 bug 回归需要在修复前失败、修复后通过；不能用 mock 的“capture 成功”代替真实设备验证。

可复用测试入口：

- `test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py`
- `test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py`
- `test/registered/unit/npu/graph_runner/test_npu_graph_capture.py`
- `test/registered/unit/npu/graph_runner/test_npu_cudagraph_backend.py`
- `test/registered/unit/npu/attention/test_npu_ascend_dsv4_backend.py`
- `test/registered/unit/npu/attention/test_npu_dsv41_low_ratio_backend.py`
- `test/registered/unit/models/test_deepseek_v4_platform_dispatch.py`
- `test/registered/unit/layers/test_engram_history.py`
- `test/registered/unit/moe/test_ascend_ops.py`

## 3. A5 最小设备探针

对每类算子建立同样的 replay 检查：初始化独立且相同的 eager/graph 状态，capture 后原地改变输入值、prefix、cache loc 和有效长度，多轮交替执行，与对应 eager 结果及缓存状态比较。不能只用 capture 的原输入 replay 一次。

| 算子/状态 | 必须覆盖 |
| --- | --- |
| ratio 1 投影/压缩写回 | 相同 bucket 的不同 live T、完整和非完整 bucket、实际输出容量与写回位置 |
| ratio 2 compressor | 奇偶 prefix/extend 组合、跨 chunk、没有新压缩块和恰好新建块、request-bank 复用 |
| source/consumer cache | 同组消费者读到本轮 source；下一 source 组不串数据；换请求时状态重置 |
| indexer/candidate/top-k | 候选数变化、边界候选、top-k sentinel、结果被同轮 attention 消费 |
| sparse attention | 页表和 cu 改变、非空 prefix、padding/零有效 query 的处理；确认实际 graph 覆盖 |
| mHC/多流 | 子层间的 pre 系数依赖、必要 side output、多段之间的 fork/join 和 event 顺序 |
| Engram | 同 shape 不同 token/history、lookup 行变化、模式对应的 collective/break 行为 |
| AllGather dispatch/combine | 固定容量路由、padding、quant scale、输出回到正确 rank；无 capture 期 host 计数 |

检查真实 KV、SWA、compressor state 和 Engram history 中无效写入区域。padding 结果为零不代表状态没有被污染。首版优先核实并复用保留请求槽 0：capture/warmup 仅写 bank 0、真实请求不分配槽 0、replay cycle table 切换到本轮 live bank。对非零真实 bank 比较 capture 前后状态；按相同初始状态运行有/无 capture 的 eager/graph 序列，覆盖奇数 chunk、bank 复用及下一次 decode。只有现有槽 0 无法满足隔离时才引入额外 bank 或快照恢复。

固定状态验收逐项检查 `[T_cap]` live mask、window indices/length、各 ratio 的 cmp length、`[B_cap]` cycle table、full/compressed/index page table、positions/loc 及 kernel metadata 的 shape/address 和尾部。padding 长度为 0，loc/index 的 skip sentinel 按实际 ABI 验证；不把 V4 的 0 sentinel 套给 V4.1。

context 验收记录页表容量来源和有效 `prefill.max_seq_len`。当超出较小图容量但仍在模型可服务范围时，应由现有 admission 一致回退；超过模型 context 则按服务契约拒绝。NPU 当前不支持 `max_context_size` capability，不能仅设置该参数假定已经获得回退。

## 4. 整模正确性矩阵

| 维度 | 用例 |
| --- | --- |
| 长度 | bucket 原长、可准入的非整 bucket、压缩边界前后、context 上限和超上限回退 |
| 历史 | 无 prefix、有 prefix、连续 chunk、prefix 复用、新请求替换旧 request slot |
| 连续 replay | 大后小、小后大、两个完全不同的 prompt 同 bucket，多轮交替 eager/replay |
| 请求数 | 首版 B=1；多请求启用后覆盖容量上限、不同长度组合和空 request slot |
| 并行 | 首个固定 TP=EP 拓扑；扩展 DP 时含不均匀行数和 idle rank；每项记录是否已支持 |
| 输出 | 首 token/logits、需要时的 prompt logprobs、hidden states、后续 decode 数值和完整响应 |
| 投机 | 无投机与启用 DSpark；target prefill 后 draft/target 的 KV layout、提交和下一轮生成 |
| 回退 | 没有 bucket、超 request/context 容量、未验证 CP/DCP/TBO/AllToAllV，所有相关 rank 决策一致 |

按实际 dtype 和 eager 自身数值波动设定并记录误差标准，不能先编造统一阈值。比较 logits/hidden states 之外还要检查生成 token 和后续缓存行为；出现生成差异时保留首个偏离 step 的输入和状态。

准确率使用仓库约定的 `sgl-eval`，服务外部 CLI 采用 `python/pyproject.toml` 的 test extra 固定版本，在独立虚拟环境运行；测试内用 `sglang.test.sgl_eval_utils.run_sgl_eval`。graph/eager 使用相同 harness、样本和模型推荐采样设置，显式传递 temperature/top_p/thinking/max tokens，记录实际参数。不得换用手写评分或旧 `run_eval`。

## 5. 图覆盖与性能验收

正确性通过后再比较性能。A/B 的唯一预期变量为 prefill graph 开关；模型、并行、量化、DSpark、Engram、请求集合、batch/chunk 调度和 prefix-cache 命中条件保持一致。decode graph 配置单独固定。

至少记录：

- prefill forward 时延分布和 CPU 算子下发开销；启动/capture 时长单列。
- TTFT、prefill 吞吐及后续 decode 指标；不以一个端到端均值代替 prefill 分析。
- graph hit/fallback 次数和原因、每步 live/bucket 行数、padding 放大以及真实请求数。
- 图数量、segment 数和 eager break 清单、图内 compressor/indexer/attention/MoE 的实际 kernel 证据。
- capture memory、workspace 和 KV/state 隔离槽开销；与相同上下文容量的 eager 基线比较。
- 多轮 warmup 后重复测量的样本数及分布，分别看小 chunk、满 bucket 和明显 padding 的输入。

首轮可选 512/1024/2048 chunk 用于对照旧工作，但 bucket 应最终按 A5 真实工作负载和 memory budget 选择，不预设这些值是最优配置。

## 6. 完成判定

同时具备以下证据后才宣布对应配置支持：

1. 正确路由到 NPUGraph；图内覆盖和剩余 break 的职责可以从日志/profiler 复核。
2. 改变输入和 metadata 后，多轮 replay 的数值、缓存状态及后续生成与 eager 对齐。
3. 超界及未支持配置有明确、集体一致的拒绝或 eager 回退，不会 hang 或悄悄切换算法。
4. DSpark/decode 等现有路径完成适用的回归，新增 prefill 状态没有污染它们。
5. 给出当前部署配置下的收益、padding 和 memory 成本；暂未改善的配置如实记录。

实验脚本放在本需求目录的 `scripts/`，结果放 `validation/` 并带环境与源码 SHA；稳定回归测试进入主仓对应测试子系统。

## 7. 当前可运行的事实收集脚本

已提供单文件 [verify_dsv41_prefill_graph.py](../scripts/verify_dsv41_prefill_graph.py)，[运行说明与覆盖边界](../scripts/README.md)。在 A5 serving 环境中执行，不加载模型权重：

```bash
python verify_dsv41_prefill_graph.py --output-dir ./dsv41-prefill-evidence
```

如需指定 checkout，增加 `--sglang-root /path/to/sglang`。revision 4 仍为 17 组；第三轮已完成全量，当前按[三组最小重跑命令](a5-20261117-run3/findings.md)补 writer 消费链即可。每组独立进程并设超时，回传输出目录中的 `dsv41-prefill-evidence.tar.gz`，失败或 BLOCKED 也保留。

Python 编译、CPU 自检和记录/归档回归已通过；revision 3 的第三轮全量 A5 结果已归档，新增 compressor/writer/mHC 图对照 exact。revision 4 的 dtype 视图回归已在本地验证修复前失败、修复后通过，实际 writer 消费链待三组设备复跑。独立 CPU reference、整模、真实 source/多流状态、DSpark KV 注入和多卡 collective 仍按各自范围验收。
