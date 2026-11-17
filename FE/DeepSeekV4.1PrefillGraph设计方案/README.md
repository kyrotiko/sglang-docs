# DeepSeek V4.1 NPU Prefill Graph

## 需求与开发基线

需求：在 Ascend NPU 上支持 DeepSeek V4.1 的 prefill 入图，复用执行计划，减少 prefill 的 CPU 算子下发开销，并保持 eager 路径的计算和缓存语义。

- 主仓：`sglang`，开发分支 `dsv41-a5`。
- **当前基线：`9898f7ce9a`**，已合入 `randgun/dsv41-a5@c790194814` 的 13 个新提交，保留先前迁入的 11 个入图提交。合并与复验调整见 [记录](analysis/randgun_merge_20261117.md)。
- 迁移前版本：`3cc67a7544`；备份分支 `backup/dsv41-a5-before-prefill-graph-20261116`。
- 首批入图来源：`prefill_graph_ascend_ops_on_main@15443a991f`；最新功能基线另合入 `randgun/dsv41-a5@c790194814`，详见合并记录。
- 已保留的上游重构：`43d0ecf9e1` 的 device-neutral graph backend bases。
- 更新日期：2026-11-17。已有[首轮](validation/a5-20261116-run1/findings.md)、[第二轮](validation/a5-20261116-run2/findings.md)和[第三轮](validation/a5-20261117-run3/findings.md) A5 证据。第三轮新环境已验证 compressor、公开 writer、mHC premix 及既有算子的图回放；writer→attention 的脚本 dtype 视图已修复，待三组重跑。独立 CPU attention 精度和整模/多卡仍未验收。

**后续基于迁入后的框架和 scope 重构实现 V4.1 增量。** NPU backend 解析、NPUGraph segment、capture/replay 生命周期、V4 固定 metadata 和 MoE prefill scope 已进入开发分支；不再作为待重写的基础设施。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [合入 randgun 后的当前基线与复验](analysis/randgun_merge_20261117.md) | 13 个新提交、281 项 CPU 回归、旧事实适用性、revision 3 定向命令；优先阅读 |
| [首批入图提交迁移记录](analysis/prefill_graph_integration.md) | 旧基线 5633d68b04 的 11 个入图提交来源、冲突处理及当时测试记录 |
| [迁移前源码分析](analysis/current_branch_gap_analysis.md) | 固定在 `3cc67a7544` 的历史快照，保留模型状态和代码证据；框架/MoE 缺口已由迁移覆盖 |
| [实现方案与阶段划分](design/implementation_plan.md) | 基于迁入代码扩展 V4.1 metadata、source/candidate、Engram 和输出契约 |
| [验收与验证计划](validation/acceptance_plan.md) | CPU 契约测试、A5 单算子与整模验证、性能 A/B、必要环境记录 |
| [单文件 A5 验证脚本与运行说明](scripts/README.md) | revision 4，共 17 组；修正 writer/consumer dtype view，当前只需补跑三组消费链 |
| [首轮 A5 事实与下一轮动作](validation/a5-20261116-run1/findings.md) | 原始证据、逐项失败归因、图回放与 CPU 参考区分、修订脚本重跑范围 |
| [第二轮 A5 事实与当前阻塞](validation/a5-20261116-run2/findings.md) | QLI/QSLI 图链 exact、BF16 诊断、部署 API inventory 和后续实现边界 |
| [第三轮全量事实与 dtype 修复](validation/a5-20261117-run3/findings.md) | Compressor、公开 writer、mHC 通过；三个消费链 FAIL 为探针视图错误，给出最小重跑命令 |

## 目标行为

对满足容量和能力约束的 prefill batch，选择已捕获的 token bucket，将当次输入和 metadata 刷新到固定存储，再 replay。bucket 内真实 token 数、prefix 长度和请求槽变化时，不能继续使用 capture 时的旧长度、旧映射或旧缓存内容。

超出已验证边界的 batch 走 eager。涉及 DP/TP/EP collective 时，相关 rank 必须就 replay/eager 的选择和通信布局达成一致。

最终覆盖 V4.1 的 Q/KV、c1/c2 压缩及写回、source/consumer 共享、indexer/candidate/top-k、sparse attention、mHC 和 MoE。Engram 的 host hashing/查表边界按实际 backend 分析，可以用显式 eager break 过渡。完整交付需要 attention/compressor/indexer 的实际图内证据。

复用现有 transformer body capture 契约，LM head、logits processor 等尾部计算沿用已有 eager tail。

## 首批 V4.1 交付边界

以下是实施边界，不是对全部部署组合的支持承诺。

| 项目 | 首批边界 | 后续扩展或回归 |
| --- | --- | --- |
| 硬件 | `dsv41-a5` 对应的 Ascend A5 环境 | V4 在 A3/A5 的既有能力不能替代 V4.1 A5 验证 |
| 执行阶段 | target 普通 `EXTEND`、chunked prefill 和有 prefix 的 prefill | decode/verify 回归；draft prefill 入图单独列验收项 |
| 图形式 | 显式启用已迁入的 breakable prefill graph | full graph 和 torch.compile 不作为前置 |
| 请求轴 | 沿用现有单请求 capacity 起步 | 多请求及零长度槽经单算子验证后扩展 |
| 并行 | CP=1、DCP=1、PP=1、无 TBO；满足每 rank 64 query heads 和 MoE 条件 | DP idle 和其他拓扑逐项验证 |
| 模型语义 | 完整 prefill 计算及正确缓存写入 | bounded replay 等改变计算行数的优化分开评估 |
| 投机 | DSpark target prefill 输出及后续生成正确 | 普通 prefill、verify、draft extend 分别维护自己的图协议 |

## 迁移后的关键判断

1. **基础框架已有。** 直接沿用 factory → NPU breakable backend → NPUGraph segment，以及现有 runner 的 staging、metadata hooks 和 eager tail。
2. **V4.1 metadata 仍需扩展。** 已迁入的容器只为 ratio 4/128 分配压缩状态，尚未覆盖 c1/c2、window/index/candidate 和 cycle-bank。
3. **模型状态依赖是主要工作。** source/consumer 共享、ratio-2 跨 chunk 状态、Engram history、mHC 系数与 DSpark aux hidden 要逐项明确所有权和刷新顺序。
4. **MoE 基于 scope 重构后的实现。** 固定 AllGather 和 AllToAllV 拒绝已迁入；后续验证 V4.1 实际几何、padding/量化和多 rank 行为。
5. **验收需要真实设备证据。** CPU 测试、服务启动或 replay 日志均不能代替 A5 上多轮数值、缓存和图内覆盖验证。

## 关联材料

- [Breakable NPU Graph 文档目录](../BreakableNPUGraphBackend设计方案/README.md)
- [旧 V4 全 attention 入图差距分析](../BreakableNPUGraphBackend设计方案/analysis/dsv4_attention_in_graph_gap_analysis_92b02637.md)
- [V4 四笔提交代码说明](../BreakableNPUGraphBackend设计方案/analysis/prefill_graph_4commits_code_walkthrough.md)
- [V4.1 学习笔记](../../learn/sglang-deepseek-v4.1.md)：基线为 `4cafabd1a3`，架构背景可参考，后端能力以本文当前基线为准。
