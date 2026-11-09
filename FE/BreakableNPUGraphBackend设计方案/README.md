# Breakable NPU Prefill Graph 文档

本目录汇总 Ascend NPU 上 breakable prefill CUDA graph 的方案、实现走读、验证脚本和实测报告。

## 目录结构

```
design/     方案与设计
analysis/   代码走读、差距分析、收益报告
scripts/
  profiling/  日志 / msprof 导出提取与分析
  verify/     真机正确性、可行性验证
  spikes/     早期 NPUGraph 行为探针
```

## 文档索引

### design/

| 文件 | 内容 |
| --- | --- |
| [breakable_npu_graph_backend_design.md](design/breakable_npu_graph_backend_design.md) | BreakableNPUGraphBackend 总体设计：段边界、capture/replay、NPU 约束 |
| [ascend_ops_prefill_graph_moe_design.md](design/ascend_ops_prefill_graph_moe_design.md) | ascend_ops `prefill=allgather` 的 MoE dispatch/combine 入图方案 |
| [ascend_ops_valid_rows_cast_design.md](design/ascend_ops_valid_rows_cast_design.md) | ascend_ops allgather 有效行 Cast 实验方案 |
| [pr_48030_body.md](design/pr_48030_body.md) | 上游 PR #48030 的正文草稿 |

### analysis/

| 文件 | 内容 |
| --- | --- |
| [dsv4_attention_in_graph_gap_analysis_92b02637.md](analysis/dsv4_attention_in_graph_gap_analysis_92b02637.md) | 基线 92b02637：DSV4 compressor/indexer/attention 全链路入图的差距分析 |
| [dsv4_prefill_graph_code_walkthrough.md](analysis/dsv4_prefill_graph_code_walkthrough.md) | 当前分支（92b02637 之后）的代码逐段说明 |
| [dsv4_prefill_graph_gain_summary.md](analysis/dsv4_prefill_graph_gain_summary.md) | 收益汇报版（给领导/评审，一页） |
| [dsv4_prefill_graph_gain_report.md](analysis/dsv4_prefill_graph_gain_report.md) | 收益技术详版：chunk 512/1024/2048 A/B、逐 step 分析、局限 |
| [prefill_graph_padding_amplification.html](analysis/prefill_graph_padding_amplification.html) | padding 放大的可视化说明 |

### scripts/

| 目录 | 脚本 | 用途 |
| --- | --- | --- |
| profiling | [extract_profile.py](scripts/profiling/extract_profile.py) | 从 msprof 导出提取 graph/eager 报告；`--label` 选择默认目录名（原 extract_graph_profile.py / extract_eager_profile.py 合并） |
| profiling | [dump_forward_breakdown.py](scripts/profiling/dump_forward_breakdown.py) | 按 forward 拆分 kernel 明细 |
| profiling | [prof_op_rows.py](scripts/profiling/prof_op_rows.py) | 从 kernel_details.csv 统计 per-op 行数，算 padding 倍数 |
| profiling | [check_dp_padding_from_log.py](scripts/profiling/check_dp_padding_from_log.py) | 从 server log 验证 DP padding / 稀疏 step |
| profiling | [compare_prefill_step_time.py](scripts/profiling/compare_prefill_step_time.py) | graph vs eager 逐 step 中位数 |
| profiling | [compare_logprobs.py](scripts/profiling/compare_logprobs.py) | 两次 server 运行的 logprob 对比 |
| verify | [verify_ascend_ops_prefill_graph.py](scripts/verify/verify_ascend_ops_prefill_graph.py) | 真实 rank 数复刻 ascend_ops allgather 链并验证入图 |
| verify | [verify_valid_rows_cast.py](scripts/verify/verify_valid_rows_cast.py) | 有效行 Cast 的 NPU 正确性验证 |
| spikes | [v2_spike_two_segments.py](scripts/spikes/v2_spike_two_segments.py) | 两段图 + 中间 eager 断点的可行性 |
| spikes | [v4_weak_ref_spike.py](scripts/spikes/v4_weak_ref_spike.py) | 段内张量弱引用 / 池生命周期 |
| spikes | [v8_raw_capture_spike.py](scripts/spikes/v8_raw_capture_spike.py) | 裸 capture_begin/capture_end 与 torch.npu.graph 的等价性 |
| spikes | [cu_rebind_spike.py](scripts/spikes/cu_rebind_spike.py) | compressor 是否每次 replay 重读 cu_seqlens |
| spikes | [compressor_graph_replay_spike.py](scripts/spikes/compressor_graph_replay_spike.py) | compressor replay vs eager 数值校验 |
| spikes | [dsv4_c4_attention_graph_spike.py](scripts/spikes/dsv4_c4_attention_graph_spike.py) | C4 attention 全链路 replay 校验 |

## 阅读顺序

1. `design/breakable_npu_graph_backend_design.md` — 先看框架和段边界。
2. `analysis/dsv4_attention_in_graph_gap_analysis_92b02637.md` — DSV4 attention 入图还差什么。
3. `analysis/dsv4_prefill_graph_code_walkthrough.md` — 当前代码怎么改的。
4. `analysis/dsv4_prefill_graph_gain_summary.md` → `analysis/dsv4_prefill_graph_gain_report.md` — 收益结论与证据。

## 约定

- 设计 / 方案放 `design/`，分析 / 走读 / 报告放 `analysis/`。
- 一次性真机实验脚本放 `scripts/spikes/`；数据处理 / 日志分析放 `scripts/profiling/`；正确性验证放 `scripts/verify/`。
- 文档内引用脚本使用相对路径；跨目录引用用 `../`。
- 确认过期的历史文档直接删除；git 历史里仍可找回。
