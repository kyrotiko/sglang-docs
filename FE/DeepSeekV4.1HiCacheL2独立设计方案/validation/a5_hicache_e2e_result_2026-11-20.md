# A5 HiCache L2 E2E 结果（2026-11-20）

## 运行配置

- 代码：`dsv41-a5-hicache-l2` @ 诊断时 `13d596d980`（临时 CRC trace）；trace 已在 `ab30299513` 清理
- 模型：DeepSeek-V4.1-Flash-1017，A5 单机 8 卡
- 启动：DP8，`--max-total-tokens 32768`，`--hicache-ratio 1.2`，`write_through`
- 探测：`TRACE_PAGES=16`，`FILLER_COUNT=95`，`REPEAT_A=2`
- 结果目录：`probe_out/e2e_target_20261120_061731`

## 功能结果

| 检查 | 结果 |
| --- | --- |
| baseline / candidate 启动 | 通过 |
| 第一次 A 请求 | 通过 |
| SWA/C1/C2/indexer/scale host 池 `item_bytes` | 7/7 正确 |
| host 命中 | `cached_tokens=768`，`cached_tokens_details={"device":0,"host":768}` |
| DP rank | 第一次 A 和重发 A 都是 rank 0 |
| `host_hit_delta` | 768 |

## 页级字节校验

`SGLANG_HICACHE_TRACE_PAGES=16` 在真实 `kernel_ascend` D2H/H2D 边界打印页 CRC。命中时 7 个池的 D2H host CRC 与 H2D host/device CRC 全部一致：

| pool | host page | D2H host_crc | H2D host_crc | H2D device_crc |
| --- | --- | --- | --- | --- |
| `deepseek_v4_c1` | 0 | `89f2f221` | `89f2f221` | `89f2f221` |
| `deepseek_v4_c1_indexer` | 0 | `3f3586f6` | `3f3586f6` | `3f3586f6` |
| `deepseek_v4_c1_indexer_scale` | 0 | `9120e6e6` | `9120e6e6` | `9120e6e6` |
| `deepseek_v4_c2` | 0 | `d90fea74` | `d90fea74` | `d90fea74` |
| `deepseek_v4_c2_indexer` | 0 | `f69d53f7` | `f69d53f7` | `f69d53f7` |
| `deepseek_v4_c2_indexer_scale` | 0 | `9fbde4f7` | `9fbde4f7` | `9fbde4f7` |
| `swa` | 2 | `033836c7` | `033836c7` | `033836c7` |

结论：host 池页几何、C1/C2 的 ratio 1/2、indexer K/scale 的 D2H/H2D 字节都正确。不是“命中了但内容错”的情况。

## 文本输出为何不能作判据

同一 server 连发两次完全相同的冷请求，baseline 两次输出不同：

```text
baseline_repeatable = false
candidate_prefill_repeatable = false
```

这是 A5 上 MoE/DeepEP 的数值不确定性，不是 HiCache 数据错误。`hit_matches_candidate` / `output_matches_baseline` 因此不能作为准入门槛；页级 CRC 才是可靠判据。

## G8 压缩状态清理 trace（2026-11-20）

同一轮 E2E 打开 `TRACE_COMPRESSOR=1`：

```text
HICACHE_COMPRESSOR op=clear req=25 ratio=2 ring_crc=250218f5   (×321)
```

- 321 次 ratio-2 request-state clear 的 ring CRC 全部相同；
- host 命中前缀 768 是 128 的整数倍，也是偶数，ratio-2 pending pair 为空；
- CPU 不变量测试 `test_ratio2_compressor_state_is_cleared_at_page_boundary` 覆盖：request slot clear 后 ring 为 0/-inf，`seq_len=768` 时 `transfer_indices` 为空，`seq_len=769` 才返回 pending pair。

结论：**host 命中请求在进入 prefill 前不会继承上一个请求的 ratio-2 pending state**。由于 A5 模型输出本身不可重复，压缩状态与冷启动的数值等价仍不能靠文本证明；这里验证的是清理/边界不变量。

## 仍未闭环

- `hicache_host_used_tokens` 在 host 命中时仍为 0，指标没有覆盖 sidecar/回载；属于可观测性缺口，不影响本次 L2 结论。
- CRC / compressor trace 是临时诊断代码（`13d596d980`、`35961346b0`），已在 `ab30299513` 移除；上面的数字保留作历史证据。
