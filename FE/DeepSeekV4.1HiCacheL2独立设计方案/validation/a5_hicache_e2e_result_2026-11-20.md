# A5 HiCache L2 E2E 结果（2026-11-20）

## 运行配置

- 代码：诊断时为 `dsv41-a5` 基线的 `dsv41-a5-hicache-l2` @ `13d596d980`（临时 CRC trace），trace 已在 `ab30299513` 清理。该分支后续已 rebase 到 `origin/main`，当前交付 HEAD 为 `46277ec334`；文中旧 commit 只属于历史基线。
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

## DSpark + HiCache（2026-11-25）

A5 上按 DSpark + HiCache 配方完成实机启动与推理功能验证，用户确认功能通过。本轮未归档启动日志与页 CRC；DSpark 路径的字节级证据仍以 2026-11-20 的 target-only CRC 为准。

### 问题

DSpark draft 是 V4 架构，SWA 行宽 608 B/token；V4.1 target 是 544 B/token。旧逻辑把 DSpark compressed-KV draft 判为 `PACKED`，draft 的 SWA buffer 被直接拼进 target 的 SWA host 池，启动时触发：

```text
ValueError: packed DSV4 draft SWA rows do not match the target's 69632 B pages: (413, 128, 1, 608)
```

仅跳过 `sidecar_draft_device_pools` 不够：在当前 main 上 DSpark 走的是 `mtp_draft_device_pools`（`PACKED`），sidecar 列表为空。

### 修复

`46277ec334` 在 `_build_hicache_draft_plan()` 的 pack 判定里增加行宽比较 `_packed_swa_rows_match_target()`：

- 544 B draft → 仍 `PACKED`；
- 608 B draft + 544 B target → 回退 `SIDECAR`，`mtp_draft_device_pools` 保持空；
- `build_hicache_draft_sidecars()` → `build_swa_draft_pools()` 建独立的 `DRAFT_SWA` host pool，`item_bytes = 608 * page_size`（page_size 128 时为 77824）；
- target `swa` host pool 保持 `544 * page_size`（69632）。

启动日志应出现：

```text
HiCache draft SWA rows differ from the target's; using a DRAFT_SWA sidecar instead of the target SWA host pool.
```

## Rebase 后 A5 冒烟（2026-11-26）

分支 `dsv41-a5-hicache-l2` @ `46277ec334`（基于 `origin/main` @ `30a3796ca8`），A5 容器 `yy_test`。使用现有 DSpark 启动脚本追加：

```bash
--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through
```

### 启动自检

8 个 DP rank 都打印：

```text
HiCache draft SWA rows differ from the target's; using a DRAFT_SWA sidecar instead of the target SWA host pool.
```

DP0 的 host 池几何（其余 rank 一致）：

| 池 | layers | pages | item_bytes | layout |
| --- | --- | --- | --- | --- |
| `swa` | 40 | 516 | 69632 | page_first |
| `draft_swa` | 3 | 516 | 77824 | page_first |
| `deepseek_v4_c1` | 1 | 8392 | 40960 | page_first |
| `deepseek_v4_c1_indexer` | 1 | 8392 | 8192 | page_first |
| `deepseek_v4_c1_indexer_scale` | 1 | 8392 | 512 | page_first |
| `deepseek_v4_c2` | 3 | 8392 | 20480 | page_first |
| `deepseek_v4_c2_indexer` | 3 | 8392 | 4096 | page_first |
| `deepseek_v4_c2_indexer_scale` | 3 | 8392 | 256 | page_first |

`swa` = 544 × 128，`draft_swa` = 608 × 128，确认 V4 draft 与 V4.1 target 已拆成两个 host 池。服务随后打印 `The server is fired up and ready to roll!`。

### 功能冒烟

- 8-token prompt 连续两次 `/generate`：都返回 16 token，HTTP 200；
- 200-token prompt 连续两次（`routed_dp_rank=0`）：第二次 `cached_tokens=128`，`cached_tokens_details={"device":128,"host":0}`，页对齐与 device 命中正常；
- 日志无 `ERROR` / `Traceback`。

本轮 `max_total_num_tokens=537088`，没有强制 device 淘汰，因此未复现 host 命中；L2 host 命中的字节级证据沿用 2026-11-20 的 target-only 结果。rebase 未改动 L2 传输路径，CPU 测试 66 passed。

## 仍未闭环

- `hicache_host_used_tokens` 在 host 命中时仍为 0，指标没有覆盖 sidecar/回载；属于可观测性缺口，不影响本次 L2 结论。
- `a5_hicache_page_roundtrip.py` 的 `c1_indexer` 仍是假阴性（测试脚本构造 host 视图的问题）；生产 E2E 的 `c1_indexer` 页 CRC 是匹配的。
- DSpark + HiCache 本轮只做了功能验证，没有归档启动日志、host 池几何与页 CRC；需要字节级证据时按 `validation/scripts/README.md` 重跑。
- CRC / compressor trace 是临时诊断代码（`13d596d980`、`35961346b0`），已在 `ab30299513` 移除；上面的数字保留作历史证据。
