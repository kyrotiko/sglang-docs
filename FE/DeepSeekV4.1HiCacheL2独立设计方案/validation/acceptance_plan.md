# 验收与验证计划

所有测试基于 `dsv41-a5-hicache-l2` 工作区；未实施 [代码改动清单](../design/code_change_plan.md) 前，G1 相关用例预期失败，可作为复现基线。

可直接上 A5 跑的探测脚本见 [scripts/README.md](scripts/README.md)：先跑 `a5_kv_exchange_probe.py` 验证 full-page 传输，再跑 `a5_hicache_e2e_probe.py` 复现 G1/G10。

## 0. 环境与启动

### 0.1 前置条件

- Ascend A5（arch35）单机，CANN 版本与 `sgl-kernel-npu` 匹配；`kv_exchange` 需按 `--arch-version 3510` 构建（见 `sgl-kernel-npu/docs/api/en/operators/mem_cache/kv_transfer.md`）。
- 安装 `cann_ops_transformer`（V4.1 算子扩展，`hardware_backend/npu/utils.py:243-253`）。
- DeepSeek-V4.1-Flash 权重，page size 128。
- 测试流量用 greedy（temperature=0），便于逐 token 对比。

### 0.2 启动命令

沿用模型现有 A5 配方（`--attention-backend` 由模型 override 解析为 `dsv4`，`arg_groups/model_overrides/deepseek_v4.py:29`）。**不要**照搬 `test_npu_deepseek_v4_flash_fp8_4p_gpqa_a5.py` 中的 `--disable-radix-cache`，它与 HiCache 互斥。

```bash
# 基线：关闭 HiCache
python -m sglang.launch_server \
  --model-path <DSV4.1-Flash> --device npu --page-size 128 \
  --tp-size <T> --mem-fraction-static <M> --trust-remote-code \
  --port 30000

# 候选：开启 L2
python -m sglang.launch_server \
  --model-path <DSV4.1-Flash> --device npu --page-size 128 \
  --tp-size <T> --mem-fraction-static <M> --trust-remote-code \
  --enable-hierarchical-cache --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --port 30001
```

`SGLANG_HICACHE_HOST_MEM_BACKEND` 不设置时 DSV4 自动取 `acl`。首批不要使用 `memfabric`（G2）。

### 0.3 启动自检

1. 日志中每个 host 池都有 `Allocating ... GB host memory for V4 paged pool '<name>'`（`memory_pool_host.py:354-363`），确认存在：
   `swa`、`deepseek_v4_c1`、`deepseek_v4_c1_indexer`、`deepseek_v4_c1_indexer_scale`、`deepseek_v4_c2`、`deepseek_v4_c2_indexer`、`deepseek_v4_c2_indexer_scale`。
2. 记录各池的 `pages`、`item_bytes`、layout、GB，作为容量核对的基线。
3. `curl http://127.0.0.1:30001/metrics | grep hicache_host_total_tokens` 有值。

## 1. P0：CPU 契约测试（无 A5）

| 用例 | 断言 | 对应缺口 |
| --- | --- | --- |
| `test_low_ratio_regions_build_host_pools` | C1/C2 KV 与 4 个 indexer K/scale host 池（共 6 个）均被装配；无 C128；`item_bytes` 等于 full page 字节（`S*row_bytes`） | G1/G5 |
| `test_low_ratio_indexer_page_geometry` | NPU 4-D device 规范化为 `[full_pages, S, 1, dim]`；`_host_page_view` 形状 `[pages, 1, S, 1, dim]`；`_to_native_page_indices` 返回 `S = P//ratio` 个槽 | G1 |
| `test_low_ratio_indexer_kernel_ascend_call` | mock 传输算子，`page_size=S`，索引覆盖 `S` 连续槽，`device_k.shape == (1, full_pages, S, 1, dim)` | G1 |
| `test_low_ratio_regions_reserved` | `reserve_dsv4_paged_host_memory` 的预留字节包含 C1/C2 与 indexer 的 `I` 倍页数 | G2 |
| `test_kv_pool_geometry_unchanged` | GPU 2-D 页行与 `item_bytes` 与修改前完全一致（回归） | G1 |

运行：

```bash
cd D:/workspace/SGLang/sglang-dsv41-a5-hicache-l2
python -m pytest test/registered/unit/mem_cache/test_dsv4_hicache_l2.py -q
```

## 2. P1：A5 功能用例

> 关键：重复请求命中 L1 不能证明 L2。每个 L2 用例都要先“挤出 device”，再用指标证明 `host_hit`。

### T1 冷启动 / 重复长前缀

- prompt：固定长文本 × 4（≥ 384 token，即 3 页）。
- 第一次：`cached_tokens == 0` 或仅 device 命中；第二次：`cached_tokens > 0`。
- 作用：基础可用性，不区分层级。

### T2 强制 L2 命中（核心）

1. 请求 A（长前缀，≥ 8 页），记录 `cached_tokens`。
2. 连续发送**足够挤出 device、但总量小于 host 容量**的不同长前缀请求：
   - 唯一前缀 token 数 > `device_pages * page_size`（确保 A 被 device 淘汰）；
   - 唯一前缀 token 数 < `hicache_ratio * device_pages * page_size`（确保 host 还能容纳 A；ratio=2 时留出余量）。
   例如取 1.2–1.5× device 容量。**不要**按 `host_ratio × device_pages` 发满：`write_through` 下 host 容量就是 `ratio × device`，发满会把最早的 A 淘汰掉，造成假阴性。
3. 重发 A 前先确认：A 的 host 页仍存在（`hicache_host_used_tokens` 未触顶；必要时按页/前缀查询 host 占用）。
4. 再发请求 A：
   - 响应 `meta_info.cached_tokens > 0`；
   - `/metrics` 中 `sglang:prefill_effective_tokens_total{mode="host_hit"}` 的 rate 在请求窗口内 > 0（`metrics_collector.py:1091-1103`）；
   - 每池 backup/load 计数与字节数有变化（首批新增指标）。
5. 若第 4 步没有 host_hit 增长，检查 write policy、host 容量与 A 是否已被 host LRU 淘汰；这是 L2 路径是否真正工作的判定点。
6. 强制命中另一组：用 `write_back` 重跑，只有 device 淘汰的页才备份，host 不会被新前缀写满；这一组与 `write_through` 分开判定。

### T3 页边界

- 前缀长度：127 / 128 / 129 / 384 / 385 token。
- 期望：只有达到整页的部分可被 host 命中；`cached_tokens` 按 128 取整。
- 129 token 的第二次请求应命中 128 token；385 命中 384。

### T4 负例

- 短 prompt（< 1 页）不命中（与现有 NPU 用例一致）。
- 不同前缀不命中，不能出现“相似但不同”的 host 命中。
- 关闭 HiCache 的基线不应报告 host_hit。

### T5 输出正确性（必须）

- 选取 20 条长前缀 prompt；分别用基线（HiCache off）和候选（强制 L2 命中）以 greedy 生成 32 token。
- 要求逐 token 完全一致（baseline 与 candidate 的 `text` 相同）。
- 额外比较**首 token logits/top-k**：G8 的压缩状态重建偏差可能要到页边界之后才显现，32-token 文本相等不足以证明。至少覆盖 127/128/129/384/385 token 的前缀。
- 若出现差异，优先排查 G1（indexer）与 G8（压缩状态重建），并保存 logits/top-k 证据。

### T6 并发、中止与失败

- 32 / 64 并发请求混合命中和非命中；
- 20% 请求在 prefill 中 abort，包含 H2D 回载途中的 abort；
- 注入一次 host 回载失败（AIV 错误/事件未完成），确认请求回退为重新计算，不使用部分回载的 KV/indexer；
- 期望：无崩溃、无死锁、无 host/device 槽泄漏；进程 RSS 稳定；D2H 在途页不被回收。

## 3. P2：性能用例

数据集：`generated-shared-prefix`，固定输入/输出长度与共享前缀比例（例如 8k 输入、1k 输出、90% 共享前缀），并发 32。

| 指标 | 采集方式 | 目标 |
| --- | --- | --- |
| TTFT P50/P90 | benchmark client / 服务端 metrics | 相比 HiCache off，前缀复用场景争取 ≥ 20% 改善（参考 NPU MLA HiCache 用例的既有标准，`test_npu_hierarchical_cache_mla.py`）；这是**目标值，不是准出硬门槛**，需先测本模型基线再定 |
| host 命中 token | `prefill_effective_tokens_total{mode="host_hit"}` | > 0 且随共享前缀比例上升 |
| 吞吐（output tok/s） | benchmark client | 回归 ≤ 2% |
| device 命中 | `{mode="device_hit"}` | 不下降（L2 不应挤占 L1 命中） |

对比组：

1. HiCache off；
2. `--hicache-ratio 2` + `write_through`；
3. `--hicache-ratio 4` + `write_through`；
4. `--hicache-ratio 2` + `write_back`（若首批开放）。

记录每组启动日志中的 host 池 GB 与节点 RSS，确认与 [容量公式](../design/l2_host_offload.md#容量与定容)一致。

## 4. P3：压力与容量

1. **Host 淘汰**：发送 1000 条互不相同的长前缀，直到 `hicache_host_used_tokens` 触顶并回落；再命中早期前缀时允许 miss，但进程稳定、无泄漏。
2. **长上下文**：单请求 32k / 64k / 128k（按模型上限），首 token 与基线一致，重复请求的 host 命中长度正确。
3. **多 rank**：TP=4/8、DP attention=4（若在首批范围）各跑一遍 T1/T2/T5；每个 rank 的 host 池独立，指标汇总后 host 总量不超过节点预算。
4. **24h 稳定性（可选）**：固定 QPS 混合流量，观察 RSS/host 池、无 OOM、无错误日志。

## 5. P4：DSpark（首批外，后续准入）

前置：提交 4 的行宽断言已合入。

1. DSpark on + HiCache on vs DSpark on + HiCache off：greedy 输出一致；
2. 投机接受率不下降（A5 既有标准 > 0.5，参考 `test_npu_deepseek_v4_flash_fp8_4p_gpqa_a5.py:149`）；
3. 精度：GPQA / GSM8K 使用 `sgl-eval`，与基线同 harness 对比；
4. 日志确认 draft SWA 层与 target 的 `page_bytes` 相等。

## 6. 证据留存

每次验证保存：

- 主仓 commit、分支、worktree 路径；
- 启动命令、环境变量、CANN / sgl-kernel-npu / cann_ops_transformer 版本；
- 启动日志（host 池分配段）与 `/metrics` 快照；
- 请求/响应 JSON（含 `meta_info.cached_tokens`）；
- benchmark 原始结果（CSV）；
- 失败用例的最小复现脚本。

## 7. 准出标准

| 类别 | 必须满足 |
| --- | --- |
| 功能 | T1、T2、T3、T4 全部通过；T2 的 host_hit 指标有增长 |
| 正确性 | T5 逐 token 一致；T6 无泄漏 |
| 测试 | P0 全绿；A5 E2E 已注册 NPU CI |
| 性能 | 首批准出以功能和正确性为主；TTFT ≥ 20% 是目标值，待基线测量后由产品确认是否作为硬门槛；吞吐回归 ≤ 2% |
| 稳定性 | 压力用例无崩溃/OOM；节点 host 内存不超预算 |
| 文档 | 启动参数、容量公式、已知限制（G2/G3/G6）写入用户文档 |

任一必须项失败则不建议合入/发布；可将首批范围收缩为“单卡、TP、无投机、write_through、ratio 2”后重新验收。
