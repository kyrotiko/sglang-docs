# A5 验证脚本

这两个脚本不改仓库代码，只在 A5 上启动本地 server / 调用 `kv_exchange` 算子，把结果写到 `probe_out/`。请把整个输出目录打包返回。

## 0. 前置条件

- A5（arch35）机器，`sgl-kernel-npu` 已按 `--arch-version 3510` 构建 `kv_exchange`。
- 当前分支/commit 是 `dsv41-a5` @ `9898f7ce9a`（或包含 HiCache L2 的同一基线）。
- 已安装 `cann_ops_transformer`，V4.1 权重可加载。
- 建议先确认 `python -c "import torch_npu; print(torch_npu.npu.is_available())"` 为 True。

## 1. 快速算子验证（约 1 分钟）

验证“full-page 传输”这个设计假设：indexer 设备 buffer 用 `[pages, 64, 1, 64]`，按 full page 重排成 `[pages, 128, 1, 64]`，host 用 `[pages, 1, 128, 1, 64]`，`page_size=128` 时 AIV 能否逐字节搬对。

```bash
cd sglang-docs/FE/DeepSeekV4.1HiCacheL2独立设计方案/validation/scripts

# 先跑参考实现，确认数据布局和期望值本身没问题
python a5_kv_exchange_probe.py --impl torch_native --out probe_out/op_ref

# 再跑 AIV 实现
python a5_kv_exchange_probe.py --impl ascendc_aiv --out probe_out/op_aiv
```

默认 `--layers 2`，这样 host 的相邻页之间会隔着另一个 layer，正好覆盖 `page_first` 的真实 stride；`--layers 1` 会退化成连续布局，不能证明页间 gap 合法。

期望：两个 `page_size`（64 和 128）的 `d2h_ok`、`h2d_ok` 都为 `True`。

如果 `ascendc_aiv` 的 128 失败而 64 通过，说明我们设计的 full-page 传输不能直接用，需要改为在 host 侧保留 native page 行；请把 `probe_out/op_aiv/*.json` 返回。

## 2. HiCache L2 端到端（约 5–15 分钟）

会启动两个 server：

- baseline：`--disable-radix-cache`，记录 prompt A 的冷启动 greedy 输出；
- candidate：`--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through`，先发 A，再用大量唯一前缀挤 device，最后重发 A，检查 `host_hit` 指标。

```bash
python a5_hicache_e2e_probe.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --tp-size 8 \
  --out-dir probe_out/e2e
```

可选项：

- `--mem-fraction-static 0.8`：按模型实际配方传。
- `--extra-args "--attention-backend dsv4"`：如果启动命令还需要额外参数，整体加引号传入。
- `--a-pages 8 --filler-pages 2 --fill-factor 1.3`：默认让唯一前缀总量约为 device 容量的 1.3 倍，仍小于 `ratio=2` 的 host 容量。
- `--max-fillers 300`：避免机器太慢时跑太久。

脚本会记录：

- 启动日志中每个 host 池的 `layers / pages / item_bytes / layout`，并与 V4.1 期望值对比（`geometry_checks`）；
- prompt A 第一次和 host 命中后的 `text`、`meta_info.cached_tokens`；
- `sglang:prefill_effective_tokens_total{mode="host_hit"}` 的增量；
- `sglang:hicache_host_used_tokens` 的变化；
- 如果 candidate 在第一次 D2H/H2D 就崩溃，会保存 candidate 日志末尾和 traceback。

当前代码的预期结果（这是我们要确认的）：

- `geometry_checks` 里 `deepseek_v4_c1` / `deepseek_v4_c2` 的 `item_bytes` **不匹配**（实际是 `128/64`，期望 `128*320 / 64*320`）→ 复现 G1a；
- 第一次 A 请求可能直接报 view/shape 错误 → 复现 G1b（NPU indexer 2-D vs 5-D）；
- 如果 candidate 没崩但 `host_hit_delta == 0`，说明 L2 路径没有真正命中，也要返回报告。

## 3. 默认 ratio 的启动路径（G10，约 1–2 分钟）

```bash
python a5_hicache_e2e_probe.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --tp-size 8 \
  --out-dir probe_out/g10 \
  --g10
```

这会额外启动一个只带 `--enable-hierarchical-cache`、不带 `--hicache-ratio` 的 server。当前代码预期启动失败，日志里应出现 auto-size 相关错误（V4 pool 不在 `_SIZEABLE_POOLS` / 没有 `layer_router`）。如果它能正常启动，说明 G10 已被其他改动修复，也请返回日志。

## 4. 返回什么

最省事：把整个 `probe_out/` 打包返回。

```bash
tar -czf probe_out.tgz probe_out
# 或 zip -r probe_out.zip probe_out
```

如果只能贴文本，至少给：

1. `probe_out/op_aiv/*.json`；
2. `probe_out/e2e/report.json`；
3. `probe_out/e2e/candidate.log` 的最后 100 行（如果 candidate 崩溃）；
4. `probe_out/e2e/baseline.log` 的前 50 行（确认 baseline 正常启动）。

脚本不会修改仓库，也不会写模型权重；只会启动本地 server、发 HTTP 请求和写 `probe_out/`。
