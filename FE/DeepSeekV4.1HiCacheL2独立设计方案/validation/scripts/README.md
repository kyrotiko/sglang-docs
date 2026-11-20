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

2026-11-18 已在 A5 上验证：`page_size=64` 与 `128`、`layers=2` 的 AIV D2H/H2D 都通过，见 [../a5_operator_probe_2026-11-18.md](../a5_operator_probe_2026-11-18.md)。

## 2. HiCache L2 端到端（约 5–15 分钟）

最省事的是直接用封装脚本（已包含下面所有环境变量和 target-only 参数）：

```bash
SGLANG_ROOT=/mnt/share/y00627667/repos/sglang \
HICACHE_RATIO=1.2 FILL_FACTOR=1.05 \
./run_a5_hicache_target_e2e.sh
```

它会先检查 `sglang` 确实从 `$SGLANG_ROOT/python` 导入，然后启动 baseline + candidate 并跑完整探测。下面是手动版说明。

会启动两个 server：

- baseline：`--disable-radix-cache`，记录 prompt A 的冷启动 greedy 输出；
- candidate：`--enable-hierarchical-cache --hicache-ratio 2.0 --hicache-write-policy write_through`，先发 A，再用大量唯一前缀挤 device，最后重发 A，检查 `host_hit` 指标。

**注意**：probe 启动的是新的 `python -m sglang.launch_server` 子进程，不会自动继承你 `serve_dspark_hicache.sh` 里的 `PYTHONPATH`/`export`。如果当前 shell 没有设置，子进程会用到 `/sgl-workspace/sglang` 之类的另一份代码，模型加载就会在 HiCache 之前失败。至少要用 `--sglang-path` 指到本分支：

```bash
python a5_hicache_e2e_probe.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --tp-size 8 \
  --mem-fraction-static 0.83 \
  --sglang-path "$HOME/repos/sglang/python" \
  --json-model-override-args '{"architectures": ["DeepseekV4ForCausalLM"]}' \
  --extra-args "--attention-backend dsv4 --enable-dp-attention --dp-size 8 --enable-dp-lm-head --reasoning-parser deepseek-v4 --language-model-only --moe-a2a-backend deepep --deepep-mode auto --kv-cache-dtype auto --enable-metrics --watchdog-timeout 9000 --max-running-requests 200 --max-prefill-tokens 16384 --chunked-prefill-size 4096 --cuda-graph-bs-decode 1 2 4 8" \
  --env SGLANG_DSV4_FP4_EXPERTS=True \
  --out-dir probe_out/e2e
```

上面是 **target-only**，不带 DSpark。V4.1 在 A5 上要求每个 rank 保留全部 64 个 query head，因此这条 A5 配方需要 `--enable-dp-attention --dp-size 8 --enable-dp-lm-head`。脚本默认 `ENABLE_DP=1`；不要用纯 TP8 跑这个模型，否则会命中 `needs all 64 query heads ... got 8`。当前首批不支持 DSpark + HiCache：draft SWA 是 V4 的 608 B/token，target 是 V4.1 的 544 B/token，必须拆成独立 draft SWA host pool（G6）。不要把这些 `--speculative-*` 参数加进 `--extra-args`。

其他 `export`（HCCL/DEEPEP/SGLANG_OPT_* 等）可以让 probe 继承：把启动脚本里 `python3 -m` 之前的 export 段复制成 `env.sh`，先 `source env.sh` 再跑 probe；或者用 `--env KEY=VALUE` 逐个补。`--env` 可重复。

关于强制 L2 命中的三个关键点：

- `--max-total-tokens 32768`：把 device KV 池压到 32k token，否则默认池 2M token，挤 device 要发几百万 token，不现实。
- `--routed-dp-rank 0`：DP8 下 round-robin 会把第一次 A 和重发 A 分到不同 rank，各自的 radix tree 不同，必然 miss；probe 默认把所有请求钉在 rank 0。
- `--filler-pages 4 --filler-count 95`：每个 filler 内容带唯一 tag，不能共享 radix 页；95 个约 370 token 的 filler 足以挤出 32k device 池，同时总量仍小于 `1.2 × 32k` 的 host 容量。

可选项：

- `--mem-fraction-static 0.83`：按模型实际配方传。
- `--extra-args`：模型/并行相关的额外参数，整体加引号传入；`--json-model-override-args` 单独用参数传，避免嵌套引号。
- `--a-pages 8`：prompt A 的长度（约 8 个 full page）。
- `--max-total-tokens` / `--filler-pages` / `--filler-count` 可按机器和内存调整。
- `--repeat-a 2`：baseline 连发两次冷请求、candidate 连发两次 prefill 请求，用来区分“模型本身不确定”和“host 回载错误”。如果 `baseline_repeatable=false`，`output_matches_baseline` 不能单独作为正确性判据。

脚本会记录：

- 启动日志中每个 host 池的 `layers / pages / item_bytes / layout`，并与 V4.1 期望值对比（`geometry_checks`）；
- prompt A 第一次和 host 命中后的 `text`、`meta_info.cached_tokens`；
- `sglang:prefill_effective_tokens_total{mode="host_hit"}` 的增量；
- `sglang:hicache_host_used_tokens` 的变化；
- 如果 candidate 在第一次 D2H/H2D 就崩溃，会保存 candidate 日志末尾和 traceback。

在**未应用设计改动**的基线 `dsv41-a5` 上，预期结果（用来复现问题）：

- `geometry_checks` 里 `deepseek_v4_c1` / `deepseek_v4_c2` 的 `item_bytes` **不匹配**（实际是 `128/64`，期望 `128*320 / 64*320`）→ 复现 G1a；
- 第一次 A 请求可能直接报 view/shape 错误 → 复现 G1b（NPU indexer 2-D vs 5-D）；
- 如果 candidate 没崩但 `host_hit_delta == 0`，说明 L2 路径没有真正命中，也要返回报告。

在**应用本设计的 commit 1–5** 后，`geometry_checks` 应全部 `ok`，第一次 A 请求应成功；若仍失败，报告里的 candidate log tail 就是定位依据。

## 2.0 临时页 checksum 日志（最贴近真实 E2E）

如果不想跑额外脚本，可以让 server 在传输边界打页 checksum：

```bash
TRACE_PAGES=16 \
SGLANG_ROOT=/mnt/share/y00627667/repos/sglang \
ENABLE_DP=1 HICACHE_RATIO=1.2 \
bash ./run_a5_hicache_target_e2e.sh
```

它会设置 `SGLANG_HICACHE_TRACE_PAGES=16`，在 `kernel_ascend` 的 D2H/H2D 边界同步后打印每个池 layer 0 的一页：

```text
HICACHE_TRACE op=D2H pool=deepseek_v4_c1 layer=0 device_page=... host_page=... device_crc=... host_crc=...
HICACHE_TRACE op=H2D pool=deepseek_v4_c1 layer=0 device_page=... host_page=... device_crc=... host_crc=...
```

判读：

- 同一个 `pool` + `host_page` 的 D2H `host_crc` 与 H2D `device_crc`/`host_crc` 应一致；
- 不一致说明真实回载路径有字节错误，比文本对比直接；
- `TRACE_PAGES=0` 时完全没有日志和额外同步，正常性能路径不受影响。

这是临时诊断代码，环境变量不设置就不生效。

## 2.1 KV 页字节级 round-trip（模型输出不可重复时用这个）

A5 上 greedy 输出本身不可重复（baseline 连发两次都能不同），所以不能用文本相等判断 host 回载是否正确。`a5_hicache_page_roundtrip.py` 绕过模型：

- 用 `_dsv4_low_ratio_regions` 构造和真实 V4.1 相同的 SWA/C1/C2/indexer/scale 全页 buffer；
- 构造真实的 `DeepSeekV4PagedHostPool`；
- 填充确定性字节；
- 走真实 AIV `kernel_ascend` D2H 备份和 H2D 回载；
- 逐字节比较 host 页和回载后的 device 页。

```bash
SGLANG_ROOT=/mnt/share/y00627667/repos/sglang \
bash ./run_a5_page_roundtrip.sh
```

或者手动 source Ascend 环境（`acl` 模块在其中）：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
SGLANG_HICACHE_HOST_MEM_BACKEND=acl \
PYTHONPATH=/mnt/share/y00627667/repos/sglang/python \
python a5_hicache_page_roundtrip.py
```

期望 7 个 case 全部：

```text
swa / c1_kv / c1_indexer / c1_scale / c2_kv / c2_indexer / c2_scale
d2h=True h2d=True mismatch=0
```

这个测试和模型采样完全无关；它通过，再加上 E2E 里的 `cached_tokens_details.host > 0`，就能证明：

- host 池几何和 page 映射对；
- D2H/H2D 字节没有丢/错；
- C1/C2 的 ratio 1/2 都覆盖。

## 3. 默认 ratio 的启动路径（G10，约 1–2 分钟）

```bash
python a5_hicache_e2e_probe.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --tp-size 8 \
  --out-dir probe_out/g10 \
  --g10
```

这会额外启动一个只带 `--enable-hierarchical-cache`、不带 `--hicache-ratio` 的 server：

- 在基线 `dsv41-a5` 上，预期启动失败，日志里出现 auto-size 相关错误（V4 pool 不在 `_SIZEABLE_POOLS` / 没有 `layer_router`）；
- 应用提交 5 后，预期能正常启动，并打印 auto-size 选出的 ratio；
- 带 C4 持久状态或 V4 draft 的模型仍应得到明确的“set --hicache-ratio”错误。

不管哪种结果，请返回 `probe_out/g10` 的日志。

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
