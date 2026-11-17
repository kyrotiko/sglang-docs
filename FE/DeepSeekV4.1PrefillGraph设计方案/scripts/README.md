# V4.1 A5 Prefill Graph 单文件验证

运行文件：[verify_dsv41_prefill_graph.py](verify_dsv41_prefill_graph.py)。只需复制这一个 `.py`，无需本目录其他脚本、模型下载或权重加载。

**当前脚本 revision 3，预期主仓 `dsv41-a5@9898f7ce9a`。** 已对齐合入 randgun 后的公开 writer/mHC 入口；完整变更和复验矩阵见 [合并后验证调整](../analysis/randgun_merge_20261117.md)。原始两轮证据保留在 [run1](../validation/a5-20261116-run1/findings.md) / [run2](../validation/a5-20261116-run2/findings.md)，没有改写历史结果。

## 合并后的推荐执行顺序

在实际 A5 serving 环境中运行，使用同一套 torch_npu、sgl-kernel-npu 和 SGLang。`--output-dir` 必须是新目录，不使用 torchrun，不需要 HCCL 初始化。

先检查新依赖是否齐全：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 --probes environment \
  --output-dir ./dsv41-after-merge-env
```

重点看 `missing_apis` 和实现解析：新基线需要 `mem_cache.kv_compress_epilog_v2`、`hyper_conn.mhc_pre_dpskv4` 及其 `cann_recipes_infer_cann920` 实现。旧 `2026.11.0.post4` 的回传 exports 没有公开 writer，不能假定拉取 SGLang 已满足依赖；精确 SDK 版本需按实际构建确认。environment 的 PASS 表示 inspection 完成，不表示所有算子可执行。

依赖可用后，优先跑五组直接相关项：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 \
  --probes environment,compressor,epilog_fp8,epilog_fp4,mhc_premix \
  --output-dir ./dsv41-after-merge-ops
```

writer 两组通过后，检查真实写入缓存能否被 attention 消费：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --probes writer_attention_r0,writer_attention_r1,writer_attention_r2 \
  --output-dir ./dsv41-after-merge-writer-read
```

本组 writer 在 replay 前 eager 准备缓存，attention 在图内执行；writer 自身 capture 由 epilog 两组验证。它不是完整 source/attention 同段捕获。

若 SDK/CANN/torch_npu 已升级，再追加 `--probes qli_c2,candidate_chain` 的轻量回归。`--list` 和 `--help` 不导入 torch。

## 全量 17 组探针命令

在实际 A5 serving 环境中更新两个仓后执行：

```bash
python "/mnt/share/y00627667/repos/sglang-docs/FE/DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py" \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 --probes all \
  --tokens 64 --hidden-size 4096 \
  --page-size 128 --index-page-size 64 \
  --candidate-blocks 2048 --state-ring-size 2 \
  --timeout 1200 \
  --output-dir "./dsv41-prefill-full-$(date +%Y%m%d-%H%M%S)"
```

全量指本文件的 17 组单卡 synthetic 探针，不是整模/多卡验收。默认浮点阈值保持不变；独立精度失败和 SDK 缺口继续如实记录，各组独立运行。运行结束按终端打印的 `Send back:` 路径回传完整证据包。

## 结果回传

每组独立子进程，默认最长 600 秒（含首次 JIT）；需要时指定 `--timeout 1200`。失败、超时或崩溃会保留最后 checkpoint 和已完成 case，继续其他组。

每个输出目录内都会尽量生成：

```text
report.json
report.md
<probe>.json / <probe>.log
verify_dsv41_prefill_graph.py
dsv41-prefill-evidence.tar.gz
```

把各目录的 `dsv41-prefill-evidence.tar.gz` 发回来即可，FAIL/BLOCKED 也保留。包内有运行脚本快照与 hash、实际 source SHA、包版本和完整日志。

## 实际覆盖范围

| 探针 | 内容 |
| --- | --- |
| environment | 包版本/NPU/CANN/source SHA、公开 API 签名/hash、auto/指定实现解析、npu/custom/cann_ops_transformer schema、主仓索引页和 runtime V4/V4.1 行宽 |
| geometry | 实际 SGLang c1 行序、c2 compact、显式 live mask、window/cmp lengths、固定 buffer 和请求槽 0；CPU 数学对照及图回放 |
| metadata | 真实 window/c1/c2/QLI/candidate/QSLI metadata 的 shape、地址和内容变化；planning 在图外 |
| compressor | 当前仍是 `compressor_v2`；有效压缩行、完整 private state、bank 切换、无新 pair、split/concat 和 ring wrap |
| epilog_fp8 / epilog_fp4 | **公开 `sgl_kernel_npu.operators.mem_cache.kv_compress_epilog_v2`**；544B/group32、320B/group16、`-1`、NaN skip、guard 和字节级 eager/replay |
| prologue | storage mode 0/1、图内 NZ 转换、skip/guard、分离 K/scale 与 combined record 一致 |
| qli_c2 | ratio-2 QLI，多输入和动态 metadata、因果范围与有序 top-k 对照 |
| candidate_chain | ratio-1 QLI→QSLI 同图数据流，candidate 内容/长度和有序 top-k；index cache 为预置合成值 |
| attention_r0/r1/r2 | 手工有效缓存，MQSMLA window/c1/c2，多轮图回放和部分行/head 的独立 CPU FP32 oracle |
| writer_attention_r0/r1/r2 | 公开 writer 生成缓存，按 544/320B ABI 解包作为 CPU reference，captured MQSMLA 读取；记录量化误差但不虚构量化精度阈值 |
| engram_break | 小型真实 EngramHasher fallback、共享 break、history/固定 hash/图内 consumer，verify 延迟提交 |
| mhc_premix | 按主仓签名调用 `hyper_conn.mhc_pre_dpskv4`，固定指定 implementation；previous-pre 折叠、返回 triplet、同 X 改 previous-pre 的系数不变性、多轮 eager/replay |

图探针按相同输入和相同初始状态先 eager、再 replay。包含 A→B→C→D→A；压缩输出只比较有效行，state/cache 比较完整存储。padding attention 输出预填 NaN，记录是否被写入、零值数和最大绝对值，不假定任意算子必须写零。

默认均为合成尺寸：T=64、主 heads=64、head_dim=512、indexer heads=32、index_dim=128、topk=512、full page=128、index PA page=64、candidate block=8、candidate capacity=2048、hidden=4096、ratio-2 ring=2。index PA page 与 candidate block 是不同维度；`--index-page-size` 必须整除 `--page-size/2`。测试 index page=128 时用 `--page-size 256`。

## 判读规则

- PASS 只对该 probe 的已记录范围成立；接口存在、auto 可解析不等于实际执行通过。
- BLOCKED 是缺依赖/NPU/公开 API。若新 writer 被 BLOCKED，检查新公开入口，旧 custom namespace 是否存在不再是阻塞标准。
- FAIL 结合最后 phase、`capture_supported` 和 traceback 判断；warmup/eager 失败不能证明 graph 不支持。
- TIMEOUT/CRASH 保留最后 checkpoint 和返回码；默认不重置设备或更改 SDK。
- 返回值 0=选中组全部通过，1=失败/超时/崩溃，2=仅依赖阻塞，130=用户中断。

`checks.capture`、`checks.eager_replay` 和 `checks.cpu_reference` 分开记录。浮点 parity 默认 `rtol=1e-3, atol=1e-3`，CPU oracle 的 reference 阈值也独立保持 `1e-3/1e-3`；整数、hash、top-k 和 cache bytes 精确比较。BF16 ULP/L2/softmax 假设均为诊断，不替换 oracle，也不为了变绿自动放宽阈值。

若真实服务有额外注册模块，用 `--import-module MODULE` 显式加入并记录；不猜测模块，不默认切换代际接口或 `torch_native`。revision 3 修复了该参数覆盖 probe 名的记录问题；前两轮该列表为空，历史证据不受影响。

## 仍未覆盖

不加载真实 checkpoint/tokenizer/lookup 表，不能替代整模、真实 allocator/source-group 生存期、compressor→RoPE→cache→index→attention 全链、mHC 外部 RMSNorm/post/完整状态和多流、DSpark packed aux hidden/draft KV、多卡 collective/admission，以及准确率/性能验收。

当前 mHC premix 的独立折叠参考只验证上一子层 pre 到输入的关系，不是完整 triplet 数学 oracle。writer→attention 检查消费格式，但没有独立量化精度阈值。脚本不修改主仓代码或启动模型服务。

## 本地无 NPU 的检查

```bash
python verify_dsv41_prefill_graph.py --self-test
```

需要 CPU torch，仅验证计数边界、KV 打包/解包、页表与 record 分离、padding、BF16 诊断；不产生 NPU 已通过结论。
