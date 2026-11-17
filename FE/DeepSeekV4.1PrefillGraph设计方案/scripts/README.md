# V4.1 A5 Prefill Graph 单文件验证

运行文件：[verify_dsv41_prefill_graph.py](verify_dsv41_prefill_graph.py)。只需复制这个 `.py`，不下载或加载模型权重。

**当前 revision 4，预期主仓 `dsv41-a5@9898f7ce9a`。** [第三轮事实](../validation/a5-20261117-run3/findings.md)已确认 compressor、公开 writer、mHC premix 和其他已捕获探针的对照 exact。writer→attention 三项在 capture 前暴露了脚本 dtype 视图问题，revision 4 已修复：FP8 writer 取 float8 view，attention 保留 uint8 view，二者共享同一存储，不做数值转换。

首两轮 [run1](../validation/a5-20261116-run1/findings.md) / [run2](../validation/a5-20261116-run2/findings.md) 与 [randgun 合并记录](../analysis/randgun_merge_20261117.md)保留为对应旧环境/代码的历史快照。第三轮的硬件和 SDK 内容均有变化，即使包版本号相同也不自动等同。

## 当前只需重跑三组

在第三轮相同的 A5 serving 环境替换脚本后运行：

```bash
python "/mnt/share/y00627667/repos/sglang-docs/FE/DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py" \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 \
  --probes writer_attention_r0,writer_attention_r1,writer_attention_r2 \
  --output-dir ./dsv41-writer-read-r4
```

使用新目录。无需重复其余 14 组；原来的 CPU attention 参考失败继续保留，writer dtype 修复不代表独立精度问题已解决。若再次换机器或 SDK，先跑 `--probes environment` 记录实际版本、source hash、公开 API 与实现解析。

运行结束回传输出目录中的 `dsv41-prefill-evidence.tar.gz`，FAIL/BLOCKED 也保留。若真实服务额外加载注册模块，显式使用 `--import-module MODULE`，不猜测模块或切换代际接口。

## 全量 17 组命令（复现或新环境使用）

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

不通过 torchrun，不需要 HCCL 初始化。全量是 17 组单卡 synthetic 探针，不是整模或多卡验收。默认每组最长 600 秒（可用 `--timeout` 调整）；各组独立子进程，失败、超时、崩溃均尽量保留 checkpoint 和已完成 case，并继续其他组。

## 结果内容

每个输出目录包含 `report.json`、`report.md`、各 probe 的 JSON/log、脚本快照与 hash，以及 `dsv41-prefill-evidence.tar.gz`。终端的 `Send back:` 给出回传包路径。

报告记录实际 source SHA/dirty 状态、包版本/API 签名与源码 hash、硬件信息、参数、误差和失败阶段。environment PASS 表示 inspection 完成，不表示所有公开函数已能执行；需同时看 `missing_apis` 和 implementation 解析。

## 探针范围

| 探针 | 内容 |
| --- | --- |
| environment | 包/NPU/CANN/source SHA、公开 API 与实现、npu/custom/cann_ops_transformer schema、索引页及 V4/V4.1 行宽 |
| geometry | SGLang c1 行序、c2 compact、显式 live mask、window/cmp lengths、固定 buffer 和请求槽 0；CPU 几何及图对照 |
| metadata | window/c1/c2/QLI/candidate/QSLI metadata 的 shape、固定地址及内容变化；planning 在图外 |
| compressor | `compressor_v2`、有效输出、完整 private state、bank 切换、pending pair、split/concat 和 ring wrap |
| epilog_fp8 / epilog_fp4 | 公开 `mem_cache.kv_compress_epilog_v2`、544B/group32 与 320B/group16、skip/NaN/guard 和字节级对照 |
| prologue | storage mode 0/1、NZ 转换、skip/guard、K/scale 与 combined records |
| qli_c2 | ratio-2 QLI，动态输入/metadata、因果边界、有序 top-k |
| candidate_chain | ratio-1 QLI→QSLI 图内数据流，candidate 内容/长度及 top-k |
| attention_r0/r1/r2 | 手工合法缓存的 MQSMLA，多轮 graph/eager 及部分行/head 的 CPU FP32 reference |
| writer_attention_r0/r1/r2 | 公开 writer 在 replay 前准备缓存，CPU 按 544/320B ABI 解码，captured MQSMLA 消费；writer 与 consumer 的 dtype view 共享存储 |
| engram_break | 小型真实 hasher fallback、固定输出、共享 break、history 和 verify 延迟提交 |
| mhc_premix | 指定 `cann_recipes_infer_cann920` 的公开 premix、previous-pre 折叠、triplet、同 X 改 previous-pre 的系数不变性 |

writer→attention 中 writer 在图外准备缓存；writer 自身 capture 由 epilog 两组验证。不宣称整个 source 链已同段入图，也不将 cache 解码误差当作独立量化精度验收。

## 几何、数值和状态判读

默认 T=64，主 heads=64，head_dim=512，indexer heads=32，index_dim=128，topk=512，full page=128，index PA page=64，candidate block=8，candidate capacity=2048，hidden=4096，ratio-2 ring=2。

index PA page 与 candidate block 不同；`--index-page-size` 必须整除 `--page-size/2`。测试 index page=128 时用 `--page-size 256`。测试更宽 private ring 可指定 `--state-ring-size 8`，但不等于验证了 DSpark。

每轮用相同输入和初始状态做 eager，再 replay。压缩只比有效行，state/cache 比完整存储。padding attention 预填 NaN并记录写入/零值情况，不假定所有算子必须写零。

- PASS 仅适用于记录的 probe、shape 和 case。
- BLOCKED 表示缺依赖/NPU/API；在当前已验证环境，旧 custom schema 缺失不影响新的公开 wrapper。
- FAIL 结合最后 phase 和 checks 判断；capture 前失败不能证明 graph 不支持。
- TIMEOUT/CRASH 保留最后 checkpoint/返回码，不自动重置设备或更改 SDK。
- 返回值：0=全通过，1=失败/超时/崩溃，2=仅阻塞，130=用户中断。

`checks.capture`、`checks.eager_replay`、`checks.cpu_reference` 分列。parity 和 reference 默认均为 `rtol=1e-3, atol=1e-3`，没有放宽；整数、hash、top-k 和 cache bytes 精确比较。BF16 ULP/L2/softmax 假设仅作诊断，不代替 FP32 oracle。

## 覆盖边界和本地检查

仍未验证真实 checkpoint/lookup 表、服务 allocator/source-group 生存期、完整 compressor→RoPE→cache→index→attention、mHC 外部 norm/post/多流、DSpark aux hidden/draft KV、多卡通信/admission，以及模型准确率和性能。

```bash
python verify_dsv41_prefill_graph.py --self-test
```

需要 CPU torch，覆盖计数边界、KV pack/unpack、页表与 record、padding、BF16 诊断及 writer/consumer 视图 alias；不输出 NPU 已通过结论。dtype 回归已在本地验证 revision 3 失败、revision 4 通过，但实际 writer→attention 仍需上述三组设备重跑。
