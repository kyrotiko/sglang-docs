# V4.1 A5 Prefill Graph 单文件验证

运行文件：[verify_dsv41_prefill_graph.py](verify_dsv41_prefill_graph.py)。只需复制这个 `.py`，不下载或加载模型权重。

**当前 revision 7；脚本内的预期基线 `d3700bf0a9` 是采集时的标签，实际主仓 SHA 以各轮 findings 为准。** [第五轮](../validation/a5-20261117-run5/findings.md)已取得 ratio 0/1 完整 idle 序列和 ratio 2 首轮 idle 的单卡正面证据；ratio 2 后续 live odd-prefix case 有 17 个 compressed cache 字节差异，仍保留 FAIL。第六轮已确认不插入 idle 也会在 compressor 原始输出出现差异；revision 7 增加隔离和同步诊断，不修改服务代码或决定 idle 回退策略。原 17 组及第三/四轮独立 CPU FP32 reference FAIL 保留，不能改写为独立精度通过。

首两轮 [run1](../validation/a5-20261116-run1/findings.md) / [run2](../validation/a5-20261116-run2/findings.md) 保留为对应旧环境/代码的历史快照。第三轮的硬件和 SDK 内容均有变化，即使包版本号相同也不自动等同。

第五至七轮的 idle 与 compressor 边界结论见上述 findings；单算子脚本仍保留 revision 7，不需要重新跑同类探针。

## 当前结论：独立 compressor 的 eager 重复也失败

[第七轮](../validation/a5-20261117-run7/findings.md)已完成。相同 GPU x/state、同 stream 且显式同步，奇数 prefix 的长 extend 在 eager 重复中已发生有效 row 0 差异；偶数 prefix 长 extend 和奇数 prefix 单 token 对照通过。合成链 replay 前加同步仍失败。这里的 eager 调用发生在同进程 capture 之后，尚未排除 capture 初始化或全局 workspace 的影响。

现有数据已可转交算子方核对 CYCLE 的 ABI/状态容量和首 pair 读写顺序，见[第七轮](../validation/a5-20261117-run7/findings.md)。不用再原样重复这些探针，也不据此禁用 idle、放宽阈值或改服务算法。脚本仍为 revision 7，留作算子方复现及对应修复后的回归。

## 第七轮已完成命令（保留复现）

[第六轮](../validation/a5-20261117-run6/findings.md)已经完成有无 idle 对照，两组都在 odd-prefix 失败，最早差异在 compressor 原始输出。两轮 idle 状态保护通过。机器换成了 950PR / driver 25.1.rc3，不能与第五轮 DT 的字节数量直接作同环境比较。

以下三组已完成，无需原样重跑；算子方复现或对应实现修复后回归可使用：

```bash
python "/mnt/share/y00627667/repos/sglang-docs/FE/DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py" \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 --tokens 64 --state-ring-size 2 \
  --probes environment,compressor_boundary,active_chain_r2,active_chain_r2_sync \
  --timeout 1200 \
  --output-dir "./dsv41-compressor-boundary-r7-$(date +%Y%m%d-%H%M%S)"
```

- `compressor_boundary` 仅运行实际 compressor，去掉 norm/writer/indexer/attention。用真实设备计算建立 prefix 0/1/2/3 的 state，每个 case 都从相同 x/state 各跑两次 eager、两次 replay；全部操作放在同一 capture stream，执行前后显式同步，保存 GPU 初始 state/input hash、逐行误差及最终 state。另检查相同 history 和首 token 下，odd-prefix 的首个 pair 是否会随 live=1→31 改变。
- `active_chain_r2` 保留 revision 6 无 idle 合成链，作为同轮环境对照。
- `active_chain_r2_sync` 仅在该合成链的 replay 前增加设备同步，其余计算和输入保持一致。同步只是诊断开关，不修改服务配置，也不据此断言原路径有跨 stream bug。

如果 eager 重复本身不一致，应先区分该算子/环境的非确定性；如果 eager 重复一致而 replay 不一致，则继续定位 graph 差异。对比 `exact_by_row`、`max_abs_by_row` 和 `first_pair_invariance`，不能只看整体最大误差。单卡结果仍不替代主仓 hooks、真实多流或 DP/DeepEP。

## 第六轮 ratio-2 定向命令（历史复现）

第五轮不必全量重跑。新脚本先用真实 compressor 的一个有效 token 建立 bank 2 的合法 prefix-1 pending pair，再执行同一组 live 输入的两种序列：插入 idle，以及不插入 idle。增加有效 compressor 原始/归一化输出、每个张量的比较结果与实际不一致坐标；比较失败会保留 FAIL 并继续收集后续 case，算子异常仍保留原失败阶段。

```bash
python "/mnt/share/y00627667/repos/sglang-docs/FE/DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py" \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 --tokens 64 \
  --probes environment,idle_chain_r2,active_chain_r2 \
  --timeout 1200 \
  --output-dir "./dsv41-idle-r2-r6-$(date +%Y%m%d-%H%M%S)"
```

`idle_state_checks` 单独记录每轮 idle 的 eager/replay 状态保护；`case_failures` 保存全部数值/字节差异。失败后不把 live 输出、cache 或 top-k 改成宽松阈值。主仓源码保持现状，脚本不应用服务侧修改。

第五轮实际 checkout 为 `24deba1e5f…`（clean），与脚本的预期标签不同。新一轮仍记录实际 SHA；不要将 expected baseline 当作实际源码证明，也不要把单卡合成链结论记为自己分支的整模验收。回传新目录中的 `dsv41-prefill-evidence.tar.gz` 即可。

## 第五轮 idle 本地链命令（历史复现）

主仓保持已推送的 `d3700bf0a9`。只替换这一个验证脚本，不应用 idle veto / 整组 eager 草稿。无需加载权重或先重跑原 17 组：

```bash
python "/mnt/share/y00627667/repos/sglang-docs/FE/DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py" \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 --tokens 64 \
  --probes environment,idle_chain_r0,idle_chain_r1,idle_chain_r2 \
  --timeout 1200 \
  --output-dir "./dsv41-idle-chain-$(date +%Y%m%d-%H%M%S)"
```

每组使用同一固定 bucket，依次执行 live full → idle → live odd-prefix/short → idle → live full；每轮先从相同初始状态跑 eager，再 replay，并将本轮状态带到下一轮。三个组分别覆盖 window-only、ratio-1 投影/writer/index-K/QLI→QSLI/attention、ratio-2 compressor/writer/index-K/QLI/attention 的合成图内链。

验证的 idle 候选协议来自当前 dummy EXTEND 物理几何：`cu=[0,T_cap]`、`KV length=T_cap`，但 `seqused=0`、bank 0、compressed/index 写入 loc 全 `-1`、SWA loc 0、window/cmp 有效长度 0，QLI/QSLI 仍执行。它不是在宣称该协议已经合法；metadata planning 或 compute 不接受时，保留失败阶段和日志。该命令不验证另一种 `cu=[0,0]` 的零物理 query ABI。

idle 必须保持所有非零 request-bank 和所有真实 SWA 页不变，compressed/index cache 全部不变，按字节检查。SWA 保留 page 0 允许 dummy 写入。idle 输出本身被忽略，只记录 finite/未写入情况，不要求无效输出为零。前后 live 的有效输出、top-k、缓存和状态仍与 eager 比较。

**该探针是单卡、本地合成链、identity RoPE。** 不绕过主仓 metadata 的异常保护，不验证实际 source-group、mHC、多流、DeepEP 或 DP8 一致通信。三个组 PASS 后再用真实模型验证多 rank 混合 live/idle；任何单卡结果都不直接决定服务是否支持 idle replay。继续回传 `dsv41-prefill-evidence.tar.gz`，FAIL/BLOCKED 也保留。

## 已完成的第四轮（保留复现命令）

下面三组已完成，当前无需重复。若后续 SDK、形状或实现变化，需要复现时使用新目录：

```bash
python "/mnt/share/y00627667/repos/sglang-docs/FE/DeepSeekV4.1PrefillGraph设计方案/scripts/verify_dsv41_prefill_graph.py" \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 \
  --probes writer_attention_r0,writer_attention_r1,writer_attention_r2 \
  --output-dir ./dsv41-writer-read-r4
```

使用新目录。无需重复其余 14 组；原来的 CPU attention 参考失败继续保留，writer dtype 修复不代表独立精度问题已解决。若再次换机器或 SDK，先跑 `--probes environment` 记录实际版本、source hash、公开 API 与实现解析。

运行结束回传输出目录中的 `dsv41-prefill-evidence.tar.gz`，FAIL/BLOCKED 也保留。若真实服务额外加载注册模块，显式使用 `--import-module MODULE`，不猜测模块或切换代际接口。

## 全量 23 组命令（复现或新环境使用）

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

不通过 torchrun，不需要 HCCL 初始化。全量是 23 组单卡 synthetic 探针，不是整模或多卡验收。默认每组最长 600 秒（可用 `--timeout` 调整）；各组独立子进程，失败、超时、崩溃均尽量保留 checkpoint 和已完成 case，并继续其他组。

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
| idle_chain_r0/r1/r2 | 同 bucket live→idle→live；公开 writer、c1 投影/c2 compressor、index-K/QLI/QSLI 与 attention 合成图内链，检查 idle 对真实 cache/state 的字节级保护；未验证多卡 |
| compressor_boundary | 独立 compressor 的合法 prefix 历史、短/长 extend、eager 重复与 replay 重复、GPU 输入 hash 和首 pair 不变性 |
| active_chain_r2_sync | 与无 idle 合成链相同，仅增加 replay 前显式设备同步的诊断对照 |
| active_chain_r2 | 与 `idle_chain_r2` 相同的合法 pending pair、输入和初始缓存，仅不插入 idle，辅助区分差异是否依赖 idle |
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

需要 CPU torch，覆盖计数边界、KV pack/unpack、页表与 record、padding、BF16 诊断及 writer/consumer 视图 alias；不输出 NPU 已通过结论。dtype 回归已在本地验证修复前失败、修复后通过，第四轮也已取得真实 writer→attention 的设备 graph/eager 对照证据，独立精度状态另行保留。
