# 可转交算子方：ratio-2 CYCLE 的 odd-prefix 首 pair 不稳定

## 问题范围

在 A5 Ascend950PR_9589 上，通过公开 `sgl_kernel_npu.operators.attention.compressor_v2` 调用 ratio-2 CYCLE：physical T=64、hidden=4096、D=512、FP32 state `[3,2,1024]`、private bank 2。

相同 GPU x/state 下，prefix=1 或 3、live=31 时，普通 eager 重复和 graph replay 均可观察到有效输出 row 0 差异；其余有效行和最终 state byte-exact。prefix=0/live=64、prefix=2/live=31、prefix=1/live=1 是通过的对照。

这是当前调用组合下的 correctness 复现，尚未把根因确定为某段已部署 kernel 源码。请先确认下面的 ABI/构建条件，再定位首 pair 历史读取与 state 写回。

## 已取得的证据

原始输入、日志、逐行误差和 SHA 见[第七轮完整证据](../validation/a5-20261117-run7/findings.md)。验证脚本 revision 7 已在 `sglang-docs/main@34798ac` 发布。

- 四次执行前实际 GPU x/state hash 一致，每次都恢复同一个设备产生的合法 prefix history。
- 所有操作在同一 capture stream 内执行，每次执行前后显式设备同步。
- `prefix=1/live=31`：eager1 对 eager2 最大绝对差 `0.13671875`，只在 row 0；replay 对 eager1 同样失败。
- `prefix=3/live=31`：eager1 对 eager2 最大绝对差 `0.132568359375`，同样只在 row 0。
- 相同 state 和 x，仅把 live 从 1 改成 31，首 pair 输出不变性在 eager 和 replay 四种模式全部失败。
- 独立 probe 不调用 RMSNorm、writer、indexer、attention、MoE 或 idle。
- 现有无 idle 合成链加 replay 前同步仍失败，不宜把外部同步作为修复。

执行顺序限制：这里的普通 eager 重复发生在同一进程先执行过一次 graph capture 之后。本轮尚无从未 capture 的新进程 pure-eager 数据；若需区分 kernel 自身、capture 初始化或全局 workspace 影响，请据此增加对照，不能宣称该因素已被排除。

## 复现命令

在包含已测公开 API 的 A5 环境，只需要现有单文件脚本，不需要模型权重、服务、HCCL 或多卡：

```bash
python verify_dsv41_prefill_graph.py \
  --device 0 --tokens 64 --hidden-size 4096 \
  --state-ring-size 2 \
  --probes compressor_boundary \
  --timeout 1200 \
  --output-dir "./compressor-odd-prefix-repro-$(date +%Y%m%d-%H%M%S)"
```

`compressor_boundary` 的已测执行顺序：

1. 用相同权重、固定种子输入，从空 state 实际执行 prefix 0/1/2/3 的历史准备，保存完整 state。
2. 用 prefix=0、live=64、bank 0 做 warmup 和 capture。
3. 对每个 case，分别恢复保存的完整 state、复制相同 x、原地更新 cu/seqused/start_pos/bank，再同步、校验实际 GPU 输入 hash。
4. 依次执行 eager1、eager2、replay1、replay2，每次都重新恢复输入/state。
5. 比较前 `floor((prefix+live)/2)-floor(prefix/2)` 个有效输出及完整非零 bank 状态；保存逐行最大误差和 exact 位。
6. 比较 odd_one 与 odd_long 的首 pair：x/state 完全相同，仅 cu/seqused 的有效 query 数不同。

脚本 SHA-256：`3822f139c464f81229fd37b76176020f308165866f7343a127c9811b3dd6d12c`。

## 环境定位

- Python 3.12.15，torch 2.13.0+cpu，torch_npu 2.13.0。
- Ascend950PR_9589，device 0，driver/npu-smi 25.1.rc3。
- sgl-kernel-npu 2026.11.0.post4，CANN 路径 `/usr/local/Ascend/cann-9.2.0`。
- 部署 public wrapper Python hash：`b8da9f4d14a1465bcd5bc17b4fb74de4ba62b4c2be919fb23178a875b26c4710`；auto 在采集环境解析为 `_cann_impl`。
- 主仓实际 SHA `24deba1e5f631ba0459dd23292881a3f9239bbbb`，clean；独立 probe 只调用公开 SDK，不依赖该主仓的模型执行路径。

## 请确认的契约与源码线索

1. 已部署 A5 实现是否接受 ratio=2、CYCLE C=2 和 live>C 的普通 prefill？公开参考文档允许 C=1..1024，描述只保存最后 min(live,C) 个 token，未给出 C>=live 的限制。请提供这个构建的实际契约，不能仅靠猜测增大 ring。
2. 在 odd prefix 下，输出 row 0 是否确实包含历史 pending token 与当前首 token？公开计数公式认为本例有 16 个有效输出，row 0 属于有效范围。
3. 请核对部署 kernel 的历史读取与 state 写回次序。只读 arch35 参考代码存在以下线索：normal 路径 `SaveState` 后调用 `ReadState`；full-load 路径也先保存新 state。ring=2、odd-prefix/live31 的保存窗口会复用历史 token 的 slot，需要保证原历史先被正确消费。
4. 请提供实际 kernel 构建 revision，并检查 fresh-process pure eager、capture 后 eager、replay 三种状态下是否共享临时 workspace。

参考路径：

- `sgl-kernel-npu/docs/api/en/operators/attention/compressor_v2.md:29/38/50`。
- `refs/ops-transformer/experimental/attention/compressor_v2/op_kernel/arch35/compressor_v2_block_vec_normal.h:88`。
- 同目录 `compressor_v2_block_vec_full_load.h:294`；`compressor_v2_block_vec.h:515/599/647`。

这些是本地只读参考。其 public wrapper hash 为 `086db298…`，不同于部署 wrapper；不能宣称参考路径已被证明就是部署二进制的根因。需按匹配的源码/二进制确认，再给出修复或合法调用方式。
