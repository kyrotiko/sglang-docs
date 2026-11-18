# A5 第七轮：独立 compressor 的 odd-prefix 首 pair 不稳定

结果时间：2026-11-17 21:25:25 UTC；来源 `C:/Users/admin/Desktop/file7`。见[原始报告](raw/report.md)、[JSON](raw/report.json)、[manifest](manifest.json)。回传 Python 只做 hash/文本校验，未执行或导入；归档脚本文本来自 hash 相同的可信本地 revision 7。

## 1. 核心事实

**在当前 ratio-2 CYCLE 调用组合中，独立 compressor 的 eager 自身已经不稳定。** 同一 case 的 GPU x/state hash 在四次执行前完全相同；同一 stream、执行前后显式设备同步。问题出现在奇数 prefix 的长 extend，且仅有效输出 row 0 不一致，其余有效行 exact。没有 norm、writer、indexer、attention 或 idle 参与这个独立探针。

| case | prefix / live | eager1 对 eager2 | eager1 对 replay1 / replay2 | 最终非零 bank 状态 |
| --- | --- | --- | --- | --- |
| even_full | 0 / 64 | exact | exact / exact | exact |
| odd_one | 1 / 1 | exact | exact / exact | exact |
| odd_long | 1 / 31 | FAIL，最大绝对差 0.13671875 | FAIL，0.13671875 / 0.118896484375 | exact |
| even_long | 2 / 31 | exact | exact / exact | exact |
| shifted_odd_long | 3 / 31 | FAIL，0.132568359375 | FAIL，0.132568359375 / 0.132080078125 | exact |
| odd_long_repeat | 1 / 31 | FAIL，0.1005859375 | FAIL，0.118896484375 / 0.13671875 | exact |

全部失败比较的 `different_rows` 均只有 `[0]`，这是本轮新增逐行检查直接记录的事实，不再仅依据前 32 个坐标采样推断。四种执行模式的首 pair 长度不变性检查全部失败：相同 history 和 x，仅把 live 从 1 改成 31，首 pair 输出就发生明显变化。

`odd_one` 和 `odd_long` 的输入 hash 也一致：

- state：`786e4cc3b4f5958b244705ec8d3fca93ac981b5493975c2616dd9b251ad9ce3c`
- x：`4a22fbb88fa28eb8f6783523a5a848b7f707cf7f746330a714e9c7135834fa04`

最终 state exact 不能用来推断首个历史 token 的读取一定正确。也不能把这里的 eager 不稳定仅标成 graph 引入的精度差异。

执行顺序边界：这组 eager1/eager2 是在同一进程完成过 warmup/capture 之后执行的普通 eager 调用；本轮没有“新进程从未 capture”的 pure-eager 组。因此可以确认 eager 阶段也观察到差异，尚不能排除 capture 初始化或全局 workspace 对后续 eager 的影响。

## 2. 合成链同步对照

两组无 idle 合成链都只有 `live_odd_prefix` 失败，full live 和 live-again exact。显式同步没有消除差异：

| odd-prefix 指标 | 原合成链 | replay 前同步 |
| --- | ---: | ---: |
| compressor 原始输出最大绝对差 | 0.13671875 | 0.13671875 |
| compressed cache 字节差异 | 214 | 109 |
| index K 字节差异 | 63 | 61 |
| attention 最大绝对差 | 1.11328125 | 1.1171875 |

两组使用相同合法 pending-pair hash `45d2527c1186a00fbdd2742fd92f2d4d1312cfb7e1ca5b127e8307e89c16c4d3`。当前结果不支持“在调用方 replay 前加一次同步就能修复”的结论。

第五、六轮的 idle 状态保护正面证据继续保留。本轮不重判 idle 支持，不修改服务回退策略；主仓 hooks、真实多流和 DP/DeepEP 仍须各自验收。

## 3. 环境与来源

- 硬件 Ascend950PR_9589，device 0；driver/npu-smi 25.1.rc3，与第六轮一致。
- Python 3.12.15；torch 2.13.0+cpu；torch_npu 2.13.0；sgl-kernel-npu 2026.11.0.post4；CANN 路径 `/usr/local/Ascend/cann-9.2.0`。
- 实际主仓 `24deba1e5f631ba0459dd23292881a3f9239bbbb`，clean；不同于预期标签 `d3700bf0a9`，不作为自己分支的整模验收，不读取或借用另一实现。
- T_cap=64、hidden=4096、D=512、ratio=2、state `[3,2,1024]` FP32、bank 2；ring=2。两个合成链另使用 full page=128、index page=64。
- 脚本 revision 7，SHA-256 `3822f139c464f81229fd37b76176020f308165866f7343a127c9811b3dd6d12c`，与报告、可信本地 LF 版本一致。
- 当前采集的 10 个公开 API Python hash 与第六轮相同；它们不替代底层 kernel 二进制版本。

## 4. ABI/参考源码核对

只读公开文档规定 CYCLE state 为 `[P,C,2*D]`，C=1..1024，每个 request 一个唯一物理 bank，保存最后 `min(live,C)` 个新 token。已查到的文档没有要求 C 必须大于 live/T_cap；不能据此直接把 ring=2 改成大 ring 作为修复。ratio 2、prefix 1、live 31 应有 16 个有效输出，row 0 是有效首 pair。

只读 arch35 参考源码中，normal 路径在 `ReadState` 前调用 `SaveState`，full-load 路径也先保存新 state；本次 odd-prefix 长 extend 的保存窗口会复用历史 pending token 的 ring slot。这与现象相符，是应请算子方确认的读写次序线索，**不是已证明的部署二进制根因**。

参考：

- `sgl-kernel-npu/docs/api/en/operators/attention/compressor_v2.md:29`、`:38`、`:50`。
- `refs/ops-transformer/experimental/attention/compressor_v2/op_kernel/arch35/compressor_v2_block_vec_normal.h:88`。
- `refs/ops-transformer/experimental/attention/compressor_v2/op_kernel/arch35/compressor_v2_block_vec_full_load.h:294`。
- 同目录 `compressor_v2_block_vec.h:515`、`:599`、`:647`。

本地 wrapper hash 为 `086db298…`，与部署 `b8da9f4d…` 不同；且本地 wrapper 的 CANN 注册仍为 arch22。不能用本地注册信息否认机器上已成功执行的 arch35 接口，也不能直接把参考源码观察写成部署 kernel 的确定缺陷。

## 5. 下一步

现有证据已足以向算子方提交可复现的 correctness 问题（复现命令见 [scripts/README.md](../../scripts/README.md)）。先请其确认部署 ABI、实际 kernel 源码/构建 revision，以及首 pair 的历史读取与环形 state 写回顺序。无需再全量重复 probe、只增加外部同步、放宽阈值或默认禁用 idle。

本轮没有新增验证脚本版本，也没有改主仓执行策略；保留全部 PASS/FAIL，等待算子契约确认或对应修复后做有针对性的设备回归。
