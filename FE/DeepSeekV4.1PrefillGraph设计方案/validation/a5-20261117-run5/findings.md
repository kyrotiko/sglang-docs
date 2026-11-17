# A5 第五轮：idle 本地链实测

结果时间：2026-11-17 11:57:42 UTC。回传目录 `C:/Users/admin/Desktop/file5`。原始报告见 [raw/report.md](raw/report.md)、[raw/report.json](raw/report.json)，来源及文件 hash 见 [manifest.json](manifest.json)。回传 Python 仅作 hash/文本校验，没有运行或导入；归档的脚本文本来自 hash 相同的可信本地 revision 5。

## 1. 当前结论

**不能继续把“idle rank 不支持 replay”作为已证实的算子限制。** 当前候选 dummy 协议已经取得单卡正面证据：ratio 0、ratio 1 的完整 live→idle→live 序列通过；ratio 2 的 capture、full live 及随后的 idle 也通过。ratio 2 在后续非 idle 的 odd-prefix case 出现字节差异，仍保留 FAIL。

| 探针 | Capture | 已完成 case | 结果 |
| --- | --- | --- | --- |
| `idle_chain_r0` | PASS | 5/5；全部 recorded comparisons exact | PASS，含两轮 idle 与后续 live |
| `idle_chain_r1` | PASS | 5/5；全部 recorded comparisons exact | PASS，含 writer/index-K/QLI→QSLI/attention 合成图内链 |
| `idle_chain_r2` | PASS | 前 2 个 case exact | FAIL：`live_odd_prefix/compressed` 有 17 个字节不一致 |

共 12 个完成并归档的 graph/eager case，记录的比较全部 exact。5 个已完成 idle case 的受保护 cache/state 在 eager 和 replay 中均未被修改；无效 attention 输出全部为 finite，但没有把无效输出必须为零作为通过条件。

这些是私有合成缓存、identity RoPE、固定单卡形状的事实，不覆盖真实 allocator/source-group、多流、mHC、DeepEP、DP8 集体通信、主仓 metadata hooks 或整模生成。独立 CPU FP32 attention reference 的原 FAIL 未被重新判定。

## 2. 环境与源码边界

| 项目 | 本轮 |
| --- | --- |
| 硬件 | Ascend950DT_9582，device 0 |
| 实际主仓 SHA | `24deba1e5f631ba0459dd23292881a3f9239bbbb`，工作树 clean |
| 脚本中预期基线 | `d3700bf0a9`；与实际主仓不同 |
| Python | 3.12.15 |
| torch / torch_npu | 2.13.0+cpu / 2.13.0 |
| sgl-kernel-npu | 2026.11.0.post4 |
| CANN 路径 / npu-smi | `/usr/local/Ascend/cann-9.2.0` / 25.1.rc1.b097 |
| 脚本 | revision 5；SHA-256 `2286d7c446da76f071f6dfe6bf62ea0ec3c3679cdacde5a5667c3eabda9d1328` |
| 参数 | T=64、hidden=4096、full page=128、index page=64、candidate capacity=2048、ring=2 |

回传脚本 hash 与报告一致，也与 intake 时可信本地脚本的 LF 规范化 hash 一致。当前采集的 10 个公开 API Python 源码 hash 与第三轮对应项全部相同；这不等于证明所有底层二进制完全相同。

实际主仓不是预期 SHA，因此本轮不能归为 `d3700bf0a9` 的整模验证。探针计算主体直接调用公开 SDK，故可保留为记录环境下的算子/合成链证据。本次不读取或借用实际未知 SHA 对应的另一实现方案。

## 3. 本次验证的 idle 协议

保留物理 bucket：`cu=[0,64]`、KV length=64；有效行数为 0，`seqused=0`，cycle bank=0，compressed/index loc 全 `-1`，SWA loc=0，window/cmp length=0。QLI/QSLI 仍执行 dummy rows，attention 忽略无效输出。

保护检查覆盖非零 state banks、所有非 dummy SWA 页，以及完整 compressed/index cache。SWA page 0 允许 dummy 写入。`idle_chain_r2/idle_after_full` 已通过该保护检查，不能把其后的 live-prefix 比较失败改称为 idle 写坏状态。

## 4. Ratio-2 的 17 字节差异尚未定位

失败输入：physical T=64、live=31、prefix=1、bank=2、KV length=32；比较对象为 `[3,64,1,320]` compressed cache。发生在 replay 后的比较阶段，capture 已完成。

revision 5 的比较会在第一个不一致张量停止。因此该 case 后续 index/output 比较，以及余下的 `idle_after_short`、`live_again` 均未完成；不能推断它们通过或失败。报告仅保存 tensor 前 12 个值，没有保存实际不一致坐标或 compressor/norm 中间值，现有数据不足以将问题归因于 compressor、RMSNorm、FP4 writer、图执行或 idle。

另有一个已确认的夹具边界：revision 5 将 private state 初始化为随机值，并在 bank 2 未产生过上一 chunk 的情况下直接使用 prefix=1。该夹具没有建立合法的 pending-pair 历史。它是否导致字节差异尚未证明，不能因此删掉本次 FAIL。

下一轮只定向 ratio 2：先由真实 compressor 产生 bank 2 的 prefix-1 pending pair，补 compressor 原始/归一化有效输出与实际不一致坐标；加入相同输入/state 的不插入 idle 对照，并收集全部张量比较。revision 6 已提供 `idle_chain_r2` 与 `active_chain_r2` 两个入口，见[命令](../../scripts/README.md)。本地编译、CPU 自检及控制流/故障注入 fixture 通过，未记为设备证据。

ratio 0/1 和原 17 组无需因本次结果全量重跑。设备事实闭合前，不选择整组 eager 回退，也不直接宣称多 rank idle 已支持。
