# A5 第四轮 writer→attention 事实

结果时间：2026-11-17 04:03:36 UTC。本轮仅重跑 revision 4 的三组 writer 消费链。原始结果：[raw/report.md](raw/report.md)、[raw/report.json](raw/report.json)；校验：[manifest.json](manifest.json)。回传 Python 只校验 hash，没有运行或导入。

## 1. 本轮结论

**第三轮的 dtype 夹具问题已修复。三组都已成功准备公开 writer 缓存、完成 capture，并在各自 5 轮 graph/eager 对照中逐元素一致。** 15 个有效输出 case 全部 `exact=true`、`max_abs=0`。

三组总状态仍为 FAIL，是独立 CPU FP32 reference 的原有差异。本轮所有记录的 CPU reference diagnostics，与第三轮对应手工缓存 attention 的 diagnostics 完全相同；没有观测到新 writer 消费路径增加的差异。该比较是报告中采样/统计记录的比较，不是跨两次运行保存完整输出 tensor 后的逐元素比较。

当前没有必要再重复相同的单卡算子 case。下一步应进入主仓固定 metadata、source/index/attention 数据流和使能条件的实际接入；独立精度标准继续单独核对，不改写成通过。

## 2. 来源和边界

| 项目 | 记录 |
| --- | --- |
| 脚本 | revision 4，SHA-256 `cea42129e530a00f9d8206fe7a982db8c8e427a2b62243f2ed8e06c9eb0e900f` |
| 期望主仓基线 | `9898f7ce9a` |
| worker 硬件 | Ascend950DT_9582 |
| torch / torch_npu | 2.13.0+cpu / 2.13.0 |
| 尺寸 | T=64，full page=128，index page=64，主 heads=64，head_dim=512 |
| 阈值 | parity 和 CPU reference 均保持 `rtol=1e-3, atol=1e-3` |

回传脚本 hash 与报告一致，与本地发布版本按 LF 规范化后逐字一致。本轮没有运行 environment，所以没有独立重采 git status、包清单或 SDK source hashes；不能将报告的 expected baseline 当成一次新的 git 实测。各 worker 的 device、torch 和实际 API signature 已记录。

## 3. 执行与视图证据

| 探针 | Capture | 有效输出 eager/replay | writer/consumer 视图 |
| --- | --- | --- | --- |
| writer_attention_r0 | PASS | 5/5 exact | 544B writer=FP8，consumer=uint8，同 storage |
| writer_attention_r1 | PASS | 5/5 exact | SWA 同上；320B compressed writer/consumer 都是 uint8，同 storage |
| writer_attention_r2 | PASS | 5/5 exact | 同上，compressed page 按 ratio 2 |

每组覆盖 full_A、short_B、page_boundary_C、different_prefix_D、repeat_A，live 行数 64/31/16/61/64，包含 prefix 和页表变化。各组的 bound input 地址检查通过。

每个 case 的 `same_storage=true`，证实修复使用 dtype view 而非数值 cast/新 buffer。公开 writer 的输出按 544B/group32 和 320B/group16 ABI 解码后，`max_abs_input_to_decoded=0`。**这些输入原本就是量化格式可表示的夹具值，不能推广为任意 BF16 输入量化无损，也没有独立量化精度阈值通过的结论。**

所有观测到的 padding 输出均为零：每组的 short_B 33 行、page_boundary_C 48 行、different_prefix_D 3 行。只对本组配置成立，不替代其他算子的 padding/state 写入契约。

writer 在 replay 前 eager 准备缓存，MQSMLA consumer 在图内执行。writer 自身的 capture 在第三轮已单独验证；本轮不是 writer 与 attention 同一 segment 的联合 capture，更不是完整 compressor/source 链。

## 4. 独立精度与旧问题一致

| 消费链 | 失败采样 | 最大绝对误差 |
| --- | --- | --- |
| window | short_B row0 heads 0/63 | 0.0018430352 |
| c1 | short_B row0 head63 | 0.0018134117 |
| c2 | short_B row0 head0 | 0.0013743341 |

失败位置、最大误差、BF16/ULP/L2 诊断与第三轮对应记录一致，完整 `cpu_reference` JSON 子对象比较相同。说明当前可观测到的精度差异延续原问题，不能当成新的 dtype 或 graph 回放错误；也不能仅凭一致就认定算子独立精度已经符合契约。

保留全部严格 reference FAIL，不放宽默认阈值。应结合算子实际精度契约/参考实现，以及后续整模 eager/graph 对照另行验收。

## 5. 事实基础已收敛到哪里

结合 [第三轮](../a5-20261117-run3/findings.md)：

- 当前 950DT/SDK 环境的 compressor、公开 544/320B writer、K prologue、QLI/QSLI、MQSMLA、mHC premix、Engram break 已取得对应范围内设备证据。
- 第三轮 59 个已执行 case 与本轮 15 个 case 的对照均 exact，但各 probe 的参考类型和覆盖边界不同，不能概括为整个模型数值已通过。
- 当前 writer 入口、dtype/storage 相容性及 writer 产出缓存的消费已执行成功；旧 custom namespace 缺失不再是此路径阻塞。
- 主仓仍未实现完整 V4.1 prefill captured metadata/source 数据流；真实 request-bank/source group 生存期、完整 mHC/Engram/DSpark、多卡通信和模型精度/性能仍需接入后验证。

本轮不需要继续修 dtype 或重复这三组。除非 SDK、形状或实现发生新变化，后续测试转到真实模型 metadata 与多轮状态闭环，独立精度问题保留为单独验收项。
