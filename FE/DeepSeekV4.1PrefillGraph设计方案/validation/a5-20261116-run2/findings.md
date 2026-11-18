# A5 第二轮探针结论

本文固定在 `5633d68b04` 和当时 SDK。2026-11-17 合入 randgun 后 writer/mHC 已换公开入口，旧 custom schema 缺失不再直接否定新路径。以下设备事实和原始结果保持不变。

结果创建时间：2026-11-16 20:03:25 UTC。原始结果：[raw/report.md](raw/report.md)、[raw/report.json](raw/report.json)；来源校验：[manifest.json](manifest.json)。回传 Python 仅用于 hash 和文本一致性校验，没有运行或导入。

## 1. 本轮结论

**修正 index PA page 后，ratio-2 QLI 和 ratio-1 QLI→QSLI candidate 链都已通过 5 轮 NPUGraph capture/replay 对照。** 全部有效有序 top-k、candidate indices 和 candidate lengths 与 eager 精确相等，且不同输入间输出确实发生变化。

window attention 再次完成 5 轮逐元素一致的 graph/eager 对照。其总状态仍为 FAIL，来自独立 CPU FP32 oracle，不是图执行差异。不能将这个 FAIL 解释成 attention 不可入图，也不能把独立精度验收改记为通过。

当前工作重心应转到 V4.1 compressor/KV writer 的真实接口契约、固定 metadata 和模型状态接入；同一组 QLI/attention case 已有证据，无需反复重跑来证明相同结论。

## 2. 来源和环境连续性

| 项目 | 本轮值 |
| --- | --- |
| 主仓 | `5633d68b04ad1bdd4faecb1c8e370e5c5ea6403c`，机器工作树干净 |
| 硬件 | Ascend950PR_9589，逻辑 device 0 |
| torch / torch_npu | 2.13.0+cpu / 2.13.0 |
| sgl-kernel-npu | 2026.11.0.post4 |
| CANN 路径 | `/usr/local/Ascend/cann-9.2.0` |
| 脚本 | revision 2，SHA-256 `db94340824e68cacf4449241a83bbd481571d9ffc558a4fd35b0de063f682f7f` |
| index PA page / candidate block | **64 / 8**，是两个独立维度；主仓运行时常量同样为 64 |
| 其他关键尺寸 | T=64，主 heads=64，indexer heads=32，topk=512，candidate capacity=2048 |
| 浮点阈值 | graph parity 和 CPU reference 均保持 `rtol=1e-3, atol=1e-3` |

回传脚本 hash 与报告一致，与本地版本按 LF 规范化后逐字一致。八个 DSL/compressor API 的源码 hash 与首轮全部一致，新增通过结果来自修正探针几何，没有更换 kernel 包。原始文件按字节保存并记录 SHA-256。

## 3. 已验证的执行链

| 探针 | Capture | 5 个 case 的 eager/replay | 独立参考与范围 |
| --- | --- | --- | --- |
| qli_c2 | PASS | 有序索引全部 exact | 因果范围检查通过；没有独立 score oracle |
| candidate_chain | PASS | QLI、QSLI、有界 candidate 区和长度全部 exact | QSLI 直接消费本轮图内 QLI 结果；K cache 为合成预置数据，尚未连接真实 source |
| attention_r0 | PASS | 全部有效输出 exact，max_abs=0 | CPU FP32 reference FAIL，见下节 |

三个执行探针都覆盖 full_A → short_B → page_boundary_C → different_prefix_D → repeat_A。真实 live 行数依次为 64、31、16、61、64，包含 prefix 和页表变化。输出变化标记均为 True，不是只拿 capture 原始输入重放。

candidate_chain 的有效形状为 QLI/QSLI `[live_T,512]`、candidate `[live_T,2048]`、length `[live_T,1]`。本轮没有并列分数造成的有序索引差异。

对于本轮 window attention，所有观测到的 padding 输出都是零：short_B 33 行、page_boundary_C 48 行、different_prefix_D 3 行，`max_abs_finite=0`。这只覆盖本轮 kernel/shape，不推广成其他算子或任意部署的 padding 契约。

这些探针使用固定存储和 replay 前原地刷新，没有 graph.update。对已覆盖的 QLI/QSLI/MQSMLA 路径，可以继续沿现有固定地址方案推进；不能据此断言尚未执行的 compressor/writer 或完整模型也不需要额外协议。

## 4. CPU attention 参考差异的进一步事实

失败仍集中在 `short_B` 的 row0（position=1），heads 0/63；其余记录的采样通过。

| 指标 | head0 | head63 |
| --- | --- | --- |
| 最大绝对误差 | 0.0018430352 | 0.0014763474 |
| RMSE | 0.0002836769 | 0.0003510043 |
| 归一化 L2 | 0.0016352315 | 0.0020217750 |
| 距最近 BF16 参考值最多 ULP | 19 | 21 |
| 位于 1 ULP 内的坐标 | 498/512 | 484/512 |

head0 最大误差坐标 121：FP32 reference=0.5057492852，最近 BF16=0.50390625，NPU 输出也为 0.50390625。**该坐标的差异正好是最终 BF16 舍入。** 但同一向量其他坐标与最近 BF16 值仍有偏差，不能推广为整段输出只存在最终舍入。

head63 最大误差坐标 52：reference=0.3032111526，最近 BF16=0.302734375，NPU=0.3046875。这里 NPU 相对最近 BF16 参考还偏移一个 BF16 ULP，说明还需核对中间精度或具体算术契约。

将 CPU softmax 概率先转 BF16 的诊断假设也没有完整复现 NPU：对应最大误差为 0.00390625 / 0.001953125。它不能替代原 oracle。当前保持严格参考 FAIL；不调整默认容差、不把回放精确一致与独立精度正确性混在一起。

## 5. SDK inventory 已确认的接口缺口

1. `attention.compressor_v2` 的 auto 解析仍明确拒绝 arch35，只支持 arch22；这一接口本轮未重新执行，首轮 warmup 已实际失败。
2. 机器确实导出 `attention.compressor`，auto 可解析，并注册了 `npu::compressor`。其签名包含 ape、norm_weight、rope_sin/cos 和显式状态表，融合 RMSNorm/RoPE；与 V4.1 当前所需的 pre-RoPE latent 和 request-bank pending-pair 路径不是同一调用契约。存在/可解析不等于可直接替换或已验证。
3. 机器导出 `mem_cache.kv_compress_epilog`，注册的是 `npu::kv_compress_epilog`；其实际文档明确为 DSV4 的 608/640B、group64 格式。
4. `custom.kv_compress_epilog_v2` 仍未注册；当前 V4.1 writer 要求 fully-quant 544B/320B。只改 op 名会引入格式不匹配，不能这样修复。
5. `custom.npu_hc_pre_v2` 也未注册，虽然公开 `npu_mhc_pre/post` schema 存在。mHC 未做执行探针，整模之前需核对真实 serving 的注册和调用路径；不能将 inventory 当成 mHC 执行失败或通过。

本轮 `import_module=[]`，没有额外注册模块。若模型服务依赖外部扩展，需要拿到实际模块/库和 V4.1 ABI；现有数据没有依据猜一个 import 或强行选择旧接口。

## 6. 下一步

- 保留 QLI/QSLI/MQSMLA 的入图正面证据，继续设计和实现对应固定 metadata 与 producer/consumer 数据流。
- 优先对齐 ratio-2 compressor 和 544/320B writer 的部署接口，确认正确输出与 state/latent 语义后再改主仓；若没有所需接口，需要明确新增实现或依赖版本，而不是继续在图层兜底。
- 对 window attention 的独立精度，核对实际算子精度契约和参考实现，再决定验证标准；现有数据不足以证明数值 bug，也不足以宣布独立精度验收通过。
- 整模的 source group、SWA/cache 写回、mHC、真实 Engram lookup、DSpark aux hidden 和多卡 collective 仍未覆盖，不能由单算子结果自动推导通过。

首轮历史证据见 [run1/findings.md](../a5-20261116-run1/findings.md)。本轮没有修改主仓或再次更改脚本默认阈值。
