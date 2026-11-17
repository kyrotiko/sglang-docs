# A5 第六轮：ratio-2 有无 idle 对照

结果时间：2026-11-17 20:48:14 UTC。来源 `C:/Users/admin/Desktop/file6`；[原始报告](raw/report.md)、[JSON](raw/report.json)、[manifest](manifest.json)。回传 Python 仅按数据校验 hash/文本，未运行或导入；归档脚本文本来自 hash 相同的可信本地 revision 6。

## 1. 已闭合的事实

**不插入 idle 也复现 odd-prefix 失败，最早记录的差异已在 compressor 原始输出。不能把问题归为 idle 污染，或仅归为 FP4 writer 精度。**

两组均完成 capture 和全部 case，合法 pending-pair 初始 state hash 相同：`45d2527c1186a00fbdd2742fd92f2d4d1312cfb7e1ca5b127e8307e89c16c4d3`。它们都只在 `live_odd_prefix` 失败；full live 和 live-again 均 exact。插入 idle 的两轮 eager/replay 状态保护全部通过，失败后的 idle 和 live-again 也实际执行并通过。

| 指标：live=31、prefix=1、bank=2 | 插入 idle | 不插入 idle |
| --- | ---: | ---: |
| 原始 compressor 有效输出最大绝对差 | 0.118896484375 | 0.13671875 |
| 归一化有效输出最大绝对差 | 2.65234375 | 3.0703125 |
| compressed cache 字节差异 | 95 | 213 |
| index K 字节差异 | 63 | 64 |
| index scale 字节差异 | 0 | 1 |
| attention 有效输出最大绝对差 | 1.01171875 | 1.11328125 |
| top-k / 受保护的非零 bank 最终状态 | exact / exact | exact / exact |
| 两轮 idle 真实 cache/state 保护 | PASS | 未执行 idle |

RMSNorm、cache 和 attention 均有后续差异，但原始 compressor 输出已经不一致；当前数据不能进一步区分 compressor 本身、输入/state 可见性、或合成链交互的原因。最终 state exact 不证明执行时的每次 state 读取都相同。

差异坐标采样指向首个跨 chunk pair：compressed cache 样本在 `[page=1,row=0,...]`，index K 同样在首行，原始输出样本在有效 row 0。仅保存前 32 个不相等坐标，不能据此宣称其余所有行一致。

合法 pending-pair 预热后仍失败，故第五轮的随机 bank 状态不能再作为唯一解释。第五轮的原 FAIL 继续保留。

## 2. 环境边界

| 项目 | 第五轮 | 第六轮 |
| --- | --- | --- |
| 硬件 | Ascend950DT_9582 | Ascend950PR_9589 |
| npu-smi / driver 报告 | 25.1.rc1.b097 | 25.1.rc3 |
| 实际主仓 | `24deba1e5f…`，clean | `24deba1e5f631ba0459dd23292881a3f9239bbbb`，clean |
| torch / torch_npu | 2.13.0+cpu / 2.13.0 | 相同 |
| sgl-kernel-npu | 2026.11.0.post4 | 相同 |
| 当前采集的公开 API Python hash | 10 项 | 10 项均相同 |

硬件和 driver 已变化，不能将 17→95/213 字节解释成同环境下仅修正夹具的影响。Python wrapper hash 相同也不证明所有底层二进制一致。device 0 报告为 OK；其他卡状态不是这份单卡失败的已证实原因。

脚本为 revision 6；SHA-256 `ea54ee20542244c55b19c7561acd7628291d2b76ed31e5c460e8e4a24e9e0aac`，与报告和 intake 时可信本地 LF 版本一致。参数仍为 T=64、hidden=4096、full page=128、index page=64、ring=2、seed=20261116。实际源码不同于脚本预期标签 `d3700bf0a9`；本轮不构成该分支的整模验收，不读取或借用另一实现。

## 3. 接下来验证什么

下一步只隔离已经出现差异的 compressor 边界：同一合法历史/state，比较 eager 重复与 graph 重复，覆盖奇数 prefix 的短/长 extend，保留逐行误差和 state 输入/输出 hash。另给现有无 idle 合成链增加 replay 前显式设备同步对照，判断 staging 完成性是否影响结果；同步仅用于定位，不作为服务修复或性能方案。revision 7 已提供 `compressor_boundary`、`active_chain_r2`、`active_chain_r2_sync`，见[运行说明](../../scripts/README.md)。编译、CPU 自检、控制流及故障注入 fixture 已通过，不计为 revision 7 设备证据。

本地 torch_npu 参考源码显示 replay 在调用方的当前 stream 执行，正常同 stream 顺序应已足够；该参考源码不等于机器上的实际二进制，也没有因此认定当前脚本存在跨 stream bug。源码依据：`refs/pytorch/torch_npu/csrc/core/npu/NPUGraph.cpp:392`、`torch_npu/npu/graphs.py:637`。

本轮不重跑 ratio 0/1、独立 CPU attention reference 或完整 17 组基元。两轮 idle 的正面证据保留，整模 metadata、真实多流和 DP/DeepEP 通信继续单独验收；不选择整组 eager 回退，也不放宽字节或数值标准。
