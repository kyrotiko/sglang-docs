# 合入 randgun/dsv41-a5 后的验证调整

日期：2026-11-17。本文是合并时的验证计划快照；随后[第三轮全量 A5 结果](../validation/a5-20261117-run3/findings.md)已取得 compressor、公开 writer 和 mHC 的正面证据。当前只需用 revision 4 修复后的脚本补三组 writer 消费链，下面的“尚未验证”描述保留其合并时语境。

## 1. 合并结果

| 项目 | 版本 |
| --- | --- |
| 合并前主仓 | `dsv41-a5@5633d68b04` |
| 拉取的远端 | `randgun/dsv41-a5@c790194814` |
| 合并后主仓 | `9898f7ce9af12fec5e2ef57d1adaa2b7533992ef` |
| 备份分支 | `backup/dsv41-a5-before-randgun-merge-20261117` |
| 合入范围 | 13 个新提交，10 个文件净变化 |
| 合并方式 | 正常 merge，无内容冲突，保留本地 11 个入图提交 |

已确认远端 tip 和合并前本地 tip 均为新 HEAD 的祖先。NPU breakable backend、V4 prefill metadata 容器、graph backend factory、MoE dispatcher 这四个入图核心文件与合并前一致；模型的 NPU attention 图分流也保留。

## 2. 与验证直接相关的改动

| 改动 | 合并后的行为 | 对旧事实的影响 |
| --- | --- | --- |
| V4.1 writer | 改用 `sgl_kernel_npu.operators.mem_cache.kv_compress_epilog_v2` 公开 wrapper | 旧 custom namespace 未注册不再是当前 writer 的直接阻塞判据；新 wrapper 必须执行复验 |
| mHC premix | 改用 `hyper_conn.mhc_pre_dpskv4(..., pre_mix=..., return_pre=True, impl="cann_recipes_infer_cann920")` | 旧 `custom.npu_hc_pre_v2` 缺失不能直接判定新路径失败；须验证新显式实现和返回系数 |
| V4 row width | `npu_kv_layout.py` 改名为 `dsv4_kv_layout.py`，V4 FP8 宽度统一由 runtime quant path 决定 | 重新记录 V4 draft 行宽和 V4.1 target 544/320B，不能混用；源码中无旧模块名遗留 |
| 投机校验 | V4.1 明确只接受 DSpark + STATIC；按模型 generation 判断，覆盖 draft 清空 ratio 的情况 | 静态通过/拒绝矩阵已做 CPU 回归，完整 DSpark target/draft 状态仍需设备验证 |
| Q 处理 | V4 专用的多流准备路径去掉冗余 `q_head_norm` 条件 | 单算子手工 Q 探针不覆盖模型准备路径，后续整模应回归；不据此作 QLI/attention 算子精度变化结论 |
| fixture/CI | 对齐 pool params、公开 wrapper 和 helper API | 旧基线的两处收集错误及 mHC 用例失败已在当前 CPU 回归中消除 |

Compressor 仍通过 `_ds41_ops().compressor = compressor_v2` 调用，ratio-2 参数和 pending-pair 路径没有实质改变。仅合并 SGLang 不能证明旧部署包的 arch35 拒绝已解决；若 SDK 更新，必须重跑。

V4.1 的 544B SWA / 320B compressed、group32/16 和 `-1` sentinel 规划没有改变；capability 仍可能选择 V4-only captured metadata，low-ratio source 仍未建立完整 prefill 固定状态。因此本次合并不是完整 V4.1 prefill graph 实现。

## 3. 本地合并回归

在 CPU-only `sglang-npu-dev` 中，每文件独立 Python 进程，避免测试 stub 串扰：

| 范围 | 通过数 |
| --- | --- |
| backend resolution、NPU segment、prefill metadata、MoE、DP padding | 49 |
| prefill runner helpers、runner、padding | 55 |
| NPU Ascend DSV4 attention | 63 |
| NPU V4.1 low-ratio attention | 101 |
| 模型平台/mHC 分流 | 11 |
| Engram history | 2 |
| **合计** | **281** |

12 个测试文件全部通过，合并后 10 个 Python 文件 AST 检查及 `git diff --check` 通过。CPU 结果证明接口使用、生命周期 bookkeeping 和回归用例一致，不证明新公开算子已在 A5 实际执行。

## 4. 探针 revision 3

已调整同一个 [验证脚本](../scripts/verify_dsv41_prefill_graph.py)：

- 预期主仓基线更新为 `9898f7ce9a`，revision=3。
- writer 探针只调用当前公开 `mem_cache.kv_compress_epilog_v2`，没有 custom namespace fallback。
- environment 补采新 writer、hyper_conn/mHC、指定 implementation 的解析，以及 runtime V4/V4.1 行宽。旧 custom schema 仅作为历史诊断。
- 新增 `mhc_premix`，按主仓的实参顺序调用新接口，检查上一子层 pre 的折叠、返回 triplet、同 X 不同 previous-pre 的系数不变性及多轮图对照。未覆盖外部 RMSNorm、post、完整 HcState 或多流。
- 新增 `writer_attention_r0/r1/r2`，由公开 writer 在 replay 前生成 KV，按 544/320B 契约解包供 CPU reference，再让 captured MQSMLA 消费。本组 writer 在图外准备，不宣称 writer/attention 已在同一 segment；writer 自身 capture 由 epilog 两组验证。
- 原来的 `attention_r*` 仍保留手工缓存夹具，便于区分读侧与写侧；默认总计 17 组。
- 修复 `--import-module` 循环覆盖 probe 名的问题：旧两轮 `import_module=[]`，不受影响；CPU 复现确认旧版会误报 CRASH，修复后正确保留原 probe 的 BLOCKED/log 路径。

新增的 writer→attention 不能由旧事实替代：旧 attention 消费的是手工打包缓存，未执行新 writer。若该链的 CPU reference 仍失败，继续保留严格精度失败，不以 replay exact 自动宣布格式和数值全部通过。

## 5. 哪些事实保留，哪些需要重验

| 项目 | 处理 |
| --- | --- |
| 首轮 geometry / metadata planning / K prologue / Engram break | 保留为 `5633d68b04` 与旧 SDK 的历史证据；对应计算主体未实质改变，不强制重复相同 case |
| 两轮 QLI/QSLI/手工缓存 attention exact | 保留；合并没有改变相应 DSL 调用和几何。若升级 SDK/CANN/torch_npu，补 QLI/QSLI smoke，不能只凭 Python hash 相同推广到新二进制 |
| CPU FP32 attention 差异 | 问题仍未闭合，不放宽默认阈值；核对算子精度契约，writer 消费链也分别记录精度与 replay 对照 |
| custom epilog/mHC 未注册 | 只保留为旧入口的历史事实，不再拿来否定新公开入口 |
| 公共 writer 的存在、格式、`-1`/NaN/guard、capture | **必须重验** |
| writer 产出缓存被 MQSMLA 消费 | **新增必验**，原手工缓存 attention 不能覆盖 |
| mHC 新公开 premix 和指定 implementation | **新增必验** |
| compressor arch35 与跨 chunk / bank 状态 | **仍需验证**，合并未改该入口；依赖新 SDK 是否提供支持 |
| DSpark target/draft 行宽及投机提交 | 当前 CPU guards 通过；完整整模阶段做设备回归，单算子脚本不替代 |

## 6. 设备上的最小执行顺序

先确认使用新主仓和新脚本。旧部署包 `2026.11.0.post4` 的第二轮 exports 没有公开 `kv_compress_epilog_v2`；**不能假定只拉 SGLang 就已满足新依赖**。先只跑环境：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --probes environment \
  --output-dir ./dsv41-after-merge-env
```

检查实际 source SHA、`missing_apis`、新 writer 与 mHC 的实现解析。若缺少公开接口，先使用包含这些功能的 sgl-kernel-npu 构建；精确版本需按实际包确认，不猜版本号或偷偷换代际接口。环境 PASS 是 inspection 状态，不等于这些函数都能执行。

依赖可用后，优先跑 5 组直接受影响项：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --probes environment,compressor,epilog_fp8,epilog_fp4,mhc_premix \
  --output-dir ./dsv41-after-merge-ops
```

writer 两组通过后，再跑真实写入到消费的三组：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --probes writer_attention_r0,writer_attention_r1,writer_attention_r2 \
  --output-dir ./dsv41-after-merge-writer-read
```

若 SDK/CANN/torch_npu 发生升级，再追加 `--probes qli_c2,candidate_chain` 的轻量回归。所有命令使用新目录，分别回传生成的 `dsv41-prefill-evidence.tar.gz`。默认浮点阈值未放宽，独立精度与 graph/eager 对照继续分开记录。
