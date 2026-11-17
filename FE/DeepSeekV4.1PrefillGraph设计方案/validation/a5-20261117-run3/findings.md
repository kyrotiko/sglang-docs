# A5 第三轮全量探针结论

后续[第四轮](../a5-20261117-run4/findings.md)已验证 dtype 修复并完成三组消费链图对照；本文保留第三轮当时的错误和下一轮计划，当前无需重复下文重跑命令。

结果时间：2026-11-17 03:36:43 UTC。主仓 `9898f7ce9af12fec5e2ef57d1adaa2b7533992ef`，工作树干净。原始结果见 [raw/report.md](raw/report.md)、[raw/report.json](raw/report.json)，来源见 [manifest.json](manifest.json)。回传脚本仅校验 hash/文本，没有运行或导入。

## 1. 核心结论

**当前环境的 compressor、544/320B 公开 writer 和 mHC premix 均已实际执行并完成图回放，对照全部 exact。旧接口阻塞在本环境中已消除，不能继续标为当前未支持。**

17 组总状态为 11 PASS、6 FAIL。6 个 FAIL 分成两类：

- 3 个手工缓存 attention：capture 和 5 轮 graph/eager 全部 exact，独立 CPU FP32 参考仍未通过。
- 3 个 writer→attention：尚未进入 capture，公开 writer 在参数校验阶段拒绝了脚本传入的 uint8 SWA cache；这是探针 dtype 视图错误，不能判为 writer 或 attention 的设备失败。

已完成 capture 的 12 组共有 59 个 case 对照，记录的比较均为 exact；metadata planning 另外有 15 组配置。尚未覆盖完整模型或多卡。

## 2. 环境与来源边界

| 项目 | 本轮 |
| --- | --- |
| 硬件 | **Ascend950DT_9582**，前两轮是 Ascend950PR_9589 |
| 主仓 | `9898f7ce9a`，clean |
| 脚本 | revision 3，SHA-256 `2b17ad5f604c64669d487f0630312c8bfce65c6659913f32a57f08c2fd2dabc7` |
| torch / torch_npu | 2.13.0+cpu / 2.13.0 |
| sgl-kernel-npu 包版本 | 仍标记为 `2026.11.0.post4` |
| SDK 内容变化 | `compressor_v2` 源码 hash 已变化；新增公开 writer/mHC 的导出与实现 |
| 核心尺寸 | T=64、B=1、full page=128、index page=64、candidate block=8、ring=2 |
| 阈值 | parity 和 CPU reference 均保持 `rtol=1e-3, atol=1e-3` |

回传脚本 hash 与报告一致，按 LF 规范化后与运行时本地已提交脚本一致。SDK 的版本号与第二轮相同，但实际代码和能力不同；硬件变体、driver、系统镜像也变了。因此这是新环境的实测结论，不能只凭相同 package version 推广回旧机器，或把差异全部归于 SGLang merge。

`compressor_v2` 的 auto 现在解析为 `_cann_impl`，公开 writer 为 `_cann_kv_compress_epilog_v2`，mHC 指定实现为 `_cann_9_2_mhc_pre_v2`。这些解析与实际执行结果都已记录。旧 custom schemas 仍可缺失，但新主仓使用公开 wrappers，因此不构成当前阻塞。

## 3. 新增通过的关键事实

| 项目 | 结果 | 范围 |
| --- | --- | --- |
| compressor_v2 | 7/7 case exact；split `[1,1]` 与 concat pair 也 exact | 有效压缩输出、完整 private state、bank 0/1/2、pending pair、ring wrap、复用 |
| epilog_fp8 | 4/4 case byte-exact | 544B、group32，`-1`、全 skip、NaN 无效输入、guard rows |
| epilog_fp4 | 4/4 case byte-exact | 320B、group16，同上 |
| mhc_premix | 5/5 case exact | 新公开入口和指定 implementation，identity/uniform/nonuniform previous-pre、折叠参考、返回 triplet |
| mHC 同 X triplet | 2 组比较 exact | previous-pre 改变输入折叠，不改变当前 X 预测的 pre/post/comb |

原有 geometry、K prologue、QLI、QLI→QSLI、Engram break 也全部通过。手工缓存 window/c1/c2 attention 共 15 个 graph/eager case exact，与旧事实一致。

这些结果不替代真实 allocator 的 capture bank 隔离、完整 source-group 生命周期、mHC 外部 norm/post/多流、真实 Engram lookup、DSpark 或整模精度。Compressor 的 state 是私有合成 bank，不是服务中真实请求池。

## 4. Writer→attention 的失败是脚本视图问题

三组都在 `public_writer_prepare:full_A` 失败，错误为：

```text
cache dtype does not match quant_mode:
mxfp8_bf16 requires float8_e4m3fn/float8_e5m2
and mxfp4_bf16 requires uint8
```

原因已由源码核实：`prepare_attention()` 为满足 MQSMLA 的输入要求，把缓存存成 uint8 byte view；`prepare_writer_cache()` 却将这个 byte view 直接交给 FP8 writer。主仓 `NPUDSV4KVLayout.V41_SWA_FP8` 的实际池 dtype 本来就是 float8_e4m3fn，写后才给 attention 看 uint8，因此主仓不是同一 dtype 错误。

revision 4 修复为同一存储的两种视图：writer 使用 `cache.view(float8_e4m3fn)`，attention 继续使用 uint8；FP4 writer 继续使用 uint8。使用 `view`，不做会改变数值/分配新存储的 dtype 转换。报告增加 writer dtype、consumer dtype 和 same-storage。

本地 CPU 外部 writer 边界回归已确认：已发布 revision 3 会复现同样 dtype 拒绝，revision 4 通过；还验证了 buffer 指针不变、byte round-trip 和 FP4 路径。新增 self-test 防止后续把 byte view 再误传给 FP8 writer。该修复尚需机器上的真实 kernel 复跑，不能提前记为链路通过。

## 5. 独立 attention 精度仍未闭合

| 手工缓存 attention | 图对照 | CPU 参考采样最大绝对误差 |
| --- | --- | --- |
| window | 5/5 exact | 0.0018430352 |
| c1 | 5/5 exact | 0.0018134117 |
| c2 | 5/5 exact | 0.0013743341 |

失败仍是 short_B、row0 的部分 head，与前两轮对应误差一致。不能将它归为 graph 新增误差，也没有因为新公开接口通过而改记独立精度通过。默认 reference 阈值保持不变，算子精度契约仍待核对。

## 6. 下一轮只需重跑三组

换用 revision 4，不必再跑已完成的其余 14 组：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /mnt/share/y00627667/repos/sglang \
  --device 0 \
  --probes writer_attention_r0,writer_attention_r1,writer_attention_r2 \
  --output-dir ./dsv41-writer-read-r4
```

继续回传该目录的 `dsv41-prefill-evidence.tar.gz`。如果 dtype 修正后出现 CPU reference FAIL，仍按分列判读 graph/eager 与独立精度，不把修正 dtype 等同于解决全部数值问题。

当前算子能力已足以推进主仓固定 metadata 与模型数据流接入；完整 V4.1 prefill graph 仍未实现，不能用这些单卡合成探针代替整模/多卡验收。
