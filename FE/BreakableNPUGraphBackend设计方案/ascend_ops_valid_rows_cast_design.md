# Ascend Ops allgather 有效行 Cast 实验方案

## 目标与依据

2026-11-07 的 DP0 profiling 中，完整容量 `[196608,4096]` 与尾批 `[195090,4096]` 的 BF16→FP32 Cast 合计 1.637 s，约占采集 Stage 的 9%。图内 Cast 平均 3.806 ms，MTE3 活动比例约 99.56%；该比例不等同于 HBM 带宽利用率。保留 GMM BF16 输出、原版 FP32 Finalize 和 FP32 ReduceScatter，仅减少无效容量行的转换。

采集混合了前面的 387 次图关联 MoE 层调用和末尾 43 次非图调用。903 条 AllGather 时间区间重复；去重后的通信区间并集为 5.775 s。尾批退图与通信异常单独排查，不计入此转换 kernel 的收益。

## 路由契约

- allgather pre-permute 调用 init routing 时使用 `expert_tokens_num_type=1`、`row_idx_type=0`、`active_num=topk_ids.numel()`；GMM 使用 `group_list_type=1`。
- 令容量为 C、hidden size 为 H、设备端本地专家计数为 counts，R=sum(counts)。优化依赖有效专家结果形成 `[0,R)` 连续前缀，且本地反向映射位于该前缀。
- 非本地路由权重为零，反向映射指向 C-1。仅当 R<C 时才允许清零该 sentinel；R=C 时最后一行是真实结果，必须保留。
- 原版 Finalize 必须不读取未转换区域。验证用 NaN 污染 `[R,C)`（sentinel 除外）检查该契约，不以正常输出文本代替算子验证。

## 实现边界

按仓库 kernel-organization 规则，新增 `python/sglang/kernels/ops/moe/npu_valid_rows_cast.py`，提供写入既有 FP32 buffer 的 `cast_valid_expert_rows(src=..., dst=..., expert_tokens=...)`。MoE 组以 NPU-only 的 lazy KernelSpec 注册，CPU 和其他平台导入组时不会加载 Triton-Ascend 实现。接入仍限定于 NPU runner。

Triton-Ascend kernel 固定启动设备 vector core 数量的 program。每个 program 在设备端读取 counts 并求和，按有效 tile 数分配连续工作区间；仅转换 `[0,R*H)`，使用 64 位元素地址。program 0 在 R<C 时清零最后一行。不存在 `.item()`、D2H、动态输出形状或按完整容量启动大量空 program。

仅在显式设置 `SGLANG_OPT_NPU_MOE_VALID_ROWS_CAST=1` 且 runner 的 w2 为 `NPUW8A8Int8MoEMethod`、EP>1 时启用。开关在 runner 初始化时读取，经每次调用的 running_state 传给 allgather post-permute；BF16、NPU、连续完整容量输入才调用新 kernel。开关关闭、其他 dtype/量化方法、EP1、其他 transport 均保留原路径。全有效行在设备端正确处理，但不能在无 host 同步时按 R 自动回退原算子；该分布下不承诺提速。

第一版仍然分配 C×H FP32 buffer，因此不减少显存预留容量。对启用范围内的 eager 和 graph 都生效，不修改 attention、scheduler、通信算法或精度。

## 验证与限制

CPU 环境验证开关/接入分支、fallback、平台注册与导入边界、语法，并通过 Triton 解释器检查前缀转换、尾部保留和 sentinel 语义；不声称通过 NPU 编译、graph capture 或性能验证。

独立脚本的 init routing 使用未量化 BF16 输入，GMM 输出为模拟数据。它验证 Cast 与真实 routing/Finalize 的接口契约，但不替代完整 A8W8 GMM 和模型的验证。

独立 NPU 脚本应覆盖：

1. 用真实 init routing 确认本地映射位于 `[0,R)`；保留原 Finalize，与全部转换的 FP32 参考逐位比对。
2. 零有效行、部分有效、极不均衡、全有效，包括最后一行真实结果必须保留。
3. 无效区域预填 NaN，确认原 Finalize 不读取 sentinel 之外的未转换数据。
4. 同一张 NPUGraph 改变输入与 counts，从少到多、从多到少、变为零，检查地址稳定且结果正确。
5. 单独比较转换耗时与转换加 Finalize 耗时。partial 默认约 1/16 本地路由，另测全有效分布；轮换大输入，每个样本计时前清写独立 512 MiB buffer，并用 NPU Event 仅包围被测算子，不计入清写耗时。

有限 BF16 值转换为 FP32 不增加舍入步骤，但实际编译器行为、路由前缀布局和 Finalize 读取范围仍须硬件验证。未验证前不默认开启，也不预估固定加速百分比。
