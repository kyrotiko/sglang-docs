# 入图提交迁移与 V4.1 开发基线

日期：2026-11-16。后续 V4.1 prefill 入图开发以本文描述的已迁移代码为基线。

## 1. 基线与来源

| 项目 | 版本 |
| --- | --- |
| 开发分支 | `sglang/dsv41-a5` |
| 迁移前 | `3cc67a7544` |
| 迁移后 | `5633d68b04` |
| 迁移前备份分支 | `backup/dsv41-a5-before-prefill-graph-20261116` |
| 来源分支 | 本地保存的 `prefill_graph_ascend_ops_on_main@15443a991f` |
| 来源功能序列 | `254d3971ee..578dc4d230` 中的 11 个入图提交 |
| 已有上游重构 | `43d0ecf9e1`，device-neutral graph backend bases（#49402），迁移前已在开发分支中 |

GitHub API 和 Git TLS 连接在本轮失败，尚未核对远端是否存在更新的 PR HEAD。本文只对已保存的来源 SHA 和实际迁移结果负责，不将缓存分支描述为实时远端状态。

11 个提交均使用 `git cherry-pick -x` 迁移，保留原提交来源。净改动为 14 个文件，包含图实现和相应测试。来源分支末尾的 main merge 没有整体带入；当前分支保留 V4.1 的 DSpark、low-ratio 和 SWA layout 实现。

## 2. 提交映射

| 原提交 | 开发分支提交 | 内容 |
| --- | --- | --- |
| `f71ce61842` | `a32109ac9e` | NPU breakable graph backend、factory、segment class 和 capture-status 接入 |
| `708b953bb9` | `55d11ac436` | V4 固定 prefill metadata、compressor/indexer/attention 入图 |
| `5c88c18cca` | `3abd43a3cd` | Ascend Ops prefill MoE 入图 |
| `6693b61df5` | `246f359fe5` | idle DP rank 保留 SUM_LEN |
| `f437fbe02f` | `74af57f3d9` | 对齐当前 segment capture 测试接口 |
| `05dd3fc391` | `255c7cd268` | V4 prefill graph 放开 Ascend 950 条件 |
| `c36d40ecb5` | `956a7f64e0` | pre-commit 格式修复 |
| `53c83ddd34` | `3ec83f2875` | 用 graph scope 推导 MoE dispatch，移除早期 per-batch marker |
| `85378bf71e` | `a65339d2cd` | MoE 使用 prefill scope，capture 发布固定 AllGather 行数 |
| `4cafabd1a3` | `c46bef8692` | 启动时拒绝图内尚不支持的 ascend_ops AllToAllV |
| `578dc4d230` | `5633d68b04` | idle-rank eager 行为改为显式环境开关并补回归 |

迁移过程中出现一处内容冲突：V4.1 在 Ascend attention constructor 后增加的 `_validate_spec_support`、`_validate_draft_window_rows`，与入图提交增加的 constructor 初始化及 capability 方法位于相同插入点。解决后保留 V4.1 两个方法及其原有调用，同时将 prefill graph 初始化放在 constructor 内、capability 方法放在类方法区域。

## 3. 迁移后可以直接复用的能力

以下路径均相对 `python/sglang/srt/`，行号针对 `5633d68b04`。

| 已有能力 | 当前代码落点 | 后续使用方式 |
| --- | --- | --- |
| NPU prefill backend 解析 | `model_executor/runner_backend/utils.py:146` | 由现有 factory 返回 `BreakableNPUGraphBackend` |
| 设备 graph class/capture-status | `model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:95`、`:109` | NPU segment 使用 NPUGraph 和 portable query |
| breakable 生命周期 | `hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py:42` | 保留现有薄 NPU backend；其父类接入 device-neutral breakable 基类 |
| capture / replay metadata hooks | `hardware_backend/npu/attention/ascend_dsv4_backend.py:4033` | 沿现有 hook 为 V4.1 增加专用状态，不另起 runner |
| V4 固定容量与原地刷新 | `hardware_backend/npu/attention/dsv4_prefill_graph.py` | 复用固定 storage、所有权和刷新模式，扩展模型语义 |
| prefill MoE 分流 | `layers/moe/token_dispatcher/ascend_ops.py:181` | 使用 `is_in_prefill_graph_scope()`，不会因 extend 标记被清掉而落到 decode |
| 固定 AllGather 行数 | `hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py:76` | capture scope 发布固定布局，dispatch/combine 消费已有 row layout |
| AllToAllV 拒绝 | 同文件 `:45` 的 construction 校验 | 保留明确限制，后续扩展需要独立动态通信协议 |
| DP idle 策略 | `model_executor/forward_batch_info.py:1782` | 保留环境开关的默认行为，V4.1 多 rank 回归验证实际适用性 |

因此，迁移前文档中的“缺 NPU factory”“segment 仍为 CUDA”“MoE 缺 prefill scope”已不再是待实现任务。

## 4. 迁移后仍需实现的 V4.1 增量

**V4 的 prefill metadata 不能直接表示 V4.1。** `dsv4_prefill_graph.py:119` 只为 ratio 4/128 分配压缩状态，`:43` 的 kernel metadata keys 也是 V4 的 c1a/c4a/c128a/lightning-indexer 集合。V4.1 的 ratio-1/2 loc、index page table、window/压缩 top-k lengths、candidate 和 cycle-bank 等尚未纳入其绑定和刷新。

现有 Ascend capability 方法在 `ascend_dsv4_backend.py:2654` 沿用了 V4 条件。该条件允许进入 V4 capture 协议，不代表 V4.1 low-ratio 状态已获支持。评审后安排先隔离 V4-only 协议，并配对现有 source/attention eager 路由；固定 metadata 与数据流共同闭合后再启用对应捕获路径。这是待实现方案，当前主仓代码尚未增加闸门。

后续工作聚焦于：

1. 在既有 prefill metadata 生命周期中增加 V4.1 固定字段，复用 decode 的行公式和布局；不直接复用其每请求固定 token 数的 allocator，也不强制新增独立 owner/Struct。
2. 将共享 KV/index K、source/index/candidate 层和 QSLI consumer 连成正确的数据流；跨 eager/captured 边界确有需要时使用具体固定输出桥接。
3. 复用并核验现有 Engram hash/history eager break、固定 hash 输出及 mHC 状态链；NPU 使用共享 breakable scope，函数名中的 CUDA 不是设备限制。
4. 优先核实并复用保留请求槽 0 对 ratio-2 capture bank 的隔离，验证真实 bank 不受污染及 replay cycle table 刷新；不预设必须新增 bank。
5. 保证 target prefill 输出的 aux hidden、有效行与 token indices 可被 DSpark 正确注入 draft KV。
6. 将 context/page-table 上限落到现有 admission 和多 rank 决策，再验证首批配置的正确性和性能。

首个设备闭环仍限定 CP=1、DCP=1、PP=1、无 TBO、完整 decoder rows，并满足当前 NPU 每 rank 64 query heads 的约束。

## 5. 验证记录

开发容器：`sglang-npu-dev`，Python 3.12.13，torch 2.10.0+cpu，pytest 8.3.2；无 NPU 设备测试。

首轮迁移相关测试 **49 passed**，覆盖：

- `test_backend_resolution.py`
- `test_breakable_npu_graph_backend.py`
- `test_dsv4_prefill_graph_metadata.py`
- `test_ascend_ops.py`
- `test_mlp_sync_pad_unpad.py`

扩大回归时，将 NPU attention stub 测试与模型测试放在同一 pytest 进程，会在 collection 阶段出现 `cannot import name 'MoEOutput' from 'sglang.srt.models.deepseek_v2'`。该合并运行没有产生有效测试结果；后续改为每个文件独立 Python 进程，以隔离模块 stub。

独立进程回归结果如下：

| 范围 | 迁移后结果 |
| --- | --- |
| prefill runner helpers | 25 passed |
| prefill runner | 18 passed |
| prefill padding | 12 passed |
| 模型平台/mHC 分流 | 10 passed、1 failed |
| Engram history | 2 passed |
| NPU Ascend DSV4 attention | collection error：测试将 `sglang.srt.speculative` stub 为非 package，无法导入 `ragged_verify` |
| NPU V4.1 low-ratio attention | collection error：`sgl_kernel_npu` stub 无 `operators` package |

合计迁移后得到 **116 passed、1 failed、2 collection errors**（包括首轮 49 项；同进程合并运行的收集失败不重复计数）。

为确认来源，将迁移前 `3cc67a7544` 的 `python/` 和 `test/` 导出到容器中新建的 `/tmp/dsv41-prefill-baseline.EjTiKP/`，以该目录的源码及测试独立运行上述三个未通过文件。结果重现完全相同的两种 collection error，以及同一条 mHC failure：`deepseek_v4_mhc` 不再有测试所 patch 的 `is_npu_arch35` 属性。因此它们是迁移前已存在的测试适配问题，本轮未将其记为迁移通过项，也未修改这些无关测试。

`git diff --check` 已通过。CPU 结果只验证迁移和 Python 状态协议，不证明 V4.1 A5 整模入图或性能已通过。
