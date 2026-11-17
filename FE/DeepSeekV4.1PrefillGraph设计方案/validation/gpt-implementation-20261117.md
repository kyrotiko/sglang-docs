# dsv41-a5-graph-gpt 实现与 CPU 回归记录

日期：2026-11-17。实现分支 `dsv41-a5-graph-gpt`，从 `dsv41-a5@9898f7ce9af12fec5e2ef57d1adaa2b7533992ef` 新建，独立 worktree 为 `D:/workspace/SGLang/sglang-dsv41-a5-graph-gpt`。实现提交：`d3700bf0a9edad40dc30b3cc4e6ea9f810822f34`；本版尚未在 A5 运行整模，不能据此宣布配置已支持。

本版仅依据[本轮代码改动清单](../design/code_change_plan.md)开发，未读取或合入其他 agent 的实现分支和方案。测试使用本分支基线的独立归档副本，再同步本分支修改；没有使用主 checkout 中其他实现的源文件。

## 1. 实际改动

生产代码修改两个文件：

- `python/sglang/srt/hardware_backend/npu/attention/dsv4_prefill_graph.py`：在现有生命周期类中增加 V4.1 c1/c2 分支，维护固定 cu、CPU 镜像、live mask、window/cmp 长度、full/SWA/index 页表、positions/loc、cycle request-bank 和 SWA loc；metadata 按实际 planner key/shape/dtype 分配和校验。原 V4 c4/c128 explicit-state 路径保留。
- `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py`：配对设置 source 入图和 captured metadata 开关；capture 使用保留 bank 0，并在现有 NPU capture hook 中初始化该 bank；replay 用 live 信息刷新固定状态，再结合静态 bucket 重建调度 metadata；明确拒绝本轮未捕获的 forward 分支。

测试扩展在现有 `test_dsv4_prefill_graph_metadata.py` 和 `test_npu_dsv41_low_ratio_backend.py`。没有新增公共 reset hook，也没有修改 `base_attn_backend.py`、`prefill_cuda_graph_runner.py`、模型路由、allocator 或上游算子库。

与方案字段表的实现细节：index 页表容量从 eager `expand_index_page_table` 已产生的 full/index 列数比例推导，复制本轮 eager 页表并清尾部，保持既有 index PA 配置；没有把 candidate block 8 当成 index PA page。c1/c2 page table 共用固定 full table。

## 2. CPU 环境与结果

运行容器：`sglang-npu-dev`；Python 3.12.13、torch 2.10.0+cpu、pytest 8.3.2。无 NPU/torch_npu，未加载模型权重。独立副本：`/tmp/dsv41-graph-gpt-La12tl1k`，`PYTHONPATH` 指向其 `python/`；执行前确认 `sglang.__file__` 来自该目录。

每个文件单独启动 pytest 进程，避免已有模块 stub 串扰。

| 测试文件（均在 `test/registered/unit/` 下） | 通过数 |
| --- | ---: |
| `model_executor/runner_backend/test_backend_resolution.py` | 7 |
| `npu/graph_runner/test_breakable_npu_graph_backend.py` | 8 |
| `npu/graph_runner/test_dsv4_prefill_graph_metadata.py` | 5 |
| `moe/test_ascend_ops.py` | 14 |
| `model_executor/test_mlp_sync_pad_unpad.py` | 16 |
| `model_executor/test_prefill_cuda_graph_runner_helpers.py` | 25 |
| `model_executor/test_prefill_cuda_graph_runner.py` | 18 |
| `model_executor/runner/test_prefill_cuda_graph_padding.py` | 12 |
| `npu/attention/test_npu_ascend_dsv4_backend.py` | 63 |
| `npu/attention/test_npu_dsv41_low_ratio_backend.py` | 106 |
| `models/test_deepseek_v4_platform_dispatch.py` | 11 |
| `layers/test_engram_history.py` | 2 |
| **合计** | **287** |

新增 6 个 case，覆盖不同的性质/生命周期：

1. vector/cu 容量越界必须拒绝，不能截断后继续 replay。
2. 同 bucket 大→小→大、不同 prefix/bank，固定地址和页表别名不变，c1 保序、c2 compact、padding 清理及调度 metadata 内容刷新。
3. capture 仅初始化 bank 0；两个 bucket 的 CPU 镜像互不覆盖；DP padded 行数不被当成真实 query 数。
4. kernel metadata shape/dtype/key 改变时拒绝继续复用旧协议。
5. token/context 容量及未捕获 forward 分支拒绝，capture 不允许误用真实请求槽。
6. source 与 attention 的使能配对，draft/关闭图保持原路由，decoder bounded replay 与本轮配置明确不兼容。

外部算子与 base pool planner 用 CPU fixture 隔离；query/sequence metadata、row helper、window/cmp、QLI/QSLI planner 调用和 BCG hooks 执行实际本分支逻辑。上述测试不能证明设备 capture 成功。

Python 编译、Ruff format、修改文件的基础 lint 和 `git diff --check` 通过。low-ratio 测试文件按原有 fixture 需要先加载模块再安装 stub，基线已有 12 条 E402；已确认基线同样报出这 12 条，保留该导入顺序并在该文件 lint 时排除 E402。未新增此类例外。

## 3. A5 接入验收重点

下一次设备验证针对本版新增协议，不需要先重复第四轮 writer dtype 探针：

- 真实模型 source→compressor/RoPE/writer→indexer/candidate→attention 的图内覆盖，及各 stream 的写读顺序。
- 同 bucket 改 prompt/live 长度/prefix，多轮大后小与小后大；检查 logits、hidden states、有效输出和实际 KV/index 缓存。
- ratio-2 奇偶 chunk、跨 chunk pending pair、真实 request-bank 切换/复用，capture 前后非零 bank 不变，下一次 decode 对齐。
- 两组 source/consumer 映射，避免上一组 candidate/top-k 被下一组覆盖。
- 使用实际固定 AllGather 拓扑检查 rank 一致 admission，DP idle 保持已有 eager 回退。
- 普通 prefill 对照闭合后，回归 DSpark target→draft 的 KV/aux hidden 与后续生成，再做模型准确率和性能 A/B。

第三/四轮独立 CPU FP32 attention reference 的 FAIL 仍保持原状态；本次 CPU 回归和 source 路由接入没有解决或重新判定它。

## 4. 首轮 A5 runtime 反馈与修复方向

用户在 `d3700bf0a9` 的 A5 环境报告服务启动成功，2026-11-17 10:39:42 首次进入所贴 graph 路径时，多数 DP rank 在 `_fill_low_ratio()` 因 idle/容量保护退出，DP1 的退出栈则在图内 eager MoE break 的 DeepEP dispatch 等待通信。失败发生于报错 rank 的 Python metadata 准备阶段，不能据此认定 DSL kernel capture 失败。

已定位的协议缺口：`_local_prefill_cuda_graph_vote()` 默认让 idle/无 batch rank 投 True；`prepare_mlp_sync_batch()` 随后会将 idle 改成 dummy EXTEND，并把 token 计数改成 MAX_LEN bucket。这样 runtime `_has_inactive_dp_rank()` 可能看不到原来的 0。DeepEP 在 `require_mlp_tp_gather=False` 时还会只保留本 rank 计数，不能靠 runtime 本地长度表保证跨 rank 一致回退。

曾提出把 V4.1 NPU 的 idle rank 改成调度投票 False、让整组 eager 回退。**该策略未经设备能力验证，用户已要求停止推进，先验证 idle replay。** 未提交草稿已另存 `/tmp/dsv41-idle-veto-draft-hLihM8.patch`，源码 worktree 恢复到已发布的 `d3700bf0a9`，没有提交或推送 base/runner/admission 改动。

本地新增投票回归确实在旧代码上复现了不支持标记未参与集体决策（True != False），但这只能证明原保护与准入边界不一致，不能证明设备或完整协议不支持 idle，也不能据此决定整组 eager。后续以设备事实决定实现。

验证入口改为单文件脚本 revision 5 的 `idle_chain_r0/r1/r2`：同 bucket live→idle→live，公开 writer、ratio-2 compressor、index-K/QLI/QSLI 与 attention 合成链在图内执行。测试采用当前 dummy EXTEND 的物理 bucket cu/KV 长度，logical seqused/sparse lengths 为 0；检查实际 cache/state 的字节级保护。使用私有合成缓存和 identity RoPE，不修改服务代码，不宣称多卡 DeepEP、真实 metadata hooks 或整模已经通过。命令与判读见[运行说明](../scripts/README.md)。本地编译、CPU 几何自检及三组控制流 fixture 已通过；fixture 结果不计作 NPU 证据，实际 A5 结果待回传。

