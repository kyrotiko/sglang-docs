**基于 92b02637：DSV4 prefill compressor / indexer / attention 全链路入 NPU 图**

分析基线：`sglang` 的 `92b02637f7732b570fe21ef03810182e4d75ef40`。
范围：`DeepseekV4AscendAttnBackend` 支持的 ratio 0 / 4 / 128，讨论 prefill；不把 V4.1 的 ratio 1 / 2 纳入已有能力。
用户明确要求：主 compressor、indexer 内部 compressor、indexer 和 sparse attention 全部进入图；图外只保留 replay 前的 metadata planning / 原地刷新。单算子实验可以拆开进行，最终实现不保留这些计算的 eager 断点。
方法：源码静态分析，并结合用户提供的 NPU 实测日志；各项验证范围和结果见第 12 节。本地 Windows 环境未运行 NPU 测试。

**1. 结论与基线边界**

`92b02637` 接通的是分段图框架，不是 DSV4 attention 的入图实现：

- `BreakableNPUGraphBackend` 仅继承公共 BCG backend，不重写 capture / replay；其文件说明明确让 attention 和 host-input 算子在段间 eager 执行。
- segment 换成 `torch.npu.NPUGraph`，使用裸 `capture_begin/capture_end`，未接入自动 dispatch / task update。
- `models/deepseek_v4.py:2887` 对 EXTEND + BCG 无条件走 `bcg_deepseek_v4_attention_with_output`。NPU 的非 unified 路径也会进入这里，sparse attention 核心不会被捕获。
- NPU DSV4 没有启用 BCG 的 captured-metadata 契约；replay 仍通过 `init_forward_metadata` 重建 metadata。
- compressor / indexer 在该基线里没有 NPU prefill 专用断点，因此它们会尝试进入 segment，但地址和动态形状尚不满足 replay 契约。

当前工作分支在基线之后有两笔提交，不能混作基线能力：

| 提交 | 实际作用 | 与目标的差距 |
| --- | --- | --- |
| `d9564f9499` | 给 c4 indexer 和主 compressor 添加 eager 断点；修 capture 缺 loc 和 live/bucket RoPE 长度不一致 | 能让 prefill graph 路径运行，但这些计算仍在图外 |
| `a153685607` | 实验性关闭上述两个 NPU 断点 | 没有修改公共模型的 attention 断点，也没有建立固定 metadata 契约，不能据此认定实现了全 attention 入图 |

最终交付覆盖以下全部计算：

| 层形态 | 必须进入图的计算 | 额外状态 |
| --- | --- | --- |
| ratio 0 | Q/KV 投影、norm / RoPE、SWA cache 写入、sparse attention、输出投影 | SWA KV |
| ratio 4 | 再包含 C4A 主 compressor、C4Li indexer compressor、indexer Q/weights/量化、lightning indexer、压缩后处理和 KV/scale 写入 | 两套 compressor state、main/indexer KV、top-k |
| ratio 128 | 再包含 C128A 主 compressor、压缩后处理和 KV 写入 | compressor state、c128 KV |

因此 `d9564f9499` 的两个 eager 断点不能作为最终实现；其 capture 占位与长度问题的修复只能按新的固定容量协议重新吸收。`92b02637` 的 BCG 框架仍可作为起点，DSV4 attention 链内部没有 eager break 并不要求立即实现新的 FULL backend。

host 侧计算 metadata，并不意味着读取该 metadata 的 attention kernel 必须 eager；关键是图绑定的存储是否稳定，动态信息是否真正由 device 读取。

**2. 先改变模型的实际分发入口**

基线定位：`python/sglang/srt/models/deepseek_v4.py:790, 841, 2887`。

应为已完成适配的后端提供明确能力标记，使 NPU 的 prefill 路径直接调用可捕获的 backend forward；其他后端继续使用自己的入口。

不能只去掉 `eager_on_graph` 装饰器、或者把现有断点函数当作普通函数调用。`deepseek_v4_attention_with_output` 会读取 CPU 上的 `global_num_token_non_padded_cpu`，切 `query[:real_num_tokens]`，并在零行时走 Python 分支。这些选择在 capture 时固定，不能服务同 bucket 的不同真实 token 数。

图内入口需要接收 bucket 形状的 Q / 输出，实际请求布局由固定 metadata 的内容表达；padding 输出必须有确定值或被设备侧 mask，不能依赖 replay 时再次执行 Python 切片。

**3. 接入 BCG 已有的 captured-metadata 契约**

基线定位：

- `layers/attention/base_attn_backend.py:234, 328, 335`：能力标记与两个 hook。
- `model_executor/runner/prefill_cuda_graph_runner.py:695, 1326, 1354`：按 ShapeKey 保存 metadata，并在 replay 前刷新。
- `hardware_backend/npu/attention/ascend_backend.py:568`：每次创建新的 `ForwardMetadata`。
- `hardware_backend/npu/attention/ascend_dsv4_backend.py:1864`：继续新建长度张量、kernel metadata 和 compress metadata。

建议在 NPU DSV4 后端启用 `use_captured_forward_metadata_for_breakable_cuda_graph`，实现：

```python
init_forward_metadata_for_breakable_cuda_graph_capture(forward_batch)
prepare_forward_metadata_for_breakable_cuda_graph_replay(
    capture_metadata, forward_batch, *, static_forward_batch=None
)
```

capture 为图读取的字段分配固定存储并保活；replay 在图外规划当次数据，用 `copy_` / 原地填充刷新这些存储，随后将 active `forward_metadata` 指向对应 bucket 对象。可以建立临时 planning 对象，但不能让图仍读取旧地址、Python 却改指向新对象。

需要覆盖的字段分两组：

| 范围 | 必须建立稳定存储和更新规则的字段 |
| --- | --- |
| attention 核心 | `actual_seq_lengths_q_pa`、`actual_seq_lengths_kv`、`swa_page_table`、`c4_page_table`、`c128_page_table`、`kernel_metadata[c1a/c4a/c128a]`、`c4_topk_indices` |
| compressor / indexer | `actual_seq_lengths_q`、`kernel_metadata[li_quant_metadata]`、`positions_cmp_padding_c4/c128`、`start_pos`、`seqused`、`c4_loc/c128_loc`、state block tables；SWA 写入相关 loc 也须保持正确绑定 |

现有 decode 实现值得复用：`_init_dsv4_graph_buffers`、`_copy_page_table_into_graph`、`_copy_1d_with_zero_tail`、`_refresh_graph_kernel_metadata` 已有固定页表、固定长度 metadata 和原地更新的做法。

但不能直接调用 decode 的整套初始化：`_init_dsv4_graph_metadata` / `_build_dsv4_graph_replay_ctx` 假设每请求一个 token，或固定数量的 speculative token，且 Q 的前缀和在该图形状下不变。prefill 每次请求数、每请求 Q 长度、prefix 长度都会变化，需要独立的 prefill 初始化 / 刷新逻辑和缓存命名空间，避免与 decode 的 `graph_metadata[bs]` 混用。

**4. 图容量需要同时约束 token 轴和请求轴**

基线定位：`prefill_cuda_graph_runner.py:1639` 的 capture dummy batch 用最少请求拼出 token bucket；固定 request slots 的分支目前用于 FULL。

例如同为 128 个 token，capture 可以是 `[128]`，真实请求却可以是 `[64, 64]`。前者 `cu=[0,128]`，后者 `cu=[0,64,128]`。仅固定 token 数不能固定 `cu`、页表第一维和 state table 第一维。

必须选择并实现一种请求轴策略：固定 `B_capacity`，或者把请求数 bucket 纳入图的 variant。固定容量方案还需要：

- 固定 `cu` 的 `B_capacity+1` 个元素和页表的 `B_capacity` 行，并定义空 request slot 的合法表示。
- 为 context / page-table 宽度设置上限；超出请求数、上下文、压缩容量的 batch 应在 replay admission 阶段回退。
- 设置与实际容量一致的 `prefill_cuda_graph_max_batch_size`，使 DP 的 replay 选择也遵循这一约束。
- 每次清掉无效页表列、未使用的请求行、top-k 和 loc 尾部，验证“大 batch → 小 batch”不会读取上一轮残留。
- 如果算子不接受零长度 dummy request，需要单独设计合法 dummy 请求及隔离存储，不能仅填零后假定安全。

**5. c4 top-k 必须连接到同一份图内数据流**

基线定位：`ascend_dsv4_backend.py:893-923` 的 indexer 将输出赋给 `self.forward_metadata.c4_topk_indices`；`:2193` 的 attention 从中读取。

indexer 与 attention 都在图内，需要固定 `[T_bucket, index_topk]` 的数据流。可以由 indexer 的捕获输出直接连接 attention，也可以用图内 `copy_` 写入预分配的稳定 top-k buffer；只有实际需要时才增加 `out` 接口。有效行以外填 `-1`，同流按程序顺序消费；多流必须有明确的依赖。

capture 时一次 Python 赋值本身并非必然错误：只要该输出和 attention 输入已经形成同一条捕获的数据流，每次 replay 都会产生新内容。需要避免的是后续更换 metadata / top-k 存储，却没有更新图绑定；也不能依赖 replay 时再执行 Python 属性赋值。

主流路径应使用 fused quantized lightning indexer。BF16 fallback 包含按请求循环、从 tensor 转 Python 整数和动态 top-k 宽度，不应直接宣称可用于通用 prefill 图。

**6. compressor 全入图需要改为“固定容量 + 有效数量”**

基线定位：`ascend_dsv4_backend.py:245-332, 442-534, 554-634`。

当前 metadata 在图外按 live batch 计算；这一步的 `.cpu().tolist()` 可以暂时保留在 replay 前，后续再用已有 CPU 长度镜像消除重复 D2H。真正的捕获约束是后续 kernel 所读地址、shape 和分支不能随 batch 改变。

设压缩比例为 r，每请求本次 prefix 和 extend 长度为 p_i、e_i，本次实际新增压缩块数为：

```text
N_r = sum(floor((p_i + e_i) / r) - floor(p_i / r))
```

它受 prefix 的余数和请求划分影响，不仅由总 token 数决定。当前 `cmp_kv[:loc.numel()]` 把这个数量固化成 capture 的切片长度，必须改造。

需要完成：

- 依据实际 compressor ABI，为每个 bucket 固定输出 / RoPE / loc 容量。当前 TH 形态的容量公式是 `min(T, T // r + B)`，不能将此容量与实际有效块数 `N_r` 混为一谈。
- 统一 X、cu、RoPE 和输出容量的语义。真实 X=64、bucket=128 时，沿用 live RoPE 17 行而图录制的 X 是 128 行，会再次出现 17/33 的口径冲突。第 12 节已验证特定 A3 环境的 C4Li / C4A / C128A 支持 `cu[-1] < X.shape[0]`；其他形状 / 环境仍需验证。
- epilog 使用固定 shape 和设备侧有效性信息，不再按 live `loc.numel()` 动态切输出或通过 boolean indexing 改变形状。
- `out_cache_loc_dsv4` 的 c4/c128 loc 按本次真实 bundle 刷新到固定存储；padding 写入只能指向已确认隔离的 dummy / padding 存储，不能污染真实 KV。
- A3 的 explicit state table 宽度按可重放的最大输入容量固定，并用 dummy state loc 填尾；A5 使用 request-bank cycle table，不能套用 A3 的二维 table。
- 除 KV epilog 外，还要保护 compressor 的 `state_cache` 写入。warmup / capture 使用的 bank 和状态需要隔离或恢复；“KV loc 设 0”并不自动保护 state。
- 在 capture 前准备好 Hadamard、融合权重视图和 RoPE cache，避免惰性初始化改变捕获路径。

`d9564f9499` 对 capture 缺 bundle 时补零 loc 的修复可用作启动准备的参考，但全入图时仍要按固定输出容量设计，不能只复制其 live 长度的占位数组。

进一步核查本地 `sgl-kernel-npu` 的 A5 compressor：

- `op_kernel/arch35/compressor_tools.h:74, 88, 98` 从设备读 `seqused`、`start_pos` 和相邻 `cu_seqlens`；分别表达有效 token 数、全局起点和请求输入容量。
- `op_kernel/arch35/compressor_kernel.h:246` 同时计算容量对应的 `tcNum` 和有效数量对应的 `compressedTcNum`；后者正是上面的新增压缩块公式。
- host 的 `op_host/compressor.cpp:65` 根据 X 的 shape 和 cu 的元素数量分配输出容量；`op_host/arch35/compressor_tiling.cpp:935` 检查 RoPE 与这一容量一致。

这为“固定 shape、动态设备 metadata”提供了源码依据，不能直接断言 compressor 必须留在图外或必须重写内核。实现时应分别核对 compressor 输入容量的 cu 与 attention/indexer 实际 Q 长度的 cu；必要时拆成两个图内固定存储，避免复用同一字段却要求相反的尾部语义。仍需实际验证 padded T / dummy request / idle 在部署算子中的行为；本地算子版本不能替代实际绑定实现的证明。

**7. graph.update 是条件需求，不是 DSV4 的必选前置改造**

基线定位：`hardware_backend/npu/graph_runner/npu_cudagraph_backend.py:435, 484-517`。

现有 decode backend 虽然开启 `auto_dispatch_capture=True`，但普通 DSV4 replay 默认跳过 host-length 的 `replay_with_input_update`；动态信息由稳定 tensor metadata 表达。这说明 prefill 应先复用 tensor metadata 路径，而不是预先断言必须改 torch_npu 或创建一种新 graph backend。

本地 torch_npu（检查时 HEAD 为 `286bee5fc1`）中：

- `torch_npu/npu/graphs.py:655` 要求 `auto_dispatch_capture=True` 才允许 `NPUGraph.update`。
- `graphs.py:456` 仅对有 handler 的算子创建 dispatch records。
- 本地内置 handler 列表未包含这里使用的 DSV4 sparse attention / compressor / quant lightning indexer；扩展包或部署版本是否另外注册，需要核查实际环境。
- BCG 的每段用裸 capture，不会自动进入 `graph_dispatch_mode`，也没有逐段 update 的 replay 协议。

如果目标版本的算子把动态长度 / tiling 计算结果固化到 host launch 参数，仅刷新 device tensor 不能改变它们，届时才需要补：每段 dispatch capture、算子 handler、稳定输出和 workspace、按实际算子出现顺序对应的 update payload、update / replay 流依赖。也可以明确使用底层 task-group update 协议；不能只给 segment 设一个布尔开关。

Ascend 官方机制说明可交叉参考：[NPUGraph](https://github.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/npugraph.md)、[handler 自定义接口](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch_npu-npu_npugraph_handlers.md)。实际决策以部署版本和单算子实验为准。

补充检查了工作区 `sgl-kernel-npu` 当前源码（HEAD `e2dec25`），该版本不等同于基线部署依赖：`csrc/utils/ge_helper.h` 有 graph capture 的 tiling cache，要求同 shape / dtype / optional inputs / attrs 先 warmup；其容量、tiling 地址保活和 workspace 大小也需要验证。这提供了可捕获实现的线索，不能代替对 `torch.ops.custom.npu_*` / `torch.ops.npu.compressor` 实际绑定实现的核对。

**8. 并行、padding 与 admission 需要一起验收**

- DP 的 idle rank 仍参与 MoE 通信，但没有真实 attention / cache 写入。capture 的非 idle Python 分支不会在 replay 自动切换，必须让 device metadata / mask 或独立 graph variant 表达 idle；非法零长度 metadata 的 kernel 不能直接启动。
- 核对 `T_global`、rank-local T、bucket T 和 cu 末项的关系。`global_num_token_non_padded_cpu` 不能不加分析地当成任意 TP / CP 配置下的本地行数。
- 多流验证 Q 准备、indexer compressor 写 cache、lightning indexer 读 cache、attention 读 top-k 的先后关系，以及 segment 结束的 fork / join。
- 基线 `layers/cp/bcg/__init__.py` 的组合表没有 DSV4，不能只添加名字后宣称支持；CP 的 local/full metadata 切换目前包含重新绑定，须另外适配。首版应明确拒绝尚未验证的 CP / TBO 组合。
- sparse attention 可能只写有效 query 行，padding 输出应确定化，避免未初始化值传播到后续量化和 MoE。

**9. 全链路的固定容量协议与代码落点**

建议每个 prefill 图维护独立状态对象，容量由 `T_capacity`、`B_capacity`、context/page-table 宽度决定，必要时通过 ShapeKey variant 区分请求数。r=4/128 的压缩容量 `C_capacity[r]` 必须匹配目标算子的实际输出 shape。

| 状态 | capture 时固定 | replay 前刷新 | 图内行为 |
| --- | --- | --- | --- |
| X / positions | bucket shape 与地址 | 输入有效行及 padding | Q/KV/indexer/compressor 均读取固定形状 |
| 请求与长度 | cu、seqused、start_pos、req/state table 的 shape 与地址 | 本次输入布局、实际有效数、prefix、request bank | 内核从设备读取长度和 state 映射 |
| 压缩输出 / RoPE | `C_capacity[r]`，与 X/B 容量一致 | 固定长度的压缩 positions，尾部合法填充 | compressor、RoPE、Hadamard / 量化以固定 shape 执行 |
| 写回 loc / 有效 mask | `[C_capacity[r]]` | 真实 `N_r` 个新增 loc，其余按协议标记无效 | 固定形状的 epilog / scatter；无效行不得写真实 cache |
| indexer top-k | `[T_capacity, index_topk]` 数据流 | 必要的初始 sentinel | lightning indexer 写，sparse attention 读，padding 为 -1 |
| kernel metadata / 页表 | 形状、地址、所有权 | 从本次 planning 结果原地复制，清理尾部 | 全链路只读稳定存储 |

`N_r` 可以利用调度器已有的 CPU prefix/extend 长度在 replay 前计算，不必从 compressor 输出再同步回 host。图内使用固定 mask / loc 协议处理有效行，而不是按 `N_r` 做 Python 动态切片。对于内核未写满的输出容量，在量化和写回前用设备侧选择操作将无效行确定化；直接乘零不能清除 NaN。epilog 若无法忽略无效行，需扩展其 mask/有效数接口，或使用经验证安全的独立 dummy 存储，不能把重复 sentinel 写入的安全性当作既定事实。

主要代码改动：

1. `ascend_dsv4_backend.py`：实现 prefill captured-metadata hooks、容量与更新；图内主/indexer compressor、完整 epilog 与 fused indexer 数据流；分开 eager 和 graph 的形状处理，避免影响 decode。
2. `models/deepseek_v4.py`：增加已适配后端的能力分发，绕过公共 sparse attention eager break；调用固定形状的 backend 路径。
3. `prefill_cuda_graph_runner.py`：必要时支持 NPU BCG 的固定 request slots / variant 与容量 admission；复用现有 metadata 保存和刷新 hook。
4. `dsv4_prefill_graph.py`：这些计算不再通过该文件的 eager wrapper 执行；回归时验证确实没有遗留调用。
5. torch_npu / kernel 仓库：仅当目标算子的单算子验证暴露捕获、padding、epilog、tiling 或动态参数能力缺口时修改对应实现，三仓独立提交。

**10. 实施顺序与验证矩阵**

| 阶段 | 工作 | 验收重点 |
| --- | --- | --- |
| A | 单算子验证 C4A / C4Li / C128A compressor、量化和 epilog、fused lightning indexer、sparse attention 的 capture / replay | 固定容量下改变有效数、prefix、cu、页表与 state bank，确认实际算子能力和需补的 kernel 接口 |
| B | 实现 prefill 固定状态与 replay 刷新，在单层串起完整 compressor → epilog → indexer → attention | 不留 eager 计算断点；确认数据流、缓存和 state 副作用正确 |
| C | 集成模型能力分发与 runner 容量约束，覆盖 ratio 0/4/128 全模型 | 变 token/请求数/前缀的同图 replay 正确，后续 decode 正确 |
| D | 扩展 DP idle、多流、attn-TP，再逐项处理 CP/TBO | 无非法写入、流依赖完整、通信顺序一致；性能收益由实测确认 |

必要的测试应覆盖：

1. 同图连续 replay 不同输入，改变 req id / 页表 / prefix，确认读取的是当次数据。
2. 同一个 128-token bucket：128 行与 64 行、`[128]` 与 `[64,64]`，再测试“大 batch → 小 batch → 大 batch”。
3. 压缩边界：3/4/5、127/128/129；前缀未对齐、无新压缩块、有新压缩块和多个 chunk。
4. 不只比较最终 logits，还比较每类层 attention 输出、top-k、压缩 KV、量化 scale、state cache，并验证后续 decode。
5. 在 metadata 刷新与 replay 间检查被图读取 tensor 的 `data_ptr` / shape / dtype 不变，尾部 sentinel 正确。
6. A3 / A5 分别验证；多流开启与关闭分别验证；DP 下一个 rank idle 也要验证。
7. 超过 request / token / context / compressed-capacity 上限时正确回退。
8. profiler 区分 host metadata planning 和 graph replay；确认主 compressor、indexer compressor、量化与 epilog、lightning indexer、sparse attention 都位于捕获段中，DSV4 attention 链内没有 eager break。不能以启动成功或文件名含 full-capture 作为入图证明。

实现主要落点是 `sglang` 的 NPU attention backend、公共模型的能力分发以及必要的 prefill runner 容量约束。只有单算子实验确认底层捕获 / 动态参数机制不足时，再分别修改 torch_npu 或对应 kernel 仓库。

**11. 下一步：先取得可信的动态 replay 结果**

第一项可执行工作是运行 `scripts/spikes/` 下的 [compressor_graph_replay_spike.py](../scripts/spikes/compressor_graph_replay_spike.py)，先验证三种 compressor 的固定容量协议。脚本内置符合 A3/A5 state-table ABI 的合成输入，直接调用安装环境的实际算子；不加载模型权重。需要复用某个 kernel checkout 的 CPU fixture 时，可额外传入 `--kernel-repo` 的真实路径。

```bash
python compressor_graph_replay_spike.py \
  --variant c4li --op npu.compressor --report c4li.json
```

算子命名空间和注册模块必须与服务一致。基线 SGLang 的 A3 路径选 `npu.compressor`，A5 路径选 `custom.compressor`。脚本在算子缺失时分别默认导入 `sgl_kernel_npu` / `custom_ops`；可用 `--import-module MODULE` 显式覆盖注册导入。注册完成后才初始化设备，并与服务一致设置 `jit_compile=False`。如果模块存在但目标算子仍缺失，报告注册阶段错误及实际可用的 compressor 命名空间，不进入 capture。

`--kernel-repo` 已改为可选；若使用，脚本在导入 NPU 库之前检查真实路径，不接受直接照抄 `/path/to/...` 的占位路径。checkout 仅提供 fixture 源码，其 HEAD 不代表实际运行的二进制；脚本记录 torch/torch_npu 版本、设备、注册模块路径、算子 schema 与可选 fixture 版本，定位结果时还应保留服务依赖版本。

先运行 C4Li，再把 `--variant` 分别改为 `c4a`、`c128a`；每次新进程运行。默认 token 容量 256、请求槽数 2、hidden 1024，A5 使用 `coff * ratio` 的小 ring。默认使用与 P1 segment 相同的裸 capture API，不调用 `graph.update()`；可用 `--capture-api context` 定位 capture API 差异。

验证方法：每个 case 从同一个初始 state 分别执行 eager 和 graph，比较真实新增压缩块对应的有效输出，以及完整 state cache。cases 覆盖有效长度缩短、cu 请求划分改变、cu 末项小于 X 容量、单请求、跨压缩边界、无新块、idle，最后回到初始 case；所有图外输入 buffer 的地址、shape、dtype 保持不变。

原有 `cu_rebind_spike.py` 用不同 cu 的两次输出是否相等推断参数是否固化，这个判据不足：无效输出行可能保留旧值，kernel 可能读 `seqused` 限制输出，state 又会跨调用累积；单独修改 cu 还可能使 `seqused` 超过该请求容量。新脚本不改动该文件，改用每个 case 对照 eager 的判定。

结果决定后续动作：

| 结果 | 后续动作 |
| --- | --- |
| 三种 variant 全部 `passed=true`，有效输出和 state 均匹配 | 固化已验证的 T/B/cu/seqused 协议；实现 prefill metadata hooks，并在单层中连接图内 epilog、fused indexer 与 attention |
| 某种布局 eager 已报错 | 先确认该布局是否违反算子 ABI，调整 dummy/capacity 表达或加 variant；不能由此认定 graph 固化了参数 |
| eager 正常、graph 对照失败 | 用报告定位是 cu、有效数量还是 state 映射引入的差异，缩小成单算子 repro，再决定补底层协议或图任务更新 |
| raw capture 失败、context capture 成功 | 先定位 segment capture 的流、warmup、tiling 生命周期差异 |

`passed=true` 表示相对于同一算子的 eager 结果满足 replay 一致性；后续仍要对照 kernel reference / 模型精度，并验证连续 chunk 的状态累积。全链路最终验收要求 compressor、量化、epilog、indexer、attention 均在图内。

本地已完成脚本语法/CLI 和三种 variant 的 case 矩阵校验；当前 Windows 环境没有 torch / torch_npu。用户提供的 NPU 结果记录在下一节。

**12. 实测记录：A3 三种 compressor 动态 replay 已通过**

证据来源：用户于 2026-11-04 提供的完整终端输出，运行命令为：

```bash
python compressor_graph_replay_spike.py \
  --variant c4li --op npu.compressor --report c4li.json
python compressor_graph_replay_spike.py \
  --variant c4a --op npu.compressor --report c4a.json
python compressor_graph_replay_spike.py \
  --variant c128a --op npu.compressor --report c128a.json
```

| 项目 | 实测配置 |
| --- | --- |
| 设备 | `Ascend910_9362`，脚本采用 A3 explicit state table |
| PyTorch / torch_npu | `2.10.0+cpu` / `2.10.0.post6` |
| 算子 | `npu.compressor`，导入安装环境的 `sgl_kernel_npu` 注册 |
| 输入 | 内置合成输入；T capacity=256、B slots=2、hidden=1024 |
| 压缩配置 | C4Li / C4A / C128A，参数见下表 |
| 捕获 | `raw`，不启用自动 dispatch capture，不调用 `graph.update()` |
| 比较方式 | 每个 case 先恢复相同初始 state，对照 eager 的有效输出和完整 state |

三种 variant 各 9 次、合计 27 次 replay 的 `output_ok`、`state_ok` 全为 true，两项最大绝对误差均为 0：

| variant | ratio | coff | head_dim | 固定输出容量 | cases | 输出 / state 最大误差 |
| --- | --- | --- | --- | --- | --- | --- |
| C4Li | 4 | 2 | 128 | 66 | 9 | 0 / 0 |
| C4A | 4 | 2 | 512 | 66 | 9 | 0 / 0 |
| C128A | 128 | 1 | 512 | 4 | 9 | 0 / 0 |

| case | capacities | used | start_pos（ratio 4） | C4Li / C4A 有效行 | C128A 有效行 |
| --- | --- | --- | --- | --- | --- |
| baseline | [128,128] | [128,128] | [0,0] | 64 | 2 |
| less_used | [128,128] | [64,32] | [1,3] | 24 | 1 |
| cu_repartition | [192,64] | [192,64] | [3,3] | 64 | 2 |
| cu_short_total | [64,64] | [64,64] | [0,3] | 32 | 1 |
| one_request | [256,0] | [128,0] | [3,0] | 32 | 1 |
| cross_boundary | [128,128] | [1,1] | [3,7] | 2 | 2 |
| no_new_blocks | [128,128] | [1,1] | [0,0] | 0 | 0 |
| idle | [0,0] | [0,0] | [0,0] | 0 | 0 |
| baseline（恢复） | [128,128] | [128,128] | [0,0] | 64 | 2 |

C128A 的相应边界起点为 127 / 255；完整 case 参数由脚本的 ratio 驱动。

这份结果支持的结论：在该设备、安装版本及测试形状下，三种 compressor 使用固定 shape / 地址时，可以通过原地修改 cu、seqused、start_pos 和 RoPE 内容正确重放；包括 cu 末项小于 X 容量、逻辑单请求、无新压缩块和 idle。对这些 case，不需要通过图任务更新动态长度，因此当前适配可以继续沿稳定设备 metadata 的路径推进。

这一轮单算子测试未覆盖 epilog；其后续实测见第 13 节。C4 fused indexer 与 sparse attention 的后续同图实测见第 16 节。仍未覆盖其他形状、设备/安装版本、连续 chunk 的状态累积、完整模型精度与性能。单算子零有效输出的两个 case 仅以完整 state 一致性作为比较依据，不表示 padding 输出已经清零。

**13. 实测记录：A3 compressor 与图内 epilog 全部通过**

用户于 2026-11-04 在同一环境完成三种 variant 的 `--with-epilog` 验证，对应命令为：

```bash
python compressor_graph_replay_spike.py --variant c4li --op npu.compressor --with-epilog --report c4li_epilog.json
python compressor_graph_replay_spike.py --variant c4a --op npu.compressor --with-epilog --report c4a_epilog.json
python compressor_graph_replay_spike.py --variant c128a --op npu.compressor --with-epilog --report c128a_epilog.json
```

图内同时捕获 compressor 与后处理：C4Li 使用 Hadamard → int8 动态量化 → K/FP16 scale scatter；C4A/C128A 使用 BF16 cache 写入。每个 case 在图外刷新固定长度的有效 mask 与 loc；图内不按真实数量切片，也不通过 Python 分支跳过零行。

实验将无效输出行强制替换为 NaN，再通过设备侧 `where` 选成零，检验清理方案不依赖未写行内容。无效行写入独立且互不重复的 scratch slot；真实 KV 区域与 scratch 区域不重叠。eager 对照只处理有效行，因此比较整个真实 cache 区域可以覆盖有效写回和 guard/历史位置不被误写；cache 内容要求精确相同，scale 使用报告中的数值容差，并检查 scratch 为有限值。

验收在原 `output_ok` / `state_ok` 上增加 `cache_ok`、`scale_ok`、`scratch_finite`，所有项通过才设置 `passed=true`。三种 variant 各 9 个 case 全部通过：

| variant | 捕获的后处理 | 输出 / state / 真实 cache 最大误差 | scale 最大误差 | scratch |
| --- | --- | --- | --- | --- |
| C4Li | Hadamard、int8 动态量化、K 和 FP16 scale scatter | 0 / 0 / 0 | 0 | 所有 case 为有限值 |
| C4A | BF16 cache 写入 | 0 / 0 / 0 | 不适用（无 scale buffer） | 所有 case 为有限值 |
| C128A | BF16 cache 写入 | 0 / 0 / 0 | 不适用（无 scale buffer） | 所有 case 为有限值 |

证据为用户贴出的三次终端输出。后两次命令复用了 `--report c4li_epilog.json`，服务器该文件最终只保留 C128A；本节根据三次终端日志记录结果，不将其误当成三个独立保存的 JSON 文件。后续不同 variant 使用不同报告名。

结合第 12 节，共有 54 次 case 执行通过：compressor 单独捕获 27 次，compressor 与 epilog 同图捕获 27 次。这支持当前 A3 配置采用固定输出容量、设备 mask 和隔离 scratch 写入，不需要为已覆盖的 compressor/epilog 增加 `graph.update()`。

这块 scratch 仅是实验中的隔离设计，生产 allocator 尚未因该脚本获得额外槽位。落主仓时必须显式预留相应空间，或提供固定形状的 masked write；不能直接把实验 loc 指向真实请求槽，也不能默认重复写 slot 0 在所有 epilog 中都安全。

**14. 实测接口：fused indexer / sparse attention 注册齐全**

compressor/epilog 已验证，接下来不再重复这一轮测试。基线 SGLang 的 A3 路径使用 `custom.npu_quant_lightning_indexer`、`custom.npu_sparse_attn_sharedkv`，以及各自的 metadata 算子；它们和 `npu.compressor` 的注册来源不同。先在服务器查询：

```bash
python compressor_graph_replay_spike.py --inspect-attention-ops --report attention_ops.json
```

查询模式独立导入 `custom_ops` / `sgl_kernel_npu`，记录各模块实际路径或导入错误，并输出四个必需算子和相关替代绑定的 schema；不创建或捕获图。查询成功只表示完成检查，图执行能力仍未验证。

用户于 2026-11-04 提供查询结果：同一 `Ascend910_9362` / torch `2.10.0+cpu` / torch_npu `2.10.0.post6` 环境中，`a3_required_ops_registered=true`，四个 `custom` 必需算子均存在。`custom_ops`、`sgl_kernel_npu` 的路径分别是当前 Python 3.12 site-packages 中的对应包。

| 算子 | 查询确认的接口要点 |
| --- | --- |
| `custom.npu_quant_lightning_indexer_metadata` | 必需 num_heads_q/num_heads_k/head_dim/query_quant_mode/key_quant_mode；query/KV 长度为可选 Tensor；返回 Tensor |
| `custom.npu_quant_lightning_indexer` | query/key/weights、Q/K dequant scale 为 Tensor；长度、page table、metadata 为 Tensor；返回 `(Tensor, Tensor)` |
| `custom.npu_sparse_attn_sharedkv_metadata` | 必需 num_heads_q/num_heads_kv/head_dim；cu、seqused 为 Tensor；支持 has_ori_kv/has_cmp_kv/cmp_topk/cmp_ratio；返回 Tensor |
| `custom.npu_sparse_attn_sharedkv` | Q、原始/压缩 KV、sparse indices、页表、长度、sinks 和 metadata 为 Tensor；返回 `(Tensor, Tensor)` |

这与基线 A3 backend 的调用相符，因此下一步直接用这些绑定构造完整链路，不切换为 `npu.lightning_indexer` 的 BF16 路径。注册与 schema 结果不等同于 capture/replay 结果。

**15. C4 单层完整链路验证脚本**

新增 [dsv4_c4_attention_graph_spike.py](../scripts/spikes/dsv4_c4_attention_graph_spike.py)。将它与最新 `compressor_graph_replay_spike.py` 放在服务器同一目录后运行：

```bash
python dsv4_c4_attention_graph_spike.py --report c4_chain.json
```

脚本使用同一张裸 NPUGraph 捕获：C4A 主 compressor、C4Li indexer compressor、两个 epilog、合成 Q/KV/weights 投影与 norm/RoPE、indexer Q Hadamard/量化、quant lightning indexer、固定 top-k buffer 交接，以及 sparse attention。每次 replay 前只做固定输入和 metadata 的规划、复制及测试状态恢复，不调用 `graph.update()`。

默认配置：X capacity=256、query capacity=258、固定两个 request slot，hidden=1024、attention heads=64、indexer heads=64、top-k=512、SWA page=128 / compressed page=32、已有 prefix=2048。Q/KV/weights 使用 rank-128 合成投影，不加载 checkpoint；该实验验证算子链 replay 一致性，不声称等价于完整模型计算。

compressor 的 cu 描述 X 的输入容量；attention/indexer 的 query 则按真实行打包，分别保留 cu 与 gather/restore buffer，避免把两种长度口径强行复用。每个无有效 query 的请求槽使用一个 dummy query，因此 query buffer 额外预留两个位置；dummy 请求只读取隔离的零 page，最终按设备 mask 和 restore 索引恢复为 X capacity 的输出。所有无效 cache 写入使用独立 scratch，页表不包含这些 scratch page。

indexer query/KV 长度、SWA/压缩页表及两份 kernel metadata 均为固定地址。metadata 算子在图外针对本次 batch 生成临时结果，再 `copy_` 到固定的 int32[1024] buffer。页表绑定会随 case 改变，历史 BF16 KV 各 page 使用不同内容，增强对错误页表/旧 metadata 的识别能力。

eager 对照使用相同算子和相同初始 cache/state，但只写真实有效行；graph 通过固定容量和 mask 写回。先在 capture 前保存全部九个 eager 参考，再预热和捕获 baseline；capture 后只进行 metadata planning、固定 buffer 更新和 replay，不再调用核心算子的 eager 路径，防止 eager 调用更新隐藏 launch/tiling cache 而影响测量。验收比较：

- `attention_ok`：恢复到原 X 布局后的输出，默认 atol=0.001、rtol=0.01。
- `topk_set_ok`：每个 query 的 top-k 集合精确一致；额外报告 `topk_exact_ok`，区分集合相同但顺序不同的情况。
- `main_state_ok` / `index_state_ok`：两套完整 compressor state。
- `main_cache_ok` / `index_cache_ok` / `swa_cache_ok`：整个真实 cache 区域精确一致，包括未写入的 guard/历史位置。
- `index_scale_ok`、`padding_output_zero`、`scratch_finite`。

九个 case 继续覆盖有效长度、请求划分、cu 总长度、压缩边界、逻辑单请求、无新块、idle 和恢复 baseline。所有 case 必需检查均通过才设置 `passed=true`。状态与 cache 每次都恢复，连续 chunk 的累积另行验证。

本地通过 Python 3.10 语法/CLI 检查，以及五项布局检查，覆盖 T=256/512、prefix=128/2048/8192、page=64/128 下的页表与物理 loc、scratch 隔离、Q gather/restore，以及第一个 request slot 为空而第二个占满的情形。本地没有 torch / torch_npu；用户提供的完整链路 NPU 运行结果见第 16 节。

该脚本目前只覆盖 C4 层；ratio 0 / 128 的 attention、模型权重精度、真实 allocator、连续 chunk、TP/DP/CP 与多流仍需后续验证。

生产实现随后落到 NPU prefill captured-metadata hooks、固定 request/context 容量和模型能力分发；同时明确解决前述 scratch 预留或 masked write 的 allocator 契约，再扩展到模型 replay。

**16. 实测记录：A3 C4 单层完整链路通过**

证据来源：用户于 2026-11-04 提供的完整终端日志，`dsv4_c4_attention_graph_spike.py --report c4_chain.json` 最终输出 `passed=true, stage=complete, cases=9`。

环境沿用前述 `Ascend910_9362`、torch `2.10.0+cpu`、torch_npu `2.10.0.post6` 和安装环境的 `custom_ops` / `sgl_kernel_npu`。测试参数为 X capacity=256、query capacity=258、request slots=2、prefix=2048、top-k=512、hidden=1024、attention/indexer heads=64，SWA page=128、C4 page=32。

| 检查项 | 九个 case 的实测结果 |
| --- | --- |
| `attention_ok` | 全部 true，最大绝对误差均为 0 |
| `topk_set_ok` / `topk_exact_ok` | 全部 true，top-k 集合与顺序都一致 |
| `main_state_ok` / `index_state_ok` | 全部 true，两套 state 的最大绝对误差均为 0 |
| `main_cache_ok` / `index_cache_ok` / `swa_cache_ok` | 全部 true，真实区域最大绝对误差均为 0 |
| `index_scale_ok` | 全部 true，最大绝对误差均为 0 |
| `padding_output_zero` / `scratch_finite` | 全部 true |

通过的 case 为 baseline、less_used、cu_repartition、cu_short_total、one_request、cross_boundary、no_new_blocks、idle，以及恢复 baseline。九份 eager 参考全部在 capture 前生成；capture 后仅更新输入/metadata 并 replay，没有再执行核心算子的 eager 路径，也没有 `graph.update()`。

由此确认：在这个 A3 安装环境和测试形状下，主/indexer compressor、epilog、fused quantized indexer 与 sparse attention 可以处于同一张裸 NPUGraph 中，并通过固定地址 metadata 响应请求长度、页表和有效行数的变化。此前设计中“这些计算必须作为 eager break”的限制属于旧 metadata 生命周期和形状协议，不能当成已验证的算子能力限制。

**环境初始化复现记录。** 用户第一次运行在 indexer metadata 生成阶段报 `aclnnQuantLightningIndexerMetadata` / `GetWorkspaceSize` 找不到，尚未进入图捕获。随后在同一 shell 执行以下环境脚本，再启动新的 Python 进程，原命令完整通过：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
python dsv4_c4_attention_graph_spike.py --report c4_chain.json
```

这里记录的是该服务器成功使用的环境组合，未分别验证四个 source 的必要性，也未将 CPU governor / sysctl 性能设置作为符号解析的前提。schema 注册只说明 wrapper 存在，实际执行还要求 ACLNN 库与 vendor 路径已加载；环境变化后使用新的 Python 进程运行。

脚本随后只增加了报告改进：保存相关库路径环境变量，并将构造 buffer/metadata 的失败阶段标为 `buffer_and_metadata_initialization`，避免误写成 registration。计算、捕获与比较逻辑未因这次通过结果而改变。

累计已有 63 次 case 执行通过（27 compressor、27 compressor+epilog、9 C4 完整链路）。这仍是合成输入的 capture/replay 一致性证据，尚不等于 checkpoint 精度、生产 allocator 和完整模型已经通过；ratio 0/128 的 attention、连续 chunk、真实权重量化、并行与多流也不在本次结论中。

**17. 下一阶段：主仓集成的具体落点**

算子能力验证已经足以开始 A3 C4 的主仓适配。继续沿用 P1 的 `BreakableNPUGraphBackend`，首版采用单流；当前证据没有要求为这条 C4 路径新增 `graph.update()` 或修改 torch_npu。

| 顺序 | 主仓改动 | 必须形成的契约 |
| --- | --- | --- |
| 1 | 在 NPU 专属模块定义 per-bucket prefill graph state，并实现两个 captured-metadata hook | 固定 T/B/context/C4/C128 容量，强引用保活；replay 原地刷新，超容量提前回退 |
| 2 | 明确 KV pool / allocator 的 dummy read page 与 scratch write 区域 | 槽位被显式预留且不分配给请求；SWA、主压缩 KV、indexer K/scale 都覆盖；不能照抄 spike 的私有 loc 到真实 pool |
| 3 | 改造 NPU compressor / indexer / sparse attention 的图内路径 | 压缩容量 cu 与 query cu 分开；固定 shape 的 mask/loc、Q/weights pack、top-k 交接、attention 输出 restore；不按 live 数量做 Python 切片 |
| 4 | 模型按后端能力分发 | `models/deepseek_v4.py` 对已适配的 NPU prefill 路径绕过公共 attention eager break；NPU source 入口也不再经过 `bcg_dsv4_*` 断点 |
| 5 | 真实模型验收与扩展 | 先覆盖 C4 的真实权重、prefix hit、连续 chunk 和后续 decode，再覆盖 ratio 0/128、DP idle 与并行组合 |

第 1 项直接使用 runner 现成的 `init_forward_metadata_for_breakable_cuda_graph_capture` / `prepare_forward_metadata_for_breakable_cuda_graph_replay`；在 `ascend_dsv4_backend.py` 里为 prefill 单独规划，复用 decode 的 copy/清尾工具，不复用其每请求一个 token 的假设。每张图的固定 metadata 对象由 runner 保留。

第 3 项接真实模型时可以保留原来 bucket 大小的投影：在 indexer 的 fused 调用前，对已经计算出的 Q 和 weights 做固定形状 pack；在 sparse attention 前 pack 主 Q，输出通过 restore 索引返回原 token 布局。两路 compressor 继续读原 X capacity，避免把 prototype 的 query 额外容量传给 compressor。SWA 写回 loc 的 padding 协议也需要同步适配。

首个可评审改动应围绕 graph state、metadata hooks 和容量检查展开；allocator 隔离与图内执行路径接齐后，才打开模型能力分发，进入整模 eager/graph 对照。验收还应确认 profiler 中这四段模型计算都处于 captured segment 内，并测量启动时间、常驻显存和实际 prefill 延迟。
