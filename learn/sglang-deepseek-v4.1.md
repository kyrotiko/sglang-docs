# DeepSeek-V4.1 学习笔记（SGLang 实现视角）

> 范围：`deepseek-ai/DeepSeek-V4.1-Flash`，HF `model_type: deepseek_v41`。
> 目标：读完后能回答「V4.1 和 V4 差在哪」「SGLang 为 V4.1 做了哪些适配和优化」
> 「看某个行为应该去哪个文件」。
> 代码行号会漂移，以文件名和符号为准。
>
> **配套动画：[V4 → V4.1 动画学习室](sglang-deepseek-v4.1-viz.html)**，可离线打开，包含逐步数据流、交互实验与自测。
> **视频讲解：[带章节导航的播放器](sglang-deepseek-v4.1-video.html) / [直接播放 MP4](sglang-deepseek-v4.1.mp4)**，中文合成旁白、画面要点字幕，完整旁白另附 SRT / VTT。
> 本次核对基线：`sglang@4cafabd1a3`。层号按代码从 0 开始；V4 配置类默认值与实际 checkpoint 参数应区分。

## 0. 一页速览

| 项 | DeepSeek-V4.1-Flash |
| --- | --- |
| 结构 | 40 层 sparse-attention MoE，384 个 routed experts（top-6）+ 1 个 shared expert |
| 注意力 | 每层一份 512 宽 KV latent，K/V 共用；64 个 query head；无独立 V 投影 |
| 压缩 | 每层另有 128 位置滑动窗口；压缩 latent 只在 source layer 产生并向下共享 |
| 压缩比 | ratio 2：第 2–19 层把 2 个位置压成 1；ratio 1：第 20 层起 1 对 1（不再池化） |
| 稀疏检索 | KV source 产 index K；index source 更新 top-512 选择，普通层复用已有选择；candidate block 用于预筛选 |
| 残差 | mHC（manifold-constrained hyper-connections），4 条并行残差流，Sinkhorn 归一化 |
| 额外记忆 | Engram：2 层 n-gram hash 查表，FP8 表合计约 189 GiB |
| 量化 | dense 权重 FP8，32×32 ue8m0 block scale；routed experts FP4；专家激活 MXFP8 |
| 上下文 | 1M（1048576） |
| 投机解码 | 自带 DSpark（3 个 draft stage + Markov head + 可选 confidence head），块大小 5 |
| 多模态 | 原生 ViT + Aligner，支持 image token |
| SGLang 入口 | 仍走 `DeepseekV4ForCausalLM`（`model_type` 分流），attention backend `dsv4` |

一句话：**V4.1 = V4 的「单 KV latent + 压缩 latent + SWA + mHC + MoE」骨架，换成更细的
c1/c2 压缩、KV/index source 分离、candidate 检索、Engram 查表、mHC 跨层预混、32×32
FP8 block 和视觉接入；支持 DSpark 投机路径，是否启用由服务配方决定。**

---

## 1. 必须理解的十个概念

### 1.1 一份 KV latent，K 和 V 共用

每层每个位置只保留一份 512 维 latent（`kv_lora_rank=512`），它同时充当 K 和 V：

```
hidden x ──► wkv ──► kv latent [T, 512] ──┬──► K (nope 448 + rope 64)
                                          └──► V (同一份)
           wq_a ──► q_lora [T, 1280] ──► q_norm ──► wq_b ──► q [T, 64 heads, 512]
```

- `head_dim = 512 = 448 (nope) + 64 (rope)`；`num_key_value_heads = 1`。
- Q 侧走低秩：本仓 V4.1-Flash 适配形状为 `q_lora_rank=1280`；输出侧走 grouped 低秩：
  `o_groups=8`，`o_lora_rank=1024`，即 `wo_a` 是 8 组分组 GEMM，再 `wo_b`。
  1280 的本地依据是 `MQALayer._normalize_q_lora` 的形状门禁及
  `test/registered/kernels/ops/attention/dsv4/test_mega_attention.py` 的 `wq_b` 测试形状；
  不是从 V4 配置类的 `q_lora_rank=1024` 默认值推出来的。实际参数仍以 checkpoint config 为准。
- V4 对每个 query head 做一次**无权重 RMSNorm**再 RoPE（`q_head_norm=True`）；
  **V4.1 去掉了这个 per-head norm**（`q_head_norm=False`），
  `_compute_q_b` 直接走 `fused_rope_inplace` / `fused_q_norm_rope(eps=None)`。
  这是 V4.1 一个容易忽略、但影响算子选择的行为差异。

代码：`MqaAttentionBase.__init__`、`MQALayer.__init__`、`_compute_q_b`
（`python/sglang/srt/models/deepseek_v4.py`）。

### 1.2 两种 KV store：压缩 latent + 滑动窗口

每层 attention 读两类历史：

1. **压缩 latent（compressed latents）**
   - 只在 `kv_source_layer_ids` 指定的 source 层产生；
   - 后面的层通过 `loc // ratio` 算术直接共享 source 的池，没有映射表；
   - 因此压缩状态随完整前缀一起生灭，prefix cache 天然可用。
2. **128 位置滑动窗口（SWA）**
   - 每层用自己的激活重新计算，不能跨层共享；
   - 但不随上下文增长，长文本下是固定成本。

V4 与 V4.1 的压缩比不同：

| 版本 | 每层压缩比 | 实现 |
| --- | --- | --- |
| V4 | 4 / 128（`c4` / `c128`） | `Compressor` + `C4Indexer` |
| V4.1 | 2 / 1（`c2` / `c1`） | `DeepseekV41Compressor` + `DeepseekV41Indexer` |

V4.1 的 ratio 2 把相邻两个 token 池化成一个（`project` 出 kv + score，softmax 加权）；
ratio 1 是逐 token 的压缩 latent，不做池化，但仍由 source 层共享。

代码：
- 分支选择：`MQALayer.__init__`（`compress_ratio in (4, 128)` vs `(1, 2)`）
- V4.1 实现：`python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py`
  （`DeepseekV41Compressor`、`DeepseekV41Indexer`）

### 1.3 source layer：kv source 与 index source 分离

这是 V4.1 最值得学的设计之一：

- `kv_source_layer_ids`（4 个）：真正**产生压缩 latent** 的层。
- `index_source_layer_ids`（8 个）：运行 indexer、用当前层 Q **更新 top-k 检索决策**的层。
- index K 的生产仍由 KV source 决定。多出来的 4 个 index source 不产新 K，
  用自己的 query 给已有 source 的 index K 重新打分；普通层复用已有选择。
- 最后一个 kv source 层（V4.1-Flash 是第 20 层）同时是 index source 和 candidate source。

```
kv source S:     x_S ──► compressor ──► latent pool（后续层共享）
index source I:  q_I ──► score(pool[S]) ──► top-k
                 └─ 若 I 本身也是 kv source，才额外产 index K
```

意义：**检索决策的频率是 KV 存储频率的两倍**，而 index K 比完整 KV 便宜得多。
代价是 consumer 层必须知道「去哪个 source 的池里取数」，这由 config 的
`kv_source_layer_ids` / `index_source_layer_ids` 驱动。

代码：`DeepseekV41Indexer.owns_k = layer_id in config.kv_source_layer_ids`；
`MQALayer` 只在 `layer_id in index_source_layer_ids` 时创建 indexer。

### 1.4 Indexer：fp4 小 side-attention + top-k

indexer 是一个很小的 side attention：

- 本仓 V4.1-Flash 适配形状为 `index_n_heads=32`，`index_head_dim=128`，`index_topk=512`；
  32 的依据见 `kernels/ops/attention/dsa/litetopk_prefill/fp4_indexer.py` 的模型形状说明。
  主 attention 的 64 heads 与 indexer 的 32 heads 不要混用；V4 config 中的 64 是另一组默认值；
- Q：`wq_b(q_lora)`，K：source latent 过 `wk` + `k_norm`；
- K 和 Q 的 RoPE 尾部会做 **fake FP4 量化**（`_rope_fq4`），索引缓存也以 FP4 存；
- 每个 head 有 `weights_proj` 学习出的权重，分数按 head 加权求和；
- 只保留 top-512 个压缩位置作为 attention 的 sparse 输入。

V4.1 在此之上增加 **candidate 方案**：

- `candidate_source_layer_id` 指定的层发布候选块，并选出本层的最终 top-k；
- 后续 index source 用自己的 query 在候选块覆盖的位置中重选最终 top-512；
- `candidate_topk_blocks` 控制块数，`index_topk` 控制最终位置数，两者不是同一个 top-k。
  SM100 sparse-table 路径能只算候选分数；部分 fallback 先算全量分数再筛选，因此
  “选择受候选约束”不一定等于“只执行候选打分”；
- 两种载体：
  - `dense_blocks`：dense block id 列表，非 SM100 的通用实现；
  - `sparse_table`：DeepGEMM paged sparse mqa-logits 需要的稀疏表，严格 SM100。

代码：`v41_indexer/` 包
（`full_topk.py`、`dense_blocks.py`、`sparse_table.py`、`scoring.py`），
入口 `make_full_topk_indexer` / `make_candidate_indexer`。

一个值得学的工程细节（#49173）：

- **dense** DeepGEMM fp8_fp4 mqa-logits 支持 SM100+（含 SM120/121）；
- **sparse** paged mqa-logits 的 kernel 断言 `arch == 10`，所以 candidate/sparse-table
  路径严格 `is_sm100()`；
- SM120 上要退回 dense blocks，不能因为「SM100+」就误开 sparse；
- dtype 也随之保持：SM100 上 index logits 可以是 bf16，其他架构保持 fp32。
  见 `test/registered/unit/layers/test_dsv41_indexer_arch_gates.py`。

### 1.5 mHC：4 条残差流 + Sinkhorn

mHC 把普通残差 `h = x + sublayer(x)` 换成：

```
R [T, hc_mult=4, hidden]   # 每个 token 有 4 条并行残差流
每个 sublayer 带一组 (pre, post, comb)：
  1. mix stats: 从 R 预测 (pre, post, comb)
  2. combine:   用上一个 sublayer 的 pre 把 4 条流压成 1 条，再 norm
  3. sublayer:  attention 或 MoE（含 TP reduction）
  4. post:      R'[i] = post[i] * y + sum_j comb[j][i] * R[j]
```

`comb` 经过 `sinkhorn_iters=20` 次 Sinkhorn 迭代，逼近双随机矩阵（每行/列和为 1），
这是「manifold-constrained」的含义。

V4 与 V4.1 的关键差异是 **pre 的时序**：

| 版本 | `hc_pre_from_prev_sublayer` | 含义 |
| --- | --- | --- |
| V4 | `False` | pre 在**当前** sublayer 内算、当前用 |
| V4.1 | `True` | 当前 sublayer 的 pre 由**上一个** sublayer 产出，本次消费 |

这个 lag 让系数预测可以和主 GEMM 重叠。当前 `post/comb` 用于本次残差更新，
只有当前 `pre` 留给下一子层。旧版绑定 same-sublayer pre 的融合不能直接复用
（`use_fused_mhc_post_pre=False`），但 V4.1 仍可融合当前 post 与下一子层的
combine、RMSNorm，实际会使用 `mhc_mega_boundary` 和 `mhc_post_combine_norm_decode`。

代码：`deepseek_v4_mhc.py`（kernel dispatch）、`deepseek_v4_boundary.py`
（`HcUpdate` / `HcRead` / `FinalRead`，负责跨层状态与 PP/CP 的 row transport）。

### 1.6 Engram：n-gram hash 查表

V4.1 在 2 个层里加入 Engram：把**前面的 token** 哈希成 n-gram 行，从两张静态 FP8 表
里取行，再 gate 进 hc 残差流。

关键设计：

- token id 在哈希**前**做规范化（NFKC/NFD、去重音、小写、空白折叠），
  所以 `" The"`、`"the"`、`"THE"` 不会分裂成不同行；
- 每层每个 lookback 一个奇数乘子（由 `(layer_id, ngram)` 的 RNG 决定），
  保证 `token_id * multiplier` 落在 int64 内；
- 哈希结果是 `[T, n_engram_layers, n_hash_cols]` 的行 id；
- 表只读、不计算；V4.1-Flash 两张表合计约 189 GiB，是 checkpoint 里最大的单块权重；
- 表可以用 FP8 或 MXFP4 存，gather 时解码；
- 图像 token 被 Engram 屏蔽（hash 作为 blocked 处理，不 gate）。

代码：`python/sglang/srt/layers/engram.py`
（`EngramLayout` / `EngramHasher` / `EngramEmbedding`），
kernel 在 `python/sglang/kernels/ops/embeddings/engram_*`。
用户文档：`docs/docs/advanced_features/engram.mdx`。

### 1.7 MoE 与量化

- 384 个 routed experts，top-6；另有 1 个 shared expert；
- routed experts 是 FP4（MXFP4 packed）；
- dense / attention 权重是 FP8，**32×32 ue8m0 block scale**；
  V4 是 128×128 block，这是 checkpoint 级别的差异（`_dequant_fp8` 用
  `weight.shape / scale.shape` 反推 block 大小）；
- V4.1 的 FP4 routed experts 在 Aiter 路径上使用 **MXFP8 激活**；
- V4.1-Flash 的 routed expert intermediate size 是 2304（MegaMoE 需要适配这个形状）；
- 视觉模型在 gate 上多一个 `e_score_correction_bias_vl`，即 `vl_correction_bias`。

### 1.8 SWA bounded replay（V4.1 的 late-layer 捷径）

普通 prefill 每层都要对整段 prompt 算 attention。V4.1 的 decoder SWA bounded replay：

- 前段层（直到最后一个 kv source，第 20 层）仍处理完整 extend；
- 后面的层只处理每个请求**最后 128 个 token**，压缩 KV / index 直接复用 source 层；
- 尾段由 `DecoderReplayTail` 从 row-bucket graph 重放；
- **它不是数值等价的**：请求 prompt logprobs 或返回所有 hidden states 时，
  整段 prompt 仍走完整计算；只返回输出 token logprob 的普通请求走 bounded 路径；
- CUDA/HIP、无 context parallelism 时默认开启；
  和 prefill CUDA graph 组合时只允许 eager / breakable；
- RL rollout 需要和 trainer 对齐 logprob 时，应显式
  `--no-enable-decoder-swa-bounded-replay`。

代码：`python/sglang/srt/models/deepseek_v4_replay_tail.py`；
自动开启/校验规则在 `arg_groups/deepseek_v4_hook.py::validate_deepseek_v41_features`。

### 1.9 DSpark：V4.1 自带的投机解码

DSpark 不是 EAGLE，也不是普通 MTP，而是 checkpoint 自带的 draft：

- 3 个 draft stage（`num_stages` 从 `dspark_target_layer_ids` 解析），表示 draft 网络深度；
- 选定 target 层的 hidden 先合并投影，再通过各 stage 的 `wkv` 生成 draft 上下文 KV，
  不能理解为每个 stage 分别固定直连一个 target 层；
- **Markov head**（`vanilla` / `gated` / `rnn`）：参与 block 内 token 提议；
- 可选 **confidence head**：动态/ragged 验证可使用置信度或在线接受估计；
  static verify 不构造 confidence head；
- cookbook 显式设置提议块大小 `gamma=5`，target 验证布局含 anchor，共 `gamma+1=6` 个位置；
  stage 数、提议长度与验证行数是不同维度，5 不能写成通用代码默认值；
- verify 支持 compact / ragged 布局：把 packed verify token 映射回各自请求，
  接受/拒绝路径依采样与验证配置选择，并据此更新提交长度；
- **KV 注入方向是 target hidden → draft KV**：target 验证产生的 hidden 按提交长度筛选后，
  经投影生成下一轮 draft 的上下文 KV；`TargetHiddenKvInjector` 写入的是
  `draft_model_runner.token_to_kv_pool`，deferred store 只改变写入时机；
- PD 下要求 static verify、Mooncake/NIXL/MoRI、decode 侧 DCP=1 / CP=1；
  prefill CP 只支持 interleave。

代码：`python/sglang/srt/models/deepseek_v4_dspark.py`、
`python/sglang/srt/speculative/dspark_components/`
（`dspark_draft.py` / `dspark_verify.py` / `dspark_planner.py` /
`dspark_kv_inject.py` / `dspark_config.py`）。
支持 `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`。

注意：原版 V4 / V4-Pro 用 EAGLE/MTP；V4-Flash-0731、V4-Pro-0813 和 V4.1-Flash
自带 DSpark head。V4.1 的功能校验会拒绝 EAGLE；V4 的 0813 等 DSpark checkpoint
曾出现 EAGLE 能启动但接受率为 0 的情况。两种情况不要混成统一行为。

### 1.10 视觉（V4.1 原生多模态）

V4.1-Flash 的 checkpoint 带 `vision_config`：

- `ViT` + `Aligner`，把图像特征对齐到语言模型 hidden；
- 特殊 token：`image_start` / `image_end` / `image_newline`，`image_token_id`；
- `model_config.py` 用 `vision_n_layers > 0` 判 `is_multimodal`，
  不依赖通用 `vision_config` 启发式；
- vision 支持 TP/EP/DP，**不支持 CP / PP**；MoE A2A 只允许 CUDA MegaMoE；
- 视觉路由让 late layer 读取 token id；Engram 对 image token 做 blocked/ungated；
- prefill graph 下 `MM_PAD_SHIFT_VALUE` 的映射在 replay 前完成，
  不能改 scheduler 的原始 input_ids。

代码：`deepseek_v41_vit.py`、`multimodal/processors/deepseek_v41.py`、
`DeepseekV4ForCausalLM.__init__` 的 vision 分支、`prepare_language_model_inputs`。

---

## 2. V4 vs V4.1 差异总表

> 注意：下表的层数/专家数是 **V4.1-Flash** 与 V4 代码默认值/Flash 形态的对照；
> Pro 或其它 checkpoint 以 HF config 为准。

| 维度 | DeepSeek-V4 | DeepSeek-V4.1 |
| --- | --- | --- |
| HF `model_type` | `deepseek_v4` | `deepseek_v41`（配置归一化后） |
| SGLang 配置类 | `DeepSeekV4Config` | `DeepseekV41Config(DeepseekV3Config)` |
| 层数 / experts | 43 / 256（config 默认；Pro 不同） | 40 / 384，top-6 + 1 shared |
| 压缩比 | 4 / 128（c4 / c128） | 2 / 1（c2 / c1） |
| 压缩 latent 共享 | 每层自己的 compressor（c4/c128） | 仅 4 个 kv source 层产生并共享 |
| index source | 与 KV 层耦合 | 8 个 index source，独立于 4 个 kv source |
| 候选检索 | 无 | candidate source 发布 top-k blocks，后续层只算 candidates |
| Q head norm | 每个 head 无权重 RMSNorm 后 RoPE | 去掉 per-head norm |
| mHC pre 时序 | 当前 sublayer 内算 | 上一个 sublayer 算、下一个用（lag） |
| Engram | 无 | 2 层 n-gram hash 表 |
| 视觉 | 单独 Flash Vision (Exp) checkpoint | 原生 ViT + Aligner |
| 定点量化 | dense FP8 128×128 block | dense FP8 32×32 ue8m0 block |
| MoE 激活 | FP8 / FP4 视 checkpoint | FP4 experts + MXFP8 激活 |
| 投机解码 | 原版 EAGLE/MTP；0731/0813 带 DSpark | 支持 DSpark，是否开启由配方决定 |
| Attention backend 默认 | `flashmla` | `flashmla_mega`（SM100/103，PD 模式或 attention TP==1，且 KV 为 fp8_e4m3） |
| KV layout | `v4` | `v41` / `v41_new`（fp8 / fp4 page） |
| 长文捷径 | 无 | decoder SWA bounded replay 默认开 |
| 不支持的组合 | 视 backend 而定 | 显式拒绝 PP>1、HiSparse、HIP 的 unified KV layout、two-batch overlap、非 DSpark 投机 |

### 2.1 为什么 V4.1 配置继承 `DeepseekV3Config`

`configs/deepseek_v41.py` 的注释写得很直白：**V3 的 config 接受 1/2 压缩比，
原生 V4 config 拒绝**。V4.1 又需要 `text_config` / `vision_config` 的扁平化，
所以这里做了三件事：

1. 把 `text_config` 展开到顶层，把 `vision_config` 的字段改名成 `vision_*`；
2. 强制 `model_type = "deepseek_v41"`；
3. 把 `architectures: ["DeepseekV41ForCausalLM"]` 改写成
   `["DeepseekV4ForCausalLM"]`，让模型注册表复用同一个实现。

---

## 3. SGLang 的适配与优化地图

### 3.1 统一模型实现 + 版本分流

`python/sglang/srt/models/deepseek_v4.py` 同时支持 V4 和 V4.1，靠
`config.model_type == "deepseek_v41"` 分流。主要分叉点：

| 位置 | V4 | V4.1 |
| --- | --- | --- |
| `MQALayer` | ratio 4/128 → `Compressor` + `C4Indexer` | ratio 1/2 → `DeepseekV41Compressor` / `DeepseekV41Indexer` |
| `_compute_q_b` | per-head q norm + RoPE | 无 q head norm，直接 RoPE |
| `use_mega_attention` | 否 | 是（`flashmla_mega`） |
| `use_fused_wo_a` | 否 | 是（满足 arch/shape 时） |
| `_apply_wo_a_bf16_matmul` | 通用 einsum / Aiter batched GEMM | `fast_path=True`，按 decode/verify/prefill 走专用 bmm/gemv |
| `_dequant_fp8` | block 128×128 | block 32×32 |
| MoE | 普通 FP4/FP8 | `vl_correction_bias`、`fp8_activations=True` |
| mHC | `hc_pre_from_prev_sublayer=False` | `True` |
| 视觉 | 仅单独 checkpoint | `self.vision` / `self.aligner` |

其它适配：

- `DeepseekV4ForCausalLM.wants_prefill_autotune()` 对 V4.1 返回 True；
  `autotune_prefill_kernels()` 在启动时对每个 M bucket 预调 resident MXFP8 linears，
  避免第一次 prefill 现调（`deepseek_v4.py` 末尾）。
- `load_weights` 在纯文本 / 无 vision 时跳过 `vision.*`、`aligner.*`、`image_*`、
  `.gate.e_score_correction_bias_vl`，并校验 `wo_a` 的 FP8 dtype。
- `arg_groups/model_overrides/deepseek_v4.py` 自动选择：
  - attention backend 固定 `dsv4`；
  - V4.1 + FP8 32×32 ue8m0 + SM100 → `fp8_gemm_runner_backend=flashinfer_cutedsl`；
  - FP4 experts → `moe_runner_backend=flashinfer_mxfp4`；
  - page size：CUDA 256，NPU 128。
- `arg_groups/model_overrides/deepseek_v4.py` 与 `deepseek_v4_hook.py` 还负责
  `--engram-config` 校验、SWA bounded replay 自动开启、DSpark/CP/PD 组合校验。

### 3.2 Attention backend 与 KV layout

`--dsv4-attn-backend` 的取值和适用面：

| backend | 适用 | 说明 |
| --- | --- | --- |
| `flashmla` | 通用 CUDA/HIP | decode + sparse prefill 的基线；attention TP > 1 时默认 |
| `flashmla_mega` | V4.1 + SM100/103 | 融合 Q-RoPE + sparse attention + inverse-RoPE + MXFP8 FlashMLA |
| `trtllm` | V4/V4.1 + SM100/103 | uniform 512 维 FP8 KV layout，decode + sparse prefill；不支持 PD |
| `cutedsl_h16` | V4 only + SM90 + BF16 + 16 heads | 实验性 sparse prefill；V4.1 明确不支持 |

`auto` 的规则（`deepseek_v4_hook.py::_default_dsv4_attention_backend`）：

```
V4.1 + CUDA + SM100/SM103 + kv_cache_dtype=fp8_e4m3
  且（PD 模式 或 attention TP == 1）
  → flashmla_mega
否则 → flashmla
```

几个容易踩的约束：

- `flashmla_mega` 要求 V4.1、SM100/103；KV pool 会把非 `v41_new` 的
  `SGLANG_DSV4_KV_LAYOUT` 归一到 `v41`（`select_dsv4_kv_layout`）；
- `trtllm` 要求 `kv_cache_dtype=fp8_e4m3`、chunked prefill、非 PD、无 CP；
- 不要手动覆盖 `--attention-backend` / `--moe-runner-backend` / `--fp8-gemm-backend`；
  覆盖会把 32-wide ue8m0 块打到 Triton fallback，decode 性能会掉一个数量级。

KV layout 的一个直观记忆（`deepseek_v4_backend.py` 的 view）：

| layout | 每 token 字节视图 |
| --- | --- |
| V4 | 584 |
| V4.1 FP8 | 528 |
| V4.1 FP4（仅 extra cache） | 288 |

MegaAttention 的实现细节：

- `DeepseekV4MegaAttnBackend`（`deepseek_v4_mega_backend.py`）：
  - `packed_query_buffer` 只分配一次且不重分配，保证捕获图中的地址稳定；
  - `_mega_paged_prefill` 只对 `num_heads=64`、ratio 1/2、`num_qo_tokens<=1024`
    且 gather tokens 足够大的形状启用；
  - decode / prefill 走 `flash_mla_mega_*`。
- prefill 的 `_mega_paged_prefill` 与 sparse prefill / candidate 路径有互斥条件，
  改这里时要同时看 `_use_sparse_prefill`。

### 3.3 V4.1 专属 indexer / compressor 算子栈

相关文件：

```
python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py       # compressor / indexer 的定义与 torch bring-up
python/sglang/srt/layers/attention/dsv4/v41_indexer/
  __init__.py        # is_sm100 / is_sm100_or_newer / make_*_indexer
  full_topk.py       # 全量 top-k（DeepGEMM 或 torch）
  dense_blocks.py    # candidate：dense block ids（非 SM100 通用）
  sparse_table.py    # candidate：DeepGEMM sparse table（严格 SM100）
  scoring.py         # 共用的打分、DeepGEMM operand、写回池
  types.py           # PrefillInputs / DecodeInputs / CandidateMetadata ...
python/sglang/kernels/ops/attention/dsv4/
  index_logits.py    # DeepGEMM mqa-logits、flat logits、dtype gate
  index_score_reduce.py
  fp4_indexer*.py    # paged fp4 indexer、RoPE+fp4 pack、schedule
  candidate_blocks.py / candidate_scores.py / candidate_table.py
  topk.py            # top-k transform / sparse topk
  sparse_prefill_kernels.py
```

学习顺序：

1. 先看 `DeepseekV41Indexer` 的 `queries` / `index_keys` / `scores`：
   弄清 Q/K 怎么来、FP4 怎么加、head weight 怎么用；
2. 再看 `v41_indexer/scoring.py`：`deep_gemm_*` 的 operand 布局、flat logits、
   写回 pool slot；
3. 最后看 `full_topk.py` / `dense_blocks.py` / `sparse_table.py` 三种策略的差异，
   以及 `make_candidate_indexer` 的选路条件。

优化点（从 commit 和代码里能直接看到的）：

- **LiteTopK prefill**：`fp4_indexer_topk` 在行数足够多时替代 dense top-k；
  `_lite_topk_wins` 给出了经验阈值（`num_rows >= 512 && rows*max_len >= 2^26`
  或 `max_len <= 6144`）；
- **candidate 限制选择范围**：优化后的 consumer 仅计算候选位置分数；
  部分 fallback 仍先算全量分数再筛选，不能一概称为不扫全量；
- **indexer TP shard**：`SGLANG_OPT_DSV4_INDEXER_TP_SHARD`，行数够大时按 attn TP 切行；
- **非分页 indexer**：`SGLANG_OPT_DSV4_NONPAGED_INDEXER`，长 query 时收益明显；
- **paged fp4 indexer**：decode 直接把 top-k 解析成 cache 地址；
- **SM120 dense 路径**：避免退回 torch fallback（每层 `tolist()` 约 125 ms）；
- **bf16 index logits**：`SGLANG_OPT_DSV41_BF16_INDEX_LOGITS` 只在 SM100 安全。

### 3.4 mHC 的融合与重叠

V4.1 的 mHC 是性能工作的重点，代码主要在：

- `deepseek_v4_mhc.py`：kernel dispatch；
- `deepseek_v4_boundary.py`：跨层 / PP / CP 的残差状态传输；
- `deepseek_v4_platform_stage.py`：非 CUDA（NPU/XPU）路径；
- `kernels/ops/layernorm/mhc_*`：具体 kernel。

已落地的优化：

- **fused decode post + combine + RMSNorm**：`mhc_post_combine_norm_decode`；
- **fused post + 下一 combine/norm boundary**：V4.1 路径也使用 `mhc_mega_boundary`；
  它与旧版绑定 same-sublayer pre 的 `use_fused_mhc_post_pre` 不是同一个契约；
- **side-stream mHC stats**：`hc_stats_stream`，Blackwell decode 上把 mix stats
  放到侧流，与主 GEMM 重叠；
- **BF16x3**：Blackwell decode batches 上 mHC mix 用 BF16x3（`#48456`）；
- **DeepGEMM HC prenorm**：`SGLANG_OPT_DEEPGEMM_HC_PRENORM`，V4.1 的
  compensated projection；SM90/SM120 走纯 Triton，SM100 走 DeepGEMM；
- **decode attention mHC mix fork**：在 MegaAttention core 处按 decode 形状分叉
  （`#49321`），以及 **fused decode mHC post 写 MegaMoE 输入**（`#49262`）；
- **fused all-reduce epilogue**：TP2/TP4 的 mHC post 有专用 epilogue。

理解这些优化的前提是 §1.5 的 `(pre, post, comb)` 三元组和 lag 语义；
否则很容易把「为什么不能复用 V4 的 fused post+pre」看漏。

### 3.5 Engram 的系统级优化

代码：`layers/engram.py` + `kernels/ops/embeddings/engram_*`，
用户侧只有 `--engram-config`。

SGLang 做了四类优化：

1. **hash / gather / gate 全 kernel 化**：
   `engram_hash_ids`、`engram_hash_extend_and_commit`、`engram_gather`、
   `fused_engram_gate`；支持 extend / target-verify / decode / CUDA graph；
2. **表的放置**：
   `placement=gpu|cpu`；CPU 时每 rank 一个 huge-page 支撑的 shard，把 HBM 让给 KV；
3. **复制与查找**：
   `replication=none|per_node`；`lookup=auto|all_reduce|reduce_scatter|all_to_all|peer_read`；
   NVLink 域内默认 `peer_read`，直接读 owning GPU 的 shard，无 collective；
4. **量化与布局**：
   FP8 或 MXFP4 表；MXFP4 保持 packed，在 gather 里解码；
   `SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS` / host table 等旧 env 已被
   `--engram-config` 取代。

### 3.6 MoE / MegaMoE

- `mega_moe.py` / `mega_gate.py`：dispatch + GEMM 融合、prefill 上的 MegaGate routing；
- V4.1 的 2304 intermediate size、MXFP4 激活需要专门适配
  （`#49388`、`#49262`）；
- `--enforce-shared-experts-fusion` 把 shared expert 作为一个额外 MXFP4 expert
  走同一个 MoE kernel；V4.1 的视觉 checkpoint 需要 `vl_correction_bias`，
  融合路径要保留它；
- EPLB / Waterfill / 冗余专家在 V4.1 上可用；
- SM90 的 all-FP8 MegaMoE 是另一条路径（`SGLANG_DSV4_FP4_EXPERTS=0`）。

### 3.7 图捕获：decode 全图 + prefill breakable

- decode 默认整图捕获；
- V4.1 prefill 支持 **breakable graph**：把不满足捕获契约的动态工作放入 eager 断点，
  满足固定地址和 metadata 刷新契约的算子仍可入段。不能把 attention / compressor /
  indexer 一概视为永久图外；candidate source 与 consumer 也有不同捕获路径；
- `PrefillCudaGraphRunner` 对 V4.1 有专门分支（`is_dsv41_model`）：
  - 多模态 input id 映射必须在 replay 前完成，不能改 scheduler ids；
  - forked 路径要在 capture 前后保持地址稳定；
- `deepseek_v4_replay_tail.py`：late-layer tail 从 row-bucket graph 重放，
  和 prefill graph 的 break 配合；
- NPU 的 breakable prefill graph 见 §3.10。

### 3.8 DSpark 的适配

- `deepseek_v4_dspark.py`：draft 模型本体（stages + Markov head + confidence head）；
- `deepseek_v4_dspark.py` 的 `DeepseekV4ForCausalLMDSpark` 复用 target 的
  `DeepseekV4ForCausalLM`，但 draft attention 走 `DSV4_DRAFT_ATTENTION_BACKEND`；
- `dspark_components/`：
  - `dspark_planner.py`：verify window 规划；
  - `dspark_draft.py`：draft step、sampling、vocab shard；
  - `dspark_verify.py`：ragged/compact verify、blockwise rejection、accept；
  - `dspark_kv_inject.py`：把已提交 target hidden 投影成 draft 上下文 KV；
  - `dspark_deferred_draft_kv.py`：延迟写 KV，配合 overlap；
- 关键 kernel 在 `kernels/ops/speculative/dspark/`：
  `dspark_accept`、`blockwise_rejection`、`dspark_verify_window`、
  `static_accept_tail`、`simulated_bonus`、`markov_sample`；
- 优化：`SGLANG_OPT_DSV41_DSPARK_*`（draft prep fork、post fold、skip prefilled
  inputs、sim draw-ahead、one-launch commit）、target verify metadata staging、
  compact verify 的 request 映射、sharded accept（AMD）。

### 3.9 Serving 侧自动配置

`apply_deepseek_v4_defaults` / `validate_deepseek_v41_features` 做了大量
「让用户少写 flag」的工作：

- `max_running_requests` 未设时默认 256；
- `dsv4_attn_backend=auto` 自动解析；
- `--engram-config` 校验（只对 V4.1；CPU/peer_read/all_to_all 的前置条件）；
- decoder SWA bounded replay 默认开启（条件不满足时自动关闭）；
- V4.1 显式拒绝：
  - speculative 非 DSpark；
  - HiSparse；
  - unified KV layout；
  - two-batch overlap；
  - PP > 1；
  - TRT-LLM + encoder SWA bounded replay；
- DSpark + PD 的组合校验（static verify、传输后端、DCP/CP）；
- 视觉只允许 TP/EP/DP，不允许 CP/PP。

### 3.10 NPU 现状（重要边界）

这个工作区里的 NPU 代码主要服务 **DSV4（V4）**，不是 V4.1：

- `hardware_backend/npu/attention/ascend_dsv4_backend.py`：Ascend 版 DSV4 backend；
- `hardware_backend/npu/attention/dsv4_prefill_graph.py`：breakable prefill graph 的
  固定容量 metadata（capture 时按 token bucket 分配，replay 原地刷新）；
- `hardware_backend/npu/dsv4/dsv4_memory_pool.py`：PA_ND layout、C4/C128 state pool；
- `hardware_backend/npu/dsv4/dsv4_rope.py`：NPU RoPE；
- `arg_groups/model_overrides/deepseek_v4.py`：NPU page size 128；
- `hardware_backend/npu/attention/ascend_dsv4_backend.py:2366` 直接限制
  `compress_ratio in (0, 4, 128)`，会拒绝 V4.1 的 ratio 1/2；这比“存在共用模型类”
  更能说明当前整模路径的缺口。
- `hardware_backend/npu/dsv4/dsv4_memory_pool.py:326` 还断言
  `kv_layout is KVLayout.V4`，拒绝 V4.1 的 FP8/FP4 page layout。
  但通用池可选择 V4 layout，因此该布局断言本身不能独立证明整模不支持；应同时看压缩比门禁。
- 当前不能宣称 V4.1 在 NPU 已完整支持。已有 NPU prefill graph、ascend_ops allgather、
  arch35 MXFP8 wo_a、通用 DSpark/mHC 模块，也不等于 c1/c2、source 共享、candidate、
  Engram 与全部状态池契约已经接通。
- 背景和设计见 `sglang-docs/FE/BreakableNPUGraphBackend设计方案/`。

---

## 4. 推荐学习路径

### 4.1 读代码的顺序

1. `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx` 的 Model Introduction；
2. `configs/deepseek_v4.py` / `configs/deepseek_v41.py`：字段和默认值；
3. `models/deepseek_v4.py` 的类图：
   `MqaAttentionBase` → `MQALayer` → `DeepseekV4DecoderLayerBase` →
   `DeepseekV4Model` → `DeepseekV4ForCausalLM`；
4. 跟着一个 prefill batch 走一遍：
   `MQALayer.forward` → `forward_low_ratio_sources` → attention backend →
   `hc_post` → MoE；
5. 再看 indexer：`DeepseekV41Compressor` → `DeepseekV41Indexer` →
   `v41_indexer/scoring` → `topk`；
6. 然后 decode / target-verify / draft 三种 forward mode 的差异；
7. 最后看 `arg_groups/deepseek_v4_hook.py`：所有组合约束都在这里。

### 4.2 读完应该能回答的 10 个问题

1. 为什么 V4.1 每层只需要一份 512 维 KV latent，而 K 和 V 能共用？
2. `kv_source_layer_ids` 和 `index_source_layer_ids` 为什么不同？
3. ratio 2 和 ratio 1 的 compressor 分别做什么？为什么 ratio 1 还要 source 层？
4. `loc // ratio` 为什么不需要映射表？prefix cache 怎么跟着存活？
5. indexer 的 top-512 是在什么空间上选的？candidate 方案省了什么？
6. `is_sm100()` 和 `is_sm100_or_newer()` 为什么不能合并？
7. mHC 的 `(pre, post, comb)` 各在什么时候算、什么时候用？V4.1 的 lag 带来什么机会和限制？
8. Engram 为什么要在 hash 前做 token 规范化？表放在 CPU 和 GPU 分别换来了什么？
9. SWA bounded replay 为什么不是数值等价的？什么请求会退回完整计算？
10. DSpark 的 draft 有哪几部分？verify 窗口、Markov head、confidence head 各自解决什么问题？

### 4.3 建议的实验

- 用 V4.1-Flash 的 low-latency recipe 起一个 TP4/TP8 服务，确认启动日志里的
  resolved backend（`dsv4` / `flashinfer_mxfp4` / `flashinfer_cutedsl` /
  `flashmla_mega`）；
- A/B：`--engram-config '{"placement":"cpu"}'` vs 默认 GPU placement，
  看 KV pool 大小和 TTFT；
- A/B：`--enable-decoder-swa-bounded-replay` vs `--no-...`，
  对比 prefill 时间和 logprob 一致性；
- A/B：DSpark on/off，记录 accept length、TPOT、吞吐；
- 在 profiler 里按 `compressor` / `indexer` / `mHC` / `MoE` / `attention` 分组看
  kernel 占比，再对照 §3 的优化项。

---

## 5. 常见坑

| 坑 | 说明 |
| --- | --- |
| 手动覆盖 backend | `--attention-backend` / `--moe-runner-backend` / `--fp8-gemm-backend` 会破坏自动选路，32-wide ue8m0 掉到 Triton fallback |
| SWA bounded replay 当等价 | 它只处理最后 128 token；要 prompt logprob 或所有 hidden states 时自动退回完整计算；RL 对齐时显式关闭 |
| DSpark checkpoint 上开 EAGLE | V4.1 会拒绝；V4 0813 等曾能启动但接受率为 0，需按 checkpoint 区分 |
| V4.1 + PD 的 KV layout 不一致 | prefill/decode 必须一致；`trtllm` 目前不支持 PD |
| V4.1 + PP / HiSparse / HIP unified KV / two-batch overlap | 启动时显式拒绝，别浪费时间调 |
| Engram 表放 CPU 需要 host RAM | 189 GiB 级别；没有 huge pages 会慢几倍；`per_node` 要求单节点同 PID namespace |
| NPU 上跑 V4.1 layout | `NPUDeepSeekV4SingleKVPool` 断言 `KVLayout.V4`，v41 fp8/fp4 layout 不支持 |
| V4.1 输出不是 bitwise 稳定 | 两个默认 kernel 按单 token 形状保护；`--enable-deterministic-inference` 被拒绝 |
| page size | CUDA 256，NPU 128；C128 子池硬编码 256-token page |

---

## 6. 代码索引

| 主题 | 入口 |
| --- | --- |
| V4.1 配置归一化 | `python/sglang/srt/configs/deepseek_v41.py` |
| V4 配置默认值 | `python/sglang/srt/configs/deepseek_v4.py` |
| 模型主体 | `python/sglang/srt/models/deepseek_v4.py` |
| mHC kernel dispatch | `python/sglang/srt/models/deepseek_v4_mhc.py` |
| mHC 跨层边界 | `python/sglang/srt/models/deepseek_v4_boundary.py` |
| 非 CUDA stage | `python/sglang/srt/models/deepseek_v4_platform_stage.py` |
| SWA bounded replay | `python/sglang/srt/models/deepseek_v4_replay_tail.py` |
| DSpark 模型 | `python/sglang/srt/models/deepseek_v4_dspark.py` |
| DSpark components | `python/sglang/srt/speculative/dspark_components/` |
| Engram | `python/sglang/srt/layers/engram.py` |
| V4.1 compressor/indexer | `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` |
| V4.1 indexer 策略 | `python/sglang/srt/layers/attention/dsv4/v41_indexer/` |
| attention backend | `python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| MegaAttention | `python/sglang/srt/layers/attention/deepseek_v4_mega_backend.py` |
| TRT-LLM backend | `python/sglang/srt/layers/attention/deepseek_v4_trtllm_backend.py` |
| KV pool | `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` |
| 自动配置 / 校验 | `python/sglang/srt/arg_groups/deepseek_v4_hook.py`、`arg_groups/model_overrides/deepseek_v4.py` |
| NPU DSV4 backend | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` |
| NPU prefill graph | `python/sglang/srt/hardware_backend/npu/attention/dsv4_prefill_graph.py` |
| NPU KV pool | `python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py` |
| 用户文档 | `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`、`docs/docs/advanced_features/engram.mdx` |

---

## 7. 术语表

| 术语 | 含义 |
| --- | --- |
| latent / KV latent | 每位置 512 维、K/V 共用的压缩表示 |
| source layer | 产生压缩 latent 或 index K 的层，后续层共享 |
| c1 / c2 / c4 / c128 | 压缩比：1 对 1 / 2 合 1 / 4 合 1 / 128 合 1 |
| indexer | 小 fp4 side-attention，给压缩位置打分并选 top-k |
| candidate | V4.1 的二级筛选：source 发布 block，consumer 只算候选 |
| mHC | manifold-constrained hyper-connections，4 条残差流 + Sinkhorn |
| Engram | n-gram hash 查表记忆，gate 进残差 |
| DSpark | V4.1 自带的三 stage 投机 draft（Markov head + verify） |
| SWA | sliding window attention，V4.1 是 128 位置窗口 |
| bounded replay | 尾段只重放最后 128 token 的 prefill 捷径 |
| MegaAttention | V4.1 SM100/103 上融合 Q-RoPE + sparse attention + MXFP8 的 backend |
| BCG | breakable CUDA graph，分段图 + 段间 eager 断点 |
