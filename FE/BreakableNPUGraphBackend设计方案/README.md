# BreakableNPUGraphBackend

让 Ascend NPU 的 prefill 可以入图：沿用 CUDA 的 breakable graph 框架，每段用一个
`torch.npu.NPUGraph`，吃 host 侧输入的算子（attention、compressor、indexer）在段
之间 eager 执行。

## 当前状态

已在 DSV4 Flash（DP16/TP16/EP16）上跑通 capture 与在线请求。

| 项 | 值 |
| --- | --- |
| 开关 | `--cuda-graph-backend-prefill breakable`（NPU 上只接受 `breakable`，其他值报 `ValueError`） |
| bucket | `--cuda-graph-bs-prefill`，**必须 >= 128**，见下 |
| 相关提交 | `d08d2b52b0` P1 接线 / `1c095d6948` DSV4 断点 / `85860762c4` rope 口径 |

已知限制：

- **bucket 必须 >= 128。** 小于压缩粒度 128 的 bucket 会被 sparse-attn metadata
  AICPU 算子拒绝（`execute kernel param invalid`）。capture batch 对该算子而言
  与一个真实的 N token 单请求 prefill 无法区分，所以 eager 下的超短 prompt 多半
  同样不可用；这是 DSV4-on-NPU 的既有限制，不是入图引入的。未做自动过滤。
- decode 不走这条路（已有全图后端）；EAGLE/draft prefill、段间 dedup 不支持。
- `docs/docs/advanced_features/breakable_cuda_graph.mdx:169` 的平台兼容表还写着
  NPU 不支持，待更新。

## 该看哪一份

| 文档 | 内容 |
| --- | --- |
| [backend design](breakable_npu_graph_backend_design.md) | 框架怎么接上 NPU、真机验证过的 NPUGraph 行为、两个必须知道的坑 |
| [DSV4 Flash prefill graph](dsv4_flash_prefill_graph.md) | DSV4 Flash 的断点契约、修过的三个 bug、未覆盖的风险 |
| [MLA / DSA 适配](mla_dsa_prefill_adaptation.md) | 把其他 MLA / DSA 模型接进来的计划（未实现） |

spike 脚本（真机跑过，结论见 backend design 的事实表）：
[v2](v2_spike_two_segments.py) 共享池连捕两段 ·
[v4](v4_weak_ref_spike.py) 弱引用跨重放 ·
[v8](v8_raw_capture_spike.py) 裸捕获与在途 kernel
