# GLM-5.2 prefill graph：采集 replay 卡住的日志

`6ceb3d9fa7` 真机反馈：能完成启动和捕图，发出几个实际命中 graph 的请求后再次挂住、NPU 100%。当前尚未验证该提交解决原问题。本轮只增加可关闭的 host 诊断与回退开关，继续采证。

诊断提交：`sglang@696f7fb65e`（`support_breakable_graph_back`）。11 项隔离 CPU 回归、Ruff、语法与差异空白检查通过，未进行 NPU 真机复现。

## 第一次：保留新 indexer 断点，抓复现

在所有服务器节点启动前设置：

```bash
export SGLANG_NPU_GRAPH_DEBUG=1
export SGLANG_NPU_DSA_INDEXER_EAGER_BREAK=1
export PYTHONUNBUFFERED=1
```

沿用原启动参数，收集完整 stdout/stderr（启动命令末尾追加 `2>&1 | tee /tmp/glm52-prefill-graph-debug.log`）。沿用能复现的负载，保存从启动、捕图、发请求到 watchdog 的完整文件，包含所有 rank 和原始 traceback。

## 日志记录什么

统一前缀 `[NPU_GRAPH_DEBUG]`，JSON 字段含 pid、rank、host 单调时钟、事件序号、phase、run、桶大小，以及张量 shape/dtype/device/存储地址。batch 只记录形状与 CPU 长度前八项，不记录请求文本。

- 详细日志默认只覆盖实际请求与 replay；启动时保留每桶完成摘要和异常。target/draft 用 `draft` 字段区分。
- 需要排查启动时额外设置 `SGLANG_NPU_GRAPH_DEBUG_CAPTURE=1`，可记录 `phase=warmup/capture`、原有同步/barrier、每个图段的 capture begin/end；桶较多时日志量很大。
- replay 的 `segment.submit.begin/return`，每个 eager break 的函数名、层号及进出。
- indexer 的当前 batch/真实 token 数、输入布局、页表地址、投影、attn-TP gather、K/scale cache 写入、长度转换、top-k、固定输出复制。
- attention 的 KV 写入、Q/KV 长度 `.to()` 及 sparse attention 提交。

`return` 仅表示 Python/API 调用返回，不能证明异步设备任务已完成。诊断不会新增 NPU 同步、设备张量 `.item()`/`.cpu()`/`.tolist()`；CPU 长度镜像允许读取。大量日志会影响 host 下发时序，仅用于复现。

## 第二次：有需要再对照原路径

保持诊断开启，将 `SGLANG_NPU_DSA_INDEXER_EAGER_BREAK=0` 后重启重捕图，可以回到加 indexer eager 断点前的路径，保留所有诊断点。先抓第一份日志，再根据最后推进位置决定是否需要这轮对照。

主诊断、indexer 断点开关默认分别为 `0`、`1`，捕图详情开关默认 `0`；未启用诊断时不输出这些日志。此轮不改变计算或同步算法。
