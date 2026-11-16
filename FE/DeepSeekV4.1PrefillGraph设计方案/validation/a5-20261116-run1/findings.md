# A5 首轮探针事实与后续动作

结果创建时间：2026-11-16 16:57:21 UTC。主仓 `dsv41-a5@5633d68b04ad1bdd4faecb1c8e370e5c5ea6403c`，机器侧工作树干净。原始结果见 [raw/report.md](raw/report.md)、[raw/report.json](raw/report.json)，文件校验见 [manifest.json](manifest.json)。回传文件只按数据读取，未运行其中的 Python 文件。

## 1. 可以据此确认什么

**NPUGraph 和关键计算已取得正面证据。** window/c1/c2 MQSMLA 三组各 5 个 case 的全部有效输出，eager 与 replay 逐元素一致（`exact=true`、`max_abs=0`）。三组总状态 FAIL 来自额外的 CPU FP32 参考检查，不是 capture 或 replay 失败。

当前尚不能宣布完整 V4.1 支持：compressor 的部署 API 拒绝 arch35；V4.1 KV epilog 未在探针进程注册；QLI/QSLI 被脚本错误的 PA 页形状拦住；整模/source 生命周期和 DSpark 未测。

## 2. 环境与来源

| 项目 | 实际值 |
| --- | --- |
| NPU | Ascend950PR_9589（A5），单卡探针 |
| 系统 | aarch64 Linux，Python 3.12.15 |
| torch / torch_npu | 2.13.0+cpu / 2.13.0 |
| sgl-kernel-npu | 2026.11.0.post4 |
| CANN 路径 | `/usr/local/Ascend/cann-9.2.0` |
| 输入 | T=64，B=1，主 heads=64，indexer heads=32，topk=512，full page=128，candidate block=8 |
| 旧脚本的索引 PA page | **8，属于脚本夹具错误**；主仓当前为 64/128 |
| 图/CPU 参考阈值 | 首轮都为 `rtol=1e-3, atol=1e-3` |
| 原脚本 SHA-256 | `58140f4821fa8106fafc0205297475891c2947bf802c5a05231a5456ee6fb387` |

回传脚本 hash 与 manifest 一致；与本地已提交脚本的差别仅为 CRLF/LF，换行规范化后逐字一致。记录的 sglang package version 与源码 SHA 不是同一概念，本轮以实际导入路径和 git SHA 为准。

机器当时还有其他 NPU 负载。探针的 wall time 包含 JIT、CPU reference、同步及其他负载，不是性能收益证据。

## 3. 逐项判读

| 探针 | 原状态 | 已确认事实 | 尚未证明 |
| --- | --- | --- | --- |
| environment | PASS | 硬件、包版本、实际 API 签名与源码 hash 可用 | API 存在不等于可执行 |
| geometry | PASS | 5 轮图回放与 CPU 几何一致；c1 行序、c2 compact、window/cmp length、显式 live mask、请求槽 0 预留通过 | 主仓尚未接入 prefill live mask；真实 compressor bank 隔离未测 |
| metadata | PASS | 15 组配置生成成功；6 类 metadata 均为 int32 `[1024]`；每类固定地址、内容随输入改变，重复 A 恢复同 hash | metadata planning 本身未入图，也无需作为首版前置 |
| compressor | FAIL / warmup | `compressor_v2` 的 registry 拒绝 `arch35`，仅列出 `arch22` | 未执行 compressor kernel，更没有验证图、ratio-2 carry 或 state bank |
| epilog_fp8/fp4 | BLOCKED | 探针导入路径下 `custom.kv_compress_epilog_v2` 未注册 | 不能推广为机器不存在其他 writer，也未验证这两种 V4.1 格式写回 |
| prologue | PASS | 两种 storage mode，4 轮图对照全部字节一致；`-1`/all-skip/guard rows、K+scale 与 combined records 一致 | identity RoPE、合成权重；完整 source 链未测 |
| qli_c2 | FAIL / warmup | DSL 在参数检查拒绝 PA page 8，要求 16 倍数 | 没有进入计算/capture；不能据此判定 QLI 不可入图 |
| candidate_chain | FAIL / warmup | 同样在第一个 QLI 被 page 8 拦住 | QSLI 尚未执行，candidate 跨算子图链未验证 |
| attention_r0/r1/r2 | FAIL（总状态） | 3×5 轮有效输出 eager/replay 均逐元素一致 | CPU reference 有少量差异，未等同于独立数值验收通过 |
| engram_break | PASS | 5 轮 hash/history/图内 consumer 精确一致；每轮一次 break；verify 不提前提交和后续提交通过 | 小型 identity token map，无真实查表、通信或整模 |

geometry 的 c2 容量 33 行与 Python 规划一致，只证明行选择/固定 buffer，不证明 compressor 实际返回此容量。padding attention 输出均被观测为 finite，但旧脚本未记录零值/最大绝对值，不能断言其为零。

## 4. Attention 的两项检查必须分开

| 算子 | eager/replay | CPU FP32 参考失败位置 | 采样最大绝对误差 |
| --- | --- | --- | --- |
| window | 5/5 case，exact | `short_B`，row0，heads 0/63 | 0.0018430352 |
| c1 | 5/5 case，exact | `short_B`，row0，head63 | 0.0018134117 |
| c2 | 5/5 case，exact | `short_B`，row0，head0 | 0.0013743341 |

`short_B` 的 prefix=1、live T=31，失败点是 position=1。BF16 输出舍入、softmax 中间精度或参考契约差异是待核实解释；现有证据不能选定根因，也不能直接放宽阈值记为通过。

修订脚本分别报告 capture、eager/replay 和 CPU reference；保持原参考阈值，增加 BF16 ULP、归一化 L2、最大误差坐标和一种明确标记为假设的 BF16-softmax 算术诊断。诊断不改变原严格参考状态。

## 5. 确认是脚本问题的修复

旧脚本将 candidate block=8 同时用于 index PA page。主仓 `deepseek_v4_memory_pool.py::dsv41_index_page_size()` 返回 64 或 128，二者均满足部署 QLI 的页约束。因此不应据旧探针修改主仓索引布局。

修订版把 `--index-page-size` 独立出来，默认 64，candidate block 仍为 8。统一修复 QLI K/scale、expanded index page table、prologue cache 和 QSLI combined records：索引页 S 对应 QSLI `[P,S/8,544]`，不是固定 `[P,1,544]`。CPU 自检覆盖多种索引页及 record 往返。

## 6. 接口缺口及主仓变更边界

`compressor_v2` 的 arch35 拒绝与当前主仓 `_ds41_ops()` 的选择直接相关，需要解决部署接口匹配后才能覆盖 V4.1 source。主仓 writer 也直接使用未注册的 `custom.kv_compress_epilog_v2`，但报告只能证明本探针导入后的状态；若真实 serving 另有注册，必须显式记录入口。

本地只读参考仓另有 `attention.compressor` 和 `mem_cache.kv_compress_epilog`，但不能只改名字：前者包含不同的融合 RMSNorm/RoPE 与显式状态寻址；后者的参考格式为旧式 608/640B，不等于 V4.1 的 544/320B。不能直接替换或静默 CPU fallback。

修订脚本的 environment 会补采公开 compressor/epilog 导出、实际签名、auto 实现解析、`npu/custom` schema 和主仓索引页常量；可用 `--import-module` 显式加入真实 serving 使用的注册模块，默认不猜测导入。获得准确契约后再改主仓调用和布局。

## 7. 下一轮最小重跑

先用修订版跑以下四组即可，保留首轮其余正面证据：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /path/to/sglang \
  --probes environment,qli_c2,candidate_chain,attention_r0 \
  --index-page-size 64 \
  --output-dir ./dsv41-prefill-evidence-r2
```

默认没有放宽任何数值阈值。若真实服务有额外 custom op 注册模块，按服务实际配置增加 `--import-module MODULE`，不要凭接口名字猜测。回传新的完整证据包后，先解决 API 路由/格式，再推进主仓 prefill 改动；不需要重跑已通过的 geometry/Engram 来重复证明同一事实。
