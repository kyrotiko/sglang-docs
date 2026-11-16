# V4.1 A5 Prefill Graph 单文件验证

运行文件：[verify_dsv41_prefill_graph.py](verify_dsv41_prefill_graph.py)。只需复制这一个 `.py`，不依赖本目录其他脚本，也不需要下载模型或加载权重。

用途：在当前 A5 **真实 serving 环境**中收集决定实现方向的算子、metadata、graph replay 和状态证据。预期主仓基线为 `dsv41-a5@5633d68b04`；版本不同时会在报告中记录实际 SHA/API，而不是自动切换到旧接口或 mock。

当前脚本为 revision 2。首轮 [结果与原始证据](../validation/a5-20261116-run1/findings.md)已归档：三组 attention 的 graph/eager 全部 exact，CPU 参考另有差异。修订版修复了把 candidate block=8 混作 index PA page 的夹具错误，默认 index page=64，并将 capture、eager/replay 与 CPU reference 独立显示；原始失败不会被重写为通过。

## 在机器上运行

进入已安装 torch_npu、sgl-kernel-npu 和 SGLang 的 A5 环境，使用单张 NPU：

```bash
python verify_dsv41_prefill_graph.py \
  --device 0 \
  --output-dir ./dsv41-prefill-evidence
```

如果需要显式使用源码 checkout，增加主仓路径：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /path/to/sglang \
  --device 0 \
  --output-dir ./dsv41-prefill-evidence
```

`--output-dir` 必须是新目录。默认每组探针最长 600 秒，包含首次 DSL/JIT 编译时间；需要时用 `--timeout 1200`。不通过 `torchrun` 启动，不需要 HCCL 初始化。

运行结束，把下面这个文件发回来即可，**即使有 FAIL/BLOCKED 也请保留**：

```text
dsv41-prefill-evidence/dsv41-prefill-evidence.tar.gz
```

包中包含 `report.json`、`report.md`、每组探针的 `.json` 和完整 `.log`，以及运行脚本快照和 hash。父进程逐组输出状态；每组独立进程，超时或崩溃会保留已完成 case 和最后阶段，继续其他组。

首次运行可以选择默认全套。基于首轮回传，本次只需用新目录补跑四组：

```bash
python verify_dsv41_prefill_graph.py \
  --sglang-root /path/to/sglang \
  --probes environment,qli_c2,candidate_chain,attention_r0 \
  --index-page-size 64 \
  --output-dir ./dsv41-prefill-evidence-r2
```

environment 会额外记录公开 compressor/epilog exports、签名/文档、auto 实现解析、npu/custom schema 和主仓实际索引页常量。若真实 serving 另有注册模块，用 `--import-module MODULE` 显式指定，可重复；不要凭旧 op 名猜模块，也不会默认加载某个模块来掩盖缺口。

可用 `--list` 列出名称，`--help` 查看参数。

## 实际验证内容

| 探针 | 事实范围 |
| --- | --- |
| `environment` | Python/torch/torch_npu/包版本、NPU/CANN 信息、实际 SGLang SHA/dirty 状态、当前 DSL API 签名与源码 hash、custom op 注册；仅检查环境，不证明算子能执行 |
| `geometry` | 当前 SGLang 的 c1 行序、c2 compact、window/cmp lengths 和 padding mask，独立 CPU 数学对照，再做 NPUGraph 多轮 replay；另检查请求槽 0 预留 |
| `metadata` | 真实 window/c1/c2/QLI/candidate/QSLI metadata API；固定 bucket 改 prefix/live T 后的 shape、dtype、地址和内容 hash。planning 本来就在图外，不要求 metadata op 本身入图 |
| `compressor` | 真实 `compressor_v2`，有效输出与完整 state 对照；private bank 0/1/2 切换、无新 pair、连续 `[1,1]` chunk、ring wrap 和 bank 复用；额外比较 split 与 concat pair |
| `epilog_fp8` / `epilog_fp4` | 真实 `kv_compress_epilog_v2` 的两种 V4.1 格式，eager/replay 字节对照；`-1` 行及全 `-1`，无效输入填 NaN，检查未触及的 guard cache |
| `prologue` | 真实 K prologue storage mode 0/1，NZ 权重转换在图内，验证 `-1`、guard rows、分离 K/scale 与 combined record 格式一致性 |
| `qli_c2` | 实际 ratio-2 QLI，修改 packed Q/K、weights、cu、KV 长度和页表后 eager/replay 对照及因果边界检查 |
| `candidate_chain` | 实际 ratio-1 QLI candidate → QSLI 同图数据流，consumer 直接读本轮 producer 的结果；同时检查 top-k、候选有效区和候选长度 |
| `attention_r0/r1/r2` | 真实 MQSMLA window/c1/c2；改变 Q、packed KV、prefix、live T、页表后 replay 对照；选取首尾有效行和 3 个 heads 与 CPU FP32 reference 比较 |
| `engram_break` | 当前真实 EngramHasher fallback 与共享 breakable graph；使用小型合成 hash 参数、identity token map，检查动态 history、固定 hash 输出、图内 consumer 和每次 replay 恰好执行一次 break；另验 verify 不提前提交及 commit_after_verify |

图探针都使用当前输入做 eager reference，再从相同初始状态 replay；包含 A → B → C → D → A，不能只用 capture 的原输入 replay 一次。压缩输出只比较有效行，state/cache 比较完整存储。attention 有效行之外预填 NaN，用于观察 kernel 是否写 padding 行，不把 padding 必须为零当作算子既定契约。

默认参数是合成尺寸，不是推荐部署配方：T=64、主 attention heads=64、head_dim=512、indexer heads=32、index_dim=128、topk=512、full page=128、**index PA page=64**、**candidate block size=8**、candidate blocks=2048、compressor hidden=4096、ratio-2 ring=2（对应非投机路径）。`--index-page-size` 必须整除 `--page-size/2`；例如测试 index page=128 时设置 `--page-size 256`。

确认默认结果后，可以用新目录扩大 bucket，例如 `--tokens 128`；压缩投影宽度可以用 `--hidden-size` 指定，DSpark 对应的较宽 private ring 可单独用 `--state-ring-size 8` 复跑。不能将这些 synthetic 变体当成真实模型或 DSpark 整体验证。

## 如何理解结果

- `PASS`：只对该探针记录的范围、shape 和 case 成立。
- `BLOCKED`：缺少运行依赖、NPU 或指定 API；不是算子数值失败。
- `FAIL`：查看 `phase`、`capture_supported` 和 traceback。在 warmup/eager 就失败，不能据此认定是 graph 不支持。
- `TIMEOUT` / `CRASH`：保留最后 checkpoint、已完成 case 和进程退出码；需要结合机器日志分析。
- 进程返回值：0 为所有选中探针通过，1 有失败/超时/崩溃，2 只有环境/API 阻塞，130 为用户中断；这些情况都会尽量产出结果包。

浮点 eager/replay 默认 `--rtol 1e-3 --atol 1e-3`；CPU FP32 参考单独使用 `--reference-rtol 1e-3 --reference-atol 1e-3`，本次没有放宽默认值。整数、hash、top-k 和 cache bytes 精确比较。失败会记录误差和小段样本，不能为了变绿而忽略误差。QLI/QSLI 的有序 top-k 不一致可能涉及并列分数，需根据原始证据分析；不把无序相等自动当作通过。

attention 的 CPU reference 是部分行/head 的独立算术对照，eager/replay 则比较全部有效输出。即使 CPU reference 有差异，也继续收集 graph parity，并在 `checks` 和摘要各列分开记录。新增 BF16 ULP、归一化 L2、最大误差坐标、最近 BF16 舍入值及一种 BF16-softmax 算术假设的诊断，均不代替 FP32 oracle 或改变严格判定。padding 另记有限值数量、零值数量和最大绝对值。

本地上游参考包与 A5 分支的某些 API 支持说明存在版本差异，尤其是 `compressor_v2` 和 custom epilog 注册。脚本使用本分支实际调用的 API 名，记录部署包签名；某组无法执行时，其余组仍能提供证据，不静默换成其他代际算子。

## 明确未覆盖

不加载真实权重，所以不能验证完整模型、真实 Engram lookup/通信、mHC 多流、真实 allocator/source-group 生命周期、DSpark packed aux hidden/draft KV 注入、多卡 MoE/DP collective 或 admission 一致性，也不测准确率和性能收益。探针为后续实现提供事实基础，不能替代整模验收。

报告中的 graph 状态来自实际探针；脚本不修改主仓代码、不替换 capability，也不启动模型服务。

## 本地无 NPU 时的检查

```bash
python verify_dsv41_prefill_graph.py --self-test
```

该模式仅检查压缩计数边界、打包往返、padding 和页表容量，需要 CPU torch；不会输出任何 NPU 已通过结论。`--help` 和 `--list` 不导入 torch。
