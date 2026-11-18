# GPT 分支：idle prefill metadata 接入

日期：2026-11-18。代码分支 `dsv41-a5-graph-gpt`，提交 `182ba91e091da65f70ab64e71b04f6d3084011c1`，基于 `d3700bf0a9`。实现范围是现有框架已物化为 B=1、bucket-shaped EXTEND 的 idle DP rank；实际多 rank A5 整模结果仍待验证。

## 1. 这次实际改了什么

生产代码仍只改原有两个文件，新增两项生命周期回归：

- `dsv4_prefill_graph.py`：增加固定 owner 的 idle refresh，使用已测物理 query/KV 长度 `T_cap`，CPU/GPU cu=`[0,T_cap]`；`seqused/start_pos/live_mask/cycle-bank/SWA loc` 归 0；c1/c2 loc=`-1`、positions=0，full/SWA 表映射到保留 page 0。index 表复用既有 expansion，保留 reserved full page 的 index subpage offsets。
- `ascend_dsv4_backend.py`：零有效行进入上述 refresh，跳过依赖真实请求的 eager planner；QLI/QSLI 的规划与 captured compute 统一读取 owner 的固定 KV bound；kernel metadata 更新后确保 idle window 第一索引为 0、其余为 `-1`。
- 既有 low-ratio 测试文件：覆盖 live→idle→live 的固定地址、页表别名、index offsets、真实 bank、top-k/candidate 引用及下一轮 live 恢复；覆盖原始长度为 0 的 idle batch 不调用真实请求 planner，且不修改该 batch 的真实 token 计数。

没有修改公共 runner/base、调度投票、MoE 算法、并行配置、state ring 大小或数值阈值。未物化的 native idle 仍按现有 runner 准入规则处理；这里不是新增一套强制准入。

## 2. 回归证据

CPU 容器 `sglang-npu-dev`，独立副本 `/tmp/dsv41-idle-integration-JqnLVVvB`；`PYTHONPATH` 指向本分支归档后的 `python/`，执行前检查实际 `sglang.__file__`。没有从其他实现 checkout 导入源码。

- 修复前：新用例在 `d3700bf0a9` 上复现与设备日志相同的 `V4.1 prefill graph got idle rows or token capacity overflow`。
- 修复后：12 个文件、**289 项全部通过**。low-ratio 108 项，固定 metadata 5 项，其他对应回归 176 项；每文件独立 Python 进程。
- 编译、格式、基础 lint 和 diff whitespace 检查通过；保留原 low-ratio fixture 的既有 E402 例外。
- 新增的两项回归检查 page size 128/256 的 index 子页映射和生命周期，不把 CPU 结果当作这些设备组合已通过。

idle 的设备协议依据是[第五轮](a5-20261117-run5/findings.md)、[第六轮](a5-20261117-run6/findings.md)的单卡合成链与状态保护。它们不替代本提交的 hooks/多流/DeepEP/整模验收。

## 3. A5 上运行自己的新提交

之前的采集主仓为 `24deba1e5f…`。本次要验证自己的 `182ba91e09`，并确认实际 Python 导入也指向对应源码。可在 A5 创建独立测试 worktree，保留其他 agent 的 checkout 不变。下面是首次创建示例：

```bash
cd /mnt/share/y00627667/repos/sglang
git fetch https://github.com/kyrotiko/sglang.git dsv41-a5-graph-gpt
git worktree add --detach ../sglang-graph-gpt-182ba91 FETCH_HEAD
cd ../sglang-graph-gpt-182ba91
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse --short HEAD
python -c 'import sglang; print(sglang.__file__)'
```

HEAD 应为 `182ba91e09`，导入路径应在新 worktree 的 `python/sglang/` 下。随后用原先已能启动的 V4.1 A5 命令重启服务，保持 TP/EP/DP、MoE、量化、DSpark 和 graph buckets 等参数一致；若启动脚本自行设置源码路径，应对应到这个测试目录。cookbook 没有给出本分支 V4.1 A5 配方，本轮不编造替代部署参数。

先使用此前真正触发 replay metadata 错误的原请求，验证一个 rank 活跃、其余 dummy rank 的场景；短请求若因原有 padding 准入规则走 eager，不据此判断 idle 接入失败。再做不同 prompt 同 bucket、live/idle 轮换和后续 decode。

需要保留：各 rank 是否都完成本轮、metadata 异常/通信等待是否仍出现、实际 graph hit、输出及真实 cache/state/后续生成的 eager 对照。服务启动或没有 Python 异常，只能算联调进展。

## 4. 已知独立问题

[第七轮](a5-20261117-run7/findings.md)的 ratio-2 odd-prefix 长 extend 首 pair 不稳定问题保持原 FAIL，算子方[复现材料](../analysis/compressor_odd_prefix_reproducer.md)不变。idle metadata 接入不宣称修复它，也不以它为由自动禁用 idle。完整正确性验收仍需该问题与真实多 rank 路径分别闭合。
