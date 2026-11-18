# A5 算子探测结果（2026-11-18）

## 环境

| 项目 | 值 |
| --- | --- |
| NPU | `Ascend950PR_9589` |
| `torch_npu` | `2.13.0` |
| `DSV41_INDEX_PAGE_SIZE` | `64` |
| host 注册方式 | ACL（`register_acl_host_memory`） |
| 探测脚本 | `validation/scripts/a5_kv_exchange_probe.py` |
| 运行命令 | `python a5_kv_exchange_probe.py --impl ascendc_aiv --out probe_out/op_aiv_l2` |

## 结果

`layers=2`，`pages=2`，`dim=64`，native index page `K=64`：

| 实现 | `page_size` | layers | D2H | H2D | 结果 |
| --- | --- | --- | --- | --- | --- |
| `ascendc_aiv` | 64 | 2 | byte-identical | byte-identical | PASS |
| `ascendc_aiv` | 128 | 2 | byte-identical | byte-identical | PASS |

早先的 `layers=1` 对照：

| 实现 | `page_size` | layers | 结果 |
| --- | --- | --- | --- |
| `ascendc_aiv` | 64 | 1 | PASS |
| `ascendc_aiv` | 128 | 1 | PASS |
| `torch_native` | 64 / 128 | 1 | PASS |

## 结论

1. **full-page 传输成立**：把设备侧 `[layers, pages*I, K=64, 1, dim=64]` 视作 full-page `[layers, pages, S=128, 1, dim=64]`，host 用 `page_first [pages, layers, S, 1, dim]`，`page_size=S=128` 时 AIV 的 D2H/H2D 都逐字节正确。
2. **`page_first` 的 layer gap 成立**：`layers=2` 时相邻 host 页之间隔着另一个 layer，算子仍能正确寻址；不需要退回“host 按 native page 行”的方案。
3. **per-layer H2D 成立**：按层传 `[1, pages, S, 1, dim]` 的 device 视图和 host 层视图，回载结果正确。

## 仍未验证

- sglang 侧的 `_dsv4_low_ratio_regions` / `DeepSeekV4PagedHostPool` 装配：NPU 4-D KV 的 `item_bytes`、indexer 的 2-D→4-D 规范化、`_host_page_view` 的实际形状。
- G10：只开 `--enable-hierarchical-cache` 不给 ratio 的启动路径。
- 端到端命中：device 淘汰后 host 命中、`host_hit` 指标、输出一致性。

注意：本次 A5 环境报告的 `sglang` 是 `0.5.23.dev636+gd0a756543`，不是设计基线 `dsv41-a5 @ 9898f7ce9a`。算子探测只依赖 `DSV41_INDEX_PAGE_SIZE` 和 `sgl-kernel-npu`，因此结论仍有效；但 sglang 侧 E2E 必须在基线 commit 上跑。
