# DeepSeek V4.1 A5 prefill graph probe

Expected source baseline: `d3700bf0a9`

| Probe | Result | Capture | Eager/replay | CPU reference | Last phase | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| environment | PASS | - | - | - | complete | 26.5 |
| idle_chain_r2 | FAIL | PASS | FAIL | - | comparison_failures_collected | 166.443 |
| active_chain_r2 | FAIL | PASS | FAIL | - | comparison_failures_collected | 179.393 |

## Failures / blocked probes

- **idle_chain_r2 / FAIL**: AssertionError: 1 case(s) failed comparison; all completed comparisons retained
- **active_chain_r2 / FAIL**: AssertionError: 1 case(s) failed comparison; all completed comparisons retained

## Interpretation

PASS applies only to a probe's recorded scope and cases. Metadata planning is eager. Environment/API presence is inspection, not device execution evidence. An operator FAIL before capture is not proof of graph incompatibility. No source files or serving configuration are modified.

## Not covered

- Full model, real checkpoint/tokenizer/Engram lookup tables
- Full mHC state/post/external norm chain and multi-stream joins
- Real serving allocator/source-group lifetime and cache sharing
- Real model compressor/RoPE/cache/indexer/attention integration
- DSpark packed aux hidden and draft KV injection
- DP/TP/EP collectives, multi-rank idle replay and admission consensus
- End-to-end accuracy, throughput, latency and capture memory budget
