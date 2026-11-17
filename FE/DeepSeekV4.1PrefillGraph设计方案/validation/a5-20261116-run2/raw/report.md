# DeepSeek V4.1 A5 prefill graph probe

Expected source baseline: `5633d68b04`

| Probe | Result | Capture | Eager/replay | CPU reference | Last phase | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| environment | PASS | - | - | - | complete | 44.632 |
| qli_c2 | PASS | PASS | PASS | - | complete | 54.313 |
| candidate_chain | PASS | PASS | PASS | - | complete | 60.974 |
| attention_r0 | FAIL | PASS | PASS | FAIL | passed:repeat_A | 92.007 |

## Failures / blocked probes

- **attention_r0 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.

## Interpretation

PASS applies only to a probe's recorded scope and cases. Metadata planning is eager. Environment/API presence is inspection, not device execution evidence. An operator FAIL before capture is not proof of graph incompatibility. No source files or serving configuration are modified.

## Not covered

- Full model, real checkpoint/tokenizer/Engram lookup tables
- mHC operator and multi-stream joins
- Real serving allocator/source-group lifetime and cache sharing
- Complete compressor/RoPE/cache/indexer/attention integration
- DSpark packed aux hidden and draft KV injection
- DP/TP/EP collectives, admission consensus, idle ranks
- End-to-end accuracy, throughput, latency and capture memory budget
