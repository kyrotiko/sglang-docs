# DeepSeek V4.1 A5 prefill graph probe

Expected source baseline: `9898f7ce9a`

| Probe | Result | Capture | Eager/replay | CPU reference | Last phase | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| writer_attention_r0 | FAIL | PASS | PASS | FAIL | passed:repeat_A | 100.196 |
| writer_attention_r1 | FAIL | PASS | PASS | FAIL | passed:repeat_A | 123.577 |
| writer_attention_r2 | FAIL | PASS | PASS | FAIL | passed:repeat_A | 121.554 |

## Failures / blocked probes

- **writer_attention_r0 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.
- **writer_attention_r1 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.
- **writer_attention_r2 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.

## Interpretation

PASS applies only to a probe's recorded scope and cases. Metadata planning is eager. Environment/API presence is inspection, not device execution evidence. An operator FAIL before capture is not proof of graph incompatibility. No source files or serving configuration are modified.

## Not covered

- Full model, real checkpoint/tokenizer/Engram lookup tables
- Full mHC state/post/external norm chain and multi-stream joins
- Real serving allocator/source-group lifetime and cache sharing
- Complete compressor/RoPE/cache/indexer/attention integration
- DSpark packed aux hidden and draft KV injection
- DP/TP/EP collectives, admission consensus, idle ranks
- End-to-end accuracy, throughput, latency and capture memory budget
