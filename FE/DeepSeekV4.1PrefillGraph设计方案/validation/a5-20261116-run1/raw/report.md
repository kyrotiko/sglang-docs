# DeepSeek V4.1 A5 prefill graph probe

Expected source baseline: `5633d68b04`

| Probe | Result | Last phase | Seconds |
| --- | --- | --- | --- |
| environment | PASS | complete | 23.333 |
| geometry | PASS | complete | 18.864 |
| metadata | PASS | complete | 20.752 |
| compressor | FAIL | warmup | 10.895 |
| epilog_fp8 | BLOCKED | start | 10.591 |
| epilog_fp4 | BLOCKED | start | 10.783 |
| prologue | PASS | complete | 21.908 |
| qli_c2 | FAIL | warmup | 20.265 |
| candidate_chain | FAIL | warmup | 16.632 |
| attention_r0 | FAIL | passed:repeat_A | 89.563 |
| attention_r1 | FAIL | passed:repeat_A | 121.741 |
| attention_r2 | FAIL | passed:repeat_A | 132.405 |
| engram_break | PASS | complete | 18.697 |

## Failures / blocked probes

- **compressor / FAIL**: UnsupportedGenerationError: attention.compressor_v2 [impl=auto] does not support generation=arch35; supported generations=['arch22']
- **epilog_fp8 / BLOCKED**: custom.kv_compress_epilog_v2 is not registered in this serving environment
- **epilog_fp4 / BLOCKED**: custom.kv_compress_epilog_v2 is not registered in this serving environment
- **qli_c2 / FAIL**: ValueError: PA block_size must be a multiple of 16 in (0, 1024]
- **candidate_chain / FAIL**: ValueError: PA block_size must be a multiple of 16 in (0, 1024]
- **attention_r0 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.
- **attention_r1 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.
- **attention_r2 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.

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
