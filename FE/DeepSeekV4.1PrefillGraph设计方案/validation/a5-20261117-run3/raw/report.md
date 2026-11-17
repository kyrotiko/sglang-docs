# DeepSeek V4.1 A5 prefill graph probe

Expected source baseline: `9898f7ce9a`

| Probe | Result | Capture | Eager/replay | CPU reference | Last phase | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| environment | PASS | - | - | - | complete | 41.041 |
| geometry | PASS | PASS | PASS | PASS | complete | 20.24 |
| metadata | PASS | - | - | - | complete | 20.256 |
| compressor | PASS | PASS | PASS | - | complete | 12.478 |
| epilog_fp8 | PASS | PASS | PASS | - | complete | 12.027 |
| epilog_fp4 | PASS | PASS | PASS | - | complete | 12.32 |
| prologue | PASS | PASS | PASS | - | complete | 22.507 |
| qli_c2 | PASS | PASS | PASS | - | complete | 52.984 |
| candidate_chain | PASS | PASS | PASS | - | complete | 57.164 |
| attention_r0 | FAIL | PASS | PASS | FAIL | passed:repeat_A | 89.413 |
| attention_r1 | FAIL | PASS | PASS | FAIL | passed:repeat_A | 121.465 |
| attention_r2 | FAIL | PASS | PASS | FAIL | passed:repeat_A | 119.816 |
| engram_break | PASS | PASS | PASS | PASS | complete | 19.199 |
| mhc_premix | PASS | PASS | PASS | - | complete | 14.107 |
| writer_attention_r0 | FAIL | - | - | - | public_writer_prepare:full_A | 13.744 |
| writer_attention_r1 | FAIL | - | - | - | public_writer_prepare:full_A | 14.804 |
| writer_attention_r2 | FAIL | - | - | - | public_writer_prepare:full_A | 15.08 |

## Failures / blocked probes

- **attention_r0 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.
- **attention_r1 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.
- **attention_r2 / FAIL**: AssertionError: Graph parity completed, but the sampled CPU attention reference mismatched; inspect cpu_reference separately.
- **writer_attention_r0 / FAIL**: ValueError: cache dtype does not match quant_mode: mxfp8_bf16 requires float8_e4m3fn/float8_e5m2 and mxfp4_bf16 requires uint8
- **writer_attention_r1 / FAIL**: ValueError: cache dtype does not match quant_mode: mxfp8_bf16 requires float8_e4m3fn/float8_e5m2 and mxfp4_bf16 requires uint8
- **writer_attention_r2 / FAIL**: ValueError: cache dtype does not match quant_mode: mxfp8_bf16 requires float8_e4m3fn/float8_e5m2 and mxfp4_bf16 requires uint8

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
