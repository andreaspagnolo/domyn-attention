# Cached decoding benchmark

`benchmark.py` compares the four attention variants using one-token decoding at
fixed prefix lengths. It measures latency and returned-cache tensor storage,
counts parameters, and reports analytical matrix-multiplication work.

There is no parallel prefill benchmark. Prefix caches are built by repeated
`decode` calls outside the timer. Full-sequence forward, training, model quality,
and complete language-model generation are not measured.

## Reproduction

Use the Python/PyTorch environment described in the README. Keep generated files
and temporary storage on the external SSD:

```sh
cd /Volumes/T7-apps/domyn_attention
mkdir -p .tmp .cache/pip
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export PYTHONDONTWRITEBYTECODE=1
python3 -B benchmark.py \
  --device cpu --dtype float32 --threads 1 \
  --d-model 128 --num-heads 4 --kv-rank 32 --q-rank 64 --rope-dim 16 \
  --window-size 64 --batch-size 1 --prefix-lengths 32 128 512 \
  --warmup 10 --repeats 50 --seed 17 --output results/decode.json
```

The JSON contains raw samples, medians, quartiles, all CLI settings, hardware,
Python/PyTorch versions, PyTorch build details, thread counts, projection weight
shapes, and hashes of the measured source files. `run_index` records actual case
order; rows are displayed in prefix/variant order for readability. A rerun can
have different timings even with identical inputs and settings.

CPU is the default. Explicit `--device mps` and `--device cuda` options synchronize
the selected accelerator; they fail rather than silently falling back if it is
unavailable. GPU measurements have not been validated in this environment.
Float32 and float64 are accepted; MPS float64 is rejected. `--prefix-lengths 0`
measures decoding the first token with an empty cache.

## Measurement protocol

1. Construct both attention architectures with PyTorch's default float32 linear
   initialization, seeded by `--seed`. Copy each model for its windowed ablation,
   preserving identical weights within MHA/SWA and MLA/hybrid pairs. Move all
   four models to the requested device/dtype and set evaluation mode.
2. Generate one standard-normal input tensor on CPU using a separate generator
   seeded with `seed + 1`. Transfer it before timing and use the same prefix and
   next-token values for every variant. All rows of a batch advance together.
3. Shuffle the variant/prefix cases using `seed`. Samples for each case are then
   collected consecutively. All four models remain resident throughout the run.
4. Build a cache from the prefix using sequential decoding. Prepare the next
   token's view outside the timer. This preparation is not timed or called prefill.
5. Invoke `decode` repeatedly on that **same unchanged prefix cache and token**.
   Discard the first `warmup` timings. Retain exactly `repeats` further timings.
   Returned outputs and updated caches are released outside the timed interval.
6. Time with `perf_counter_ns`. Synchronize the device before the start and after
   the call; the latter synchronization is inside the interval. CPU synchronization
   is a no-op. This measures synchronous API wall time, not isolated kernel time.
7. Report the median and 25th/75th percentiles, computed using Python's inclusive
   quantiles. Save every measured sample in microseconds.

Timed work includes the projection layers, RoPE, attention, output projection,
cache concatenation and trimming, internal allocations, and Python call overhead.
Input generation/transfer, model construction, prefix construction, storage
inspection, statistics, and JSON writing are excluded. `decode` disables autograd;
there is no `torch.compile`, autocast, or custom attention kernel.

This is a repeated **hot-cache, fixed-prefix latency** experiment. It does not
measure a growing generation trajectory, cold-start latency, or generation
throughput. CPU frequency, core placement, thermal state, and other applications
are not controlled. Quartiles describe variation within one case's sample block;
they are not confidence intervals across independent experimental runs.

## Dimensions and parameter counts

Let `D` be model width, `H` the head count, `d=D/H`, `r` the KV latent width,
`q` the query latent width, and `R` the MLA positional width per head. All
projections are bias-free. For the recorded settings:

| Property | MHA / SWA | MLA / SWA+MLA |
| --- | ---: | ---: |
| Model width `D` | 128 | 128 |
| Heads `H` | 4 | 4 |
| Value width per head `d` | 32 | 32 |
| Query/key width per head | 32 | 48 = 32 content + 16 positional |
| Rotated coordinates per head | 32 | 16 |
| KV latent width `r` | — | 32, shared across heads |
| Query latent width `q` | — | 64, shared across heads |
| Parameters | 65,536 | 51,200 |

These are equal-width comparisons, not equal-parameter or equal-FLOP comparisons.
Windowing changes neither projection shapes nor parameter counts. The JSON lists
each projection's weight shape as `[output_features, input_features]`.

The parameter-count formulas follow directly from these projection shapes:

```text
P_MHA = 4 D^2
P_MLA = D^2 + 3 D r + 2 D q + H R q + D R
```

The MLA terms correspond respectively to output projection; KV down-projection
plus K/V up-projections; query down/up-projections; positional-query projection;
and positional-key projection. There are no embeddings, transformer blocks, or
language-model head in these counts.

## Theoretical work and memory

Let `B` be batch size, `T` the number of prefix tokens, and `N` the number of
visible tokens **after** appending the new token. Global attention uses `N=T+1`;
windowed attention uses `N=min(T+1,W)`. One multiply-accumulate is counted as one
MAC, not one FLOP. For this implementation's single-token decoder:

| Variant | Projection MACs | Attention mixing MACs | Returned-cache elements |
| --- | --- | --- | --- |
| MHA | `B P_MHA` | `2 B H (T+1) d` | `2 B (T+1) D` |
| SWA | `B P_MHA` | `2 B H N d` | `2 B N D` |
| MLA | `B P_MLA` | `B H (T+1) (2r+R)` | `B (T+1) (r+R)` |
| SWA+MLA | `B P_MLA` | `B H N (2r+R)` | `B N (r+R)` |

Each projection weight participates once per batch row in this decoder, including
the key-to-query transformation and value expansion after latent aggregation.
For MLA mixing, scoring content costs `B H N r`, scoring position costs `B H N R`,
and aggregating latents costs `B H N r`. Softmax, rotations, elementwise operations,
and memory copies are excluded from these MAC counts. The JSON marks these counts
as theoretical; no hardware profiler was used to measure operations.

At fixed dimensions, global decoding's context-dependent work and cache grow
linearly with `T`; windowed decoding bounds them by `W`. MLA reduces cache width
when `r+R < 2D`. That does **not** guarantee less mixing arithmetic: `r` can be
larger than an individual head width. In the recorded configuration, MLA has
320 mixing MACs per visible token versus MHA's 256.

Cache storage is measured with `untyped_storage().nbytes()` on the two distinct
cached tensors. Both before-call and after-call bytes are saved. The console's
cache column reports the **returned cache after the call**. Its measured size is
checked against the formula on every case. These numbers exclude metadata,
model parameters, allocator overhead, temporary tensors, and the base cache
still held by the benchmark. They are not peak memory measurements.

### Implementation overhead

The current updater concatenates the old cache with the new entry and clones a
trimmed slice when eviction is necessary. For `N_before` retained prefix tokens
and cached width `K` (`2D` for MHA, `r+R` for MLA), concatenation writes
`B (N_before+1) K` elements. Eviction adds a clone of `B W K` elements. These
copies and allocations are timed. Updating a windowed cache therefore costs
`O(B W K)`, not constant work independent of the window. This is a deliberate
clarity tradeoff, not an architectural requirement of sliding-window attention.

The dense full-sequence path is a different workload. For sequence length `L`,
its mixing MACs are `2 B H L^2 d` for MHA/SWA and `B H L^2 (2d+R)` for
MLA/hybrid. It materializes `O(B H L^2)` scores even when a window is selected.
A sparse implementation could reduce the attended pair count toward
`O(L min(L,W))`; the current dense mask does not implement that saving. This path
is not timed here and does not return a cache.

## Recorded CPU run

The committed [raw results](../results/decode_cpu_float32.json) were collected on
2026-10-23 at 11:17 UTC. Hardware: MacBook Air (MacBookAir10,1), Apple M1, 8 GB RAM.
Environment: macOS 26.5.2 ARM64, Python 3.12.3, PyTorch 2.10.0, CPU float32,
one intra-op thread and eight inter-op threads. Float32 matmul precision was
`highest`. MPS and CUDA were unavailable in the execution environment.

Settings: `B=1, D=128, H=4, r=32, q=64, R=16, W=64`; 10 warmups and 50 measured
calls per case. Model/order seed: 17; input seed: 18. These are the explicit
settings in the reproduction command above. All results are for untrained
attention modules on synthetic inputs.

| Prefix tokens | Variant | Visible tokens | Cache KiB | Median µs | Q25–Q75 µs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | MHA | 33 | 33.000 | 123.10 | 122.89–123.41 |
| 32 | SWA | 33 | 33.000 | 122.79 | 122.62–122.88 |
| 32 | MLA | 33 | 6.188 | 148.75 | 148.47–151.16 |
| 32 | SWA+MLA | 33 | 6.188 | 148.33 | 148.17–149.26 |
| 128 | MHA | 129 | 129.000 | 126.42 | 126.25–126.57 |
| 128 | SWA | 64 | 64.000 | 131.52 | 131.30–131.91 |
| 128 | MLA | 129 | 24.188 | 150.21 | 150.00–150.49 |
| 128 | SWA+MLA | 64 | 12.000 | 157.10 | 155.64–159.01 |
| 512 | MHA | 513 | 513.000 | 142.46 | 142.38–142.70 |
| 512 | SWA | 64 | 64.000 | 131.75 | 131.58–131.99 |
| 512 | MLA | 513 | 96.188 | 162.08 | 161.69–162.68 |
| 512 | SWA+MLA | 64 | 12.000 | 155.96 | 155.83–156.90 |

At a 512-token prefix, the measured returned-cache storage is 525,312 bytes for
MHA and 12,288 bytes for the hybrid. The hybrid retains only 64 tokens and uses
compressed entries; this is not a comparison at equal retained context. Its
median latency is higher than MHA's in this run despite the smaller cache.

At the 128-token prefix, both windowed variants also have higher median latency
than their global counterparts despite less attention mixing. The implementation
contains extra eviction copies and several separate operators; these observations
do not isolate which operation causes any timing difference. No profiling or
optimization ablation was performed.

These measurements demonstrate cache-footprint reductions and the actual behavior
of this small eager implementation on this CPU. They do not establish that MLA or
the hybrid is universally faster/slower, predict accelerator performance, or
evaluate the quality cost of compression and restricted context.

## Validation

Run `python3 -B -m unittest discover -s scripts -p 'test_*.py' -v` from the repository root. All 15 tests passed for this increment: the 12
attention/decoding tests plus three benchmark tests. The latter verify fixed
prefix reuse, warmup exclusion, backing-storage accounting, and complete CLI
reporting of all four variants with known parameter and MAC counts. They do not
assert speed thresholds. The saved run's source hashes were checked against the
measured files. Model integration remains outside this increment.
