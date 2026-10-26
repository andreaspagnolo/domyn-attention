# Hybrid Sliding Window and Multi-head Latent Attention

A minimal PyTorch implementation of a **hybrid attention mechanism combining Sliding Window Attention (SWA) and Multi-head Latent Attention (MLA)**, developed as a take-home assignment for the Junior AI Research Engineer position at Domyn.

The goal is to implement the mechanisms faithfully, verify their mathematical correctness, and compare their **KV-cache footprint and single-token decoding latency** against standard Multi-Head Attention (MHA). The project prioritizes clarity and reproducibility over production-level optimization.

## Attention variants

| Variant | Attention scope | KV cache |
| --- | --- | --- |
| **MHA** | All previous tokens (causal) | Full per-head Keys and Values |
| **SWA** | Most recent `W` tokens | Window of full per-head Keys and Values |
| **MLA** | All previous tokens (causal) | Shared compressed KV latents and positional Keys |
| **SWA + MLA** | Most recent `W` tokens | Window of compressed KV latents and positional Keys |

The hybrid combines two complementary ideas: **SWA limits how many positions each query can attend to; MLA reduces how much information must be cached per position**. All four variants are implemented using two attention modules (`MHA` and `MLA`), each with an optional sliding-window size.

## Implementation

### Sliding Window Attention

For a query at position `t`, a window of size `W` permits attention only to Key positions `j` satisfying:

```text
max(0, t - W + 1) <= j <= t
```

The current token counts toward the window. With `W=1`, a token attends only to itself; with `W` at least as large as the sequence, SWA is equivalent to full causal attention. The full-sequence implementation uses a **dense local causal mask** for simplicity and correctness.

### Multi-head Latent Attention

MLA follows **DeepSeek-V2, Section 2.1**. Instead of storing full historical Keys and Values, it learns a shared low-rank KV representation `c_kv`. Query compression is also implemented.

To preserve relative positional information without losing the benefits of KV compression, **Decoupled RoPE** separates content and positional components. Each head has its own positional Query, while the positional Key is shared across heads:

```text
c_kv[t] = W_DKV x[t]                         # shared KV latent
c_q[t]  = W_DQ  x[t]                         # shared Query latent

q_content[t,i] = W_UQ[i] c_q[t]
k_content[t,i] = W_UK[i] c_kv[t]
v[t,i]         = W_UV[i] c_kv[t]

q_position[t,i] = RoPE_t(W_QR[i] c_q[t])
k_position[t]   = RoPE_t(W_KR x[t])          # shared across heads

score[t,j,i] = (
    dot(q_content[t,i], k_content[j,i])
    + dot(q_position[t,i], k_position[j])
) / sqrt(head_dim + rope_dim)
```

The two score contributions are added **before a single Softmax**. Keeping RoPE out of the content Keys is essential for efficient compressed-cache decoding.

### Compressed KV-cache decoding

Both attention modules provide the same inference interface:

```python
output, cache = model.decode(token, cache)  # cache=None for the first token
```

Here `token` and `output` have shape `(batch, 1, d_model)`. Each call returns a new cache without mutating the previous one.

- **MHA/SWA** store rotated Keys and unrotated Values, each with shape `(B, H, N, d)`.
- **MLA/hybrid** store only a shared KV latent `(B, N, r)` and a shared rotated positional Key `(B, N, R)`; no historical full K/V or Query latents are cached.
- **Windowed variants** retain at most `W` positions. Absolute RoPE positions continue increasing after older entries are evicted.

For each head, MLA avoids reconstructing historical content Keys and Values using these equivalent rearrangements:

```text
q^T (W_UK c_j)          = (W_UK^T q)^T c_j
sum_j a_j (W_UV c_j)   = W_UV (sum_j a_j c_j)
```

Thus the Key up-projection moves to the current Query, and the Value up-projection is applied **after** aggregating the cached latents. The attention scaling remains `1 / sqrt(head_dim + rope_dim)`.

The decoder uses simple cache concatenation and trimming; it does **not** use a preallocated circular buffer.

## Results

### Decoding latency across context lengths

The experiment measures **single-token cached decoding** at nine prefix lengths from **128 to 32,768 tokens**, with a fixed sliding window of **64**. All measurements come from one run using an **Apple M1 MacBook Air (8 GB RAM)**, CPU float32, batch size 1, and one intra-op thread. Model settings are `d_model=128`, `num_heads=4`, `kv_rank=32`, `q_rank=64`, and `rope_dim=16`.

Each case uses the same prefix cache for 10 warmups and 50 measured calls. Prefix-cache construction is excluded from timing. The plot shows measured medians and interquartile ranges on logarithmic axes; the whiskers are not confidence intervals, and the connecting lines are not a fitted scaling law.

![Measured cached decoding latency for MHA, SWA, MLA, and SWA+MLA across nine prefix lengths through 32,768 tokens](results/context_scaling_32768_cpu_float32.svg)

Measured median latencies (µs):

| Prefix tokens | MHA | SWA | MLA | SWA+MLA |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 130.48 | 144.73 | 155.83 | 160.85 |
| 256 | 135.42 | 145.52 | 167.65 | 160.33 |
| 512 | 155.46 | 134.65 | 165.83 | 158.73 |
| 1,024 | 187.02 | 146.65 | 190.90 | 159.29 |
| 2,048 | 285.67 | 145.04 | 265.71 | 160.00 |
| 4,096 | 642.83 | 145.58 | 316.29 | 160.17 |
| 8,192 | 1,543.21 | 144.65 | 520.60 | 159.85 |
| 16,384 | 2,255.81 | 134.65 | 855.35 | 160.08 |
| 32,768 | 4,558.67 | 133.85 | 1,722.71 | 158.81 |

**Experimental findings:** the hybrid first has a lower median than MHA at **1,024 tokens** (159.29 versus 187.02 µs), while it is slightly slower at 512 (158.73 versus 155.46 µs). The sampled ordering therefore reverses between 512 and 1,024; this single run does not establish an exact or universal break-even threshold. Across the sweep, the hybrid stays near **159–161 µs**, while full MHA and MLA become slower as context grows. SWA alone has a lower median than the hybrid at every sampled length.

### Cache storage and interpretation

After decoding one token from the **32,768-token prefix**, measured cache storage and parameter counts are:

| Variant | Parameters | Cached tokens | Returned KV cache (KiB) |
| --- | ---: | ---: | ---: |
| MHA | 65,536 | 32,769 | 32,769.00 |
| SWA | 65,536 | 64 | 64.00 |
| MLA | 51,200 | 32,769 | 6,144.19 |
| SWA+MLA | 51,200 | 64 | 12.00 |

The hybrid combines a shorter retained context with compression. Compared with SWA at the **same 64-token window**, it uses **5.33× less cache storage**, but takes longer in this run. Cache measurements cover the backing storage of returned tensors; they exclude temporary allocations, peak memory, and full process memory.

**Theoretical interpretation:** at fixed model dimensions, full-attention decoding work and cache storage grow linearly with context; windowing bounds them by `W`. MLA reduces cached features per token, but changes projection dimensions and operator costs, so smaller caches do not guarantee lower latency. Concatenation and trimming are included in these timings: full caches require growing copies, and windowed updates still copy `O(W)` entries. No profiling was performed to isolate these costs. The dense full-sequence SWA path remains quadratic and is not timed here.

These are untrained attention modules on synthetic inputs. The results do not establish language-model quality or universal speedups. [Raw samples and complete settings](results/context_scaling_32768_cpu_float32.json) accompany the plot; [BENCHMARKS.md](docs/BENCHMARKS.md) documents the measurement protocol, projection dimensions, and theoretical operation counts.

## Setup and usage

The recorded environment uses **Python 3.12.3 and PyTorch 2.10.0**. Run these commands from the repository root to create a virtual environment and keep temporary files and package caches in the project directory:

```bash
mkdir -p .tmp .cache/pip
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export PYTHONDONTWRITEBYTECODE=1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Run the tests** from the repository root:

```bash
.venv/bin/python -B -m unittest discover -s scripts -p 'test_*.py' -v
```

The documented test suite contains **15 tests**, covering independent attention references, input/parameter gradients, RoPE, causal and window boundaries, full-sequence versus cached decoding, cache eviction/storage, and benchmark accounting. CPU float32 and float64 are used for numerical verification; there are no speed thresholds in the tests.

**Reproduce the context-length experiment:**

```bash
.venv/bin/python -B benchmark.py --device cpu --dtype float32 --threads 1 \
  --d-model 128 --num-heads 4 --kv-rank 32 --q-rank 64 --rope-dim 16 \
  --batch-size 1 --window-size 64 \
  --prefix-lengths 128 256 512 1024 2048 4096 8192 16384 32768 \
  --warmup 10 --repeats 50 --seed 17 --output results/context_scaling_32768_cpu_float32.json
```

This command replaces the recorded JSON with the new measurements. Sequential cache construction at the largest prefixes can take several minutes even though it is excluded from the reported latency.

**Regenerate the plot** with optional Matplotlib (version 3.11.1 was used for the committed SVG). It is not required for attention, tests, or benchmarking:

```bash
.venv/bin/python -m pip install matplotlib==3.11.1
mkdir -p .cache/matplotlib
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python -B scripts/plot_context.py results/context_scaling_32768_cpu_float32.json
```

Example of incremental decoding with the hybrid:

```python
import torch
from attention import MLA

model = MLA(64, 4, kv_rank=16, q_rank=32, rope_dim=8, window_size=4).eval()
x = torch.randn(2, 10, 64)
cache = None
outputs = []
for token in x.split(1, dim=1):
    output, cache = model.decode(token, cache)
    outputs.append(output)

decoded = torch.cat(outputs, dim=1)
torch.testing.assert_close(decoded, model(x), rtol=1e-4, atol=1e-6)
```

The `forward()` method processes complete sequences and is differentiable; `decode()` handles one token at a time with autograd disabled. A separate cache-producing parallel prefill API is not implemented.

## Scope and limitations

This implementation emphasizes **mathematical correctness, understandable code, and reproducible measurements** rather than production performance.

- **Dense SWA forward:** the full-sequence path masks out-of-window positions but still computes a dense attention matrix, retaining quadratic score computation and storage. Only the cached single-token decoder limits the number of attended positions directly.
- **MLA execution paths:** the differentiable full-sequence path expands content K/V from latents; cached decoding operates directly on compressed representations.
- **Cache update overhead:** concatenation and trimming copy data. Returned windowed caches stay bounded, but updates are not constant-time and temporary allocations are not captured by cache-payload measurements.
- **Controlled comparison:** model width and head count are held fixed, but MLA has different query/key projection dimensions and parameter counts. The results are not equal-parameter or equal-FLOP benchmarks.
- **Scope:** this is an attention-mechanism implementation, not a reproduction of the full DeepSeek-V2 or Mistral models. It does not implement sparse attention kernels, long-context RoPE scaling, padding masks, MoE, or language-model quality evaluation.

## References

- [Mistral 7B — Sliding Window Attention](https://arxiv.org/abs/2310.06825)
- [DeepSeek-V2 — MLA, Section 2.1 and Appendix C](https://arxiv.org/abs/2405.04434)
- [RoFormer — Rotary Position Embeddings](https://arxiv.org/abs/2104.09864)
- [nanoGPT](https://github.com/karpathy/nanoGPT) — suggested minimal Transformer reference; no nanoGPT code is included in the current attention implementation.
