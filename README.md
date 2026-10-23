# Domyn hybrid attention — attention, decoding, and benchmarks

The mathematical attention modules and single-token KV-cache decoding are
implemented and tested. The four variants use two classes in `attention.py`:

| Variant | Class | `window_size` |
| --- | --- | --- |
| MHA | `MHA` | `None` |
| SWA | `MHA` | Positive integer |
| MLA | `MLA` | `None` |
| SWA + MLA | `MLA` | Positive integer |

The hybrid applies a local causal mask inside MLA. All inputs and outputs have
shape `(batch, sequence, d_model)`.

## Benchmarks

`benchmark.py` compares cached single-token decoding for the four variants,
reporting latency, actual returned-cache storage, parameters, projection shapes,
and analytical MAC counts. From the configured project environment:

```sh
python3 -B benchmark.py --output results/decode.json
```

See [BENCHMARKS.md](BENCHMARKS.md) for the exact protocol, theoretical complexity,
experimental settings, limitations, and the recorded CPU results. Raw samples
are in [results/decode_cpu_float32.json](results/decode_cpu_float32.json).
No parallel prefill, training, or full-model timing is reported.

## Run the tests

Validated with Python 3.12.3 and PyTorch 2.10.0 on CPU. The existing installation
provided PyTorch, so this increment required no package download or installation.
MPS was unavailable in the execution environment; no GPU results are claimed.

From the project directory on the external SSD:

```sh
cd /Volumes/T7-apps/domyn_attention
mkdir -p .tmp .cache/pip
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export PYTHONDONTWRITEBYTECODE=1
python3 -B -m unittest -v
```

To reproduce with a new environment, create it on the SSD, install the pinned
dependency, and use that interpreter for the same tests:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest -v
```

There are no datasets, pretrained weights, or test dependencies to download.
The tests fix the random seed before each test. Float64 output and gradient
comparisons use `rtol=1e-9, atol=1e-10`; float32 comparisons use
`rtol=1e-4, atol=1e-6`. Additional exact identities use zero tolerance.

## Mathematical contract

Let `D=d_model`, `H=num_heads`, and `d=D/H`. A window of `W` includes the current
token, so query position `t` can attend to precisely these positions:

```text
max(0, t-W+1) <= j <= t
```

With no window, every `0 <= j <= t` is visible. A window of one permits only
self-attention, and a window at least as long as the sequence gives full causal
attention. The window convention is explicit to avoid ambiguity about whether
the current token counts toward its size.

The following equations use column vectors, as in the papers. PyTorch stores
`nn.Linear` weights as `(output_features, input_features)` and computes
`x @ weight.T` on batched row vectors.

### MHA

Each head has independent query, key, and value projection rows. Queries and keys
are rotated at their token positions; values are not rotated:

```text
q[t,i] = RoPE_t(W_Q[i] x[t])
k[t,i] = RoPE_t(W_K[i] x[t])
v[t,i] = W_V[i] x[t]
score[t,j,i] = dot(q[t,i], k[j,i]) / sqrt(d)
```

For each query and head, softmax is over the permitted key positions. The
weighted value sums are concatenated and passed through the output projection.

### MLA

The implementation follows DeepSeek-V2 section 2.1, equations (9)-(19).
`r=kv_rank` and `r_q=q_rank` are shared latent dimensions, not per-head widths.
Choose them below `D` when demonstrating compression; positive larger widths
are mathematically valid but do not provide that bottleneck.

```text
c_kv[t] = W_DKV x[t]                       # r features
c_q[t]  = W_DQ x[t]                        # r_q features

q_content[t,i] = W_UQ[i] c_q[t]            # d features per head
k_content[t,i] = W_UK[i] c_kv[t]           # d features per head
v[t,i]         = W_UV[i] c_kv[t]           # d features per head

q_position[t,i] = RoPE_t(W_QR[i] c_q[t])   # rope_dim features per head
k_position[t]   = RoPE_t(W_KR x[t])        # rope_dim features, shared by heads

score[t,j,i] = (
    dot(q_content[t,i], k_content[j,i])
    + dot(q_position[t,i], k_position[j])
) / sqrt(d + rope_dim)
```

Softmax, value aggregation, and the output projection follow the same rule as
MHA. Concatenating content and positional coordinates implements the sum of dot
products above. The scale uses their combined width, not the latent width.

Keeping position out of the content keys is essential for moving the key
up-projection onto the query side during compressed-cache decoding. Applying
RoPE to the content keys would obstruct that rearrangement. The positional key
is projected directly from `x`, as specified in the paper.

## KV-cache decoding

Both classes expose the same inference interface:

```python
output, cache = model.decode(token, cache)
```

`token` and `output` have shape `(batch, 1, d_model)`. Pass `cache=None` to start
a new sequence, then retain the returned cache for the next call. The caller
owns the state; decoding does not mutate earlier caches or store sequence state
inside the module. A small named tuple gives each cached tensor an explicit name.

All rows in a batch advance together. Keep the same model weights, window size,
batch, dtype, and device while reusing a cache. `decode` disables autograd;
training continues to use `forward`.

The cache records `position`, the next absolute token position. This equals the
total number of processed tokens even after older entries are evicted. Queries
and newly cached keys receive RoPE at this position; cached keys are already
rotated and are not rotated again.

For example, a prompt can be processed through the same single-token interface:

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

This includes ten positions and several evictions from a four-token window.
There is no separate parallel prefill API in this increment.

### MHA cache

MHA stores rotated keys and unrotated values, each of shape `(B, H, N, d)`.
Each call projects only the new token, appends its key/value, and drops entries
outside the window. The newest query can attend to every retained entry,
including itself, so decoding needs no causal mask.

### MLA cache and rearrangement

MLA stores only:

- `latent`: the shared KV latent, shape `(B, N, r)`;
- `rope_key`: the shared rotated positional key, shape `(B, N, rope_dim)`.

Neither tensor has a head axis. Query latents, historical expanded keys, and
historical expanded values are not cached.

The decoding computation uses the associative law, as described in DeepSeek-V2
Appendix C. For one head, with the same weights as the full-sequence forward:

```text
q_content^T (W_UK c_j) = (W_UK^T q_content)^T c_j

sum_j a_j (W_UV c_j) = W_UV (sum_j a_j c_j)
```

The first identity moves the key up-projection onto the current query. Its dot
product with each cached latent gives the content score; the positional dot
product is added separately. The scale remains `sqrt(d + rope_dim)`, because
rearranging a product must not change its numerical value or normalization.

The second identity aggregates latents with the attention probabilities before
applying the value up-projection. Only one output vector per head is expanded;
historical content K/V are never reconstructed during decoding. These identities
depend on the projections being linear. The implementation uses the existing
weights directly, without building or maintaining pre-fused projection matrices.

### Cache storage and update cost

Let `N` be the number of retained tokens: all processed tokens without a window,
or at most `W` with a window. The tensor payload per layer is:

| Cache | Elements | Bytes |
| --- | --- | --- |
| MHA / SWA | `2 * B * N * H * d` | Elements times dtype size |
| MLA / hybrid | `B * N * (r + rope_dim)` | Elements times dtype size |

The tests verify the contents and underlying tensor-storage bytes against these
formulas after every decoding step. This counts cache tensor storage, excluding
Python metadata, model parameters, temporary tensors, and allocator overhead.
MLA saves cache payload relative to this MHA when `r + rope_dim < 2 * H * d`.

Updates use `torch.cat` followed, when needed, by trimming and cloning. Cloning
is necessary because a sliced tensor can keep the entire larger storage alive.
The returned windowed cache therefore owns storage for at most `W` entries.
Existing caches are not modified, so two continuations can reuse a common prefix.

This simple update copies retained data and temporarily holds additional tensors.
Keeping old cache objects also retains their storage. Bounded returned-cache
size is not a claim about peak process memory or constant-time updates. Global
caches grow with sequence length; windowed caches stop growing at `W`.
[The decoding benchmark](BENCHMARKS.md) measures latency including these updates.

## Implementation choices and limits

- **Dense masks:** `_attention` computes every score before masking. Windowed
  variants have the correct visibility pattern but still use quadratic score
  computation and storage. This increment demonstrates no sparse speedup.
- **Expanded MLA forward:** content keys and values are materialized from the
  latent tensors for the differentiable full-sequence calculation. The separate
  inference-only `decode` method uses the compressed representation directly.
- **RoPE:** adjacent coordinate pairs rotate with a fixed base of 10,000.
  Positions start at zero for each full-sequence call and continue from the
  cache's absolute position during decoding. There is no long-context rescaling.
  MHA rotates its full query/key head; MLA rotates only its positional coordinates.
- **Independent projections:** separate `nn.Linear` layers keep the mapping to
  the equations visible. Projection fusion is unnecessary at this stage.
- **Bias-free projections and no dropout:** the projection equations have no
  additive biases, and omitting dropout keeps reference comparisons
  deterministic. PyTorch's default linear initialization is retained.
- **Attention only:** residual connections, transformer normalization, and the
  language-model loss belong to later integration. DeepSeek's additional
  latent normalization/scaling from its training setup is omitted: this is the
  section 2.1 formulation, not a reproduction of the entire trained model.
- **Fixed-length, unpadded batches:** every row in a batch has the same sequence
  length. Padding masks, per-row decoding positions, and mixed-precision policies
  are outside this increment. CPU float32 and float64 are tested.
- **Comparison dimensions:** MHA query/key heads have width `d`; MLA query/key
  heads have width `d + rope_dim`. Both have value width `d`. Equal `D` and `H`
  therefore do not imply equal parameter counts or FLOPs. The benchmark reports
  these differences explicitly.

## What the tests establish

Twelve tests in `test_attention.py` cover shapes (including a one-token sequence),
windows, and dtypes. The eight full-sequence tests establish:

1. RoPE agrees with explicit rotation matrices and preserves vector norms.
2. MHA agrees with PyTorch's scaled dot-product attention for the same weights.
3. Both modules' outputs, input gradients, and every parameter gradient agree
   with a separate reference that loops over queries and heads. That reference
   slices visible tokens instead of using the implementation's mask. MLA scores
   use separate content and positional dot products.
4. Changes to future tokens cannot affect past outputs.
5. Changes before a query's window cannot affect that query; gradients outside
   its permitted positions are zero.
6. A window of one reduces attention to the value and output projections.
7. A window covering the sequence exactly matches full causal attention.
8. Invalid dimensions and nonpositive windows are rejected.

The locality check concerns one layer. It does not imply that a stack of local
layers has the same receptive field. No test expects independently parameterized
MHA and MLA models to produce identical outputs.

The four decoding tests establish:

1. Token-by-token outputs match the full-sequence forward for all four variants
   in float32 and float64, with batch sizes one and three. Windows include one,
   three, and a size larger than the entire sequence. Eleven decoding steps
   exercise repeated eviction. Outputs and cache tensors retain no autograd graph.
2. Cache contents, absolute positions, and backing-storage bytes match the
   expected projections and formulas at every step, including after eviction.
3. A cache can be reused for two independent continuations without mutation;
   `None` resets the sequence.
4. Decoding rejects inputs that are not shaped `(batch, 1, d_model)`.

The RoPE test also checks nonzero offsets against independent rotation matrices.

Three tests in `test_benchmark.py` check the fixed-prefix timing protocol,
backing-storage measurement, and CLI reporting of dimensions and operation
counts. There are 15 tests in total; none asserts a timing speed threshold.

## Review boundary

Attention, single-token KV-cache decoding, and reproducible decoding benchmarks
are complete. This increment stops before model integration. Sparse attention,
parallel prefill, circular preallocated buffers, and extensive training remain
optional extensions.

## References

- [Mistral 7B, section 2](https://arxiv.org/abs/2310.06825)
- [DeepSeek-V2, section 2.1 and Appendix C](https://arxiv.org/abs/2405.04434)
- [PyTorch 2.10 scaled dot-product attention](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)

This increment is written directly from the equations; no nanoGPT code has been
copied. nanoGPT remains the proposed reference for later minimal model integration.
