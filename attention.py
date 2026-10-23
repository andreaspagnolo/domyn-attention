"""Full-sequence and cached causal attention, with an optional sliding window.

Inputs and outputs have shape (batch, sequence, d_model). This module implements
the attention equations only: residual connections and normalization belong to
the surrounding transformer block. All projections are bias-free, and there is
no dropout. Both choices keep the equations and numerical comparisons explicit.

MLA follows DeepSeek-V2 section 2.1, equations (9)-(19):
https://arxiv.org/abs/2405.04434
The window follows Mistral's local causal attention, with W counting the current
token: https://arxiv.org/abs/2310.06825
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


class MHACache(NamedTuple):
    """Rotated keys and values: (batch, heads, stored_tokens, head_dim).

    position is the next absolute token position, including evicted tokens.
    """

    key: Tensor
    value: Tensor
    position: int


class MLACache(NamedTuple):
    """Shared latent and rotated key: (batch, stored_tokens, respective_width).

    There is no head axis. position counts all processed tokens, not just those
    retained in the window. Query latents and expanded K/V are never stored.
    """

    latent: Tensor
    rope_key: Tensor
    position: int


def _rope(x: Tensor, start_position: int = 0) -> Tensor:
    """Rotate adjacent feature pairs in (..., sequence, even_head_dim).

    Full-sequence calls start at zero; decoding uses the absolute cache position.
    Pair i rotates by position * 10000**(-2*i/head_dim). MLA content is not rotated.
    """
    length, dim = x.shape[-2:]
    positions = torch.arange(
        start_position, start_position + length, dtype=x.dtype, device=x.device
    )
    frequencies = 10000.0 ** (
        -torch.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim
    )
    angles = positions[:, None] * frequencies[None, :]
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _attention(q: Tensor, k: Tensor, v: Tensor, window_size: int | None) -> Tensor:
    """Dense scaled dot-product attention on (batch, heads, sequence, features).

    W includes the query's own position: a key j is visible to query t iff
    0 <= t-j < W. With no window, only the causal condition t-j >= 0 applies.
    Every row includes itself, so softmax never receives an entirely masked row.
    """
    positions = torch.arange(q.size(-2), device=q.device)
    distance = positions[:, None] - positions[None, :]
    allowed = distance >= 0
    if window_size is not None:
        allowed = allowed & (distance < window_size)

    # A dense mask changes visibility, not the quadratic cost of this product.
    # For MLA, q's width includes both content and positional coordinates.
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
    return weights @ v


def _append_cache(previous: Tensor | None, current: Tensor, window_size: int | None) -> Tensor:
    """Append along sequence, then retain at most W entries in independent storage."""
    if previous is not None:
        current = torch.cat((previous, current), dim=-2)
    if window_size is not None and current.size(-2) > window_size:
        # A slice alone would keep the larger underlying allocation alive.
        # clone makes the returned cache's storage match its retained elements.
        current = current[..., -window_size:, :].clone()
    return current


class MHA(nn.Module):
    """Multi-head attention with RoPE on each head's entire query/key vector.

    Each head has d_model / num_heads features. window_size=None gives full
    causal attention; a positive window_size gives dense masked SWA.
    """

    def __init__(self, d_model: int, num_heads: int, window_size: int | None = None):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(
                "d_model and num_heads must be positive; num_heads must divide d_model"
            )
        if (d_model // num_heads) % 2 != 0:
            raise ValueError("MHA head dimension must be even for RoPE")
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be positive or None")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        # Split independent heads, keeping sequence as the matrix row dimension.
        q = _rope(self.q_proj(x).reshape(shape).transpose(1, 2))
        k = _rope(self.k_proj(x).reshape(shape).transpose(1, 2))
        v = self.v_proj(x).reshape(shape).transpose(1, 2)
        y = _attention(q, k, v, self.window_size)
        return self.out_proj(y.transpose(1, 2).reshape(batch, length, self.d_model))

    @torch.no_grad()
    def decode(self, x: Tensor, cache: MHACache | None = None) -> tuple[Tensor, MHACache]:
        """Decode one token (batch, 1, d_model); None starts a fresh sequence.

        The returned cache replaces the previous one without modifying it.
        Keep model weights, window, batch, dtype, and device fixed for its lifetime.
        This is inference-only; use forward for differentiable full sequences.
        """
        if x.ndim != 3 or x.size(1) != 1 or x.size(2) != self.d_model:
            raise ValueError("decode expects x with shape (batch, 1, d_model)")
        position = 0 if cache is None else cache.position
        batch = x.size(0)
        shape = (batch, 1, self.num_heads, self.head_dim)
        q = _rope(self.q_proj(x).reshape(shape).transpose(1, 2), position)
        k = _rope(self.k_proj(x).reshape(shape).transpose(1, 2), position)
        v = self.v_proj(x).reshape(shape).transpose(1, 2)
        k = _append_cache(None if cache is None else cache.key, k, self.window_size)
        v = _append_cache(None if cache is None else cache.value, v, self.window_size)

        # Every retained token is visible to this newest query. A square causal
        # mask (or a 1-by-S upper-left causal mask) would be incorrect here.
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)).softmax(dim=-1)
        y = weights @ v
        y = self.out_proj(y.transpose(1, 2).reshape(batch, 1, self.d_model))
        return y, MHACache(k, v, position + 1)


class MLA(nn.Module):
    """Multi-head latent attention with decoupled RoPE (DeepSeek-V2 section 2.1).

    kv_rank and q_rank are shared latent widths, not per-head widths. Content
    queries, content keys, and values have head_dim=d_model/num_heads features.
    The additional rope_dim coordinates carry position. The positional key is
    shared across heads, whereas positional queries are head-specific.

    forward expands content K/V from their joint latent. decode operates on a
    compressed cache directly, without reconstructing historical content K/V.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_rank: int,
        q_rank: int,
        rope_dim: int,
        window_size: int | None = None,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(
                "d_model and num_heads must be positive; num_heads must divide d_model"
            )
        if kv_rank <= 0 or q_rank <= 0:
            raise ValueError("kv_rank and q_rank must be positive")
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError("rope_dim must be positive and even")
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be positive or None")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_dim = rope_dim
        self.window_size = window_size

        # nn.Linear applies x @ weight.T; the paper writes column-vector W @ x.
        self.kv_down = nn.Linear(d_model, kv_rank, bias=False)  # W^DKV, eq. (9)
        self.k_up = nn.Linear(kv_rank, d_model, bias=False)     # W^UK, eq. (10)
        self.v_up = nn.Linear(kv_rank, d_model, bias=False)     # W^UV, eq. (11)
        self.q_down = nn.Linear(d_model, q_rank, bias=False)    # W^DQ, eq. (12)
        self.q_up = nn.Linear(q_rank, d_model, bias=False)     # W^UQ, eq. (13)
        self.q_rope = nn.Linear(q_rank, num_heads * rope_dim, bias=False)  # eq. (14)
        self.k_rope = nn.Linear(d_model, rope_dim, bias=False)            # eq. (15)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)           # eq. (19)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        content_shape = (batch, length, self.num_heads, self.head_dim)
        c_kv = self.kv_down(x)
        c_q = self.q_down(x)
        q_content = self.q_up(c_q).reshape(content_shape).transpose(1, 2)
        k_content = self.k_up(c_kv).reshape(content_shape).transpose(1, 2)
        v = self.v_up(c_kv).reshape(content_shape).transpose(1, 2)

        q_position = self.q_rope(c_q).reshape(batch, length, self.num_heads, self.rope_dim)
        q_position = _rope(q_position.transpose(1, 2))
        # W^KR acts on x directly, not on the KV latent. Only one positional
        # key is produced per token; expand is a view sharing it across heads.
        k_position = _rope(self.k_rope(x).unsqueeze(1))
        k_position = k_position.expand(-1, self.num_heads, -1, -1)

        # Concatenation makes the score q_content.k_content + q_position.k_position.
        # It materializes expanded keys here, but does not alter the parameters.
        q = torch.cat((q_content, q_position), dim=-1)
        k = torch.cat((k_content, k_position), dim=-1)
        y = _attention(q, k, v, self.window_size)
        return self.out_proj(y.transpose(1, 2).reshape(batch, length, self.d_model))

    @torch.no_grad()
    def decode(self, x: Tensor, cache: MLACache | None = None) -> tuple[Tensor, MLACache]:
        """Decode one token (batch, 1, d_model) with a compressed inference cache.

        None starts a fresh sequence. As for MHA.decode, keep model/configuration
        fixed while reusing the cache. Prior caches are not modified.
        """
        if x.ndim != 3 or x.size(1) != 1 or x.size(2) != self.d_model:
            raise ValueError("decode expects x with shape (batch, 1, d_model)")
        position = 0 if cache is None else cache.position
        batch = x.size(0)
        c_q = self.q_down(x)
        q_content = self.q_up(c_q).reshape(batch, 1, self.num_heads, self.head_dim)
        q_content = q_content.transpose(1, 2)
        q_position = self.q_rope(c_q).reshape(batch, 1, self.num_heads, self.rope_dim)
        q_position = _rope(q_position.transpose(1, 2), position)
        latent = _append_cache(
            None if cache is None else cache.latent, self.kv_down(x), self.window_size
        )
        rope_key = _append_cache(
            None if cache is None else cache.rope_key,
            _rope(self.k_rope(x), position), self.window_size,
        )

        rank = latent.size(-1)
        # q^T (W_UK c) = (W_UK^T q)^T c. Move W_UK to the current
        # query, avoiding expanded keys for every cached token.
        # (B,H,1,d) @ (H,d,r) -> (B,H,1,r); H broadcasts across the batch.
        w_uk = self.k_up.weight.reshape(self.num_heads, self.head_dim, rank)
        q_latent = q_content @ w_uk
        scores = q_latent @ latent.unsqueeze(1).transpose(-2, -1)
        scores = scores + q_position @ rope_key.unsqueeze(1).transpose(-2, -1)
        # The scale is still the ORIGINAL concatenated Q/K width, not r+rope_dim.
        weights = (scores / math.sqrt(self.head_dim + self.rope_dim)).softmax(dim=-1)

        # sum_j a_j (W_UV c_j) = W_UV (sum_j a_j c_j). Aggregate the
        # shared latents per head, then expand only one result per head.
        pooled = weights @ latent.unsqueeze(1)  # (B,H,1,r)
        w_uv = self.v_up.weight.reshape(self.num_heads, self.head_dim, rank)
        y = pooled @ w_uv.transpose(-2, -1)     # (B,H,1,d)
        y = self.out_proj(y.transpose(1, 2).reshape(batch, 1, self.d_model))
        return y, MLACache(latent, rope_key, position + 1)
