"""Small, deterministic CPU tests; no model training or external data required."""

import math
import unittest
from itertools import product

import torch
from torch.nn import functional as F

from attention import MHA, MLA, _rope


def rotation(position, dim, dtype):
    """Independent RoPE reference using explicit 2-by-2 rotation matrices."""
    blocks = []
    for i in range(dim // 2):
        angle = position / (10000.0 ** (2 * i / dim))
        c, s = math.cos(angle), math.sin(angle)
        blocks.append(torch.tensor([[c, -s], [s, c]], dtype=dtype))
    return torch.block_diag(*blocks)


def reference_rope(x):
    return torch.stack(
        [x[..., t, :] @ rotation(t, x.size(-1), x.dtype).T for t in range(x.size(-2))],
        dim=-2,
    )


def reference_forward(model, x):
    """Evaluate each query/head separately, slicing only the visible tokens.

    No production RoPE, mask, or attention helper is used. For MLA, content and
    positional dot products are evaluated separately rather than concatenated.
    All operations remain differentiable to check every projection's gradient.
    """
    batch, length, _ = x.shape
    shape = (batch, length, model.num_heads, model.head_dim)
    if isinstance(model, MHA):
        q = (x @ model.q_proj.weight.T).reshape(shape)
        k = (x @ model.k_proj.weight.T).reshape(shape)
        v = (x @ model.v_proj.weight.T).reshape(shape)
        q = reference_rope(q.transpose(1, 2)).transpose(1, 2)
        k = reference_rope(k.transpose(1, 2)).transpose(1, 2)
        scale = math.sqrt(model.head_dim)
    else:
        c_q = x @ model.q_down.weight.T
        c_kv = x @ model.kv_down.weight.T
        q = (c_q @ model.q_up.weight.T).reshape(shape)
        k = (c_kv @ model.k_up.weight.T).reshape(shape)
        v = (c_kv @ model.v_up.weight.T).reshape(shape)
        q_r = (c_q @ model.q_rope.weight.T).reshape(
            batch, length, model.num_heads, model.rope_dim
        )
        q_r = reference_rope(q_r.transpose(1, 2)).transpose(1, 2)
        k_r = reference_rope(x @ model.k_rope.weight.T)
        scale = math.sqrt(model.head_dim + model.rope_dim)

    rows = []
    for t in range(length):
        start = 0 if model.window_size is None else max(0, t + 1 - model.window_size)
        heads = []
        for h in range(model.num_heads):
            scores = []
            for j in range(start, t + 1):
                score = (q[:, t, h] * k[:, j, h]).sum(dim=-1)
                if isinstance(model, MLA):
                    score = score + (q_r[:, t, h] * k_r[:, j]).sum(dim=-1)
                scores.append(score / scale)
            probabilities = torch.stack(scores, dim=-1).softmax(dim=-1)
            heads.append((probabilities[..., None] * v[:, start:t + 1, h]).sum(dim=1))
        rows.append(torch.cat(heads, dim=-1))
    return torch.stack(rows, dim=1) @ model.out_proj.weight.T


def make_model(kind, window_size=None, dtype=torch.float64):
    if kind == "mha":
        model = MHA(d_model=8, num_heads=2, window_size=window_size)
    else:
        # Unequal, non-head-divisible latent widths detect accidental reshaping
        # of the shared latents as though they were per-head representations.
        model = MLA(
            d_model=8, num_heads=2, kv_rank=3, q_rank=5,
            rope_dim=2, window_size=window_size,
        )
    return model.to(dtype=dtype)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_rope_matches_rotation_matrices_and_preserves_norm(self):
        x = torch.randn(2, 3, 5, 6, dtype=torch.float64)
        for offset in (0, 7, 31):
            with self.subTest(offset=offset):
                actual = _rope(x, offset)
                expected = torch.stack(
                    [x[..., t, :] @ rotation(t + offset, 6, x.dtype).T for t in range(5)],
                    dim=-2,
                )
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))
        torch.testing.assert_close(_rope(x)[..., 0, :], x[..., 0, :], rtol=0, atol=0)

    def test_mha_matches_pytorch_sdpa(self):
        for window in (None, 1, 3, 8):
            with self.subTest(window=window):
                model = make_model("mha", window)
                x = torch.randn(2, 5, 8, dtype=torch.float64)
                shape = (2, 5, 2, 4)
                q = reference_rope((x @ model.q_proj.weight.T).reshape(shape).transpose(1, 2))
                k = reference_rope((x @ model.k_proj.weight.T).reshape(shape).transpose(1, 2))
                v = (x @ model.v_proj.weight.T).reshape(shape).transpose(1, 2)
                # SDPA boolean masks use True for allowed entries. Construct
                # the visible columns by slices, independently of the module.
                allowed = torch.zeros(5, 5, dtype=torch.bool)
                for t in range(5):
                    first = 0 if window is None else max(0, t + 1 - window)
                    allowed[t, first:t + 1] = True
                expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
                expected = expected.transpose(1, 2).reshape(2, 5, 8) @ model.out_proj.weight.T
                torch.testing.assert_close(model(x), expected, rtol=1e-10, atol=1e-12)

    def test_outputs_and_all_gradients_match_independent_reference(self):
        for dtype in (torch.float64, torch.float32):
            rtol, atol = (1e-9, 1e-10) if dtype == torch.float64 else (1e-4, 1e-6)
            for kind in ("mha", "mla"):
                for length, window in ((1, None), (5, None), (5, 1), (5, 3), (5, 8)):
                    with self.subTest(dtype=dtype, kind=kind, length=length, window=window):
                        model = make_model(kind, window, dtype)
                        x = torch.randn(2, length, 8, dtype=dtype, requires_grad=True)
                        actual = model(x)
                        expected = reference_forward(model, x)
                        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                        # A nonuniform probe exercises all output coordinates.
                        probe = torch.randn_like(actual)
                        named_inputs = [("input", x), *model.named_parameters()]
                        inputs = [value for _, value in named_inputs]
                        actual_grads = torch.autograd.grad(actual, inputs, grad_outputs=probe)
                        expected_grads = torch.autograd.grad(expected, inputs, grad_outputs=probe)
                        for (name, _), actual_grad, expected_grad in zip(
                            named_inputs, actual_grads, expected_grads
                        ):
                            with self.subTest(gradient=name):
                                self.assertTrue(torch.isfinite(actual_grad).all().item())
                                torch.testing.assert_close(
                                    actual_grad, expected_grad, rtol=rtol, atol=atol
                                )

    def test_future_tokens_cannot_change_past_outputs(self):
        for kind in ("mha", "mla"):
            for window in (None, 1, 3):
                with self.subTest(kind=kind, window=window):
                    model = make_model(kind, window)
                    x = torch.randn(2, 7, 8, dtype=torch.float64)
                    changed = x.clone()
                    changed[:, 4:] += 10 * torch.randn_like(changed[:, 4:])
                    torch.testing.assert_close(
                        model(x)[:, :4], model(changed)[:, :4], rtol=0, atol=0
                    )

    def test_tokens_before_window_cannot_change_query_output(self):
        for kind in ("mha", "mla"):
            for window in (1, 3):
                with self.subTest(kind=kind, window=window):
                    model = make_model(kind, window)
                    x = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
                    target = 4
                    first = target + 1 - window
                    changed = x.detach().clone()
                    changed[:, :first] += 10 * torch.randn_like(changed[:, :first])
                    y = model(x)[:, target]
                    torch.testing.assert_close(y, model(changed)[:, target], rtol=0, atol=0)
                    gradient, = torch.autograd.grad(y.sum(), x)
                    # This is single-layer locality, not a claim about the
                    # receptive field of a stack of local-attention layers.
                    for excluded in (gradient[:, :first], gradient[:, target + 1:]):
                        torch.testing.assert_close(
                            excluded, torch.zeros_like(excluded), rtol=0, atol=0
                        )

    def test_window_one_reduces_to_value_and_output_projections(self):
        for kind in ("mha", "mla"):
            with self.subTest(kind=kind):
                model = make_model(kind, 1)
                x = torch.randn(2, 5, 8, dtype=torch.float64)
                values = model.v_proj(x) if kind == "mha" else model.v_up(model.kv_down(x))
                torch.testing.assert_close(model(x), model.out_proj(values), rtol=0, atol=0)

    def test_window_covering_sequence_equals_full_attention(self):
        for kind in ("mha", "mla"):
            model = make_model(kind)
            x = torch.randn(2, 5, 8, dtype=torch.float64)
            expected = model(x)
            for window in (5, 8):
                with self.subTest(kind=kind, window=window):
                    model.window_size = window
                    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)

    def test_invalid_dimensions_and_windows_are_rejected(self):
        invalid = [
            lambda: MHA(0, 2),
            lambda: MHA(8, 0),
            lambda: MHA(7, 2),
            lambda: MHA(6, 2),  # Odd RoPE width per head.
            lambda: MHA(8, 2, window_size=0),
            lambda: MLA(0, 2, 3, 5, 2),
            lambda: MLA(8, 0, 3, 5, 2),
            lambda: MLA(7, 2, 3, 5, 2),
            lambda: MLA(8, 2, 0, 5, 2),
            lambda: MLA(8, 2, 3, 0, 2),
            lambda: MLA(8, 2, 3, 5, 0),
            lambda: MLA(8, 2, 3, 5, 3),
            lambda: MLA(8, 2, 3, 5, 2, window_size=-1),
        ]
        for index, construct in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                construct()


class DecodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_decode_matches_full_sequence_without_autograd(self):
        # Batch 3 differs from the 2 heads, exposing accidental broadcasting.
        for kind, window, dtype, batch in product(
            ("mha", "mla"), (None, 1, 3, 20),
            (torch.float64, torch.float32), (1, 3),
        ):
            with self.subTest(kind=kind, window=window, dtype=dtype, batch=batch):
                model = make_model(kind, window, dtype)
                x = torch.randn(batch, 11, 8, dtype=dtype, requires_grad=True)
                expected = model(x).detach()
                cache = None
                outputs = []
                for t in range(x.size(1)):
                    y, cache = model.decode(x[:, t:t + 1], cache)
                    outputs.append(y)
                    self.assertFalse(y.requires_grad)
                    for tensor in cache[:2]:
                        self.assertFalse(tensor.requires_grad)
                        self.assertIsNone(tensor.grad_fn)
                rtol, atol = (1e-9, 1e-10) if dtype == torch.float64 else (1e-4, 1e-6)
                torch.testing.assert_close(
                    torch.cat(outputs, dim=1), expected, rtol=rtol, atol=atol
                )

    def test_cache_contents_absolute_positions_and_storage(self):
        for kind, window, batch in product(("mha", "mla"), (None, 1, 3, 20), (1, 3)):
            with self.subTest(kind=kind, window=window, batch=batch):
                model = make_model(kind, window)
                x = torch.randn(batch, 11, 8, dtype=torch.float64)
                # Construct the complete expected cache independently, using
                # absolute positions and then slicing the currently visible span.
                with torch.no_grad():
                    if kind == "mha":
                        shape = (batch, 11, model.num_heads, model.head_dim)
                        first = reference_rope(
                            (x @ model.k_proj.weight.T).reshape(shape).transpose(1, 2)
                        )
                        second = (x @ model.v_proj.weight.T).reshape(shape).transpose(1, 2)
                        elements_per_token = 2 * model.d_model
                    else:
                        first = x @ model.kv_down.weight.T
                        second = reference_rope(x @ model.k_rope.weight.T)
                        elements_per_token = model.kv_down.out_features + model.rope_dim

                cache = None
                for t in range(11):
                    _, cache = model.decode(x[:, t:t + 1], cache)
                    retained = t + 1 if window is None else min(t + 1, window)
                    start = t + 1 - retained
                    self.assertEqual(cache.position, t + 1)
                    for actual, complete in zip(cache[:2], (first, second)):
                        torch.testing.assert_close(
                            actual, complete[..., start:t + 1, :], rtol=1e-9, atol=1e-10
                        )
                        # Checking storage, not only shape, detects a trimmed
                        # view that still retains an oversized allocation.
                        self.assertEqual(
                            actual.untyped_storage().nbytes(),
                            actual.numel() * actual.element_size(),
                        )
                    expected_bytes = batch * retained * elements_per_token * x.element_size()
                    actual_bytes = sum(tensor.untyped_storage().nbytes() for tensor in cache[:2])
                    self.assertEqual(actual_bytes, expected_bytes)

    def test_cache_can_be_reused_without_mutation_and_reset(self):
        for kind, window in product(("mha", "mla"), (None, 1, 3)):
            with self.subTest(kind=kind, window=window):
                model = make_model(kind, window)
                x = torch.randn(2, 7, 8, dtype=torch.float64)
                cache = None
                for t in range(5):
                    _, cache = model.decode(x[:, t:t + 1], cache)
                saved_tensors = [tensor.clone() for tensor in cache[:2]]
                # Two continuations share the same prefix cache. Each must
                # produce the corresponding full-sequence result independently.
                for next_index in (5, 6):
                    next_token = x[:, next_index:next_index + 1]
                    y, next_cache = model.decode(next_token, cache)
                    expected = model(torch.cat((x[:, :5], next_token), dim=1))[:, -1:]
                    torch.testing.assert_close(y, expected, rtol=1e-9, atol=1e-10)
                    self.assertEqual(next_cache.position, 6)
                    self.assertEqual(cache.position, 5)
                    for actual, saved in zip(cache[:2], saved_tensors):
                        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
                y, fresh_cache = model.decode(x[:, :1])
                self.assertEqual(fresh_cache.position, 1)
                torch.testing.assert_close(y, model(x[:, :1]), rtol=1e-9, atol=1e-10)

    def test_decode_requires_exactly_one_token(self):
        for kind in ("mha", "mla"):
            model = make_model(kind)
            for shape in ((2, 8), (2, 0, 8), (2, 2, 8), (2, 1, 7)):
                with self.subTest(kind=kind, shape=shape), self.assertRaises(ValueError):
                    model.decode(torch.zeros(shape, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
