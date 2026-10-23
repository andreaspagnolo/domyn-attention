"""Fixed-prefix, single-token decode latency; no prefill or training benchmark."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import torch

from attention import MHA, MLA


def cache_storage_bytes(cache):
    """Count backing storage, not just view sizes; these caches have two storages."""
    return 0 if cache is None else sum(t.untyped_storage().nbytes() for t in cache[:2])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_decode(model, x, warmup, repeats):
    """Build a prefix outside the timer, then repeatedly decode the SAME next token.

    x contains the prefix followed by one new token. decode does not mutate its
    input cache, so every sample has exactly the same history and input. Returned
    outputs/caches are released after timing, before the next invocation.
    """
    prefix_tokens = x.size(1) - 1
    cache = None
    for position in range(prefix_tokens):
        output, cache = model.decode(x[:, position:position + 1], cache)
        del output
    token = x[:, -1:]
    result = {
        "prefix_tokens": prefix_tokens,
        "cache_tokens_before": 0 if cache is None else cache[0].size(-2),
        "cache_bytes_before": cache_storage_bytes(cache),
    }
    samples = []
    for iteration in range(warmup + repeats):
        synchronize(x.device)
        start = time.perf_counter_ns()
        output, updated_cache = model.decode(token, cache)
        synchronize(x.device)
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        if iteration >= warmup:
            samples.append(elapsed_us)
        if iteration == warmup:
            # Inspect storage after the first measured call, outside its timer.
            result["cache_tokens_after"] = updated_cache[0].size(-2)
            result["cache_bytes_after"] = cache_storage_bytes(updated_cache)
        del output, updated_cache

    q25, _, q75 = statistics.quantiles(samples, n=4, method="inclusive")
    result.update(
        median_us=statistics.median(samples), q25_us=q25, q75_us=q75,
        samples_us=samples,
    )
    return result


def describe_model(model):
    """Report dimensions explicitly: equal model width is not equal attention cost."""
    is_mla = isinstance(model, MLA)
    d, h = model.head_dim, model.num_heads
    rank = model.kv_down.out_features if is_mla else None
    rope = model.rope_dim if is_mla else d
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "window_size": model.window_size,
        "qk_head_dim": d + rope if is_mla else d,
        "value_head_dim": d,
        "rope_dim_per_head": rope,
        "kv_rank": rank,
        "q_rank": model.q_down.out_features if is_mla else None,
        "cached_elements_per_token_per_batch_row": rank + rope if is_mla else 2 * h * d,
        "mixing_macs_per_visible_token_per_batch_row": h * (2 * rank + rope) if is_mla else 2 * h * d,
        "projection_weight_shapes_out_in": {
            name: list(layer.weight.shape) for name, layer in model.named_children()
        },
    }


def hardware_info(device):
    info = {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        # Save only hardware characteristics; omit host names and serial numbers.
        try:
            output = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            hardware = json.loads(output.stdout)["SPHardwareDataType"][0]
            info["mac_hardware"] = {
                key: hardware[key] for key in (
                    "machine_name", "machine_model", "chip_type",
                    "number_processors", "physical_memory",
                ) if key in hardware
            }
        except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired):
            info["mac_hardware"] = None
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--kv-rank", type=int, default=32)
    parser.add_argument("--q-rank", type=int, default=64)
    parser.add_argument("--rope-dim", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="results/decode.json")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 2:
        parser.error("warmup must be nonnegative and repeats must be at least two")
    if args.batch_size < 1 or args.threads < 1 or args.window_size < 1:
        parser.error("batch-size, threads, and window-size must be positive")
    if min(args.prefix_lengths) < 0:
        parser.error("prefix-lengths must be nonnegative (zero means no prior cache)")
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS is unavailable; select an available device explicitly")
        if args.dtype == "float64":
            parser.error("MPS does not support float64; use float32 or CPU")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select an available device explicitly")

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    try:
        mha = MHA(args.d_model, args.num_heads)
        mla = MLA(args.d_model, args.num_heads, args.kv_rank, args.q_rank, args.rope_dim)
    except ValueError as error:
        parser.error(str(error))
    # Each windowed ablation has exactly its full-attention counterpart's weights.
    swa, hybrid = copy.deepcopy(mha), copy.deepcopy(mla)
    swa.window_size = hybrid.window_size = args.window_size
    models = {"MHA": mha, "SWA": swa, "MLA": mla, "SWA+MLA": hybrid}
    models = {name: model.to(device=device, dtype=dtype).eval() for name, model in models.items()}
    descriptions = {name: describe_model(model) for name, model in models.items()}
    # A separate CPU generator keeps inputs independent of model/rank choices.
    x = torch.randn(
        args.batch_size, max(args.prefix_lengths) + 1, args.d_model,
        generator=torch.Generator().manual_seed(args.seed + 1), dtype=dtype,
    ).to(device)

    directory = Path(__file__).resolve().parent
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "settings": vars(args), "hardware": hardware_info(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_build": torch.__config__.show(),
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("attention.py", "benchmark.py")
        },
        "protocol": "Fixed-prefix hot-cache decode; sequential untimed cache construction; "
                    "seed-shuffled cases; consecutive samples per case; synchronized wall time.",
    }
    cases = [(name, length) for length in args.prefix_lengths for name in models]
    random.Random(args.seed).shuffle(cases)
    results = []
    for run_index, (name, length) in enumerate(cases):
        row = measure_decode(models[name], x[:, :length + 1], args.warmup, args.repeats)
        description = descriptions[name]
        n = length + 1 if description["window_size"] is None else min(length + 1, args.window_size)
        expected_bytes = (
            args.batch_size * n * x.element_size()
            * description["cached_elements_per_token_per_batch_row"]
        )
        # These are analytical multiply-accumulate counts, not profiler readings.
        # Each projection weight is used once per batch row in this decoder.
        projection_macs = args.batch_size * description["parameters"]
        mixing_macs = args.batch_size * n * description["mixing_macs_per_visible_token_per_batch_row"]
        if row["cache_bytes_after"] != expected_bytes or row["cache_tokens_after"] != n:
            raise AssertionError(f"{name}: returned cache does not match its storage formula")
        row.update(
            variant=name, run_index=run_index, expected_cache_bytes_after=expected_bytes,
            theoretical_projection_macs=projection_macs,
            theoretical_mixing_macs=mixing_macs,
            theoretical_total_macs=projection_macs + mixing_macs,
        )
        results.append(row)

    # Display in context/variant order, while retaining actual run order in JSON.
    order = list(models)
    results.sort(key=lambda row: (row["prefix_tokens"], order.index(row["variant"])))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"metadata": metadata, "models": descriptions, "results": results}, indent=2) + "\n")
    print("Variant  Prefix  Visible  Parameters  Cache KiB  Median us  [Q25, Q75] us")
    for row in results:
        parameters = descriptions[row["variant"]]["parameters"]
        print(
            f"{row['variant']:<8} {row['prefix_tokens']:>6} {row['cache_tokens_after']:>8} "
            f"{parameters:>11} {row['cache_bytes_after'] / 1024:>10.3f} "
            f"{row['median_us']:>10.2f} [{row['q25_us']:.2f}, {row['q75_us']:.2f}]"
        )
    print(f"Raw samples and settings: {output_path}")


if __name__ == "__main__":
    main()
