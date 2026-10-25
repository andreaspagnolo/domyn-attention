"""Check the measurement protocol and reporting, without asserting timing speed."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from attention import MHA, MLA, MHACache
from benchmark import cache_storage_bytes, measure_decode


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_samples_reuse_fixed_prefix_and_warmup_is_excluded(self):
        for kind in ("mha", "mla"):
            for window in (None, 3):
                for prefix in (0, 5):
                    with self.subTest(kind=kind, window=window, prefix=prefix):
                        if kind == "mha":
                            model = MHA(8, 2, window)
                            bytes_per_token = 256  # B=2, 2*D=16, float64=8 bytes.
                        else:
                            model = MLA(8, 2, 3, 5, 2, window)
                            bytes_per_token = 80   # B=2, r+rope=5, float64=8 bytes.
                        model = model.double().eval()
                        x = torch.randn(2, prefix + 1, 8, dtype=torch.float64)
                        with patch.object(model, "decode", wraps=model.decode) as decode:
                            result = measure_decode(model, x, warmup=2, repeats=3)
                        self.assertEqual(decode.call_count, prefix + 2 + 3)
                        timed_calls = decode.call_args_list[prefix:]
                        base_cache = timed_calls[0].args[1]
                        for call in timed_calls:
                            self.assertIs(call.args[1], base_cache)
                            torch.testing.assert_close(call.args[0], x[:, -1:], rtol=0, atol=0)
                        if prefix:
                            self.assertEqual(base_cache.position, prefix)
                        else:
                            self.assertIsNone(base_cache)
                        before = prefix if window is None else min(prefix, window)
                        after = prefix + 1 if window is None else min(prefix + 1, window)
                        self.assertEqual(result["cache_tokens_before"], before)
                        self.assertEqual(result["cache_tokens_after"], after)
                        self.assertEqual(result["cache_bytes_before"], bytes_per_token * before)
                        self.assertEqual(result["cache_bytes_after"], bytes_per_token * after)
                        self.assertEqual(len(result["samples_us"]), 3)
                        self.assertTrue(all(sample >= 0 for sample in result["samples_us"]))
                        self.assertLessEqual(result["q25_us"], result["median_us"])
                        self.assertLessEqual(result["median_us"], result["q75_us"])

    def test_storage_measurement_counts_underlying_allocation(self):
        key = torch.zeros(1, 2, 6, 4, dtype=torch.float64)
        value = torch.zeros_like(key)
        # Deliberately retain views of larger allocations. Counting numel alone
        # would incorrectly report 384 bytes instead of the actual 768 here.
        cache = MHACache(key[..., -3:, :], value[..., -3:, :], 6)
        self.assertEqual(cache_storage_bytes(cache), 768)
        self.assertEqual(cache_storage_bytes(None), 0)

    def test_cli_records_all_variants_dimensions_and_analytical_counts(self):
        directory = Path(__file__).resolve().parents[1]
        # Keep subprocess output and temporary files on the external SSD.
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            output = Path(temporary) / "smoke.json"
            command = [
                sys.executable, "-B", str(directory / "benchmark.py"),
                "--device", "cpu", "--d-model", "8", "--num-heads", "2",
                "--kv-rank", "3", "--q-rank", "5", "--rope-dim", "2",
                "--window-size", "3", "--prefix-lengths", "0", "5",
                "--warmup", "0", "--repeats", "2", "--output", str(output),
            ]
            environment = dict(os.environ, TMPDIR=temporary, PYTHONDONTWRITEBYTECODE="1")
            process = subprocess.run(
                command, cwd=directory, env=environment,
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads(output.read_text())

        self.assertEqual(set(report["models"]), {"MHA", "SWA", "MLA", "SWA+MLA"})
        self.assertEqual(len(report["results"]), 8)
        self.assertEqual({row["run_index"] for row in report["results"]}, set(range(8)))
        self.assertEqual(report["metadata"]["intraop_threads"], 1)
        for row in report["results"]:
            is_mla = row["variant"] in ("MLA", "SWA+MLA")
            local = row["variant"] in ("SWA", "SWA+MLA")
            n = min(row["prefix_tokens"] + 1, 3) if local else row["prefix_tokens"] + 1
            model = report["models"][row["variant"]]
            parameters = 252 if is_mla else 256
            self.assertEqual(model["parameters"], parameters)
            self.assertEqual(model["qk_head_dim"], 6 if is_mla else 4)
            self.assertEqual(model["value_head_dim"], 4)
            self.assertEqual(row["cache_bytes_after"], (20 if is_mla else 64) * n)
            self.assertEqual(row["expected_cache_bytes_after"], row["cache_bytes_after"])
            self.assertEqual(row["theoretical_projection_macs"], parameters)
            self.assertEqual(row["theoretical_mixing_macs"], 16 * n)
            self.assertEqual(row["theoretical_total_macs"], parameters + 16 * n)
            self.assertEqual(len(row["samples_us"]), 2)


if __name__ == "__main__":
    unittest.main()
