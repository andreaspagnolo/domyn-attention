"""Plot the measured context sweep: medians and interquartile ranges, with no fit."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # File output only; no GUI or display is required.
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "results", nargs="?", type=Path,
    default=Path("results/context_scaling_32768_cpu_float32.json"),
)
args = parser.parse_args()
report = json.loads(args.results.read_text())
settings = report["metadata"]["settings"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for variant, color, marker in (
    ("MHA", "#0072B2", "o"), ("SWA", "#009E73", "s"),
    ("MLA", "#E69F00", "^"), ("SWA+MLA", "#CC79A7", "D"),
):
    rows = sorted(
        (row for row in report["results"] if row["variant"] == variant),
        key=lambda row: row["prefix_tokens"],
    )
    ax.errorbar(
        [row["prefix_tokens"] for row in rows], [row["median_us"] for row in rows],
        yerr=[
            [row["median_us"] - row["q25_us"] for row in rows],
            [row["q75_us"] - row["median_us"] for row in rows],
        ],
        label=variant, color=color, marker=marker, linewidth=1.8,
        markersize=6, capsize=3, elinewidth=1, zorder=3,
    )

ax.set_xscale("log", base=2)
lengths = sorted(settings["prefix_lengths"])
ax.set_xticks(lengths, [f"{length:,}" for length in lengths])
# Log latency keeps the short-context crossover and long-context growth visible.
ax.set_yscale("log", base=2)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.set_xlabel("Prefix length before decoding the next token (tokens; log scale)")
ax.set_ylabel("Single-token decode latency (µs; log scale)")
ax.set_title("Cached decoding as context grows", loc="left", fontsize=15, pad=35)
ax.text(
    0, 1.025,
    f"{settings['device'].upper()} · {settings['dtype']} · batch {settings['batch_size']} · "
    f"width {settings['d_model']} · {settings['num_heads']} heads · window {settings['window_size']}",
    transform=ax.transAxes, fontsize=10, color="#444444",
)
ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False, ncol=2)
fig.text(
    0.11, 0.04,
    f"Markers: measured medians; whiskers: 25th–75th percentiles. "
    f"{settings['repeats']} calls after {settings['warmup']} warmups.\n"
    "Lines only connect sampled points; no fitted curve. Cache construction is excluded.",
    fontsize=9, color="#555555",
)
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.83)
output = args.results.with_suffix(".svg")
fig.savefig(output, facecolor="white")
plt.close(fig)
print(f"Saved {output} (Matplotlib {matplotlib.__version__})")
