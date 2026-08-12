from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import __utils__ as ut


INPUT_DIR = ut.PROJECT_ROOT / "_ten_run_average"
OUTPUT_PATH = INPUT_DIR / "_ten_run_entropy_probability_tradeoff.png"
TABLE_PATH = INPUT_DIR / "_ten_run_entropy_probability_tradeoff.csv"


def create_pooled_tradeoff_plot():
    paths = sorted(INPUT_DIR.glob("run_*_entropy_probability_tradeoff.csv"))
    if not paths:
        raise FileNotFoundError("No ten-run entropy-probability CSV files were found.")

    frames = []
    for path in paths:
        run = int(path.name.split("_")[1])
        frame = pd.read_csv(path)
        frame["run"] = run
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["lambda_log_probability"] = (
        data["probability_lambda"].astype(float)
        * data["log_probability"].astype(float)
    )
    data["pooled_net_gain"] = (
        data["von_neumann_entropy_gain"].astype(float)
        + data["lambda_log_probability"]
    )
    data.to_csv(TABLE_PATH, index=False)

    true_mask = data["true_assigned_class"].astype(bool).to_numpy()
    entropy_gain = data["von_neumann_entropy_gain"].to_numpy(float)
    probability_term = data["lambda_log_probability"].to_numpy(float)

    figure, axis = plt.subplots(figsize=(11, 7))
    lambda_values_numeric = sorted(data["probability_lambda"].unique())
    lambda_colors = dict(zip(lambda_values_numeric, plt.cm.tab10.colors))
    for lambda_value in lambda_values_numeric:
        lambda_mask = np.isclose(
            data["probability_lambda"].to_numpy(float), lambda_value
        )
        absent = lambda_mask & ~true_mask
        present = lambda_mask & true_mask
        axis.scatter(
            entropy_gain[absent], probability_term[absent],
            marker="x", s=30, linewidth=1.1, alpha=0.42,
            color=lambda_colors[lambda_value], zorder=3,
        )
        axis.scatter(
            entropy_gain[present], probability_term[present],
            marker="o", s=30, linewidth=0, alpha=0.55,
            color=lambda_colors[lambda_value], zorder=4,
        )

    x_min = min(0.0, float(entropy_gain.min()))
    x_max = float(entropy_gain.max())
    x_padding = max((x_max - x_min) * 0.04, 1e-8)
    x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 500)
    y_min = float(probability_term.min())
    y_padding = max(abs(y_min) * 0.04, 1e-8)
    zero_boundary = -x_grid
    axis.plot(
        x_grid, zero_boundary, color="#1769aa", linewidth=2.2,
        zorder=2,
    )

    gain_scale = max(float(np.quantile(np.abs(data["pooled_net_gain"]), 0.9)), 1e-8)
    for level in (-gain_scale, -gain_scale / 2, gain_scale / 2, gain_scale):
        axis.plot(
            x_grid, level - x_grid, linestyle="--", linewidth=1.0, alpha=0.55,
            color="#2e8b57" if level > 0 else "#d1495b",
        )

    lambda_values = ", ".join(
        f"{value:g}" for value in sorted(data["probability_lambda"].unique())
    )
    axis.set(
        xlim=(x_min - x_padding, x_max + x_padding),
        ylim=(y_min - y_padding, y_padding),
        xlabel=r"Marginal entropy gain, $\Delta H$",
        ylabel=r"Weighted log-probability, $\lambda\log(\hat P_+)$",
        title=(
            f"Pooled entropy-probability tradeoff across {len(paths)} runs "
            f"({true_mask.sum():,}/{len(data):,} true positives)\n"
            rf"Observed $\lambda$: {lambda_values}"
        ),
    )
    axis.grid(alpha=0.2)
    lambda_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            color=lambda_colors[value], label=rf"$\lambda={value:g}$",
        )
        for value in lambda_values_numeric
    ]
    status_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            color="#555555", label=f"True label present ({true_mask.sum():,})",
        ),
        Line2D(
            [0], [0], marker="x", linestyle="none", markersize=7,
            color="#555555", label=f"True label absent ({(~true_mask).sum():,})",
        ),
        Line2D(
            [0], [0], color="#2e8b57", linestyle="--",
            label="Positive net-gain contour",
        ),
        Line2D([0], [0], color="#1769aa", linewidth=2.2, label="Net gain = 0"),
        Line2D(
            [0], [0], color="#d1495b", linestyle="--",
            label="Negative net-gain contour",
        ),
    ]
    lambda_legend = axis.legend(
        handles=lambda_handles, title="Selection lambda", fontsize=8,
        title_fontsize=9, loc="lower left",
    )
    axis.add_artist(lambda_legend)
    axis.legend(handles=status_handles, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved pooled ten-run tradeoff plot -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    create_pooled_tradeoff_plot()
