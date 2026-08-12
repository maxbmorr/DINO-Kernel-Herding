from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "_ten_run_average" / "_ten_run_results.csv"
OUTPUT_PATH = ROOT / "_ten_run_average" / "_ten_run_caterpillar.png"


def create_caterpillar_plot():
    results = pd.read_csv(INPUT_PATH)
    required_models = {"M_0", "M_1", "M_rand"}
    if set(results["model"]) != required_models:
        raise ValueError(f"Expected models {sorted(required_models)}.")

    wide = results.pivot(index="run", columns="model", values="macro_auc")
    metadata = results.drop_duplicates("run").set_index("run")
    effects = pd.DataFrame({
        "M_1": wide["M_1"] - wide["M_0"],
        "M_rand": wide["M_rand"] - wide["M_0"],
    })

    colors = {"M_1": "#d1495b", "M_rand": "#2e8b57"}
    labels = {
        "M_1": r"Optimized augmentation: $M_1-M_0$",
        "M_rand": r"Random forced-positive: $M_{rand}-M_0$",
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)

    for axis, model_name in zip(axes, ("M_1", "M_rand")):
        ordered = effects[model_name].sort_values()
        y = np.arange(len(ordered))
        mean = float(ordered.mean())
        std = float(ordered.std(ddof=1))
        margin = float(student_t.ppf(0.975, len(ordered) - 1) * std / np.sqrt(len(ordered)))

        axis.axvspan(mean - margin, mean + margin, color=colors[model_name], alpha=0.14,
                     label="95% CI of mean")
        axis.axvline(mean, color=colors[model_name], linewidth=2,
                     label=f"Mean = {mean:+.4f}")
        axis.axvline(0, color="#333333", linestyle="--", linewidth=1.2,
                     label="No AUC change")
        axis.scatter(ordered.to_numpy(), y, s=65, color=colors[model_name],
                     edgecolor="white", linewidth=0.8, zorder=3)

        tick_labels = [
            f"Run {run}: {metadata.loc[run, 'classes'].replace('|', ' / ')}"
            for run in ordered.index
        ]
        axis.set(
            yticks=y,
            yticklabels=tick_labels,
            xlabel=r"Macro ROC AUC change relative to $M_0$",
            title=labels[model_name],
        )
        axis.grid(axis="x", alpha=0.22)
        axis.legend(loc="lower right", fontsize=9)

    x_limit = max(abs(axis.get_xlim()[0]) for axis in axes)
    x_limit = max(x_limit, max(abs(axis.get_xlim()[1]) for axis in axes))
    for axis in axes:
        axis.set_xlim(-x_limit, x_limit)
    figure.suptitle("Ten-run caterpillar plot of augmentation effects", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved caterpillar plot -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    create_caterpillar_plot()
