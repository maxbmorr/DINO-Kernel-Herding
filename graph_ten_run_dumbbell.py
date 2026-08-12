from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "_ten_run_average" / "_ten_run_results.csv"
OUTPUT_PATH = ROOT / "_ten_run_average" / "_ten_run_dumbbell.png"


def create_dumbbell_plot():
    results = pd.read_csv(INPUT_PATH)
    models = ("M_0", "M_1", "M_rand")
    colors = {"M_0": "#1769aa", "M_1": "#d1495b", "M_rand": "#2e8b57"}
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    for axis, metric, title in zip(
        axes,
        ("macro_auc", "micro_auc"),
        ("Macro ROC AUC", "Micro ROC AUC"),
    ):
        wide = results.pivot(index="run", columns="model", values=metric)
        metadata = results.drop_duplicates("run").set_index("run")
        order = wide["M_1"].sub(wide["M_0"]).sort_values().index
        wide = wide.loc[order]
        y = np.arange(len(wide))

        for position, (_, row) in enumerate(wide.iterrows()):
            axis.plot(
                [row[model] for model in models],
                [position] * len(models),
                color="#a8a8a8",
                linewidth=1.6,
                zorder=1,
            )
        for model in models:
            axis.scatter(
                wide[model], y, s=62, color=colors[model],
                edgecolor="white", linewidth=0.8, label=model, zorder=2,
            )

        axis.set(
            yticks=y,
            yticklabels=[
                f"Run {run}: {metadata.loc[run, 'classes'].replace('|', ' / ')}"
                for run in wide.index
            ],
            xlabel=title,
            title=f"Paired {title} by seed",
        )
        axis.grid(axis="x", alpha=0.22)
        axis.legend(loc="lower right")

    figure.suptitle("Within-seed model comparison", fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved paired dumbbell plot -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    create_dumbbell_plot()
