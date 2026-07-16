from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import __utils__ as ut


SWEEP_DIR = ut.PROJECT_ROOT / "_surprisal_tradeoff_evaluation"
SELECTIONS_PATH = SWEEP_DIR / "all_tradeoff_selections.csv"
GROUND_TRUTH_DIR = ut.PROJECT_ROOT / "saved_vectors"


def _parse_label_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(label_id) for label_id in str(value).split("|") if label_id}


def _load_evaluation_rows():
    if not SELECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SELECTIONS_PATH}. Run evaluate_surprisal_tradeoff.py first."
        )

    selections = pd.read_csv(SELECTIONS_PATH)
    _, _, _, _, metadata, _ = ut.load_DINO_vectors(GROUND_TRUTH_DIR)
    truth = metadata[["path", "all_label_ids", "all_label_names"]].copy()
    truth = truth.drop_duplicates(subset="path")
    truth["truth_row_found"] = True
    rows = selections.drop(
        columns=["all_label_ids", "all_label_names"], errors="ignore"
    ).merge(truth, on="path", how="left", validate="many_to_one")

    missing_truth = rows["truth_row_found"].isna()
    if missing_truth.any():
        raise ValueError(
            f"Could not find ground truth for {int(missing_truth.sum())} selections."
        )
    rows = rows.drop(columns="truth_row_found")

    rows["selection_is_correct"] = [
        int(int(label_id) in _parse_label_ids(all_label_ids))
        for label_id, all_label_ids in zip(
            rows["subset_label_id"], rows["all_label_ids"]
        )
    ]
    return rows


def _summaries(rows):
    total_classes = rows["subset_label_id"].nunique()
    summary = (
        rows.groupby("surprisal_lambda", as_index=False)
        .agg(
            selected_count=("selection_is_correct", "size"),
            correct_count=("selection_is_correct", "sum"),
            classes_with_selections=("subset_label_id", "nunique"),
        )
        .sort_values("surprisal_lambda")
    )
    summary["incorrect_count"] = (
        summary["selected_count"] - summary["correct_count"]
    )
    summary["selection_precision"] = (
        summary["correct_count"] / summary["selected_count"]
    )

    correct_classes = (
        rows.loc[rows["selection_is_correct"] == 1]
        .groupby("surprisal_lambda")["subset_label_id"]
        .nunique()
    )
    summary["classes_with_correct_selection"] = (
        summary["surprisal_lambda"].map(correct_classes).fillna(0).astype(int)
    )
    summary["correct_class_coverage"] = (
        summary["classes_with_correct_selection"] / total_classes
    )

    per_class = (
        rows.groupby(
            ["subset_label_id", "subset_label_name", "surprisal_lambda"],
            as_index=False,
        )
        .agg(
            selected_count=("selection_is_correct", "size"),
            correct_count=("selection_is_correct", "sum"),
            selection_precision=("selection_is_correct", "mean"),
        )
    )
    return summary, per_class


def _plot_summary(summary):
    labels = [f"{value:g}" for value in summary["surprisal_lambda"]]
    positions = np.arange(len(summary))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].bar(positions, summary["selection_precision"], color="#267a6b")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Lambda")
    axes[0].set_ylabel("Correct selections / all selections")
    axes[0].set_title("Selection Precision")
    for position, value in zip(positions, summary["selection_precision"]):
        axes[0].text(position, min(value + 0.025, 0.98), f"{value:.1%}", ha="center")

    axes[1].bar(
        positions,
        summary["correct_count"],
        label="Correct",
        color="#267a6b",
    )
    axes[1].bar(
        positions,
        summary["incorrect_count"],
        bottom=summary["correct_count"],
        label="Incorrect",
        color="#c85454",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("Selection records")
    axes[1].set_title("Correct and Incorrect Selections")
    axes[1].legend()

    axes[2].scatter(
        summary["selected_count"],
        summary["selection_precision"],
        s=80,
        color="#315f91",
    )
    annotation_offsets = {
        0.0: (5, 5),
        0.001: (5, 5),
        0.005: (7, -14),
        0.01: (7, -28),
        0.025: (7, -42),
    }
    for _, row in summary.iterrows():
        axes[2].annotate(
            f"lambda={row['surprisal_lambda']:g}",
            (row["selected_count"], row["selection_precision"]),
            xytext=annotation_offsets[float(row["surprisal_lambda"])],
            textcoords="offset points",
        )
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Total selections")
    axes[2].set_ylabel("Selection precision")
    axes[2].set_title("Quantity vs Correctness")

    figure.suptitle("Lambda Selection Accuracy Using Held-Out COCO Labels")
    figure.tight_layout()
    figure.savefig(SWEEP_DIR / "lambda_selection_accuracy.png", dpi=180)
    plt.close(figure)


def _plot_class_heatmap(per_class):
    heatmap = per_class.pivot(
        index="subset_label_name",
        columns="surprisal_lambda",
        values="selection_precision",
    ).sort_index()
    figure_height = max(10, 0.23 * len(heatmap))
    figure, axis = plt.subplots(figsize=(8, figure_height))
    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#e6e6e6")
    image = axis.imshow(
        heatmap.to_numpy(),
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=color_map,
    )
    axis.set_xticks(
        np.arange(len(heatmap.columns)),
        [f"{value:g}" for value in heatmap.columns],
    )
    axis.set_yticks(np.arange(len(heatmap.index)), heatmap.index, fontsize=7)
    axis.set_xlabel("Lambda")
    axis.set_ylabel("Selected subset")
    axis.set_title("Per-Class Selection Precision (gray = no selections)")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Correct selections / all selections")
    figure.tight_layout()
    figure.savefig(SWEEP_DIR / "lambda_class_accuracy_heatmap.png", dpi=180)
    plt.close(figure)


def evaluate_lambda_selection_accuracy():
    rows = _load_evaluation_rows()
    summary, per_class = _summaries(rows)
    rows.to_csv(SWEEP_DIR / "labeled_tradeoff_selections.csv", index=False)
    summary.to_csv(SWEEP_DIR / "lambda_accuracy_summary.csv", index=False)
    per_class.to_csv(SWEEP_DIR / "lambda_class_accuracy.csv", index=False)
    _plot_summary(summary)
    _plot_class_heatmap(per_class)
    print(summary.to_string(index=False))
    print(f"Saved label-only evaluation -> {SWEEP_DIR}")
    return rows, summary, per_class


if __name__ == "__main__":
    evaluate_lambda_selection_accuracy()
