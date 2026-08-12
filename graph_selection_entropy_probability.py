from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SELECTION_PATH = ROOT / "saved_vectors" / "retrain" / "optimization_selection.csv"
METADATA_PATH = ROOT / "saved_vectors" / "retrain" / "metadata.csv"
OUTPUT_DIR = ROOT / "_organized_calibrated_images" / "AUC_evaluation"
OUTPUT_PATH = OUTPUT_DIR / "_selection_entropy_probability_tradeoff.png"
TABLE_PATH = OUTPUT_DIR / "_selection_entropy_probability_tradeoff.csv"


def _labels(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return set(str(value).split("|"))


def create_tradeoff_plot():
    selection = pd.read_csv(SELECTION_PATH)
    metadata = pd.read_csv(METADATA_PATH, usecols=["path", "all_label_names"])
    data = selection.merge(metadata, on="path", how="left", validate="many_to_one")
    data["true_assigned_class"] = [
        class_name in _labels(label_names)
        for class_name, label_names in zip(
            data["subset_label_name"], data["all_label_names"]
        )
    ]
    used_lambdas = data["probability_lambda"].astype(float).unique()
    if len(used_lambdas) != 1:
        raise ValueError(
            f"Expected one lambda in the saved selection, found {used_lambdas}."
        )
    selected_lambda = float(used_lambdas[0])
    data["net_gain_at_selected_lambda"] = (
        data["von_neumann_entropy_gain"]
        + selected_lambda * data["log_probability"]
    )
    data.to_csv(TABLE_PATH, index=False)

    class_names = data["subset_label_name"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(class_names), 1,
        figsize=(11, max(5.5, 4.8 * len(class_names))),
        squeeze=False,
    )

    for axis, class_name in zip(axes.ravel(), class_names):
        class_data = data[data["subset_label_name"] == class_name]
        true_mask = class_data["true_assigned_class"].to_numpy(dtype=bool)
        delta_entropy = class_data["von_neumann_entropy_gain"].to_numpy(dtype=float)
        log_probability = class_data["log_probability"].to_numpy(dtype=float)

        axis.scatter(
            delta_entropy[~true_mask], log_probability[~true_mask],
            marker="x", s=58, linewidth=1.7, color="#d1495b",
            label="True label absent", zorder=3,
        )
        axis.scatter(
            delta_entropy[true_mask], log_probability[true_mask],
            marker="o", s=58, edgecolor="white", linewidth=0.8,
            color="#2e8b57", label="True label present", zorder=4,
        )

        x_max = max(delta_entropy.max() * 1.08, 1e-6)
        x_grid = np.linspace(0.0, x_max, 400)
        # Keep the view centered on observed selections; extreme lambda rays
        # are clipped naturally instead of compressing all points vertically.
        y_min = min(log_probability.min() * 1.25, -0.1)

        zero_boundary = -x_grid / selected_lambda
        visible_boundary = np.clip(zero_boundary, y_min, 0.0)
        axis.fill_between(
            x_grid, visible_boundary, 0.0,
            color="#2e8b57", alpha=0.07,
            label="Positive net gain", zorder=0,
        )
        axis.fill_between(
            x_grid, y_min, visible_boundary,
            color="#d1495b", alpha=0.09,
            label="Negative net gain", zorder=0,
        )

        net_gains = class_data["net_gain_at_selected_lambda"].to_numpy(float)
        contour_scale = max(float(np.quantile(np.abs(net_gains), 0.9)), 1e-6)
        contour_levels = np.linspace(-contour_scale, contour_scale, 7)
        for level in contour_levels:
            contour = (level - x_grid) / selected_lambda
            is_zero = np.isclose(level, 0.0, atol=contour_scale * 1e-9)
            axis.plot(
                x_grid, contour,
                linestyle="-" if is_zero else "--",
                linewidth=2.2 if is_zero else 1.0,
                alpha=0.95 if is_zero else 0.55,
                color="#1769aa" if is_zero else (
                    "#2e8b57" if level > 0 else "#d1495b"
                ),
                label=(
                    "Net gain = 0" if is_zero
                    else f"Net gain = {level:+.2e}"
                ),
            )

        axis.set(
            xlim=(-0.025 * x_max, x_max),
            ylim=(y_min, 0.02),
            xlabel=r"Marginal entropy gain, $\Delta H$",
            ylabel=r"Log positive probability, $\log(\hat P_+)$",
            title=(
                f"{class_name}: entropy–probability tradeoff "
                f"({int(true_mask.sum())}/{len(true_mask)} true positives)"
            ),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2, loc="lower right")

    figure.suptitle(
        rf"Iso-gain contours for $\lambda={selected_lambda:g}$: "
        r"$\Delta H+\lambda\log(\hat P_+)=c$",
        fontsize=15,
    )
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved entropy-probability tradeoff plot -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    create_tradeoff_plot()
