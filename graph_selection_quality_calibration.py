from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SELECTION_PATH = ROOT / "saved_vectors" / "retrain" / "optimization_selection.csv"
METADATA_PATH = ROOT / "saved_vectors" / "retrain" / "metadata.csv"
OUTPUT_DIR = ROOT / "_organized_calibrated_images" / "AUC_evaluation"
OUTPUT_PATH = OUTPUT_DIR / "_selection_quality_calibration.png"
TABLE_PATH = OUTPUT_DIR / "_selection_quality_calibration.csv"


def _label_set(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return set(str(value).split("|"))


def create_selection_quality_calibration():
    selection = pd.read_csv(SELECTION_PATH)
    metadata = pd.read_csv(METADATA_PATH, usecols=["path", "all_label_names"])
    audited = selection.merge(metadata, on="path", how="left", validate="many_to_one")
    if audited["all_label_names"].isna().all():
        raise ValueError("Selected paths do not match retraining metadata.")
    audited["is_true_positive"] = [
        class_name in _label_set(labels)
        for class_name, labels in zip(
            audited["subset_label_name"], audited["all_label_names"]
        )
    ]

    edges = np.linspace(0.0, 1.0, 11)
    audited["probability_bin"] = pd.cut(
        audited["positive_probability"], edges, include_lowest=True
    )
    grouped = audited.groupby("probability_bin", observed=True)
    calibration = grouped.agg(
        selected_count=("is_true_positive", "size"),
        mean_predicted_probability=("positive_probability", "mean"),
        observed_positive_fraction=("is_true_positive", "mean"),
    ).reset_index()

    # Wilson score interval for the observed positive proportion in each bin.
    z = 1.959963984540054
    n = calibration["selected_count"].to_numpy(dtype=float)
    p = calibration["observed_positive_fraction"].to_numpy(dtype=float)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    calibration["ci95_low"] = np.clip(center - half_width, 0, 1)
    calibration["ci95_high"] = np.clip(center + half_width, 0, 1)
    calibration["probability_bin"] = calibration["probability_bin"].astype(str)

    brier = float(np.mean(
        (audited["positive_probability"] - audited["is_true_positive"].astype(int)) ** 2
    ))
    ece = float(np.sum(
        calibration["selected_count"] / len(audited)
        * np.abs(
            calibration["observed_positive_fraction"]
            - calibration["mean_predicted_probability"]
        )
    ))
    calibration["overall_brier_score"] = brier
    calibration["overall_ece"] = ece
    calibration.to_csv(TABLE_PATH, index=False)

    figure, (calibration_axis, count_axis) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = calibration["mean_predicted_probability"].to_numpy()
    y = calibration["observed_positive_fraction"].to_numpy()
    calibration_axis.plot([0, 1], [0, 1], color="#555555", linestyle="--",
                          label="Perfect calibration")
    calibration_axis.errorbar(
        x, y,
        yerr=np.vstack([
            np.maximum(y - calibration["ci95_low"].to_numpy(), 0.0),
            np.maximum(calibration["ci95_high"].to_numpy() - y, 0.0),
        ]),
        marker="o", markersize=7, linewidth=2, capsize=4,
        color="#d1495b", label="Optimized selections",
    )
    calibration_axis.set(
        xlim=(0, 1), ylim=(0, 1.01),
        ylabel="Observed true-positive fraction",
        title="Selection-quality calibration",
    )
    calibration_axis.text(
        0.03, 0.95, f"Brier = {brier:.3f}\nECE = {ece:.3f}\nn = {len(audited)}",
        transform=calibration_axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    calibration_axis.grid(alpha=0.2)
    calibration_axis.legend(loc="lower right")

    count_axis.bar(
        x, calibration["selected_count"], width=0.075,
        color="#1769aa", alpha=0.8,
    )
    count_axis.set(
        xlabel="Mean predicted positive probability",
        ylabel="Selections",
        xlim=(0, 1),
    )
    count_axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved selection-quality calibration -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    create_selection_quality_calibration()
