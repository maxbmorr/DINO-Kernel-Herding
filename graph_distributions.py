from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import __utils__ as ut


OUTPUT_DIR = ut.PROJECT_ROOT / "distribution_graphs"
TARGET_DIR = ut.PROJECT_ROOT / "saved_vectors" / "testing"


def _load_csv(name):
    path = TARGET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Run main.py first. Missing: {path}")
    return pd.read_csv(path)


def _save_current(name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved {output_path}")


def plot_subset_probability_distributions(scores):
    method = scores["probability_method"].iloc[0] if "probability_method" in scores else "subset"
    plt.figure(figsize=(9, 5))
    plt.hist(scores["subset_probability"], bins=40, alpha=0.75, label=method)
    if "nce_subset_probability" in scores:
        plt.hist(scores["nce_subset_probability"], bins=40, alpha=0.35, label="NCE")
    plt.xlabel("NCE positive-membership probability")
    plt.ylabel("Image count")
    plt.title("NCE Positive-Membership Probability Distribution")
    plt.legend()
    _save_current("subset_probability_histogram.png")


def plot_subset_counts(scores, summary):
    predicted_counts = (
        scores["predicted_subset_label_name"]
        .value_counts()
        .head(20)
        .sort_values()
    )
    plt.figure(figsize=(10, 7))
    plt.barh(predicted_counts.index, predicted_counts.values)
    plt.xlabel("Target image count")
    plt.ylabel("Predicted subset")
    plt.title("Top Predicted Subset Counts")
    _save_current("top_predicted_subset_counts.png")

    selected_counts = summary.sort_values("selected_count", ascending=False).head(20)
    selected_counts = selected_counts.sort_values("selected_count")
    plt.figure(figsize=(10, 7))
    plt.barh(selected_counts["subset_label_name"], selected_counts["selected_count"])
    plt.xlabel("Selected image count")
    plt.ylabel("Subset")
    plt.title("Top Optimization Selection Counts")
    _save_current("top_optimization_selection_counts.png")


def plot_optimization_distributions(selection, summary):
    if len(selection) == 0:
        print("No optimization selections found. Skipping optimization graphs.")
        return

    plt.figure(figsize=(9, 5))
    plt.hist(selection["objective_gain"], bins=40, alpha=0.75)
    plt.xlabel("Objective gain")
    plt.ylabel("Selected image count")
    plt.title("Optimization Objective Gain Distribution")
    _save_current("objective_gain_histogram.png")

    plt.figure(figsize=(9, 5))
    plt.hist(selection["von_neumann_entropy_gain"], bins=40, alpha=0.75)
    plt.xlabel("Von Neumann entropy gain")
    plt.ylabel("Selected image count")
    plt.title("Von Neumann Entropy Gain Distribution")
    _save_current("von_neumann_entropy_gain_histogram.png")

    plt.figure(figsize=(9, 5))
    plt.scatter(
        selection["subset_probability"],
        selection["von_neumann_entropy_gain"],
        s=22,
        alpha=0.65,
    )
    plt.xlabel("Subset probability")
    plt.ylabel("Von Neumann entropy gain")
    plt.title("Selected Images: Probability vs Entropy Gain")
    _save_current("selected_probability_vs_entropy_gain.png")

    rank_summary = (
        selection.groupby("optimization_rank", as_index=False)
        .agg(
            mean_objective_gain=("objective_gain", "mean"),
            mean_entropy_gain=("von_neumann_entropy_gain", "mean"),
        )
    )
    probability_lambda = selection["probability_lambda"].iloc[0]

    plt.figure(figsize=(8, 5))
    plt.plot(
        rank_summary["optimization_rank"],
        rank_summary["mean_objective_gain"],
        marker="o",
        label="objective gain",
    )
    plt.plot(
        rank_summary["optimization_rank"],
        rank_summary["mean_entropy_gain"],
        marker="o",
        label="entropy gain",
    )
    plt.xlabel("Optimization rank")
    plt.ylabel("Mean gain")
    plt.title(f"Mean Gain by Greedy Selection Rank (lambda={probability_lambda:g})")
    plt.legend()
    _save_current("mean_gain_by_selection_rank.png")

    plt.figure(figsize=(8, 5))
    plt.plot(
        rank_summary["optimization_rank"],
        rank_summary["mean_objective_gain"],
        marker="o",
    )
    plt.xlabel("Optimization rank")
    plt.ylabel("Mean objective gain")
    plt.title(f"Mean Objective Gain by Greedy Selection Rank (lambda={probability_lambda:g})")
    _save_current("mean_objective_gain_by_selection_rank.png")

    plt.figure(figsize=(8, 5))
    plt.plot(
        rank_summary["optimization_rank"],
        rank_summary["mean_entropy_gain"],
        marker="o",
    )
    plt.xlabel("Optimization rank")
    plt.ylabel("Mean von Neumann entropy gain")
    plt.title(f"Mean Entropy Gain by Greedy Selection Rank (lambda={probability_lambda:g})")
    _save_current("mean_entropy_gain_by_selection_rank.png")

    entropy_delta = summary.copy()
    entropy_delta["entropy_delta"] = (
        entropy_delta["final_von_neumann_entropy"]
        - entropy_delta["base_von_neumann_entropy"]
    )
    entropy_delta = entropy_delta.sort_values("entropy_delta", ascending=False).head(20)
    entropy_delta = entropy_delta.sort_values("entropy_delta")
    plt.figure(figsize=(10, 7))
    plt.barh(entropy_delta["subset_label_name"], entropy_delta["entropy_delta"])
    plt.xlabel("Final entropy - base entropy")
    plt.ylabel("Subset")
    plt.title("Largest Von Neumann Entropy Changes by Subset")
    _save_current("top_entropy_changes_by_subset.png")


def create_distribution_graphs():
    scores = _load_csv("subset_probability_scores.csv")
    selection = _load_csv("optimization_selection.csv")
    summary = _load_csv("optimization_summary.csv")

    plot_subset_probability_distributions(scores)
    plot_subset_counts(scores, summary)
    plot_optimization_distributions(selection, summary)

    print(f"Distribution graphs saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    create_distribution_graphs()
