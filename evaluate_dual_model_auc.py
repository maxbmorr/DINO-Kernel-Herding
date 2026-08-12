from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import __utils__ as ut
import subset_probability as sp


OUTPUT_DIR = ut.PROJECT_ROOT / "_organized_calibrated_images" / "AUC_evaluation"
def _resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ut.PROJECT_ROOT / path
    return path


def _probability_matrix(model, X, class_ids):
    X_scaled = model.scaler.transform(X)
    return np.column_stack([
        model.classifiers_by_class[int(label_id)].calibrated_probability(X_scaled)
        for label_id in class_ids
    ])


def _threshold_vector(model, class_ids):
    return np.asarray([
        model.classifiers_by_class[int(label_id)].threshold
        for label_id in class_ids
    ], dtype=float)


def _macro_roc(y_true, probabilities):
    false_positive_grid = np.linspace(0.0, 1.0, 1001)
    interpolated_true_positive = []
    for position in range(y_true.shape[1]):
        false_positive, true_positive, _ = roc_curve(
            y_true[:, position],
            probabilities[:, position],
        )
        interpolated = np.interp(
            false_positive_grid,
            false_positive,
            true_positive,
        )
        interpolated[0] = 0.0
        interpolated_true_positive.append(interpolated)

    mean_true_positive = np.mean(interpolated_true_positive, axis=0)
    mean_true_positive[-1] = 1.0
    return (
        false_positive_grid,
        mean_true_positive,
        auc(false_positive_grid, mean_true_positive),
    )


def _exact_macro_auc(y_true, probabilities):
    return float(np.mean([
        roc_auc_score(y_true[:, position], probabilities[:, position])
        for position in range(y_true.shape[1])
    ]))


def evaluate_dual_model_auc(
    model_0_path="saved_models/calibrated/M_0.joblib",
    model_1_path="saved_models/calibrated/M_1.joblib",
    model_rand_path="saved_models/calibrated/M_rand.joblib",
    test_dir="saved_vectors/test",
    selection_path="saved_vectors/retrain/optimization_selection.csv",
    best_hyperparameters_path="_hyperparameter_sensitivity/best_hyperparameters.csv",
    output_dir=OUTPUT_DIR,
):
    model_0 = joblib.load(_resolve_path(model_0_path))
    model_1 = joblib.load(_resolve_path(model_1_path))
    model_rand = joblib.load(_resolve_path(model_rand_path))
    models = {"M_0": model_0, "M_1": model_1, "M_rand": model_rand}
    test_dir = _resolve_path(test_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_path = _resolve_path(selection_path)
    selection = pd.read_csv(selection_path)
    selected_image_count = int(selection["path"].dropna().nunique())
    per_class_selection_counts = selection.groupby("subset_label_id").size()
    selection_k = (
        int(per_class_selection_counts.iloc[0])
        if len(per_class_selection_counts)
        and per_class_selection_counts.nunique() == 1
        else np.nan
    )
    best_hyperparameters = pd.read_csv(
        _resolve_path(best_hyperparameters_path)
    ).iloc[0]

    X_test, _, _, _, metadata, _ = ut.load_DINO_vectors(test_dir)
    shared_class_ids = sorted(
        set(int(value) for value in model_0.class_ids)
        & set(int(value) for value in model_1.class_ids)
        & set(int(value) for value in model_rand.class_ids)
    )
    valid_class_ids = []
    true_columns = []
    for label_id in shared_class_ids:
        targets = sp.multilabel_targets(metadata, label_id)
        if targets.min() == targets.max():
            continue
        valid_class_ids.append(label_id)
        true_columns.append(targets)
    if not valid_class_ids:
        raise ValueError("No test classes contain both positive and negative examples.")

    y_true = np.column_stack(true_columns)
    probabilities = {
        name: _probability_matrix(model, X_test, valid_class_ids)
        for name, model in models.items()
    }
    thresholds = {
        name: _threshold_vector(model, valid_class_ids)
        for name, model in models.items()
    }

    class_name_by_id = {
        int(label_id): name
        for label_id, name in zip(model_0.class_ids, model_0.class_names)
    }
    per_class_rows = []
    for position, label_id in enumerate(valid_class_ids):
        row = {
            "label_id": label_id,
            "label_name": class_name_by_id[label_id],
            "positive_test_count": int(y_true[:, position].sum()),
        }
        for name in models:
            row[f"{name}_auc"] = roc_auc_score(
                y_true[:, position], probabilities[name][:, position]
            )
            row[f"{name}_accuracy"] = accuracy_score(
                y_true[:, position],
                probabilities[name][:, position] >= thresholds[name][position],
            )
            class_predictions = (
                probabilities[name][:, position] >= thresholds[name][position]
            )
            row[f"{name}_precision"] = precision_score(
                y_true[:, position], class_predictions, zero_division=0
            )
            row[f"{name}_recall"] = recall_score(
                y_true[:, position], class_predictions, zero_division=0
            )
            row[f"{name}_f1"] = f1_score(
                y_true[:, position], class_predictions, zero_division=0
            )
            row[f"{name}_threshold"] = thresholds[name][position]
        row["M_1_auc_change_vs_M_0"] = row["M_1_auc"] - row["M_0_auc"]
        row["M_rand_auc_change_vs_M_0"] = row["M_rand_auc"] - row["M_0_auc"]
        row["M_1_accuracy_change_vs_M_0"] = (
            row["M_1_accuracy"] - row["M_0_accuracy"]
        )
        row["M_rand_accuracy_change_vs_M_0"] = (
            row["M_rand_accuracy"] - row["M_0_accuracy"]
        )
        per_class_rows.append(row)
    macro_rocs = {name: _macro_roc(y_true, values) for name, values in probabilities.items()}
    micro_rocs = {
        name: roc_curve(y_true.ravel(), values.ravel())
        for name, values in probabilities.items()
    }
    roc_grid = np.linspace(0.0, 1.0, 1001)
    roc_curve_data = {"false_positive_rate": roc_grid}
    for name in models:
        roc_curve_data[f"{name}_macro_true_positive_rate"] = macro_rocs[name][1]
        micro_tpr = np.interp(
            roc_grid, micro_rocs[name][0], micro_rocs[name][1]
        )
        micro_tpr[0] = 0.0
        micro_tpr[-1] = 1.0
        roc_curve_data[f"{name}_micro_true_positive_rate"] = micro_tpr
    pd.DataFrame(roc_curve_data).to_csv(
        output_dir / "_roc_curves.csv", index=False
    )
    summary_rows = []
    for name, values in probabilities.items():
        binary = values >= thresholds[name][None, :]
        summary_rows.append({
            "model": name,
            "test_image_count": len(X_test),
            "evaluated_class_count": len(valid_class_ids),
            "selection_k_per_class": 0 if name == "M_0" else selection_k,
            "selected_unique_image_count": (
                0 if name == "M_0" else selected_image_count
            ),
            "macro_auc": _exact_macro_auc(y_true, values),
            "micro_auc": roc_auc_score(y_true.ravel(), values.ravel()),
            "macro_accuracy": float(np.mean([
                accuracy_score(y_true[:, position], binary[:, position])
                for position in range(y_true.shape[1])
            ])),
            "micro_accuracy": accuracy_score(y_true.ravel(), binary.ravel()),
            "exact_match_accuracy": accuracy_score(y_true, binary),
            "macro_precision": precision_score(
                y_true, binary, average="macro", zero_division=0
            ),
            "micro_precision": precision_score(
                y_true, binary, average="micro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_true, binary, average="macro", zero_division=0
            ),
            "micro_recall": recall_score(
                y_true, binary, average="micro", zero_division=0
            ),
            "macro_f1": f1_score(
                y_true, binary, average="macro", zero_division=0
            ),
            "micro_f1": f1_score(
                y_true, binary, average="micro", zero_division=0
            ),
            "mean_decision_threshold": float(thresholds[name].mean()),
            "min_decision_threshold": float(thresholds[name].min()),
            "max_decision_threshold": float(thresholds[name].max()),
        })
    summary = pd.DataFrame(summary_rows)

    combined_rows = []
    for row in summary_rows:
        combined_rows.append({"scope": "overall", **row})
    for row in per_class_rows:
        for name in models:
            combined_rows.append({
                "scope": "class",
                "model": name,
                "label_id": row["label_id"],
                "label_name": row["label_name"],
                "positive_test_count": row["positive_test_count"],
                "test_image_count": len(X_test),
                "selection_k_per_class": 0 if name == "M_0" else selection_k,
                "selected_unique_image_count": (
                    0 if name == "M_0" else selected_image_count
                ),
                "class_auc": row[f"{name}_auc"],
                "class_accuracy": row[f"{name}_accuracy"],
                "class_precision": row[f"{name}_precision"],
                "class_recall": row[f"{name}_recall"],
                "class_f1": row[f"{name}_f1"],
                "decision_threshold": row[f"{name}_threshold"],
                "auc_change_vs_M_0": (
                    row[f"{name}_auc"] - row["M_0_auc"]
                ),
                "accuracy_change_vs_M_0": (
                    row[f"{name}_accuracy"] - row["M_0_accuracy"]
                ),
            })
    combined = pd.DataFrame(combined_rows)
    combined["probability_lambda"] = best_hyperparameters["lambda"]
    combined["selected_data_weight"] = best_hyperparameters["selected_data_weight"]
    combined["logistic_regression_C"] = best_hyperparameters["C"]
    combined["hyperparameter_selection_rule"] = best_hyperparameters[
        "selection_rule"
    ]
    combined_path = output_dir / "_model_comparison.csv"
    combined.to_csv(combined_path, index=False)
    for legacy_name in ("auc_summary.csv", "auc_per_class.csv"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    macro_axis, micro_axis, accuracy_axis, class_axis = axes.ravel()
    colors = {"M_0": "#1769aa", "M_1": "#d1495b", "M_rand": "#2e8b57"}
    for name, macro_roc in macro_rocs.items():
        model_auc = summary.loc[summary["model"] == name, "macro_auc"].iloc[0]
        macro_axis.plot(
            macro_roc[0], macro_roc[1], color=colors[name], linewidth=1.5,
            label=f"{name} macro AUC = {model_auc:.4f}",
        )
    macro_axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1.2)
    macro_axis.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Macro ROC",
    )
    macro_axis.grid(alpha=0.2)
    macro_axis.legend(loc="lower right")

    for name, micro_roc in micro_rocs.items():
        model_auc = summary.loc[summary["model"] == name, "micro_auc"].iloc[0]
        micro_axis.plot(
            micro_roc[0], micro_roc[1], color=colors[name], linewidth=1.5,
            label=f"{name} micro AUC = {model_auc:.4f}",
        )
    micro_axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1.2)
    micro_axis.set(
        xlim=(0, 1), ylim=(0, 1.01),
        xlabel="False positive rate", ylabel="True positive rate",
        title="Micro ROC",
    )
    micro_axis.grid(alpha=0.2)
    micro_axis.legend(loc="lower right")

    metric_columns = ["macro_accuracy", "micro_accuracy", "exact_match_accuracy"]
    metric_labels = ["Macro", "Micro", "Exact match"]
    x = np.arange(len(metric_columns))
    width = 0.24
    for offset, name in enumerate(models):
        values = summary.loc[summary["model"] == name, metric_columns].iloc[0]
        accuracy_axis.bar(
            x + (offset - 1) * width, values, width,
            color=colors[name], label=name,
        )
    accuracy_axis.set(
        xticks=x, xticklabels=metric_labels, ylim=(0, 1.05),
        ylabel="Accuracy", title="Overall accuracy (optimized thresholds)",
    )
    accuracy_axis.grid(axis="y", alpha=0.2)
    accuracy_axis.legend()

    class_names = [class_name_by_id[label_id] for label_id in valid_class_ids]
    class_x = np.arange(len(class_names))
    for offset, name in enumerate(models):
        values = [row[f"{name}_accuracy"] for row in per_class_rows]
        class_axis.bar(
            class_x + (offset - 1) * width, values, width,
            color=colors[name], label=name,
        )
    class_axis.set(
        xticks=class_x, xticklabels=class_names, ylim=(0, 1.05),
        ylabel="Accuracy", title="Accuracy by class (optimized thresholds)",
    )
    class_axis.tick_params(axis="x", rotation=30)
    class_axis.grid(axis="y", alpha=0.2)
    class_axis.legend()

    figure.suptitle(
        f"Model comparison on untouched test data | K={selection_k:g} per class",
        fontsize=16,
    )
    figure.tight_layout()
    dashboard_path = output_dir / "_model_comparison.png"
    figure.savefig(dashboard_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    for legacy_name in ("auc_macro.png", "auc_micro.png"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    print(summary.to_string(index=False))
    print(f"Saved all numerical results -> {combined_path}")
    print(f"Saved combined graph dashboard -> {dashboard_path}")
    return summary


if __name__ == "__main__":
    evaluate_dual_model_auc()
