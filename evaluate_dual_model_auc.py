from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

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
    test_dir="saved_vectors/test",
    output_dir=OUTPUT_DIR,
):
    model_0 = joblib.load(_resolve_path(model_0_path))
    model_1 = joblib.load(_resolve_path(model_1_path))
    test_dir = _resolve_path(test_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test, _, _, _, metadata, _ = ut.load_DINO_vectors(test_dir)
    shared_class_ids = sorted(
        set(int(value) for value in model_0.class_ids)
        & set(int(value) for value in model_1.class_ids)
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
    probabilities_0 = _probability_matrix(model_0, X_test, valid_class_ids)
    probabilities_1 = _probability_matrix(model_1, X_test, valid_class_ids)

    class_name_by_id = {
        int(label_id): name
        for label_id, name in zip(model_0.class_ids, model_0.class_names)
    }
    per_class_rows = []
    for position, label_id in enumerate(valid_class_ids):
        auc_0 = roc_auc_score(y_true[:, position], probabilities_0[:, position])
        auc_1 = roc_auc_score(y_true[:, position], probabilities_1[:, position])
        per_class_rows.append({
            "label_id": label_id,
            "label_name": class_name_by_id[label_id],
            "positive_test_count": int(y_true[:, position].sum()),
            "M_0_auc": auc_0,
            "M_1_auc": auc_1,
            "auc_change": auc_1 - auc_0,
        })
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "auc_per_class.csv",
        index=False,
    )

    macro_0 = _macro_roc(y_true, probabilities_0)
    macro_1 = _macro_roc(y_true, probabilities_1)
    micro_false_positive_0, micro_true_positive_0, _ = roc_curve(
        y_true.ravel(),
        probabilities_0.ravel(),
    )
    micro_false_positive_1, micro_true_positive_1, _ = roc_curve(
        y_true.ravel(),
        probabilities_1.ravel(),
    )
    micro_auc_0 = auc(micro_false_positive_0, micro_true_positive_0)
    micro_auc_1 = auc(micro_false_positive_1, micro_true_positive_1)
    exact_macro_auc_0 = _exact_macro_auc(y_true, probabilities_0)
    exact_macro_auc_1 = _exact_macro_auc(y_true, probabilities_1)

    summary = pd.DataFrame([
        {
            "model": "M_0",
            "test_image_count": len(X_test),
            "evaluated_class_count": len(valid_class_ids),
            "macro_auc": exact_macro_auc_0,
            "micro_auc": micro_auc_0,
        },
        {
            "model": "M_1",
            "test_image_count": len(X_test),
            "evaluated_class_count": len(valid_class_ids),
            "macro_auc": exact_macro_auc_1,
            "micro_auc": micro_auc_1,
        },
    ])
    summary.to_csv(output_dir / "auc_summary.csv", index=False)

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(
        macro_0[0], macro_0[1], color="#1769aa", linewidth=1.2,
        label=f"M_0 macro AUC = {exact_macro_auc_0:.4f}",
    )
    axis.plot(
        macro_1[0], macro_1[1], color="#d1495b", linewidth=1.2,
        label=f"M_1 macro AUC = {exact_macro_auc_1:.4f}",
    )
    axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1.2)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Macro ROC: M_0 vs M_1 on untouched test images",
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "auc_macro.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(
        micro_false_positive_0, micro_true_positive_0,
        color="#1769aa", linewidth=1.2,
        label=f"M_0 micro AUC = {micro_auc_0:.4f}",
    )
    axis.plot(
        micro_false_positive_1, micro_true_positive_1,
        color="#d1495b", linewidth=1.2,
        label=f"M_1 micro AUC = {micro_auc_1:.4f}",
    )
    axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1.2)
    axis.set(
        xlim=(0, 1), ylim=(0, 1.01),
        xlabel="False positive rate", ylabel="True positive rate",
        title="Micro ROC: M_0 vs M_1 on untouched test images",
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "auc_micro.png", dpi=180)
    plt.close(figure)

    print(summary.to_string(index=False))
    print(f"Saved macro AUC graph -> {output_dir / 'auc_macro.png'}")
    print(f"Saved micro AUC graph -> {output_dir / 'auc_micro.png'}")
    return summary


if __name__ == "__main__":
    evaluate_dual_model_auc()
