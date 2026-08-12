import argparse
import random
import secrets
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import __utils__ as ut
import dual_model_calibration as dmc
import evaluate_dual_model_auc as dual_auc
import export_calibrated_images as calibrated_exporter
import export_selected_images as exporter
import graph_selection_entropy_probability as entropy_probability_graph
import graph_selection_quality_calibration as selection_quality_graph
import hyperparameter_sweep as hp_sweep
import optimization as opt
import subset_probability as sp


# Retained experiment switches
CREATE_DINO_DATASET = False
REUSE_EXISTING_SPLIT = False

# COCO input. Prefer a dataset inside this project, then the shared Downloads copy.
_PROJECT_DIR = Path(__file__).resolve().parent
_COCO_ROOT_CANDIDATES = [
    _PROJECT_DIR / "coco",
    _PROJECT_DIR.parent.parent / "coco",
    _PROJECT_DIR.parent.parent / "DINO-Kernel-Herding" / "coco",
]


def _find_coco_root():
    for root in _COCO_ROOT_CANDIDATES:
        if (
            (root / "train2017").is_dir()
            and (root / "annotations" / "instances_train2017.json").is_file()
        ):
            return root
    searched = "\n".join(f"  - {root}" for root in _COCO_ROOT_CANDIDATES)
    raise FileNotFoundError(
        "COCO train2017 images and instances_train2017.json were not found. "
        f"Searched:\n{searched}"
    )

# Proof-of-concept class subset. Each run records its generated seed.
CLASS_COUNT = 2

# Fixed proof-of-concept sampling cap (classes with fewer images use all of them).
_MIN_TRAIN_IMAGES_PER_CLASS = 550
_MAX_TRAIN_IMAGES_PER_CLASS = 550
_MAX_CLASS_SELECTION_ATTEMPTS = 1000
MIN_KNOWN_POSITIVE_FRACTION = 0.05
TRAINING_NEGATIVE_FRACTION = 0.10
TEST_IMAGE_COUNT = 5000

# Labeled reference representation
USE_KERNEL_HERDED_REFERENCES = True
KERNEL_HERDED_REFERENCE_COUNT = 100

# Candidate K values. M_1 uses the globally optimized nonzero count per class.
K_VALUES = tuple(range(20, 201, 20))

# Eigenvalue calculation
# Direct LAPACK is faster at the current reference size. Secular mode is
# experimental and automatically falls back to direct decomposition on failure.
USE_SECULAR_EIGENVALUE_UPDATES = False


def _reference_method():
    return "kernel_herding" if USE_KERNEL_HERDED_REFERENCES else "all"


def _eigenvalue_method():
    return "secular" if USE_SECULAR_EIGENVALUE_UPDATES else "direct"


def _random_selected_classes():
    metadata = pd.read_csv(ut.PROJECT_ROOT / "saved_vectors" / "metadata.csv")
    label_lists = metadata["all_label_names"].fillna("").astype(str).str.split("|")
    counts = {}
    for labels in label_lists:
        for class_name in set(labels):
            if class_name:
                counts[class_name] = counts.get(class_name, 0) + 1
    available_classes = sorted(
        name
        for name, count in counts.items()
        if count >= _MIN_TRAIN_IMAGES_PER_CLASS
    )
    if CLASS_COUNT > len(available_classes):
        raise ValueError(
            f"CLASS_COUNT={CLASS_COUNT} exceeds the "
            f"{len(available_classes)} available classes."
        )
    selection_seed = secrets.randbits(32)
    rng = random.Random(selection_seed)
    for attempt in range(1, _MAX_CLASS_SELECTION_ATTEMPTS + 1):
        selected = rng.sample(available_classes, CLASS_COUNT)
        selected_set = set(selected)
        usable_counts = {
            class_name: sum(
                class_name in labels and len(set(labels) & selected_set) == 1
                for labels in label_lists
            )
            for class_name in selected
        }
        feasible_count = min(
            ut.max_feasible_positive_samples_per_class(
                count,
                CLASS_COUNT,
                _MAX_TRAIN_IMAGES_PER_CLASS,
                MIN_KNOWN_POSITIVE_FRACTION,
                TRAINING_NEGATIVE_FRACTION,
                TEST_IMAGE_COUNT,
            )
            for count in usable_counts.values()
        )
        if feasible_count >= _MIN_TRAIN_IMAGES_PER_CLASS:
            break
    else:
        raise RuntimeError(
            "Could not find a random class set with at least "
            f"{_MIN_TRAIN_IMAGES_PER_CLASS} usable images per class after "
            f"{_MAX_CLASS_SELECTION_ATTEMPTS} attempts."
        )
    print(
        f"Selected proof-of-concept classes (seed={selection_seed}, "
        f"attempt={attempt}):"
    )
    print(", ".join(selected))
    print(
        "Usable images before balancing: "
        + ", ".join(
            f"{class_name}={usable_counts[class_name]}"
            for class_name in selected
        )
    )
    return selected, selection_seed


def run_pipeline():
    hp_sweep.reset_search_grids()
    if CREATE_DINO_DATASET:
        coco_root = _find_coco_root()
        vectors, labels, paths, classes = ut.create_dataset(
            data_location=coco_root / "train2017",
            annotation_path=coco_root / "annotations" / "instances_train2017.json",
            bckgnd_rmv=False,
        )
        print("Vector shape:", vectors.shape)
        print("Label shape:", labels.shape)
        print("Class count:", len(classes))
        print("Image count:", len(paths))

    selection_manifest_path = (
        ut.PROJECT_ROOT / "saved_vectors" / "selected_classes.csv"
    )
    split_files = [
        ut.PROJECT_ROOT / "saved_vectors" / split / "metadata.csv"
        for split in ("train", "retrain", "test")
    ]
    if (
        REUSE_EXISTING_SPLIT
        and not CREATE_DINO_DATASET
        and selection_manifest_path.exists()
        and all(path.exists() for path in split_files)
    ):
        selection_manifest = pd.read_csv(selection_manifest_path)
        selected_classes = selection_manifest["label_name"].tolist()
        selection_seed = int(selection_manifest["random_seed"].iloc[0])
        print(
            f"Resuming existing split (seed={selection_seed}): "
            + ", ".join(selected_classes)
        )
    else:
        selected_classes, selection_seed = _random_selected_classes()
        ut.split_DINO_vectors_with_multilabel_training(
            "saved_vectors",
            selected_class_names=selected_classes,
            min_samples_per_class=_MIN_TRAIN_IMAGES_PER_CLASS,
            samples_per_class=_MAX_TRAIN_IMAGES_PER_CLASS,
            random_state=selection_seed,
            min_known_positive_fraction=MIN_KNOWN_POSITIVE_FRACTION,
            training_negative_fraction=TRAINING_NEGATIVE_FRACTION,
            test_size=TEST_IMAGE_COUNT,
        )
        pd.DataFrame({
            "random_seed": [selection_seed] * len(selected_classes),
            "label_name": selected_classes,
        }).to_csv(selection_manifest_path, index=False)

    best, _, _ = hp_sweep.run_hyperparameter_sweep(
        "saved_vectors/train", selection_counts_per_class=K_VALUES
    )
    fixed_c = float(best["C"])
    probability_lambda = float(best["lambda"])
    selected_data_weight = float(best["selected_data_weight"])
    selection_k = int(best["selection_k_per_class"])
    print(
        "Applying selected hyperparameters: "
        f"lambda={probability_lambda:g}, "
        f"trust={selected_data_weight:g}, C={fixed_c:g}, K={selection_k}"
    )

    baseline_model, _, _ = sp.score_directory(
        learning_dir="saved_vectors/train",
        target_dir="saved_vectors/retrain",
        tune_regularization=False,
        fixed_c=fixed_c,
        use_hard_negative_mining=False,
        selected_class_names=selected_classes,
    )

    opt.optimize_subset_selection(
        learning_dir="saved_vectors/train",
        target_dir="saved_vectors/retrain",
        selection_count=selection_k,
        probability_lambda=probability_lambda,
        kernel="rbf",
        # Select exactly K per class (unless a class has fewer than K candidates).
        stop_when_objective_decreases=False,
        reference_method=_reference_method(),
        max_labeled_reference_per_subset=KERNEL_HERDED_REFERENCE_COUNT,
        eigenvalue_method=_eigenvalue_method(),
        selected_class_names=selected_classes,
    )

    model_0, model_1, model_rand = dmc.calibrate_baseline_and_augmented_models(
        training_dir="saved_vectors/train",
        retraining_dir="saved_vectors/retrain",
        tune_regularization=False,
        fixed_c=fixed_c,
        use_hard_negative_mining=False,
        baseline_model=baseline_model,
        selected_class_names=selected_classes,
        selected_data_weight=selected_data_weight,
        random_state=selection_seed,
    )
    calibrated_exporter.export_calibrated_test_images(
        model_0,
        model_1,
        model_rand,
        test_dir="saved_vectors/test",
    )
    evaluation_summary = dual_auc.evaluate_dual_model_auc(
        test_dir="saved_vectors/test",
    )

    exporter.export_selected_images()
    evaluation_summary = evaluation_summary.copy()
    evaluation_summary["random_seed"] = selection_seed
    evaluation_summary["classes"] = "|".join(selected_classes)
    evaluation_summary["lambda"] = probability_lambda
    evaluation_summary["selected_data_weight"] = selected_data_weight
    evaluation_summary["C"] = fixed_c
    return evaluation_summary


def run_ten_experiments():
    output_dir = ut.PROJECT_ROOT / "_ten_run_average"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_roc_curves = []
    comparison_dir = (
        ut.PROJECT_ROOT / "_organized_calibrated_images" / "AUC_evaluation"
    )

    for run_number in range(1, 11):
        print(f"\n===== Independent run {run_number}/10 =====")
        summary = run_pipeline()
        selection_quality_path = (
            selection_quality_graph.create_selection_quality_calibration()
        )
        shutil.copy2(
            selection_quality_path,
            output_dir / f"run_{run_number:02d}_selection_quality_calibration.png",
        )
        shutil.copy2(
            selection_quality_graph.TABLE_PATH,
            output_dir / f"run_{run_number:02d}_selection_quality_calibration.csv",
        )
        entropy_probability_path = entropy_probability_graph.create_tradeoff_plot()
        shutil.copy2(
            entropy_probability_path,
            output_dir / f"run_{run_number:02d}_entropy_probability_tradeoff.png",
        )
        shutil.copy2(
            entropy_probability_graph.TABLE_PATH,
            output_dir / f"run_{run_number:02d}_entropy_probability_tradeoff.csv",
        )
        summary.insert(0, "run", run_number)
        all_results.append(summary)
        accumulated = pd.concat(all_results, ignore_index=True)
        accumulated.to_csv(output_dir / "_ten_run_results.csv", index=False)
        roc_curves = pd.read_csv(comparison_dir / "_roc_curves.csv")
        roc_curves.insert(0, "run", run_number)
        all_roc_curves.append(roc_curves)
        pd.concat(all_roc_curves, ignore_index=True).to_csv(
            output_dir / "_ten_run_roc_curves.csv", index=False
        )
        shutil.copy2(
            comparison_dir / "_model_comparison.csv",
            output_dir / f"run_{run_number:02d}_model_comparison.csv",
        )
        shutil.copy2(
            comparison_dir / "_model_comparison.png",
            output_dir / f"run_{run_number:02d}_model_comparison.png",
        )
        shutil.copy2(
            comparison_dir / "_roc_curves.csv",
            output_dir / f"run_{run_number:02d}_roc_curves.csv",
        )

    results = pd.concat(all_results, ignore_index=True)
    metric_columns = [
        "macro_auc", "micro_auc", "macro_accuracy", "micro_accuracy",
        "exact_match_accuracy", "macro_precision", "micro_precision",
        "macro_recall", "micro_recall", "macro_f1", "micro_f1",
    ]
    average = results.groupby("model")[metric_columns].agg(["mean", "std"])
    average.columns = [f"{metric}_{stat}" for metric, stat in average.columns]
    average = average.reset_index()
    average["run_count"] = 10
    critical_value = float(student_t.ppf(0.975, df=9))
    for metric in metric_columns:
        margin = critical_value * average[f"{metric}_std"] / np.sqrt(10)
        average[f"{metric}_ci95_low"] = average[f"{metric}_mean"] - margin
        average[f"{metric}_ci95_high"] = average[f"{metric}_mean"] + margin
    average.to_csv(output_dir / "_ten_run_average.csv", index=False)

    roc_runs = pd.concat(all_roc_curves, ignore_index=True)
    colors = {"M_0": "#1769aa", "M_1": "#d1495b", "M_rand": "#2e8b57"}
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    roc_average_rows = []
    for axis, averaging in zip(axes, ("macro", "micro")):
        for model_name in ("M_0", "M_1", "M_rand"):
            column = f"{model_name}_{averaging}_true_positive_rate"
            grouped = roc_runs.groupby("false_positive_rate")[column]
            mean_tpr = grouped.mean()
            std_tpr = grouped.std()
            count = grouped.count()
            margin = student_t.ppf(0.975, count - 1) * std_tpr / np.sqrt(count)
            fpr = mean_tpr.index.to_numpy(dtype=float)
            lower = np.clip((mean_tpr - margin).to_numpy(), 0.0, 1.0)
            upper = np.clip((mean_tpr + margin).to_numpy(), 0.0, 1.0)
            axis.plot(
                fpr, mean_tpr.to_numpy(), linewidth=2,
                color=colors[model_name], label=model_name,
            )
            axis.fill_between(
                fpr, lower, upper, color=colors[model_name], alpha=0.18,
            )
            roc_average_rows.extend({
                "averaging": averaging,
                "model": model_name,
                "false_positive_rate": x,
                "mean_true_positive_rate": mean,
                "ci95_low": low,
                "ci95_high": high,
                "run_count": int(n),
            } for x, mean, low, high, n in zip(
                fpr, mean_tpr.to_numpy(), lower, upper, count.to_numpy()
            ))
        axis.plot([0, 1], [0, 1], color="#555555", linestyle=":")
        axis.set(
            xlim=(0, 1), ylim=(0, 1.01),
            xlabel="False positive rate", ylabel="True positive rate",
            title=f"Mean {averaging.capitalize()} ROC with 95% CI",
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right")
    pd.DataFrame(roc_average_rows).to_csv(
        output_dir / "_ten_run_roc_average.csv", index=False
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "_ten_run_average.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)
    print(f"Saved ten-run averages -> {output_dir}")
    return average


def _parse_arguments():
    parser = argparse.ArgumentParser(description="DINO kernel-herding experiment")
    parser.add_argument(
        "--10run",
        dest="ten_run",
        action="store_true",
        help="run ten independent experiments and average their metrics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    if arguments.ten_run:
        run_ten_experiments()
    else:
        run_pipeline()
