from pathlib import Path
from dataclasses import dataclass
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import __utils__ as ut
import optimization as opt
import subset_probability as sp


OUTPUT_DIR = ut.PROJECT_ROOT / "_hyperparameter_sensitivity"
LAMBDA_VALUES = [0.001, 0.005, 0.01, 0.025]
SELECTED_DATA_WEIGHTS = [0.01, 0.1, 1.0]
C_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
MAX_C_BOUNDARY_EXPANSIONS = 8
CV_FOLDS = 3
CV_RANDOM_STATE = 42
CANDIDATE_FRACTION = 0.25
SELECTION_COUNT_PER_CLASS = None
REFERENCE_COUNT = 30


@dataclass
class FastSweepModel:
    scaler: StandardScaler
    class_ids: np.ndarray
    classifiers_by_class: dict


def _selected_mapping(class_mapping):
    manifest = pd.read_csv(ut.PROJECT_ROOT / "saved_vectors" / "selected_classes.csv")
    names = set(manifest["label_name"])
    return class_mapping[class_mapping["label_name"].isin(names)].reset_index(drop=True)


def _probabilities(model, X):
    scaled = model.scaler.transform(X)
    columns = []
    for label_id in model.class_ids:
        classifier = model.classifiers_by_class[int(label_id)]
        if hasattr(classifier, "calibrated_probability"):
            columns.append(classifier.calibrated_probability(scaled))
        else:
            columns.append(classifier.predict_proba(scaled)[:, 1])
    return np.column_stack(columns)


def _auc_metrics(model, X, metadata):
    probabilities = _probabilities(model, X)
    targets = np.column_stack([
        sp.multilabel_targets(metadata, int(label_id)) for label_id in model.class_ids
    ])
    valid = [i for i in range(targets.shape[1]) if np.unique(targets[:, i]).size == 2]
    if not valid:
        raise ValueError("A CV fold has no classes with both outcomes.")
    targets = targets[:, valid]
    probabilities = probabilities[:, valid]
    macro = float(np.mean([
        roc_auc_score(targets[:, i], probabilities[:, i])
        for i in range(targets.shape[1])
    ]))
    micro = float(roc_auc_score(targets.ravel(), probabilities.ravel()))
    return macro, micro, len(valid)


def _selection_contexts(X_base, metadata_base, X_candidates, class_ids):
    base_scaler = StandardScaler()
    X_base_scaled = base_scaler.fit_transform(X_base)
    X_candidate_scaled = base_scaler.transform(X_candidates)
    gamma = 1.0 / X_base_scaled.shape[1]
    contexts = []
    for position, label_id in enumerate(class_ids):
        reference_mask = sp.multilabel_targets(metadata_base, int(label_id)).astype(bool)
        if not reference_mask.any():
            continue
        reference = opt._reference_subset(
            X_base_scaled[reference_mask], REFERENCE_COUNT,
            method="kernel_herding", kernel="rbf", gamma=gamma,
        )
        contexts.append((position, reference, X_candidate_scaled, gamma))
    return contexts


def _select_candidates(contexts, probabilities, lambda_value):
    selected = set()
    for position, reference, X_candidate_scaled, gamma in contexts:
        choices = opt.greedy_maximize_von_neumann_entropy(
            reference,
            X_candidate_scaled,
            np.arange(len(X_candidate_scaled)),
            probabilities[:, position],
            SELECTION_COUNT_PER_CLASS,
            "rbf", gamma,
            probability_lambda=lambda_value,
            stop_when_objective_decreases=True,
            eigenvalue_method="direct",
        )
        selected.update(item["target_index"] for item in choices)
    return np.array(sorted(selected), dtype=int)


def _fit_fast(X, metadata, class_mapping, c_value, sample_weight=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if sample_weight is None:
        sample_weight = np.ones(len(X), dtype=float)
    classifiers = {}
    class_ids = []
    for _, row in class_mapping.iterrows():
        label_id = int(row["label_id"])
        y = sp.multilabel_targets(metadata, label_id)
        if np.unique(y).size < 2:
            continue
        classifier = LogisticRegression(
            C=c_value, max_iter=2000, class_weight="balanced"
        )
        classifier.fit(X_scaled, y, sample_weight=sample_weight)
        classifiers[label_id] = classifier
        class_ids.append(label_id)
    if not class_ids:
        raise ValueError("No sweep classes contain both outcomes.")
    return FastSweepModel(scaler, np.asarray(class_ids), classifiers)


def _fit_calibrated_baseline(X, metadata, class_mapping, c_value):
    return sp.fit_subset_probability_model_from_data(
        X, metadata, class_mapping,
        tune_regularization=False,
        fixed_c=c_value,
        use_hard_negative_mining=False,
    )


def _plot_sensitivity(aggregate, best):
    settings = [
        ("lambda", LAMBDA_VALUES, "Lambda"),
        ("selected_data_weight", SELECTED_DATA_WEIGHTS, "Selected-data trust weight"),
        ("C", C_VALUES, "L2 inverse regularization (C)"),
    ]
    overview, overview_axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for overview_axis, (parameter, values, label) in zip(overview_axes, settings):
        controlled = aggregate.copy()
        for other in ("lambda", "selected_data_weight", "C"):
            if other != parameter:
                controlled = controlled[np.isclose(controlled[other], best[other])]
        controlled = controlled.sort_values(parameter)
        figure, axis = plt.subplots(figsize=(8, 5.5))
        axis.errorbar(
            controlled[parameter], controlled["macro_auc_mean"],
            yerr=controlled["macro_auc_se"], marker="o", capsize=3,
            label="Macro AUC ± SE",
        )
        axis.errorbar(
            controlled[parameter], controlled["micro_auc_mean"],
            yerr=controlled["micro_auc_se"], marker="s", capsize=3,
            label="Micro AUC ± SE",
        )
        if parameter in {"C", "lambda", "selected_data_weight"}:
            axis.set_xscale("log")
        axis.axvline(best[parameter], color="#555555", linestyle=":", label="Selected value")
        axis.set(xlabel=label, ylabel="Cross-validated AUC", title=f"Sensitivity to {label}")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"sensitivity_{parameter}.png", dpi=180)
        plt.close(figure)

        overview_axis.errorbar(
            controlled[parameter], controlled["macro_auc_mean"],
            yerr=controlled["macro_auc_se"], marker="o", capsize=3,
            label="Macro AUC ± SE",
        )
        overview_axis.errorbar(
            controlled[parameter], controlled["micro_auc_mean"],
            yerr=controlled["micro_auc_se"], marker="s", capsize=3,
            label="Micro AUC ± SE",
        )
        if parameter in {"C", "lambda", "selected_data_weight"}:
            overview_axis.set_xscale("log")
        overview_axis.axvline(best[parameter], color="#555555", linestyle=":")
        overview_axis.set(xlabel=label, ylabel="CV AUC", title=label)
        overview_axis.grid(alpha=0.2)
    overview_axes[0].legend()
    overview.suptitle("Hyperparameter sensitivity around the selected configuration")
    overview.tight_layout()
    overview.savefig(OUTPUT_DIR / "hyperparameter_sensitivity_overview.png", dpi=180)
    plt.close(overview)


def run_hyperparameter_sweep(
    training_dir="saved_vectors/train",
    _c_boundary_expansion_count=0,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, _, _, _, metadata, class_mapping = ut.load_DINO_vectors(training_dir)
    class_mapping = _selected_mapping(class_mapping)
    selection_manifest = pd.read_csv(
        ut.PROJECT_ROOT / "saved_vectors" / "selected_classes.csv"
    )
    data_seed = int(selection_manifest["random_seed"].iloc[0])
    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    checkpoint_path = OUTPUT_DIR / "cv_sweep_results.csv"
    key_columns = ["fold", "lambda", "selected_data_weight", "C"]
    if checkpoint_path.exists():
        results = pd.read_csv(checkpoint_path)
        if "data_seed" in results and set(results["data_seed"]) == {data_seed}:
            results = results[
                results["lambda"].isin(LAMBDA_VALUES)
                & results["selected_data_weight"].isin(SELECTED_DATA_WEIGHTS)
                & results["C"].isin(C_VALUES)
                & results["fold"].between(1, CV_FOLDS)
            ].copy()
            rows = results.to_dict("records")
            completed_keys = {
                tuple(float(row[column]) for column in key_columns)
                for _, row in results.iterrows()
            }
        else:
            print("Ignoring checkpoint from a different random data split.")
            rows = []
            completed_keys = set()
    else:
        rows = []
        completed_keys = set()
    total_combinations = CV_FOLDS * len(C_VALUES) * len(LAMBDA_VALUES) * len(SELECTED_DATA_WEIGHTS)
    completed = len(completed_keys)
    initial_completed = completed
    started = perf_counter()
    print(f"Sweep progress: {completed}/{total_combinations} ({100 * completed / total_combinations:.1f}%)")

    for fold, (development_indices, validation_indices) in enumerate(splitter.split(X), 1):
        rng = np.random.default_rng(CV_RANDOM_STATE + fold)
        shuffled = development_indices.copy()
        rng.shuffle(shuffled)
        candidate_count = max(1, int(round(len(shuffled) * CANDIDATE_FRACTION)))
        candidate_indices = shuffled[:candidate_count]
        base_indices = shuffled[candidate_count:]
        X_base, metadata_base = X[base_indices], metadata.iloc[base_indices].reset_index(drop=True)
        X_candidate = X[candidate_indices]
        metadata_candidate = metadata.iloc[candidate_indices].reset_index(drop=True)
        contexts = _selection_contexts(
            X_base, metadata_base, X_candidate, class_mapping["label_id"].to_numpy()
        )

        for c_value in C_VALUES:
            c_keys = {
                tuple(float(value) for value in (fold, lambda_value, trust_weight, c_value))
                for lambda_value in LAMBDA_VALUES
                for trust_weight in SELECTED_DATA_WEIGHTS
            }
            if c_keys.issubset(completed_keys):
                print(
                    f"[Fold {fold}/{CV_FOLDS}] C={c_value:g} already checkpointed",
                    flush=True,
                )
                continue
            print(
                f"[Fold {fold}/{CV_FOLDS}] fitting calibrated baseline with C={c_value:g}",
                flush=True,
            )
            # Lambda multiplies log calibrated probability, so cache one fully
            # calibrated baseline for each (fold, C). Trust-weight variants
            # still use fast ranking models during CV.
            baseline = _fit_calibrated_baseline(
                X_base, metadata_base, class_mapping, c_value
            )
            candidate_probabilities = _probabilities(baseline, X_candidate)
            for lambda_value in LAMBDA_VALUES:
                selected = _select_candidates(
                    contexts, candidate_probabilities, lambda_value
                )
                if len(selected):
                    X_augmented = np.concatenate([X_base, X_candidate[selected]])
                    metadata_augmented = pd.concat(
                        [metadata_base, metadata_candidate.iloc[selected]],
                        ignore_index=True,
                    )
                else:
                    # Selecting nothing is a valid acquisition outcome. Score
                    # the unchanged baseline-size training data in every fold
                    # instead of dropping the configuration from CV.
                    X_augmented = X_base
                    metadata_augmented = metadata_base
                for trust_weight in SELECTED_DATA_WEIGHTS:
                    key = tuple(float(value) for value in (
                        fold, lambda_value, trust_weight, c_value
                    ))
                    if key in completed_keys:
                        continue
                    weights = np.concatenate([
                        np.ones(len(X_base)),
                        np.full(len(selected), trust_weight),
                    ])
                    model = _fit_fast(
                        X_augmented, metadata_augmented, class_mapping,
                        c_value, sample_weight=weights,
                    )
                    macro, micro, class_count = _auc_metrics(
                        model, X[validation_indices],
                        metadata.iloc[validation_indices].reset_index(drop=True),
                    )
                    rows.append({
                        "data_seed": data_seed,
                        "fold": fold, "lambda": lambda_value,
                        "selected_data_weight": trust_weight, "C": c_value,
                        "selected_count": len(selected),
                        "evaluated_class_count": class_count,
                        "macro_auc": macro, "micro_auc": micro,
                    })
                    completed_keys.add(key)
                    completed += 1
                    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                    elapsed = perf_counter() - started
                    newly_completed = max(1, completed - initial_completed)
                    average = elapsed / newly_completed
                    remaining_minutes = average * (total_combinations - completed) / 60
                    print(
                        f"Sweep progress: {completed}/{total_combinations} "
                        f"({100 * completed / total_combinations:.1f}%) | "
                        f"C={c_value:g} | ETA ~{remaining_minutes:.1f} min",
                        flush=True,
                    )

    results = pd.DataFrame(rows)
    results.to_csv(checkpoint_path, index=False)
    aggregate = (
        results.groupby(["lambda", "selected_data_weight", "C"], as_index=False)
        .agg(
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_std=("macro_auc", "std"),
            fold_count=("macro_auc", "count"),
            micro_auc_mean=("micro_auc", "mean"),
            micro_auc_std=("micro_auc", "std"),
            mean_selected_count=("selected_count", "mean"),
        )
        .sort_values(["macro_auc_mean", "micro_auc_mean"], ascending=False)
    )
    aggregate["macro_auc_se"] = (
        aggregate["macro_auc_std"] / np.sqrt(aggregate["fold_count"])
    )
    aggregate["micro_auc_se"] = (
        aggregate["micro_auc_std"] / np.sqrt(aggregate["fold_count"])
    )
    aggregate["complete_cv"] = aggregate["fold_count"] == CV_FOLDS
    aggregate.to_csv(OUTPUT_DIR / "cv_sweep_summary.csv", index=False)
    eligible = aggregate[aggregate["complete_cv"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "No hyperparameter configuration has results from every CV fold. "
            "Resume the sweep to fill the checkpoint before selecting a winner."
        )
    raw_best = eligible.iloc[0].copy()
    c_at_lower_boundary = np.isclose(raw_best["C"], min(C_VALUES))
    c_at_upper_boundary = np.isclose(raw_best["C"], max(C_VALUES))
    if c_at_lower_boundary or c_at_upper_boundary:
        if _c_boundary_expansion_count >= MAX_C_BOUNDARY_EXPANSIONS:
            raise RuntimeError(
                "C optimum remained on a search boundary after "
                f"{MAX_C_BOUNDARY_EXPANSIONS} expansions. Current grid: {C_VALUES}"
            )
        new_c = (
            min(C_VALUES) / 10.0
            if c_at_lower_boundary
            else max(C_VALUES) * 10.0
        )
        C_VALUES.append(float(new_c))
        C_VALUES.sort()
        direction = "lower" if c_at_lower_boundary else "upper"
        print(
            f"Raw best C={raw_best['C']:g} is on the {direction} boundary. "
            f"Expanding C grid with {new_c:g} and continuing the sweep.",
            flush=True,
        )
        return run_hyperparameter_sweep(
            training_dir,
            _c_boundary_expansion_count=_c_boundary_expansion_count + 1,
        )
    boundary_parameters = []
    for parameter, grid in (
        ("lambda", LAMBDA_VALUES),
        ("selected_data_weight", SELECTED_DATA_WEIGHTS),
        ("C", C_VALUES),
    ):
        if np.isclose(raw_best[parameter], min(grid)) or np.isclose(
            raw_best[parameter], max(grid)
        ):
            boundary_parameters.append(parameter)
    one_se_threshold = raw_best["macro_auc_mean"] - raw_best["macro_auc_se"]
    competitive = eligible[
        eligible["macro_auc_mean"] >= one_se_threshold
    ].copy()
    moderate_lambda = float(np.exp(np.mean(np.log(LAMBDA_VALUES))))
    competitive["lambda_log_distance_from_center"] = np.abs(
        np.log(competitive["lambda"]) - np.log(moderate_lambda)
    )
    competitive = competitive.sort_values(
        [
            "selected_data_weight",
            "C",
            "lambda_log_distance_from_center",
            "mean_selected_count",
            "macro_auc_mean",
            "micro_auc_mean",
        ],
        # Within the one-standard-error competitive set, prefer the most
        # liberal trust in newly selected data.
        ascending=[False, True, True, True, False, False],
    )
    selected = competitive.iloc[0].copy()
    selected["selection_rule"] = "one_standard_error_liberal_trust"
    selected["raw_best_macro_auc_mean"] = raw_best["macro_auc_mean"]
    selected["raw_best_macro_auc_se"] = raw_best["macro_auc_se"]
    selected["one_se_macro_auc_threshold"] = one_se_threshold
    selected["competitive_configuration_count"] = len(competitive)
    selected["raw_optimum_boundary_parameters"] = (
        "|".join(boundary_parameters) if boundary_parameters else "none"
    )

    pd.DataFrame([raw_best]).to_csv(
        OUTPUT_DIR / "raw_highest_auc_hyperparameters.csv", index=False
    )
    competitive.to_csv(OUTPUT_DIR / "one_se_competitive_configurations.csv", index=False)
    pd.DataFrame([selected]).to_csv(
        OUTPUT_DIR / "best_hyperparameters.csv", index=False
    )
    best = selected.to_dict()
    _plot_sensitivity(eligible, best)
    print("Raw highest-mean-macro-AUC configuration:")
    print(pd.DataFrame([raw_best]).to_string(index=False))
    if boundary_parameters:
        print(
            "Warning: raw optimum is on the search boundary for: "
            + ", ".join(boundary_parameters)
            + ". Expand those grids before treating the optimum as settled."
        )
    print("Selected by the one-standard-error rule:")
    print(pd.DataFrame([selected]).to_string(index=False))
    print(f"Final selected C={selected['C']:g}")
    return best, results, eligible


if __name__ == "__main__":
    run_hyperparameter_sweep()
