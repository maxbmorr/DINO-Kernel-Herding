from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import __utils__ as ut
import optimization as opt
from progress import ProgressBar
import subset_probability as sp


OUTPUT_DIR = ut.PROJECT_ROOT / "_hyperparameter_sensitivity"
LAMBDA_MIN = 1e-4
INITIAL_LAMBDA_VALUES = (0.001, 0.005, 0.01, 0.025)
INITIAL_SELECTED_DATA_WEIGHTS = (0.01, 0.1, 1.0)
INITIAL_C_VALUES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
LAMBDA_VALUES = list(INITIAL_LAMBDA_VALUES)
SELECTED_DATA_WEIGHTS = list(INITIAL_SELECTED_DATA_WEIGHTS)
C_VALUES = list(INITIAL_C_VALUES)
MAX_BOUNDARY_EXPANSION_ROUNDS = 5
CV_FOLDS = 3
CV_RANDOM_STATE = 42
CANDIDATE_FRACTION = 0.25
SELECTION_COUNTS_PER_CLASS = tuple(range(20, 201, 20))
HYPERPARAMETER_TUNING_SELECTION_COUNT = 40
REFERENCE_COUNT = 30


def _overall_progress_label(completed, total, width=18):
    ratio = completed / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    return f"Overall [{bar}] {completed}/{total} ({ratio:.1%})"


def reset_search_grids():
    LAMBDA_VALUES[:] = INITIAL_LAMBDA_VALUES
    SELECTED_DATA_WEIGHTS[:] = INITIAL_SELECTED_DATA_WEIGHTS
    C_VALUES[:] = INITIAL_C_VALUES


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
    candidate_kernel = opt._kernel_matrix(
        X_candidate_scaled, X_candidate_scaled, "rbf", gamma
    )
    contexts = []
    for position, label_id in enumerate(class_ids):
        reference_mask = sp.multilabel_targets(metadata_base, int(label_id)).astype(bool)
        if not reference_mask.any():
            continue
        reference = opt._reference_subset(
            X_base_scaled[reference_mask], REFERENCE_COUNT,
            method="kernel_herding", kernel="rbf", gamma=gamma,
        )
        contexts.append((
            position, reference, X_candidate_scaled, gamma,
            opt._kernel_matrix(reference, reference, "rbf", gamma),
            opt._kernel_matrix(reference, X_candidate_scaled, "rbf", gamma),
            candidate_kernel,
        ))
    return contexts


def _select_candidate_prefixes(
    contexts, probabilities, lambda_value, selection_counts, progress_prefix="Selection"
):
    choices_by_class = []
    for (
        position, reference, X_candidate_scaled, gamma,
        reference_kernel, reference_candidate_kernel, candidate_kernel,
    ) in contexts:
        choices = opt.greedy_maximize_von_neumann_entropy(
            reference,
            X_candidate_scaled,
            np.arange(len(X_candidate_scaled)),
            probabilities[:, position],
            max(selection_counts),
            "rbf", gamma,
            probability_lambda=lambda_value,
            stop_when_objective_decreases=False,
            eigenvalue_method="direct",
            precomputed_reference_kernel=reference_kernel,
            precomputed_reference_candidate_kernel=reference_candidate_kernel,
            precomputed_candidate_kernel=candidate_kernel,
            progress_label=f"{progress_prefix}, class {position + 1}",
        )
        choices_by_class.append(choices)
    prefixes = {}
    for selection_count in selection_counts:
        selected = {
            item["target_index"]
            for choices in choices_by_class
            for item in choices[:selection_count]
        }
        prefixes[selection_count] = np.array(sorted(selected), dtype=int)
    return prefixes


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
        ("selection_k_per_class", SELECTION_COUNTS_PER_CLASS, "Selections per class (K)"),
    ]
    overview, overview_axes = plt.subplots(2, 2, figsize=(14, 10))
    overview_axes = overview_axes.ravel()
    for overview_axis, (parameter, values, label) in zip(overview_axes, settings):
        controlled = aggregate.copy()
        for other in ("lambda", "selected_data_weight", "C", "selection_k_per_class"):
            if other != parameter:
                target = (
                    HYPERPARAMETER_TUNING_SELECTION_COUNT
                    if other == "selection_k_per_class" and parameter != "selection_k_per_class"
                    else best[other]
                )
                controlled = controlled[np.isclose(controlled[other], target)]
        if parameter == "selection_k_per_class":
            controlled = controlled[
                controlled[parameter].isin(SELECTION_COUNTS_PER_CLASS)
            ]
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
    selection_counts_per_class=SELECTION_COUNTS_PER_CLASS,
    _boundary_expansion_count=0,
    _fixed_parameters=None,
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
    selection_counts_per_class = tuple(sorted(set(int(k) for k in selection_counts_per_class)))
    if not selection_counts_per_class or selection_counts_per_class[0] <= 0:
        raise ValueError("K values must be positive integers; K=0 is not allowed.")
    selection_grid_id = "|".join(map(str, selection_counts_per_class))
    if _fixed_parameters is None:
        active_lambdas = LAMBDA_VALUES
        active_weights = SELECTED_DATA_WEIGHTS
        active_cs = C_VALUES
        active_selection_counts = (HYPERPARAMETER_TUNING_SELECTION_COUNT,)
    else:
        active_lambdas = (float(_fixed_parameters["lambda"]),)
        active_weights = (float(_fixed_parameters["selected_data_weight"]),)
        active_cs = (float(_fixed_parameters["C"]),)
        active_selection_counts = selection_counts_per_class
    key_columns = ["fold", "lambda", "selected_data_weight", "C", "selection_k_per_class"]
    if checkpoint_path.exists():
        results = pd.read_csv(checkpoint_path)
        if (
            "data_seed" in results
            and set(results["data_seed"]) == {data_seed}
            and "selection_grid" in results
            and set(results["selection_grid"].astype(str)) == {selection_grid_id}
        ):
            for grid, column in (
                (LAMBDA_VALUES, "lambda"),
                (SELECTED_DATA_WEIGHTS, "selected_data_weight"),
                (C_VALUES, "C"),
            ):
                grid.extend(
                    float(value) for value in results[column].unique()
                    if float(value) not in grid
                    and (column != "lambda" or float(value) >= LAMBDA_MIN)
                )
                grid.sort()
            inferred_expansions = max(
                len(LAMBDA_VALUES) - len(INITIAL_LAMBDA_VALUES),
                len(SELECTED_DATA_WEIGHTS) - len(INITIAL_SELECTED_DATA_WEIGHTS),
                len(C_VALUES) - len(INITIAL_C_VALUES),
            )
            _boundary_expansion_count = max(
                _boundary_expansion_count, inferred_expansions
            )
            results = results[
                results["lambda"].isin(LAMBDA_VALUES)
                & results["selected_data_weight"].isin(SELECTED_DATA_WEIGHTS)
                & results["C"].isin(C_VALUES)
                & results["selection_k_per_class"].isin(
                    (*selection_counts_per_class, HYPERPARAMETER_TUNING_SELECTION_COUNT)
                )
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
    required_keys = {
        tuple(float(value) for value in (fold, lambda_value, trust_weight, c_value, selection_count))
        for fold in range(1, CV_FOLDS + 1)
        for lambda_value in active_lambdas
        for trust_weight in active_weights
        for c_value in active_cs
        for selection_count in active_selection_counts
    }
    total_combinations = len(required_keys)
    completed = len(completed_keys & required_keys)
    progress = ProgressBar(total_combinations, "Hyperparameter sweep", completed)

    for fold, (development_indices, validation_indices) in enumerate(splitter.split(X), 1):
        rng = np.random.default_rng(CV_RANDOM_STATE + fold)
        shuffled = development_indices.copy()
        rng.shuffle(shuffled)
        candidate_count = max(
            max(selection_counts_per_class),
            int(round(len(shuffled) * CANDIDATE_FRACTION)),
        )
        candidate_count = min(candidate_count, len(shuffled) - 1)
        if candidate_count < max(selection_counts_per_class):
            raise ValueError(
                f"CV fold {fold} has only {candidate_count} candidate images, "
                f"but K={max(selection_counts_per_class)} was requested."
            )
        candidate_indices = shuffled[:candidate_count]
        base_indices = shuffled[candidate_count:]
        X_base, metadata_base = X[base_indices], metadata.iloc[base_indices].reset_index(drop=True)
        X_candidate = X[candidate_indices]
        metadata_candidate = metadata.iloc[candidate_indices].reset_index(drop=True)
        contexts = _selection_contexts(
            X_base, metadata_base, X_candidate, class_mapping["label_id"].to_numpy()
        )

        for c_value in active_cs:
            c_keys = {
                tuple(float(value) for value in (fold, lambda_value, trust_weight, c_value, selection_count))
                for lambda_value in active_lambdas
                for trust_weight in active_weights
                for selection_count in active_selection_counts
            }
            if c_keys.issubset(completed_keys):
                continue
            # Lambda multiplies log calibrated probability, so cache one fully
            # calibrated baseline for each (fold, C). Trust-weight variants
            # still use fast ranking models during CV.
            baseline = _fit_calibrated_baseline(
                X_base, metadata_base, class_mapping, c_value
            )
            candidate_probabilities = _probabilities(baseline, X_candidate)
            for lambda_value in active_lambdas:
                selected_by_k = _select_candidate_prefixes(
                    contexts, candidate_probabilities, lambda_value,
                    active_selection_counts,
                    progress_prefix=(
                        f"{_overall_progress_label(completed, total_combinations)} | "
                        f"Fold {fold}, C={c_value:g}, lambda={lambda_value:g}, "
                        f"ranking to K={max(active_selection_counts)}"
                    ),
                )
                for selection_count, selected in selected_by_k.items():
                    X_augmented = np.concatenate([X_base, X_candidate[selected]])
                    metadata_augmented = pd.concat(
                        [metadata_base, metadata_candidate.iloc[selected]],
                        ignore_index=True,
                    )
                    for trust_weight in active_weights:
                        key = tuple(float(value) for value in (
                            fold, lambda_value, trust_weight, c_value, selection_count
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
                            "data_seed": data_seed, "selection_grid": selection_grid_id,
                            "fold": fold, "lambda": lambda_value,
                            "selected_data_weight": trust_weight, "C": c_value,
                            "selected_count": len(selected),
                            "selection_k_per_class": selection_count,
                            "evaluated_class_count": class_count,
                            "macro_auc": macro, "micro_auc": micro,
                        })
                        completed_keys.add(key)
                        completed += 1
                        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                        progress.update(detail=(
                            f"fold={fold}, C={c_value:g}, lambda={lambda_value:g}, K={selection_count}"
                        ))

    progress.close()

    results = pd.DataFrame(rows)
    results.to_csv(checkpoint_path, index=False)
    aggregate = (
        results.groupby(["lambda", "selected_data_weight", "C", "selection_k_per_class"], as_index=False)
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
    if _fixed_parameters is None:
        selection_eligible = eligible[
            eligible["selection_k_per_class"] == HYPERPARAMETER_TUNING_SELECTION_COUNT
        ].copy()
    else:
        selection_eligible = eligible[
            np.isclose(eligible["lambda"], _fixed_parameters["lambda"])
            & np.isclose(eligible["selected_data_weight"], _fixed_parameters["selected_data_weight"])
            & np.isclose(eligible["C"], _fixed_parameters["C"])
            & eligible["selection_k_per_class"].isin(selection_counts_per_class)
        ].copy()
    best_macro = selection_eligible["macro_auc_mean"].max()
    tied = selection_eligible[np.isclose(
        selection_eligible["macro_auc_mean"], best_macro, rtol=0.0, atol=1e-12
    )].copy()
    best_micro = tied["micro_auc_mean"].max()
    tied = tied[np.isclose(
        tied["micro_auc_mean"], best_micro, rtol=0.0, atol=1e-12
    )].copy()
    if len(tied) > 1:
        def boundary_clearance(row):
            clearance = 0
            for parameter, grid in (
                ("lambda", LAMBDA_VALUES),
                ("selected_data_weight", SELECTED_DATA_WEIGHTS),
                ("C", C_VALUES),
            ):
                position = int(np.argmin(np.abs(np.asarray(grid) - row[parameter])))
                clearance += min(position, len(grid) - 1 - position)
            return clearance

        tied["_boundary_clearance"] = tied.apply(boundary_clearance, axis=1)
        tied = tied.sort_values(
            ["_boundary_clearance", "selection_k_per_class"],
            ascending=[False, True],
        )
    raw_best = tied.iloc[0].drop(labels=["_boundary_clearance"], errors="ignore").copy()
    boundary_parameters = []
    parameter_grids = (
        ("lambda", LAMBDA_VALUES),
        ("selected_data_weight", SELECTED_DATA_WEIGHTS),
        ("C", C_VALUES),
    )
    for parameter, grid in parameter_grids if _fixed_parameters is None else ():
        lower = np.isclose(raw_best[parameter], min(grid)) and not (
            parameter == "lambda" and min(grid) <= LAMBDA_MIN
        )
        upper = np.isclose(raw_best[parameter], max(grid))
        if lower or upper:
            boundary_parameters.append((parameter, grid, lower, upper))

    if boundary_parameters:
        if _boundary_expansion_count >= MAX_BOUNDARY_EXPANSION_ROUNDS:
            grid_description = "; ".join(
                f"{name}={grid}" for name, grid in parameter_grids
            )
            raise RuntimeError(
                "ROC-AUC optimum remained on a search boundary after "
                f"{MAX_BOUNDARY_EXPANSION_ROUNDS} expansion rounds. "
                f"Current grids: {grid_description}"
            )
        for parameter, grid, lower, upper in boundary_parameters:
            additions = []
            if lower:
                new_lower = min(grid) / 10.0
                if parameter == "lambda":
                    new_lower = max(new_lower, LAMBDA_MIN)
                additions.append(float(new_lower))
            if upper:
                additions.append(float(max(grid) * 10.0))
            grid.extend(value for value in additions if value not in grid)
            grid.sort()
            directions = " and ".join(
                direction for direction, active in (("lower", lower), ("upper", upper))
                if active
            )
            print(
                f"Best {parameter}={raw_best[parameter]:g} is on the "
                f"{directions} boundary; added {additions}.",
                flush=True,
            )
        return run_hyperparameter_sweep(
            training_dir,
            selection_counts_per_class=selection_counts_per_class,
            _boundary_expansion_count=_boundary_expansion_count + 1,
        )
    if _fixed_parameters is None:
        fixed_parameters = {
            "lambda": float(raw_best["lambda"]),
            "selected_data_weight": float(raw_best["selected_data_weight"]),
            "C": float(raw_best["C"]),
        }
        print(
            f"Stage 1 selected at K={HYPERPARAMETER_TUNING_SELECTION_COUNT}: "
            f"lambda={fixed_parameters['lambda']:g}, "
            f"trust={fixed_parameters['selected_data_weight']:g}, "
            f"C={fixed_parameters['C']:g}. Now optimizing K.",
            flush=True,
        )
        return run_hyperparameter_sweep(
            training_dir,
            selection_counts_per_class=selection_counts_per_class,
            _boundary_expansion_count=_boundary_expansion_count,
            _fixed_parameters=fixed_parameters,
        )
    # Select the configuration that directly maximizes cross-validated ROC AUC.
    # The aggregate is already sorted by macro AUC and then micro AUC.
    selected = raw_best.copy()
    selected["selection_rule"] = "staged_maximum_cv_macro_roc_auc"
    selected["raw_optimum_boundary_parameters"] = (
        "none"
    )

    pd.DataFrame([raw_best]).to_csv(
        OUTPUT_DIR / "raw_highest_auc_hyperparameters.csv", index=False
    )
    pd.DataFrame([selected]).to_csv(
        OUTPUT_DIR / "best_hyperparameters.csv", index=False
    )
    best = selected.to_dict()
    _plot_sensitivity(eligible, best)
    print("Raw highest-mean-macro-AUC configuration:")
    print(pd.DataFrame([raw_best]).to_string(index=False))
    print("Selected by maximum cross-validated macro ROC AUC:")
    print(pd.DataFrame([selected]).to_string(index=False))
    print(f"Final selected C={selected['C']:g}")
    return best, results, eligible


if __name__ == "__main__":
    run_hyperparameter_sweep()
