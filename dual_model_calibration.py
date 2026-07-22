from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import __utils__ as ut
import subset_probability as sp


def _resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ut.PROJECT_ROOT / path
    return path.resolve()


def _normalized_path(path):
    return str(_resolve_path(path)).casefold()


def _selected_retraining_indices(retraining_metadata, selection):
    if "path" not in selection.columns:
        raise ValueError("optimization_selection.csv must contain a path column.")

    retraining_paths = {
        _normalized_path(path): index
        for index, path in enumerate(retraining_metadata["path"])
    }
    selected_paths = selection["path"].dropna().drop_duplicates()
    missing_paths = [
        path
        for path in selected_paths
        if _normalized_path(path) not in retraining_paths
    ]
    if missing_paths:
        examples = ", ".join(str(path) for path in missing_paths[:3])
        raise ValueError(
            "Selected images do not match the current retraining split. "
            f"Examples: {examples}"
        )

    return np.array(
        [retraining_paths[_normalized_path(path)] for path in selected_paths],
        dtype=int,
    )


def _save_model(output_dir, name, model, training_manifest):
    model_path = output_dir / f"{name}.joblib"
    metrics_path = output_dir / f"{name}_calibration_metrics.csv"
    manifest_path = output_dir / f"{name}_training_manifest.csv"
    joblib.dump(model, model_path)
    model.calibration_metrics.to_csv(metrics_path, index=False)
    training_manifest.to_csv(manifest_path, index=False)
    print(f"Saved {name} calibrated model -> {model_path}")


def calibrate_baseline_and_augmented_models(
    training_dir="saved_vectors/train",
    retraining_dir="saved_vectors/retrain",
    selection_path="saved_vectors/retrain/optimization_selection.csv",
    output_dir="saved_models/calibrated",
    tune_regularization=True,
    fixed_c=1.0,
    use_hard_negative_mining=False,
    hard_negative_fraction=0.1,
    hard_negative_weight=3.0,
    baseline_model=None,
    selected_class_names=None,
    selected_data_weight=1.0,
):
    training_dir = _resolve_path(training_dir)
    retraining_dir = _resolve_path(retraining_dir)
    selection_path = _resolve_path(selection_path)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_learning, _, _, _, learning_metadata, class_mapping = ut.load_DINO_vectors(
        training_dir
    )
    X_retraining, _, _, _, retraining_metadata, _ = ut.load_DINO_vectors(
        retraining_dir
    )
    if selected_class_names is not None:
        class_mapping = class_mapping[
            class_mapping["label_name"].isin(selected_class_names)
        ].reset_index(drop=True)
    selection = pd.read_csv(selection_path)
    selected_indices = _selected_retraining_indices(retraining_metadata, selection)
    if len(selected_indices) == 0:
        raise ValueError("No newly labeled images were selected for M_1.")

    common_fit_arguments = {
        "tune_regularization": tune_regularization,
        "fixed_c": fixed_c,
        "use_hard_negative_mining": use_hard_negative_mining,
        "hard_negative_fraction": hard_negative_fraction,
        "hard_negative_weight": hard_negative_weight,
    }

    if baseline_model is None:
        print(f"Calibrating M_0 on {len(X_learning)} original labeled images")
        model_0 = sp.fit_subset_probability_model_from_data(
            X_learning,
            learning_metadata,
            class_mapping,
            **common_fit_arguments,
        )
    else:
        print("Using the M_0 model already fitted for subset probabilities")
        model_0 = baseline_model
    manifest_0 = learning_metadata[["path"]].copy()
    manifest_0["training_source"] = "original_labeled"
    _save_model(output_dir, "M_0", model_0, manifest_0)

    selected_metadata = retraining_metadata.iloc[selected_indices].reset_index(drop=True)
    X_augmented = np.concatenate(
        [X_learning, X_retraining[selected_indices]],
        axis=0,
    )
    augmented_metadata = pd.concat(
        [learning_metadata, selected_metadata],
        ignore_index=True,
    )
    print(
        f"Calibrating M_1 on {len(X_learning)} original plus "
        f"{len(selected_indices)} newly labeled images"
    )
    model_1 = sp.fit_subset_probability_model_from_data(
        X_augmented,
        augmented_metadata,
        class_mapping,
        sample_weight=np.concatenate([
            np.ones(len(X_learning), dtype=float),
            np.full(len(selected_indices), selected_data_weight, dtype=float),
        ]),
        **common_fit_arguments,
    )
    manifest_1 = pd.concat(
        [
            manifest_0,
            selected_metadata[["path"]].assign(training_source="newly_labeled"),
        ],
        ignore_index=True,
    )
    _save_model(output_dir, "M_1", model_1, manifest_1)

    return model_0, model_1
