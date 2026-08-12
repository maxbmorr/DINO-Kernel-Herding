from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

import __utils__ as ut


OUTPUT_DIR = ut.PROJECT_ROOT / "_organized_calibrated_images"


def _safe_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _model_predictions(model, X):
    X_scaled = model.scaler.transform(X)
    probabilities = np.column_stack([
        model.classifiers_by_class[int(label_id)].calibrated_probability(X_scaled)
        for label_id in model.class_ids
    ])
    thresholds = np.asarray([
        model.classifiers_by_class[int(label_id)].threshold
        for label_id in model.class_ids
    ])
    passes_threshold = probabilities >= thresholds[None, :]
    passes_any_threshold = passes_threshold.any(axis=1)
    # Choose only among classes that passed their own threshold. This matters
    # because independently optimized class thresholds can differ substantially.
    eligible_probabilities = np.where(passes_threshold, probabilities, -np.inf)
    best_positions = np.argmax(eligible_probabilities, axis=1)
    predicted_ids = model.class_ids[best_positions].astype(int)
    predicted_names = np.asarray([
        model.class_names[position] for position in best_positions
    ], dtype=object)
    predicted_ids[~passes_any_threshold] = -1
    predicted_names[~passes_any_threshold] = "negative"
    return predicted_ids, predicted_names.tolist()


def _export_model_predictions(model_name, model, X_test, paths, output_dir):
    predicted_ids, predicted_names = _model_predictions(
        model,
        X_test,
    )
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for path, label_id, label_name in zip(
        paths,
        predicted_ids,
        predicted_names,
    ):
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing test image: {source_path}")

        class_dir = model_dir / _safe_name(label_name)
        class_dir.mkdir(exist_ok=True)
        destination_path = class_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        manifest_rows.append({
            "path": str(source_path),
            "predicted_label_id": int(label_id),
            "predicted_label_name": label_name,
            "exported_path": str(destination_path),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(model_dir / "prediction_manifest.csv", index=False)
    print(
        f"Sorted {len(manifest)} test images by {model_name} prediction -> "
        f"{model_dir}"
    )


def export_calibrated_test_images(
    model_0,
    model_1,
    model_rand,
    test_dir="saved_vectors/test",
    output_dir=OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ut.PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in ("M_0", "M_1", "M_rand"):
        model_dir = output_dir / model_name
        if model_dir.exists():
            shutil.rmtree(model_dir)

    test_dir = Path(test_dir)
    if not test_dir.is_absolute():
        test_dir = ut.PROJECT_ROOT / test_dir
    X_test = np.load(test_dir / "vectors.npy")
    paths = pd.read_csv(test_dir / "metadata.csv", usecols=["path"])["path"]
    _export_model_predictions("M_0", model_0, X_test, paths, output_dir)
    _export_model_predictions("M_1", model_1, X_test, paths, output_dir)
    _export_model_predictions("M_rand", model_rand, X_test, paths, output_dir)
    print("No test labels or evaluation metrics were used during export.")
    return output_dir
