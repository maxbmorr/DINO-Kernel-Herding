from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import __utils__ as ut


REGULARIZATION_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]


@dataclass
class CalibratedSubsetClassifier:
    classifier: LogisticRegression
    calibrator: LogisticRegression
    threshold: float

    def raw_probability(self, X):
        return self.classifier.predict_proba(X)[:, 1]

    def calibrated_probability(self, X):
        scores = self.classifier.decision_function(X).reshape(-1, 1)
        return self.calibrator.predict_proba(scores)[:, 1]


@dataclass
class SubsetProbabilityModel:
    scaler: StandardScaler
    class_ids: np.ndarray
    class_names: list
    classifiers_by_class: dict
    calibration_metrics: pd.DataFrame


def _parse_label_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(label_id) for label_id in str(value).split("|") if label_id != ""}


def multilabel_targets(metadata, label_id):
    if "all_label_ids" in metadata.columns:
        return np.array(
            [int(label_id in _parse_label_ids(value)) for value in metadata["all_label_ids"]],
            dtype=int,
        )
    return (metadata["label_id"].to_numpy() == label_id).astype(int)


def _expected_calibration_error(y_true, probabilities, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def _precision_threshold(y_true, probabilities, target_precision):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= target_precision)
    if len(eligible):
        best = eligible[np.argmax(recall[eligible])]
        return float(thresholds[best])

    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[np.argmax(f1)])


def _fit_tuned_classifier(X, y, random_state):
    positive_count = int(y.sum())
    negative_count = int((y == 0).sum())
    inner_fold_count = min(3, positive_count, negative_count)
    if inner_fold_count < 2:
        raise ValueError(
            "At least two positive and two negative examples are required "
            "for regularization tuning."
        )

    inner_splitter = StratifiedKFold(
        n_splits=inner_fold_count,
        shuffle=True,
        random_state=random_state,
    )
    search = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        param_grid={"C": REGULARIZATION_C_VALUES},
        scoring="neg_log_loss",
        cv=inner_splitter,
        refit=True,
        n_jobs=1,
    )
    search.fit(X, y)
    return search.best_estimator_, float(search.best_params_["C"])


def _refit_with_hard_negatives(
    classifier,
    X,
    y,
    hard_negative_fraction,
    hard_negative_weight,
):
    negative_indices = np.flatnonzero(y == 0)
    negative_scores = classifier.decision_function(X[negative_indices])
    hard_negative_count = max(
        1,
        int(np.ceil(len(negative_indices) * hard_negative_fraction)),
    )
    hardest_positions = np.argpartition(
        negative_scores,
        -hard_negative_count,
    )[-hard_negative_count:]
    hard_negative_indices = negative_indices[hardest_positions]

    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[hard_negative_indices] = hard_negative_weight
    refit_classifier = LogisticRegression(
        C=float(classifier.C),
        max_iter=2000,
        class_weight="balanced",
    )
    refit_classifier.fit(X, y, sample_weight=sample_weight)
    return refit_classifier, hard_negative_count


def fit_calibrated_classifier(
    X,
    y,
    target_precision=0.8,
    random_state=42,
    tune_regularization=True,
    fixed_c=1.0,
    use_hard_negative_mining=True,
    hard_negative_fraction=0.1,
    hard_negative_weight=3.0,
):
    if fixed_c <= 0:
        raise ValueError("fixed_c must be positive.")
    if not 0 < hard_negative_fraction <= 1:
        raise ValueError("hard_negative_fraction must be in (0, 1].")
    if hard_negative_weight < 1:
        raise ValueError("hard_negative_weight must be at least 1.")

    positive_count = int(y.sum())
    negative_count = int((y == 0).sum())
    fold_count = min(5, positive_count, negative_count)
    if fold_count < 2:
        raise ValueError(
            "At least two positive and two negative examples are required "
            "for out-of-fold calibration."
        )

    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=random_state,
    )
    out_of_fold_scores = np.empty(len(y), dtype=float)
    outer_fold_c_values = []
    outer_hard_negative_counts = []
    for fold_number, (fit_indices, fold_indices) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        if tune_regularization:
            fold_classifier, fold_c = _fit_tuned_classifier(
                X[fit_indices],
                y[fit_indices],
                random_state + fold_number,
            )
        else:
            fold_c = float(fixed_c)
            fold_classifier = LogisticRegression(
                C=fold_c,
                max_iter=2000,
                class_weight="balanced",
            )
            fold_classifier.fit(X[fit_indices], y[fit_indices])
        if use_hard_negative_mining:
            fold_classifier, hard_negative_count = _refit_with_hard_negatives(
                fold_classifier,
                X[fit_indices],
                y[fit_indices],
                hard_negative_fraction,
                hard_negative_weight,
            )
        else:
            hard_negative_count = 0
        outer_fold_c_values.append(fold_c)
        outer_hard_negative_counts.append(hard_negative_count)
        out_of_fold_scores[fold_indices] = fold_classifier.decision_function(
            X[fold_indices]
        )

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(out_of_fold_scores.reshape(-1, 1), y)
    calibrated = calibrator.predict_proba(
        out_of_fold_scores.reshape(-1, 1)
    )[:, 1]
    threshold = _precision_threshold(y, calibrated, target_precision)

    if tune_regularization:
        classifier, selected_c = _fit_tuned_classifier(
            X,
            y,
            random_state + 1000,
        )
        regularization_selection = "nested_stratified_cv_log_loss"
        regularization_c_candidates = "|".join(
            f"{value:g}" for value in REGULARIZATION_C_VALUES
        )
    else:
        selected_c = float(fixed_c)
        classifier = LogisticRegression(
            C=selected_c,
            max_iter=2000,
            class_weight="balanced",
        )
        classifier.fit(X, y)
        regularization_selection = "fixed"
        regularization_c_candidates = f"{selected_c:g}"

    if use_hard_negative_mining:
        classifier, final_hard_negative_count = _refit_with_hard_negatives(
            classifier,
            X,
            y,
            hard_negative_fraction,
            hard_negative_weight,
        )
    else:
        final_hard_negative_count = 0

    model = CalibratedSubsetClassifier(classifier, calibrator, threshold)
    metrics = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "fit_count": int(len(y)),
        "calibration_method": "stratified_out_of_fold_sigmoid",
        "calibration_fold_count": int(fold_count),
        "calibration_count": int(len(y)),
        "calibration_positive_count": positive_count,
        "regularization_selection": regularization_selection,
        "regularization_c_candidates": regularization_c_candidates,
        "outer_fold_c_values": "|".join(
            f"{value:g}" for value in outer_fold_c_values
        ),
        "selected_c": selected_c,
        "hard_negative_mining": bool(use_hard_negative_mining),
        "hard_negative_fraction": hard_negative_fraction,
        "hard_negative_weight": hard_negative_weight,
        "outer_hard_negative_counts": "|".join(
            str(value) for value in outer_hard_negative_counts
        ),
        "final_hard_negative_count": final_hard_negative_count,
        "threshold": threshold,
        "brier_score": brier_score_loss(y, calibrated),
        "expected_calibration_error": _expected_calibration_error(
            y, calibrated
        ),
        "average_precision": average_precision_score(y, calibrated),
    }
    return model, metrics


def fit_subset_probability_model(
    learning_dir="saved_vectors/train",
    min_class_samples=6,
    target_precision=0.8,
    tune_regularization=True,
    fixed_c=1.0,
    use_hard_negative_mining=True,
    hard_negative_fraction=0.1,
    hard_negative_weight=3.0,
):
    X_train, _, _, _, metadata, class_mapping = ut.load_DINO_vectors(learning_dir)
    return fit_subset_probability_model_from_data(
        X_train,
        metadata,
        class_mapping,
        min_class_samples=min_class_samples,
        target_precision=target_precision,
        tune_regularization=tune_regularization,
        fixed_c=fixed_c,
        use_hard_negative_mining=use_hard_negative_mining,
        hard_negative_fraction=hard_negative_fraction,
        hard_negative_weight=hard_negative_weight,
    )


def fit_subset_probability_model_from_data(
    X_train,
    metadata,
    class_mapping,
    min_class_samples=6,
    target_precision=0.8,
    tune_regularization=True,
    fixed_c=1.0,
    use_hard_negative_mining=True,
    hard_negative_fraction=0.1,
    hard_negative_weight=3.0,
):
    X_train = np.asarray(X_train)
    metadata = metadata.reset_index(drop=True)
    class_mapping = class_mapping.reset_index(drop=True)
    if len(X_train) != len(metadata):
        raise ValueError("X_train and metadata must have the same row count.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    class_ids = []
    class_names = []
    classifiers_by_class = {}
    metric_rows = []
    print(f"Fitting calibrated multi-label models for {len(class_mapping)} classes")

    for class_number, class_row in class_mapping.iterrows():
        label_id = int(class_row["label_id"])
        class_name = class_row["label_name"]
        y = multilabel_targets(metadata, label_id)
        positive_count = int(y.sum())
        negative_count = int((y == 0).sum())
        if positive_count < min_class_samples or negative_count < min_class_samples:
            print(
                f"[Calibration {class_number + 1}/{len(class_mapping)}] "
                f"Skipping '{class_name}': {positive_count} positives, "
                f"{negative_count} negatives"
            )
            continue

        print(
            f"[Calibration {class_number + 1}/{len(class_mapping)}] "
            f"Fitting '{class_name}' with {positive_count} multi-label positives"
        )
        classifier, metrics = fit_calibrated_classifier(
            X_train_scaled,
            y,
            target_precision=target_precision,
            random_state=42 + label_id,
            tune_regularization=tune_regularization,
            fixed_c=fixed_c,
            use_hard_negative_mining=use_hard_negative_mining,
            hard_negative_fraction=hard_negative_fraction,
            hard_negative_weight=hard_negative_weight,
        )
        classifiers_by_class[label_id] = classifier
        class_ids.append(label_id)
        class_names.append(class_name)
        metric_rows.append({
            "label_id": label_id,
            "label_name": class_name,
            **metrics,
        })

    if not class_ids:
        raise ValueError("No classes had enough examples for calibration.")

    calibration_metrics = pd.DataFrame(metric_rows)
    print(f"Fit {len(class_ids)} calibrated subset models on {len(X_train)} images")
    return SubsetProbabilityModel(
        scaler=scaler,
        class_ids=np.array(class_ids),
        class_names=class_names,
        classifiers_by_class=classifiers_by_class,
        calibration_metrics=calibration_metrics,
    )


def score_embeddings(model, X, metadata):
    X_scaled = model.scaler.transform(X)
    calibrated_columns = []
    raw_columns = []
    thresholds = []
    for label_id in model.class_ids:
        classifier = model.classifiers_by_class[int(label_id)]
        calibrated_columns.append(classifier.calibrated_probability(X_scaled))
        raw_columns.append(classifier.raw_probability(X_scaled))
        thresholds.append(classifier.threshold)

    probabilities = np.column_stack(calibrated_columns)
    raw_probabilities = np.column_stack(raw_columns)
    thresholds = np.asarray(thresholds)
    order = np.argsort(probabilities, axis=1)
    best_positions = order[:, -1]
    second_positions = order[:, -2] if probabilities.shape[1] > 1 else best_positions
    row_indices = np.arange(len(X))
    best_probabilities = probabilities[row_indices, best_positions]
    second_probabilities = probabilities[row_indices, second_positions]
    passes_threshold = probabilities >= thresholds[None, :]

    scores = pd.DataFrame({
        "path": metadata["path"].values,
        "predicted_subset_label_id": model.class_ids[best_positions],
        "predicted_subset_label_name": [
            model.class_names[position] for position in best_positions
        ],
        "positive_probability": best_probabilities,
        "subset_probability": best_probabilities,
        "raw_positive_probability": raw_probabilities[row_indices, best_positions],
        "second_subset_label_id": model.class_ids[second_positions],
        "second_subset_label_name": [
            model.class_names[position] for position in second_positions
        ],
        "second_positive_probability": second_probabilities,
        "top_two_probability_margin": best_probabilities - second_probabilities,
        "predicted_class_threshold": thresholds[best_positions],
        "passes_predicted_class_threshold": passes_threshold[
            row_indices, best_positions
        ],
        "classes_above_threshold": passes_threshold.sum(axis=1),
        "probability_method": "calibrated_multilabel_ovr",
    })
    for column in ("label_id", "label_name", "all_label_ids", "all_label_names"):
        if column in metadata.columns:
            scores[f"actual_{column}"] = metadata[column].values

    probability_matrix = pd.DataFrame({
        f"{class_name}_positive_probability": probabilities[:, position]
        for position, class_name in enumerate(model.class_names)
    })
    probability_matrix.insert(0, "path", metadata["path"].values)
    raw_probability_matrix = pd.DataFrame({
        f"{class_name}_raw_positive_probability": raw_probabilities[:, position]
        for position, class_name in enumerate(model.class_names)
    })
    raw_probability_matrix.insert(0, "path", metadata["path"].values)
    return scores, probability_matrix, raw_probability_matrix


def evaluate_probabilities(model, probability_matrix, metadata):
    rows = []
    for position, label_id in enumerate(model.class_ids):
        y_true = multilabel_targets(metadata, int(label_id))
        probabilities = probability_matrix.iloc[:, position + 1].to_numpy()
        threshold = model.classifiers_by_class[int(label_id)].threshold
        predicted = probabilities >= threshold
        true_positive = int(((y_true == 1) & predicted).sum())
        rows.append({
            "label_id": int(label_id),
            "label_name": model.class_names[position],
            "positive_count": int(y_true.sum()),
            "threshold": threshold,
            "brier_score": brier_score_loss(y_true, probabilities),
            "expected_calibration_error": _expected_calibration_error(
                y_true, probabilities
            ),
            "average_precision": average_precision_score(y_true, probabilities)
            if y_true.any() else np.nan,
            "precision_at_threshold": true_positive / max(1, int(predicted.sum())),
            "recall_at_threshold": true_positive / max(1, int(y_true.sum())),
        })
    return pd.DataFrame(rows)


def score_directory(
    learning_dir="saved_vectors/train",
    target_dir="saved_vectors/retrain",
    output_name="subset_probability_scores.csv",
    matrix_output_name="subset_probability_matrix.csv",
    tune_regularization=True,
    fixed_c=1.0,
    use_hard_negative_mining=True,
    hard_negative_fraction=0.1,
    hard_negative_weight=3.0,
):
    model = fit_subset_probability_model(
        learning_dir=learning_dir,
        tune_regularization=tune_regularization,
        fixed_c=fixed_c,
        use_hard_negative_mining=use_hard_negative_mining,
        hard_negative_fraction=hard_negative_fraction,
        hard_negative_weight=hard_negative_weight,
    )
    X_target, _, _, _, metadata, _ = ut.load_DINO_vectors(target_dir)
    scores, probability_matrix, raw_probability_matrix = score_embeddings(
        model, X_target, metadata
    )
    evaluation_metrics = evaluate_probabilities(model, probability_matrix, metadata)

    output_dir = Path(target_dir)
    if not output_dir.is_absolute():
        output_dir = ut.PROJECT_ROOT / output_dir
    scores.to_csv(output_dir / output_name, index=False)
    probability_matrix.to_csv(output_dir / matrix_output_name, index=False)
    raw_probability_matrix.to_csv(
        output_dir / "subset_probability_raw_matrix.csv", index=False
    )
    model.calibration_metrics.to_csv(
        output_dir / "subset_probability_calibration_metrics.csv", index=False
    )
    evaluation_metrics.to_csv(
        output_dir / "subset_probability_evaluation_metrics.csv", index=False
    )

    print(f"Saved calibrated subset scores -> {output_dir / output_name}")
    print(f"Saved calibrated probability matrix -> {output_dir / matrix_output_name}")
    print(
        "Probabilities use COCO multi-label targets and held-out sigmoid calibration."
    )
    return model, scores, probability_matrix
