from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import __utils__ as ut


@dataclass
class SubsetProbabilityModel:
    scaler: StandardScaler
    class_ids: np.ndarray
    class_names: list
    nce_by_class: dict


def _filter_labeled_rows(X, label_ids, metadata):
    labeled_mask = label_ids >= 0
    return X[labeled_mask], label_ids[labeled_mask], metadata[labeled_mask]


def fit_nce_density(X_class, noise_ratio=5, random_state=42):
    variance = np.var(X_class, axis=0)
    variance = np.maximum(variance, 1e-4)
    mean = np.mean(X_class, axis=0)

    rng = np.random.default_rng(random_state)
    noise_count = max(1, int(len(X_class) * noise_ratio))
    noise = rng.normal(
        loc=mean,
        scale=np.sqrt(variance),
        size=(noise_count, X_class.shape[1]),
    )

    X_binary = np.vstack([X_class, noise])
    y_binary = np.concatenate([
        np.ones(len(X_class), dtype=int),
        np.zeros(noise_count, dtype=int),
    ])

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_binary, y_binary)

    return {
        "classifier": classifier,
        "mean": mean,
        "variance": variance,
        "noise_ratio": noise_count / len(X_class),
    }


def gaussian_log_probability(X, mean, variance):
    centered = X - mean
    log_det = np.sum(np.log(2 * np.pi * variance))
    quadratic = np.sum((centered * centered) / variance, axis=1)
    return -0.5 * (log_det + quadratic)


def nce_log_density(nce_model, X):
    logit = nce_model["classifier"].decision_function(X)
    noise_log_probability = gaussian_log_probability(
        X,
        nce_model["mean"],
        nce_model["variance"],
    )
    return logit + noise_log_probability + np.log(nce_model["noise_ratio"])


def nce_positive_probability(nce_model, X):
    return nce_model["classifier"].predict_proba(X)[:, 1]


def fit_subset_probability_model(
    learning_dir="saved_vectors/learning",
    min_class_samples=2,
):
    X_train, label_ids, _, _, metadata, class_mapping = ut.load_DINO_vectors(learning_dir)
    X_train, label_ids, metadata = _filter_labeled_rows(X_train, label_ids, metadata)

    if len(label_ids) == 0:
        raise ValueError("No labeled learning images found for subset probability model.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    class_ids = []
    class_names = []
    nce_by_class = {}

    unique_label_ids = sorted(np.unique(label_ids))
    print(f"Fitting NCE positive-probability models for {len(unique_label_ids)} classes")

    for class_number, label_id in enumerate(unique_label_ids, start=1):
        class_indices = np.where(label_ids == label_id)[0]
        if len(class_indices) < min_class_samples:
            print(
                f"[NCE {class_number}/{len(unique_label_ids)}] "
                f"Skipping label {label_id}: only {len(class_indices)} samples"
            )
            continue

        class_name = class_mapping.loc[
            class_mapping["label_id"] == label_id,
            "label_name",
        ].iloc[0]
        print(
            f"[NCE {class_number}/{len(unique_label_ids)}] "
            f"Fitting '{class_name}' with {len(class_indices)} labeled images"
        )
        X_class = X_train_scaled[class_indices]

        nce_by_class[int(label_id)] = fit_nce_density(
            X_class,
            random_state=42 + int(label_id),
        )

        class_ids.append(int(label_id))
        class_names.append(class_name)

    if not class_ids:
        raise ValueError("No classes had enough labeled samples to fit subset models.")

    print(f"Fit NCE subset probability model on {len(label_ids)} labeled images")
    print(f"Classes modeled: {len(class_ids)}")

    return SubsetProbabilityModel(
        scaler=scaler,
        class_ids=np.array(class_ids),
        class_names=class_names,
        nce_by_class=nce_by_class,
    )


def score_embeddings(
    model,
    X,
    metadata,
    probability_method="nce",
):
    if probability_method != "nce":
        raise ValueError("probability_method must be 'nce'.")

    X_scaled = model.scaler.transform(X)
    print(
        f"Scoring {len(X_scaled)} target images against "
        f"{len(model.class_ids)} NCE subset models"
    )

    positive_probability_columns = []
    log_density_columns = []
    for class_number, label_id in enumerate(model.class_ids, start=1):
        class_name = model.class_names[class_number - 1]
        print(
            f"[NCE score {class_number}/{len(model.class_ids)}] "
            f"Scoring '{class_name}'"
        )
        class_model = model.nce_by_class[int(label_id)]
        positive_probability_columns.append(
            nce_positive_probability(class_model, X_scaled)
        )
        log_density_columns.append(nce_log_density(class_model, X_scaled))

    nce_positive_probabilities = np.column_stack(positive_probability_columns)
    nce_log_densities = np.column_stack(log_density_columns)
    positive_probabilities = nce_positive_probabilities

    best_positions = np.argmax(positive_probabilities, axis=1)
    best_label_ids = model.class_ids[best_positions]
    best_label_names = [model.class_names[position] for position in best_positions]

    scores = pd.DataFrame({
        "path": metadata["path"].values,
        "predicted_subset_label_id": best_label_ids,
        "predicted_subset_label_name": best_label_names,
        "positive_probability": positive_probabilities[
            np.arange(len(X)),
            best_positions,
        ],
        "subset_probability": positive_probabilities[
            np.arange(len(X)),
            best_positions,
        ],
        "probability_method": probability_method,
        "nce_positive_probability": nce_positive_probabilities[
            np.arange(len(X)),
            best_positions,
        ],
        "nce_subset_probability": nce_positive_probabilities[
            np.arange(len(X)),
            best_positions,
        ],
        "nce_log_density": nce_log_densities[np.arange(len(X)), best_positions],
    })

    if "label_id" in metadata.columns:
        scores["actual_label_id"] = metadata["label_id"].values
    if "label_name" in metadata.columns:
        scores["actual_label_name"] = metadata["label_name"].values

    probability_matrix = pd.DataFrame({
        f"{class_name}_positive_probability": positive_probabilities[:, position]
        for position, class_name in enumerate(model.class_names)
    })
    probability_matrix.insert(0, "path", metadata["path"].values)

    return scores, probability_matrix


def score_directory(
    learning_dir="saved_vectors/learning",
    target_dir="saved_vectors/testing",
    output_name="subset_probability_scores.csv",
    matrix_output_name="subset_probability_matrix.csv",
    probability_method="nce",
):
    model = fit_subset_probability_model(
        learning_dir=learning_dir,
    )

    X_target, _, _, _, metadata, _ = ut.load_DINO_vectors(target_dir)
    scores, probability_matrix = score_embeddings(
        model,
        X_target,
        metadata,
        probability_method=probability_method,
    )

    output_dir = Path(target_dir)
    if not output_dir.is_absolute():
        output_dir = ut.PROJECT_ROOT / output_dir

    scores.to_csv(output_dir / output_name, index=False)
    probability_matrix.to_csv(output_dir / matrix_output_name, index=False)

    print(f"Saved subset probability scores -> {output_dir / output_name}")
    print(f"Saved subset probability matrix -> {output_dir / matrix_output_name}")
    print(f"Subset probability method: {probability_method}")
    print(
        "Positive probabilities are not normalized across classes; "
        "use them as P_hat_+(x) in the optimization objective."
    )

    return model, scores, probability_matrix
