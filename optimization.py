from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import __utils__ as ut


def _resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ut.PROJECT_ROOT / path
    return path


def _rbf_kernel(X_left, X_right, gamma):
    left_norm = np.sum(X_left * X_left, axis=1)[:, None]
    right_norm = np.sum(X_right * X_right, axis=1)[None, :]
    distances = np.maximum(left_norm + right_norm - (2 * X_left @ X_right.T), 0.0)
    return np.exp(-gamma * distances)


def _cosine_kernel(X_left, X_right):
    left_norm = np.linalg.norm(X_left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(X_right, axis=1, keepdims=True).T
    cosine_similarity = (X_left @ X_right.T) / (left_norm * right_norm + 1e-12)
    return (cosine_similarity + 1.0) / 2.0


def _kernel_matrix(X_left, X_right, kernel, gamma):
    if kernel == "rbf":
        return _rbf_kernel(X_left, X_right, gamma)
    if kernel == "cosine":
        return _cosine_kernel(X_left, X_right)
    raise ValueError("kernel must be 'rbf' or 'cosine'.")


def von_neumann_entropy(X, kernel="rbf", gamma=None):
    if len(X) <= 1:
        return 0.0
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    kernel_matrix = _kernel_matrix(X, X, kernel, gamma)
    density_matrix = kernel_matrix / (np.trace(kernel_matrix) + 1e-12)
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    eigenvalues = np.clip(eigenvalues, 1e-12, 1.0)

    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    max_entropy = np.log(len(eigenvalues))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def _reference_subset(X_class, max_reference_count, random_state):
    if len(X_class) <= max_reference_count:
        return X_class

    rng = np.random.default_rng(random_state)
    reference_indices = rng.choice(
        len(X_class),
        size=max_reference_count,
        replace=False,
    )
    return X_class[np.sort(reference_indices)]


def greedy_maximize_von_neumann_entropy(
    X_labeled_class,
    X_candidates,
    candidate_indices,
    candidate_positive_probabilities,
    selection_count,
    kernel,
    gamma,
    probability_lambda=0.01,
    stop_when_objective_decreases=True,
):
    selected_indices = []
    selected_candidate_positions = []
    current_X = X_labeled_class.copy()
    current_entropy = von_neumann_entropy(current_X, kernel=kernel, gamma=gamma)
    current_log_probability_sum = 0.0
    current_objective = current_entropy

    for rank in range(selection_count):
        best_position = None
        best_objective = -np.inf
        best_entropy = -np.inf
        best_gain = -np.inf
        best_log_probability = -np.inf

        for position, candidate_index in enumerate(candidate_indices):
            if position in selected_candidate_positions:
                continue

            trial_X = np.vstack([current_X, X_candidates[position:position + 1]])
            trial_entropy = von_neumann_entropy(trial_X, kernel=kernel, gamma=gamma)
            log_probability = float(
                np.log(candidate_positive_probabilities[position] + 1e-12)
            )
            trial_log_probability_sum = current_log_probability_sum + log_probability
            # This is the value being maximized for each candidate:
            # H(labeled subset union S) + lambda * sum(log(P_hat_+(x))).
            trial_objective = (
                trial_entropy + probability_lambda * trial_log_probability_sum
            )
            objective_gain = trial_objective - current_objective

            if trial_objective > best_objective:
                best_position = position
                best_objective = trial_objective
                best_entropy = trial_entropy
                best_gain = objective_gain
                best_log_probability = log_probability

        if best_position is None:
            break
        if stop_when_objective_decreases and best_gain <= 0.0:
            break

        selected_candidate_positions.append(best_position)
        selected_indices.append({
            "target_index": int(candidate_indices[best_position]),
            "optimization_rank": rank + 1,
            "objective": float(best_objective),
            "objective_gain": float(best_gain),
            "von_neumann_entropy": float(best_entropy),
            "von_neumann_entropy_gain": float(best_entropy - current_entropy),
            "log_probability": float(best_log_probability),
            "log_probability_sum": float(
                current_log_probability_sum + best_log_probability
            ),
        })
        current_X = np.vstack([current_X, X_candidates[best_position:best_position + 1]])
        current_entropy = best_entropy
        current_log_probability_sum += best_log_probability
        current_objective = best_objective

    return selected_indices


def optimize_subset_selection(
    learning_dir="saved_vectors/learning",
    target_dir="saved_vectors/testing",
    probability_scores_path=None,
    probability_matrix_path=None,
    selection_count=4,
    probability_lambda=0.01,
    kernel="rbf",
    stop_when_objective_decreases=True,
    max_candidate_pool_per_subset=None,
    max_labeled_reference_per_subset=200,
    random_state=42,
):
    learning_dir = _resolve_path(learning_dir)
    target_dir = _resolve_path(target_dir)

    if probability_scores_path is None:
        probability_scores_path = target_dir / "subset_probability_scores.csv"
    else:
        probability_scores_path = _resolve_path(probability_scores_path)
    if probability_matrix_path is None:
        probability_matrix_path = target_dir / "subset_probability_matrix.csv"
    else:
        probability_matrix_path = _resolve_path(probability_matrix_path)

    X_learning, learning_label_ids, _, _, learning_metadata, class_mapping = (
        ut.load_DINO_vectors(learning_dir)
    )
    X_target, _, _, _, target_metadata, _ = ut.load_DINO_vectors(target_dir)
    probability_scores = pd.read_csv(probability_scores_path)
    probability_matrix = pd.read_csv(probability_matrix_path)

    if len(probability_scores) != len(X_target):
        raise ValueError(
            "subset_probability_scores.csv row count does not match target vectors."
        )
    if len(probability_matrix) != len(X_target):
        raise ValueError(
            "subset_probability_matrix.csv row count does not match target vectors."
        )

    labeled_mask = learning_label_ids >= 0
    X_learning = X_learning[labeled_mask]
    learning_label_ids = learning_label_ids[labeled_mask]

    if len(learning_label_ids) == 0:
        raise ValueError("No labeled learning images found for optimization.")

    scaler = StandardScaler()
    X_learning_scaled = scaler.fit_transform(X_learning)
    X_target_scaled = scaler.transform(X_target)
    gamma = 1.0 / X_learning_scaled.shape[1]

    selected_rows = []
    summary_rows = []
    optimization_rows = [
        class_row
        for _, class_row in class_mapping.iterrows()
        if f"{class_row['label_name']}_positive_probability"
        in probability_matrix.columns
    ]
    print(
        f"Running global von Neumann optimization for "
        f"{len(optimization_rows)} subsets"
    )
    print(
        f"Optimization settings: selection_count={selection_count}, "
        f"lambda={probability_lambda}, kernel={kernel}, "
        f"stop_when_objective_decreases={stop_when_objective_decreases}"
    )
    if probability_lambda == 0:
        print(
            "Warning: lambda is 0, so P_hat_+(x) is reported but not used "
            "as a soft penalty. The optimizer is maximizing entropy only."
        )

    for class_number, class_row in enumerate(optimization_rows, start=1):
        label_id = int(class_row["label_id"])
        class_name = class_row["label_name"]
        probability_column = f"{class_name}_positive_probability"

        label_id = int(label_id)
        class_learning_mask = learning_label_ids == label_id
        if class_learning_mask.sum() == 0:
            print(
                f"[Optimization {class_number}/{len(optimization_rows)}] "
                f"Skipping '{class_name}': no labeled reference images"
            )
            continue

        candidate_scores = pd.DataFrame({
            "target_index": np.arange(len(X_target)),
            "positive_probability": probability_matrix[
                probability_column
            ].to_numpy(),
        })

        if len(candidate_scores) == 0:
            print(
                f"[Optimization {class_number}/{len(optimization_rows)}] "
                f"Skipping '{class_name}': no target images"
            )
            continue

        if max_candidate_pool_per_subset is not None:
            candidate_scores = candidate_scores.sort_values(
                "positive_probability",
                ascending=False,
            ).head(max_candidate_pool_per_subset)
        candidate_indices = candidate_scores["target_index"].to_numpy()

        class_reference = _reference_subset(
            X_learning_scaled[class_learning_mask],
            max_labeled_reference_per_subset,
            random_state + label_id,
        )
        base_entropy = von_neumann_entropy(class_reference, kernel=kernel, gamma=gamma)
        print(
            f"[Optimization {class_number}/{len(optimization_rows)}] "
            f"Subset '{class_name}': "
            f"{int(class_learning_mask.sum())} labeled reference images, "
            f"{len(candidate_scores)} global candidate images, "
            f"base H={base_entropy:.6f}"
        )

        selected = greedy_maximize_von_neumann_entropy(
            X_labeled_class=class_reference,
            X_candidates=X_target_scaled[candidate_indices],
            candidate_indices=candidate_indices,
            candidate_positive_probabilities=candidate_scores[
                "positive_probability"
            ].to_numpy(),
            selection_count=min(selection_count, len(candidate_indices)),
            kernel=kernel,
            gamma=gamma,
            probability_lambda=probability_lambda,
            stop_when_objective_decreases=stop_when_objective_decreases,
        )
        print(
            f"[Optimization {class_number}/{len(optimization_rows)}] "
            f"Subset '{class_name}' selected {len(selected)} images"
        )

        for selected_item in selected:
            target_index = selected_item["target_index"]
            probability_row = probability_scores.loc[target_index].to_dict()
            metadata_row = target_metadata.iloc[target_index].to_dict()
            print(
                f"  selected rank {selected_item['optimization_rank']}/"
                f"{selection_count}: "
                f"H gain={selected_item['von_neumann_entropy_gain']:.6f}, "
                f"objective gain={selected_item['objective_gain']:.6f}, "
                f"P_hat_+={probability_matrix.loc[target_index, probability_column]:.6f}, "
                f"image={metadata_row['path']}"
            )
            selected_rows.append({
                "subset_label_id": label_id,
                "subset_label_name": class_name,
                "path": metadata_row["path"],
                "optimization_rank": selected_item["optimization_rank"],
                "kernel": kernel,
                "stop_when_objective_decreases": stop_when_objective_decreases,
                "objective": selected_item["objective"],
                "objective_gain": selected_item["objective_gain"],
                "probability_lambda": probability_lambda,
                "base_von_neumann_entropy": base_entropy,
                "von_neumann_entropy": selected_item["von_neumann_entropy"],
                "von_neumann_entropy_gain": selected_item["von_neumann_entropy_gain"],
                "log_probability": selected_item["log_probability"],
                "log_probability_sum": selected_item["log_probability_sum"],
                "positive_probability": probability_matrix.loc[
                    target_index,
                    probability_column,
                ],
                "subset_probability": probability_matrix.loc[
                    target_index,
                    probability_column,
                ],
                "best_predicted_subset_label_id": probability_row[
                    "predicted_subset_label_id"
                ],
                "best_predicted_subset_label_name": probability_row[
                    "predicted_subset_label_name"
                ],
                "best_predicted_subset_probability": probability_row[
                    "subset_probability"
                ],
                "actual_label_id": metadata_row.get("label_id", -1),
                "actual_label_name": metadata_row.get("label_name", "unlabeled"),
            })

        summary_rows.append({
            "subset_label_id": label_id,
            "subset_label_name": class_name,
            "labeled_reference_count": int(class_learning_mask.sum()),
            "candidate_count": int(len(candidate_scores)),
            "selected_count": int(len(selected)),
            "kernel": kernel,
            "stop_when_objective_decreases": stop_when_objective_decreases,
            "probability_lambda": probability_lambda,
            "base_von_neumann_entropy": base_entropy,
            "final_von_neumann_entropy": (
                selected[-1]["von_neumann_entropy"] if selected else base_entropy
            ),
            "final_objective": (
                selected[-1]["objective"] if selected else base_entropy
            ),
        })

    selected_frame = pd.DataFrame(selected_rows)
    summary_frame = pd.DataFrame(summary_rows)

    selected_output = target_dir / "optimization_selection.csv"
    summary_output = target_dir / "optimization_summary.csv"
    selected_frame.to_csv(selected_output, index=False)
    summary_frame.to_csv(summary_output, index=False)

    print(f"Saved von Neumann optimization selection -> {selected_output}")
    print(f"Saved von Neumann optimization summary -> {summary_output}")
    print(f"Optimization selected {len(selected_frame)} images")

    return selected_frame, summary_frame
