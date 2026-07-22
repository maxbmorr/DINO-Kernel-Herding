from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import brentq
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


def _kernel_diagonal(X, kernel):
    if kernel == "rbf":
        return np.ones(len(X), dtype=X.dtype)
    if kernel == "cosine":
        squared_norm = np.sum(X * X, axis=1)
        return ((squared_norm / (squared_norm + 1e-12)) + 1.0) / 2.0
    raise ValueError("kernel must be 'rbf' or 'cosine'.")


def _von_neumann_entropy_from_eigenvalues(eigenvalues):
    if len(eigenvalues) <= 1:
        return 0.0

    eigenvalues = eigenvalues / (np.sum(eigenvalues) + 1e-12)
    eigenvalues = np.clip(eigenvalues, 1e-12, 1.0)

    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    max_entropy = np.log(len(eigenvalues))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def _von_neumann_entropy_from_kernel(kernel_matrix):
    return _von_neumann_entropy_from_eigenvalues(
        np.linalg.eigvalsh(kernel_matrix)
    )


def _group_secular_poles(eigenvalues, transformed_cross):
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    pole_tolerance = 1e-14 * scale
    weight_tolerance = 1e-28 * scale * scale
    poles = []
    weights = []
    retained_eigenvalues = []

    start = 0
    while start < len(eigenvalues):
        end = start + 1
        while (
            end < len(eigenvalues)
            and abs(eigenvalues[end] - eigenvalues[start]) <= pole_tolerance
        ):
            end += 1

        pole = float(np.mean(eigenvalues[start:end]))
        weight = float(np.sum(transformed_cross[start:end] ** 2))
        multiplicity = end - start
        if weight <= weight_tolerance:
            retained_eigenvalues.extend([pole] * multiplicity)
        else:
            retained_eigenvalues.extend([pole] * (multiplicity - 1))
            poles.append(pole)
            weights.append(weight)
        start = end

    return (
        np.asarray(poles, dtype=float),
        np.asarray(weights, dtype=float),
        retained_eigenvalues,
    )


def _secular_bordered_eigenvalues(
    current_eigenvalues,
    current_eigenvectors,
    cross_kernel,
    self_kernel,
):
    transformed_cross = current_eigenvectors.T @ cross_kernel
    poles, weights, retained = _group_secular_poles(
        current_eigenvalues,
        transformed_cross,
    )
    if len(poles) == 0:
        return np.sort(np.asarray([*retained, float(self_kernel)]))

    def secular_function(value):
        return float(
            self_kernel
            - value
            - np.sum(weights / (poles - value))
        )

    scale = max(
        1.0,
        float(np.max(np.abs(poles))),
        abs(float(self_kernel)),
        float(np.sqrt(np.sum(weights))),
    )
    roots = []
    lower = min(float(poles[0]), float(self_kernel)) - scale
    left_of_first = float(np.nextafter(poles[0], -np.inf))
    while secular_function(lower) * secular_function(left_of_first) > 0:
        lower -= scale
        scale *= 2.0
    roots.append(
        brentq(
            secular_function,
            lower,
            left_of_first,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=100,
        )
    )

    for left_pole, right_pole in zip(poles[:-1], poles[1:]):
        left = float(np.nextafter(left_pole, np.inf))
        right = float(np.nextafter(right_pole, -np.inf))
        roots.append(
            brentq(
                secular_function,
                left,
                right,
                xtol=1e-13,
                rtol=1e-13,
                maxiter=100,
            )
        )

    right_of_last = float(np.nextafter(poles[-1], np.inf))
    upper = max(float(poles[-1]), float(self_kernel)) + scale
    while secular_function(right_of_last) * secular_function(upper) > 0:
        upper += scale
        scale *= 2.0
    roots.append(
        brentq(
            secular_function,
            right_of_last,
            upper,
            xtol=1e-13,
            rtol=1e-13,
            maxiter=100,
        )
    )

    eigenvalues = np.sort(np.asarray([*retained, *roots], dtype=float))
    if len(eigenvalues) != len(current_eigenvalues) + 1:
        raise ValueError("Secular update returned the wrong eigenvalue count.")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Secular update returned non-finite eigenvalues.")
    return eigenvalues


def von_neumann_entropy(X, kernel="rbf", gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return _von_neumann_entropy_from_kernel(_kernel_matrix(X, X, kernel, gamma))


def kernel_herd_reference_subset(X_class, reference_count, kernel, gamma):
    if len(X_class) <= reference_count:
        return X_class

    kernel_matrix = _kernel_matrix(X_class, X_class, kernel, gamma)
    kernel_mean = kernel_matrix.mean(axis=1)
    selected = []
    selected_mask = np.zeros(len(X_class), dtype=bool)
    accumulated_similarity = np.zeros(len(X_class), dtype=float)

    for step in range(reference_count):
        scores = kernel_mean - accumulated_similarity / (step + 1)
        scores[selected_mask] = -np.inf
        best_position = int(np.argmax(scores))
        selected.append(best_position)
        selected_mask[best_position] = True
        accumulated_similarity += kernel_matrix[:, best_position]

    return X_class[np.array(selected)]


def _reference_subset(
    X_class,
    max_reference_count,
    method="all",
    kernel="rbf",
    gamma=None,
):
    if method == "all" or len(X_class) <= max_reference_count:
        return X_class

    if method == "kernel_herding":
        if gamma is None:
            gamma = 1.0 / X_class.shape[1]
        return kernel_herd_reference_subset(
            X_class,
            max_reference_count,
            kernel,
            gamma,
        )
    raise ValueError("reference_method must be 'all' or 'kernel_herding'.")


def _metadata_has_label(metadata, label_id):
    if "all_label_ids" not in metadata.columns:
        return metadata["label_id"].to_numpy() == label_id

    label_token = str(label_id)
    return metadata["all_label_ids"].fillna("").astype(str).map(
        lambda value: label_token in value.split("|")
    ).to_numpy()


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
    eigenvalue_method="direct",
    progress_label=None,
):
    if probability_lambda <= 0:
        raise ValueError(
            "probability_lambda must be positive so subset trust is enforced."
        )
    if eigenvalue_method not in {"direct", "secular"}:
        raise ValueError("eigenvalue_method must be 'direct' or 'secular'.")

    X_labeled_class = np.asarray(X_labeled_class, dtype=np.float64)
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    selected_indices = []
    selected_candidate_positions = []
    # These blocks do not change between greedy rounds. Caching them preserves
    # the objective while avoiding a full kernel rebuild for every candidate.
    current_kernel = _kernel_matrix(
        X_labeled_class, X_labeled_class, kernel, gamma
    )
    reference_candidate_kernel = _kernel_matrix(
        X_labeled_class, X_candidates, kernel, gamma
    )
    candidate_self_kernel = _kernel_diagonal(X_candidates, kernel)
    selected_candidate_kernel_rows = []
    current_entropy = _von_neumann_entropy_from_kernel(current_kernel)
    current_log_probability_sum = 0.0
    current_objective = current_entropy

    selection_limit = (
        len(candidate_indices) if selection_count is None
        else min(selection_count, len(candidate_indices))
    )
    for rank in range(selection_limit):
        remaining_count = len(candidate_indices) - len(selected_candidate_positions)
        round_started = perf_counter()
        report_every = max(1, int(np.ceil(remaining_count / 10)))
        evaluated_count = 0
        if progress_label is not None:
            print(
                f"[{progress_label}] selection round {rank + 1}: "
                f"evaluating {remaining_count} remaining candidates",
                flush=True,
            )
        if eigenvalue_method == "secular":
            current_eigenvalues, current_eigenvectors = np.linalg.eigh(
                current_kernel
            )
        best_position = None
        best_objective = -np.inf
        best_entropy = -np.inf
        best_gain = -np.inf
        best_log_probability = -np.inf

        for position, candidate_index in enumerate(candidate_indices):
            if position in selected_candidate_positions:
                continue

            cross_kernel = reference_candidate_kernel[:, position]
            if selected_candidate_kernel_rows:
                cross_kernel = np.concatenate([
                    cross_kernel,
                    np.array([
                        row[position] for row in selected_candidate_kernel_rows
                    ]),
                ])
            if eigenvalue_method == "secular":
                try:
                    trial_eigenvalues = _secular_bordered_eigenvalues(
                        current_eigenvalues,
                        current_eigenvectors,
                        cross_kernel,
                        candidate_self_kernel[position],
                    )
                    trial_entropy = _von_neumann_entropy_from_eigenvalues(
                        trial_eigenvalues
                    )
                except (
                    ValueError,
                    FloatingPointError,
                    OverflowError,
                    ZeroDivisionError,
                    np.linalg.LinAlgError,
                ):
                    trial_kernel = np.empty(
                        (len(current_kernel) + 1, len(current_kernel) + 1),
                        dtype=current_kernel.dtype,
                    )
                    trial_kernel[:-1, :-1] = current_kernel
                    trial_kernel[:-1, -1] = cross_kernel
                    trial_kernel[-1, :-1] = cross_kernel
                    trial_kernel[-1, -1] = candidate_self_kernel[position]
                    trial_entropy = _von_neumann_entropy_from_kernel(
                        trial_kernel
                    )
            else:
                trial_kernel = np.empty(
                    (len(current_kernel) + 1, len(current_kernel) + 1),
                    dtype=current_kernel.dtype,
                )
                trial_kernel[:-1, :-1] = current_kernel
                trial_kernel[:-1, -1] = cross_kernel
                trial_kernel[-1, :-1] = cross_kernel
                trial_kernel[-1, -1] = candidate_self_kernel[position]
                trial_entropy = _von_neumann_entropy_from_kernel(trial_kernel)
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

            evaluated_count += 1
            if progress_label is not None and (
                evaluated_count % report_every == 0
                or evaluated_count == remaining_count
            ):
                elapsed = perf_counter() - round_started
                fraction = evaluated_count / remaining_count
                eta = elapsed * (1.0 - fraction) / max(fraction, 1e-12)
                print(
                    f"[{progress_label}] round {rank + 1}: "
                    f"{evaluated_count}/{remaining_count} candidates "
                    f"({100 * fraction:.0f}%), elapsed={elapsed:.1f}s, "
                    f"ETA={eta:.1f}s",
                    flush=True,
                )

        if best_position is None:
            if progress_label is not None:
                print(
                    f"[{progress_label}] stopped: no candidate remained.",
                    flush=True,
                )
            break
        if stop_when_objective_decreases and best_gain <= 0.0:
            if progress_label is not None:
                print(
                    f"[{progress_label}] stopped after {rank} selections: "
                    f"best remaining objective gain={best_gain:.8f} is not positive.",
                    flush=True,
                )
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
        chosen_cross_kernel = reference_candidate_kernel[:, best_position]
        if selected_candidate_kernel_rows:
            chosen_cross_kernel = np.concatenate([
                chosen_cross_kernel,
                np.array([
                    row[best_position] for row in selected_candidate_kernel_rows
                ]),
            ])
        expanded_kernel = np.empty(
            (len(current_kernel) + 1, len(current_kernel) + 1),
            dtype=current_kernel.dtype,
        )
        expanded_kernel[:-1, :-1] = current_kernel
        expanded_kernel[:-1, -1] = chosen_cross_kernel
        expanded_kernel[-1, :-1] = chosen_cross_kernel
        expanded_kernel[-1, -1] = candidate_self_kernel[best_position]
        current_kernel = expanded_kernel
        selected_candidate_kernel_rows.append(
            _kernel_matrix(
                X_candidates[best_position:best_position + 1],
                X_candidates,
                kernel,
                gamma,
            )[0]
        )
        current_entropy = best_entropy
        current_log_probability_sum += best_log_probability
        current_objective = best_objective
        if progress_label is not None:
            print(
                f"[{progress_label}] accepted selection {rank + 1}: "
                f"objective gain={best_gain:.8f}, "
                f"entropy gain={selected_indices[-1]['von_neumann_entropy_gain']:.8f}, "
                f"round time={perf_counter() - round_started:.1f}s",
                flush=True,
            )

    return selected_indices


def optimize_subset_selection(
    learning_dir="saved_vectors/train",
    target_dir="saved_vectors/retrain",
    probability_scores_path=None,
    probability_matrix_path=None,
    selection_count=None,
    probability_lambda=0.01,
    kernel="rbf",
    stop_when_objective_decreases=True,
    eigenvalue_method="direct",
    max_candidate_pool_per_subset=None,
    max_labeled_reference_per_subset=200,
    reference_method="all",
    selected_class_names=None,
    allowed_candidate_indices=None,
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
    learning_metadata = learning_metadata.loc[labeled_mask].reset_index(drop=True)

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
        and (
            selected_class_names is None
            or class_row["label_name"] in selected_class_names
        )
    ]
    print(
        f"Running global von Neumann optimization for "
        f"{len(optimization_rows)} subsets"
    )
    print(
        f"Optimization settings: selection_count={selection_count}, "
        f"lambda={probability_lambda}, kernel={kernel}, "
        f"reference_method={reference_method}, "
        f"eigenvalue_method={eigenvalue_method}, "
        f"stop_when_objective_decreases={stop_when_objective_decreases}"
    )

    for class_number, class_row in enumerate(optimization_rows, start=1):
        label_id = int(class_row["label_id"])
        class_name = class_row["label_name"]
        probability_column = f"{class_name}_positive_probability"

        label_id = int(label_id)
        class_learning_mask = _metadata_has_label(learning_metadata, label_id)
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
        if allowed_candidate_indices is not None:
            allowed = np.asarray(allowed_candidate_indices, dtype=int)
            candidate_scores = candidate_scores[
                candidate_scores["target_index"].isin(allowed)
            ].reset_index(drop=True)

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
            method=reference_method,
            kernel=kernel,
            gamma=gamma,
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
            selection_count=selection_count,
            kernel=kernel,
            gamma=gamma,
            probability_lambda=probability_lambda,
            stop_when_objective_decreases=stop_when_objective_decreases,
            eigenvalue_method=eigenvalue_method,
            progress_label=(
                f"Optimization {class_number}/{len(optimization_rows)}: {class_name}"
            ),
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
                f"{selection_count if selection_count is not None else 'open'}: "
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
                "reference_method": reference_method,
                "eigenvalue_method": eigenvalue_method,
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
            "reference_method": reference_method,
            "eigenvalue_method": eigenvalue_method,
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
