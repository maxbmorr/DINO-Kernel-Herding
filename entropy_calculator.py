from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import __utils__ as ut


@dataclass
class EntropyModel:
    scaler: StandardScaler
    class_ids: np.ndarray
    class_names: list
    priors: np.ndarray
    nce_by_class: dict
    training_embeddings_by_class: dict
    entropy_threshold: float
    density_threshold: float
    outlier_density_threshold: float
    representation_gap_threshold: float
    class_thresholds: dict
    gamma: float
    von_neumann_neighbor_count: int


def _filter_labeled_rows(X, label_ids, metadata):
    labeled_mask = label_ids >= 0
    return X[labeled_mask], label_ids[labeled_mask], metadata[labeled_mask]


def safe_folder_name(name):
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return safe_name.strip("_") or "unknown"


def _logsumexp(values, axis=1):
    max_values = np.max(values, axis=axis, keepdims=True)
    return max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True))


def _normalized_entropy(probabilities):
    class_count = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    if class_count <= 1:
        return np.zeros_like(entropy)
    return np.maximum(entropy / np.log(class_count), 0.0)


def _von_neumann_entropy_from_kernel(kernel_matrix):
    density_matrix = kernel_matrix / (np.trace(kernel_matrix) + 1e-12)
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    eigenvalues = np.clip(eigenvalues, 1e-12, 1.0)
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    max_entropy = np.log(len(eigenvalues))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def local_von_neumann_entropy(X_scaled, class_embeddings, gamma, neighbor_count=25):
    if class_embeddings is None or len(class_embeddings) == 0:
        return np.zeros(len(X_scaled))

    entropies = []
    for row in X_scaled:
        row = row.reshape(1, -1)
        similarities = _rbf_kernel(row, class_embeddings, gamma).ravel()
        local_count = min(neighbor_count, len(class_embeddings))
        neighbor_positions = np.argpartition(similarities, -local_count)[-local_count:]
        local_points = np.vstack([class_embeddings[neighbor_positions], row])
        local_kernel = _rbf_kernel(local_points, local_points, gamma)
        entropies.append(_von_neumann_entropy_from_kernel(local_kernel))

    return np.array(entropies)


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


def fit_entropy_model(
    learning_dir="saved_vectors/learning",
    calibration_dir="saved_vectors/testing",
    entropy_percentile=75,
    density_percentile=75,
    outlier_density_percentile=1,
    min_class_samples=2,
):
    X_train, train_label_ids, _, _, train_metadata, class_mapping = ut.load_DINO_vectors(learning_dir)
    X_train, train_label_ids, train_metadata = _filter_labeled_rows(
        X_train,
        train_label_ids,
        train_metadata,
    )

    if len(train_label_ids) == 0:
        raise ValueError("No labeled learning images found for NCE entropy model.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    nce_by_class = {}
    training_embeddings_by_class = {}
    class_ids = []
    class_names = []
    priors = []

    for label_id in sorted(np.unique(train_label_ids)):
        class_indices = np.where(train_label_ids == label_id)[0]
        if len(class_indices) < min_class_samples:
            continue

        class_name = class_mapping.loc[
            class_mapping["label_id"] == label_id,
            "label_name",
        ].iloc[0]

        nce_by_class[int(label_id)] = fit_nce_density(
            X_train_scaled[class_indices],
            random_state=42 + int(label_id),
        )
        training_embeddings_by_class[int(label_id)] = X_train_scaled[class_indices]
        class_ids.append(int(label_id))
        class_names.append(class_name)
        priors.append(len(class_indices) / len(train_label_ids))

    if not nce_by_class:
        raise ValueError("No classes had enough labeled samples to fit NCE models.")

    model = EntropyModel(
        scaler=scaler,
        class_ids=np.array(class_ids),
        class_names=class_names,
        priors=np.array(priors),
        nce_by_class=nce_by_class,
        training_embeddings_by_class=training_embeddings_by_class,
        entropy_threshold=0.0,
        density_threshold=0.0,
        outlier_density_threshold=0.0,
        representation_gap_threshold=0.0,
        class_thresholds={},
        gamma=1.0 / X_train_scaled.shape[1],
        von_neumann_neighbor_count=25,
    )

    calibration_scores = score_directory(model, calibration_dir, save_csv=False)
    model.entropy_threshold = float(
        np.percentile(calibration_scores["entropy"], entropy_percentile)
    )
    model.density_threshold = float(
        np.percentile(calibration_scores["best_log_density"], density_percentile)
    )
    model.outlier_density_threshold = float(
        np.percentile(calibration_scores["best_log_density"], outlier_density_percentile)
    )
    model.representation_gap_threshold = float(
        np.percentile(calibration_scores["representation_gap"], 75)
    )
    model.class_thresholds = learn_class_thresholds(
        calibration_scores,
        entropy_percentile=entropy_percentile,
        density_percentile=density_percentile,
        outlier_density_percentile=outlier_density_percentile,
    )
    apply_class_thresholds(model, calibration_scores)

    print(f"Fit NCE entropy model on {len(train_label_ids)} labeled images from {learning_dir}")
    print(f"Classes modeled: {len(model.class_ids)}")
    print(
        f"High von Neumann entropy threshold "
        f"({entropy_percentile}%): {model.entropy_threshold:.3f}"
    )
    print(f"High NCE density threshold ({density_percentile}%): {model.density_threshold:.3f}")
    print(
        f"Complete outlier density threshold "
        f"({outlier_density_percentile}%): {model.outlier_density_threshold:.3f}"
    )
    print(f"Representation gap threshold (75%): {model.representation_gap_threshold:.3f}")
    print(f"Class-specific thresholds learned: {len(model.class_thresholds)}")

    return model


def threshold_label_column(scores):
    if "actual_label_id" in scores.columns and (scores["actual_label_id"] >= 0).any():
        return "actual_label_id"
    return "predicted_label_id"


def learn_class_thresholds(
    calibration_scores,
    entropy_percentile=75,
    density_percentile=5,
    outlier_density_percentile=1,
    min_samples=2,
):
    label_column = threshold_label_column(calibration_scores)
    thresholds = {}

    for label_id, class_scores in calibration_scores.groupby(label_column):
        if len(class_scores) < min_samples:
            continue

        thresholds[int(label_id)] = {
            "entropy_threshold": float(
                np.percentile(class_scores["entropy"], entropy_percentile)
            ),
            "density_threshold": float(
                np.percentile(class_scores["best_log_density"], density_percentile)
            ),
            "outlier_density_threshold": float(
                np.percentile(class_scores["best_log_density"], outlier_density_percentile)
            ),
            "representation_gap_threshold": float(
                np.percentile(class_scores["representation_gap"], 75)
            ),
            "sample_count": int(len(class_scores)),
        }

    return thresholds


def apply_class_thresholds(model, scores, X_scaled=None):
    label_column = threshold_label_column(scores)
    comparison_ids = scores[label_column].fillna(scores["predicted_label_id"]).astype(int)

    entropy_thresholds = []
    density_thresholds = []
    outlier_thresholds = []
    representation_thresholds = []

    for label_id in comparison_ids:
        class_threshold = model.class_thresholds.get(int(label_id), {})
        entropy_thresholds.append(
            max(
                class_threshold.get("entropy_threshold", model.entropy_threshold),
                model.entropy_threshold,
            )
        )
        density_thresholds.append(
            class_threshold.get("density_threshold", model.density_threshold)
        )
        outlier_thresholds.append(
            class_threshold.get(
                "outlier_density_threshold",
                model.outlier_density_threshold,
            )
        )
        representation_thresholds.append(
            class_threshold.get(
                "representation_gap_threshold",
                model.representation_gap_threshold,
            )
        )

    scores["threshold_label_id"] = comparison_ids
    if label_column == "actual_label_id" and "actual_label_name" in scores.columns:
        scores["threshold_label_name"] = scores["actual_label_name"]
    else:
        scores["threshold_label_name"] = scores["predicted_label_name"]
    scores["entropy_threshold"] = entropy_thresholds
    scores["density_threshold"] = density_thresholds
    scores["outlier_density_threshold"] = outlier_thresholds
    scores["representation_gap_threshold"] = representation_thresholds

    if X_scaled is not None:
        representation_scores = []
        von_neumann_entropies = []
        for row_index, label_id in enumerate(comparison_ids):
            class_embeddings = model.training_embeddings_by_class.get(int(label_id))
            if class_embeddings is None or len(class_embeddings) == 0:
                representation_scores.append(0.0)
                von_neumann_entropies.append(0.0)
                continue

            similarities = _rbf_kernel(
                X_scaled[row_index:row_index + 1],
                class_embeddings,
                model.gamma,
            ).ravel()
            representation_scores.append(float(np.max(similarities)))
            von_neumann_entropies.append(
                local_von_neumann_entropy(
                    X_scaled[row_index:row_index + 1],
                    class_embeddings,
                    model.gamma,
                    neighbor_count=model.von_neumann_neighbor_count,
                )[0]
            )

        scores["representation_score"] = representation_scores
        scores["representation_gap"] = 1.0 - scores["representation_score"]
        scores["entropy"] = von_neumann_entropies

    scores["high_entropy"] = scores["entropy"] >= scores["entropy_threshold"]
    scores["high_density"] = scores["best_log_density"] >= scores["density_threshold"]
    scores["complete_outlier"] = (
        scores["best_log_density"] <= scores["outlier_density_threshold"]
    )
    scores["in_distribution"] = ~scores["complete_outlier"]
    scores["poorly_represented"] = (
        scores["representation_gap"] >= scores["representation_gap_threshold"]
    )
    scores["interesting"] = (
        scores["high_entropy"]
        & scores["in_distribution"]
        & scores["poorly_represented"]
    )
    scores["herding_candidate"] = scores["interesting"] & ~scores["complete_outlier"]
    return scores


def score_embeddings(model, X, metadata):
    X_scaled = model.scaler.transform(X)

    log_densities = np.column_stack([
        nce_log_density(model.nce_by_class[int(label_id)], X_scaled)
        for label_id in model.class_ids
    ])
    log_priors = np.log(model.priors + 1e-12)
    log_scores = log_densities + log_priors
    probabilities = np.exp(log_scores - _logsumexp(log_scores, axis=1))

    class_entropy = _normalized_entropy(probabilities)
    best_class_positions = np.argmax(probabilities, axis=1)
    best_label_ids = model.class_ids[best_class_positions]
    best_label_names = [model.class_names[position] for position in best_class_positions]
    best_probabilities = probabilities[np.arange(len(X)), best_class_positions]
    best_log_densities = log_densities[np.arange(len(X)), best_class_positions]

    scores = pd.DataFrame({
        "path": metadata["path"].values,
        "predicted_label_id": best_label_ids,
        "predicted_label_name": best_label_names,
        "class_probability": best_probabilities,
        "class_entropy": class_entropy,
        "best_log_density": best_log_densities,
    })

    if "label_id" in metadata.columns:
        scores["actual_label_id"] = metadata["label_id"].values
    if "label_name" in metadata.columns:
        scores["actual_label_name"] = metadata["label_name"].values

    return apply_class_thresholds(model, scores, X_scaled=X_scaled)


def score_directory(model, input_dir="saved_vectors/testing", save_csv=True):
    X, _, _, _, metadata, _ = ut.load_DINO_vectors(input_dir)
    scores = score_embeddings(model, X, metadata)

    if save_csv:
        output_path = Path(input_dir)
        if not output_path.is_absolute():
            output_path = ut.PROJECT_ROOT / output_path
        scores.to_csv(output_path / "entropy_scores.csv", index=False)
        print(f"Saved entropy scores -> {output_path / 'entropy_scores.csv'}")

    return scores


def _rbf_kernel(X_left, X_right, gamma):
    left_norm = np.sum(X_left * X_left, axis=1)[:, None]
    right_norm = np.sum(X_right * X_right, axis=1)[None, :]
    distances = np.maximum(left_norm + right_norm - (2 * X_left @ X_right.T), 0.0)
    return np.exp(-gamma * distances)


def kernel_herding_select(
    X,
    candidate_indices,
    target_indices=None,
    selection_count=3,
    gamma=None,
):
    if target_indices is None:
        target_indices = np.arange(len(X))

    candidate_indices = np.asarray(candidate_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)

    if len(candidate_indices) == 0:
        return []

    selection_count = min(selection_count, len(candidate_indices))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_candidates = X_scaled[candidate_indices]
    X_target = X_scaled[target_indices]

    if gamma is None:
        gamma = 1.0 / X_scaled.shape[1]

    candidate_to_target = _rbf_kernel(X_candidates, X_target, gamma)
    target_mean_similarity = candidate_to_target.mean(axis=1)

    selected_positions = []
    selected_mask = np.zeros(len(candidate_indices), dtype=bool)
    selected_similarity_sum = np.zeros(len(candidate_indices))

    for step in range(selection_count):
        if step == 0:
            herding_scores = target_mean_similarity
        else:
            herding_scores = target_mean_similarity - (selected_similarity_sum / step)

        herding_scores[selected_mask] = -np.inf
        best_position = int(np.argmax(herding_scores))
        selected_positions.append(best_position)
        selected_mask[best_position] = True

        selected_kernel = _rbf_kernel(
            X_candidates,
            X_candidates[best_position:best_position + 1],
            gamma,
        ).ravel()
        selected_similarity_sum += selected_kernel

    return candidate_indices[selected_positions].tolist()


def select_interesting_with_kernel_herding(
    input_dir,
    scores,
    selection_count=25,
    per_class=False,
    candidate_pool="all",
    save_csv=True,
):
    X, _, _, _, _, _ = ut.load_DINO_vectors(input_dir)
    if candidate_pool == "all":
        candidate_indices = np.arange(len(scores))
    elif candidate_pool == "interesting":
        candidate_indices = scores.index[scores["interesting"]].to_numpy()
    elif candidate_pool == "candidate":
        candidate_indices = scores.index[scores["herding_candidate"]].to_numpy()
    else:
        raise ValueError(
            "candidate_pool must be one of: 'all', 'interesting', or 'candidate'."
        )

    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(scores))

    if per_class:
        selected_indices = []
        for _, class_scores in scores.loc[candidate_indices].groupby("threshold_label_id"):
            class_candidate_indices = class_scores.index.to_numpy()
            class_count = min(selection_count, len(class_candidate_indices))
            selected_indices.extend(
                kernel_herding_select(
                    X,
                    candidate_indices=class_candidate_indices,
                    target_indices=class_candidate_indices,
                    selection_count=class_count,
                )
            )
    else:
        selected_indices = kernel_herding_select(
            X,
            candidate_indices=candidate_indices,
            target_indices=candidate_indices,
            selection_count=selection_count,
        )

    selected = scores.loc[selected_indices].copy()
    selected.insert(0, "herding_rank", range(1, len(selected) + 1))

    scores["herding_selected"] = False
    scores["herding_rank"] = np.nan
    for rank, index in enumerate(selected_indices, start=1):
        scores.loc[index, "herding_selected"] = True
        scores.loc[index, "herding_rank"] = rank

    if save_csv:
        output_path = Path(input_dir)
        if not output_path.is_absolute():
            output_path = ut.PROJECT_ROOT / output_path
        subset_summary = (
            scores.groupby(["threshold_label_id", "threshold_label_name"], dropna=False)
            .agg(
                total_images=("path", "size"),
                interesting_subset=("herding_candidate", "sum"),
                herding_selected=("herding_selected", "sum"),
            )
            .reset_index()
            .sort_values(["interesting_subset", "herding_selected"], ascending=False)
        )
        subset_indices = scores.index[scores["herding_candidate"]].to_numpy()
        scores.loc[subset_indices].to_csv(output_path / "interesting_subset.csv", index=False)
        selected.to_csv(output_path / "herding_selection.csv", index=False)
        subset_summary.to_csv(output_path / "class_subset_summary.csv", index=False)
        scores.to_csv(output_path / "entropy_scores.csv", index=False)
        print(f"Saved interesting candidate subset -> {output_path / 'interesting_subset.csv'}")
        print(
            f"Saved Kernel Herding selection "
            f"({candidate_pool} pool) -> {output_path / 'herding_selection.csv'}"
        )
        print(f"Saved class subset summary -> {output_path / 'class_subset_summary.csv'}")

    return selected, scores


def save_subset_visualizations(input_dir, scores, set_name="target_subset"):
    output_path = Path(input_dir)
    if not output_path.is_absolute():
        output_path = ut.PROJECT_ROOT / output_path

    X, _, _, _, _, _ = ut.load_DINO_vectors(input_dir)
    subset_mask = scores["herding_candidate"].fillna(False).to_numpy(dtype=bool)
    selected_mask = scores["herding_selected"].fillna(False).to_numpy(dtype=bool)

    if subset_mask.sum() == 0:
        subset_mask = scores["interesting"].fillna(False).to_numpy(dtype=bool)

    if subset_mask.sum() == 0:
        print("No interesting subset found to graph.")
        return

    plt.figure(figsize=(9, 6))
    plt.scatter(
        scores.loc[subset_mask, "best_log_density"],
        scores.loc[subset_mask, "entropy"],
        s=34,
        alpha=0.70,
        label="candidate subset",
    )
    plt.scatter(
        scores.loc[selected_mask, "best_log_density"],
        scores.loc[selected_mask, "entropy"],
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.7,
        label="Kernel Herding selected",
    )
    plt.xlabel("Best NCE log density")
    plt.ylabel("Normalized von Neumann entropy")
    plt.title(f"{set_name}: High-Entropy In-Distribution Poorly Represented Subset")
    plt.legend()
    plt.tight_layout()
    subset_entropy_plot = output_path / f"{set_name}_entropy_density_plot.png"
    plt.savefig(subset_entropy_plot, dpi=160)
    plt.close()

    X_scaled = StandardScaler().fit_transform(X)
    points_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        points_2d[subset_mask, 0],
        points_2d[subset_mask, 1],
        s=34,
        alpha=0.70,
        label="candidate subset",
    )
    plt.scatter(
        points_2d[selected_mask, 0],
        points_2d[selected_mask, 1],
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.7,
        label="Kernel Herding selected",
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"{set_name}: Poorly Represented Subset in DINO Space")
    plt.legend()
    plt.tight_layout()
    subset_pca_plot = output_path / f"{set_name}_pca_plot.png"
    plt.savefig(subset_pca_plot, dpi=160)
    plt.close()

    print(f"Saved subset entropy/density plot -> {subset_entropy_plot}")
    print(f"Saved subset PCA plot -> {subset_pca_plot}")


def save_class_subset_visualizations(input_dir, scores):
    output_path = Path(input_dir)
    if not output_path.is_absolute():
        output_path = ut.PROJECT_ROOT / output_path

    graph_root = output_path / "class_subset_graphs"
    graph_root.mkdir(parents=True, exist_ok=True)

    X, _, _, _, _, _ = ut.load_DINO_vectors(input_dir)
    X_scaled = StandardScaler().fit_transform(X)
    points_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    subset_mask = scores["herding_candidate"].fillna(False).to_numpy(dtype=bool)
    selected_mask = scores["herding_selected"].fillna(False).to_numpy(dtype=bool)
    class_names = scores["threshold_label_name"].fillna("unknown")

    saved_count = 0
    graph_mask = subset_mask | selected_mask
    for class_name in sorted(class_names[graph_mask].unique()):
        all_class_mask = class_names == class_name
        class_mask = subset_mask & (class_names == class_name)
        class_selected_mask = selected_mask & (class_names == class_name)
        if class_mask.sum() == 0 and class_selected_mask.sum() == 0:
            continue

        class_dir = graph_root / safe_folder_name(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        class_scores = scores.loc[all_class_mask]
        entropy_threshold = float(class_scores["entropy_threshold"].iloc[0])
        representation_threshold = float(
            class_scores["representation_gap_threshold"].iloc[0]
        )

        plt.figure(figsize=(8, 5))
        plt.scatter(
            scores.loc[all_class_mask, "representation_gap"],
            scores.loc[all_class_mask, "entropy"],
            s=24,
            alpha=0.25,
            label=f"all {class_name} target images",
        )
        plt.scatter(
            scores.loc[class_mask, "representation_gap"],
            scores.loc[class_mask, "entropy"],
            s=42,
            alpha=0.75,
            label="poorly represented subset",
        )
        plt.scatter(
            scores.loc[class_selected_mask, "representation_gap"],
            scores.loc[class_selected_mask, "entropy"],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.7,
            label="Kernel Herding selected",
        )
        plt.axhline(
            entropy_threshold,
            color="tab:green",
            linestyle="--",
            linewidth=1.2,
            label=f"high vN entropy >= {entropy_threshold:.3f}",
        )
        plt.axvline(
            representation_threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"high rep gap >= {representation_threshold:.3f}",
        )
        plt.xlabel("Representation gap")
        plt.ylabel("Normalized von Neumann entropy")
        plt.title(f"{class_name}: High-Entropy Poorly Represented Candidate Subset")
        plt.legend()
        plt.tight_layout()
        subset_metrics_plot = class_dir / "representation_gap_entropy.png"
        plt.savefig(subset_metrics_plot, dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(
            points_2d[all_class_mask, 0],
            points_2d[all_class_mask, 1],
            s=24,
            alpha=0.25,
            label=f"all {class_name} target images",
        )
        plt.scatter(
            points_2d[class_mask, 0],
            points_2d[class_mask, 1],
            s=42,
            alpha=0.75,
            label="poorly represented subset",
        )
        plt.scatter(
            points_2d[class_selected_mask, 0],
            points_2d[class_selected_mask, 1],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.7,
            label="Kernel Herding selected",
        )
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title(f"{class_name}: Subset in DINO Embedding Space")
        plt.legend()
        plt.tight_layout()
        subset_pca_plot = class_dir / "pca_subset.png"
        plt.savefig(subset_pca_plot, dpi=160)
        plt.close()
        saved_count += 1

    print(f"Saved class subset graph folders -> {graph_root} ({saved_count} classes)")


def save_visualizations(input_dir, scores, set_name="target"):
    output_path = Path(input_dir)
    if not output_path.is_absolute():
        output_path = ut.PROJECT_ROOT / output_path

    X, _, _, _, _, _ = ut.load_DINO_vectors(input_dir)
    if "herding_selected" in scores.columns:
        selected_mask = scores["herding_selected"].fillna(False).to_numpy(dtype=bool)
    else:
        selected_mask = np.zeros(len(scores), dtype=bool)
    interesting_mask = scores["interesting"].fillna(False).to_numpy(dtype=bool)
    outlier_mask = scores["complete_outlier"].fillna(False).to_numpy(dtype=bool)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        scores["best_log_density"],
        scores["entropy"],
        s=18,
        alpha=0.35,
        label="all images",
    )
    plt.scatter(
        scores.loc[interesting_mask, "best_log_density"],
        scores.loc[interesting_mask, "entropy"],
        s=26,
        alpha=0.65,
        label="interesting",
    )
    plt.scatter(
        scores.loc[outlier_mask, "best_log_density"],
        scores.loc[outlier_mask, "entropy"],
        s=36,
        marker="x",
        label="complete outlier",
    )
    plt.scatter(
        scores.loc[selected_mask, "best_log_density"],
        scores.loc[selected_mask, "entropy"],
        s=80,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Kernel Herding selected",
    )
    plt.xlabel("Best NCE log density")
    plt.ylabel("Normalized von Neumann entropy")
    plt.title(f"{set_name}: vN Entropy vs NCE Density")
    plt.legend()
    plt.tight_layout()
    entropy_plot = output_path / f"{set_name}_entropy_density_plot.png"
    plt.savefig(entropy_plot, dpi=160)
    plt.close()

    X_scaled = StandardScaler().fit_transform(X)
    points_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=18, alpha=0.30, label="all images")
    plt.scatter(
        points_2d[interesting_mask, 0],
        points_2d[interesting_mask, 1],
        s=26,
        alpha=0.65,
        label="interesting",
    )
    plt.scatter(
        points_2d[outlier_mask, 0],
        points_2d[outlier_mask, 1],
        s=36,
        marker="x",
        label="complete outlier",
    )
    plt.scatter(
        points_2d[selected_mask, 0],
        points_2d[selected_mask, 1],
        s=80,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Kernel Herding selected",
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"{set_name}: DINO Embedding Space")
    plt.legend()
    plt.tight_layout()
    pca_plot = output_path / f"{set_name}_pca_plot.png"
    plt.savefig(pca_plot, dpi=160)
    plt.close()

    print(f"Saved entropy/density plot -> {entropy_plot}")
    print(f"Saved PCA herding plot -> {pca_plot}")


def print_interesting_summary(scores, top_n=10):
    interesting_scores = scores[scores["herding_candidate"]].sort_values(
        ["entropy", "representation_gap", "best_log_density"],
        ascending=[False, False, False],
    )

    print(f"Interesting candidate subset: {len(interesting_scores)} of {len(scores)}")
    for _, row in interesting_scores.head(top_n).iterrows():
        print(
            f"{row['path']} | predicted={row['predicted_label_name']} "
            f"| vN_entropy={row['entropy']:.3f} "
            f"| class_entropy={row['class_entropy']:.3f} "
            f"| probability={row['class_probability']:.3f} "
            f"| log_density={row['best_log_density']:.1f} "
            f"| representation_gap={row['representation_gap']:.3f}"
        )


def run_entropy_analysis(
    learning_dir="saved_vectors/learning",
    calibration_dir="saved_vectors/testing",
    target_dir="saved_vectors/testing",
    herding_count=3,
    herding_candidate_pool="all",
    herding_per_class=False,
    top_n=10,
):
    model = fit_entropy_model(
        learning_dir=learning_dir,
        calibration_dir=calibration_dir,
    )

    learning_scores = score_directory(model, learning_dir)
    save_visualizations(learning_dir, learning_scores, set_name="learning")

    target_scores = score_directory(model, target_dir)
    selected, target_scores = select_interesting_with_kernel_herding(
        target_dir,
        target_scores,
        selection_count=herding_count,
        per_class=herding_per_class,
        candidate_pool=herding_candidate_pool,
    )
    save_visualizations(target_dir, target_scores, set_name="target")
    save_subset_visualizations(target_dir, target_scores, set_name="target_subset")
    save_class_subset_visualizations(target_dir, target_scores)
    print_interesting_summary(target_scores, top_n=top_n)
    print(f"Kernel Herding selected {len(selected)} images")
    print(f"Kernel Herding candidate pool: {herding_candidate_pool}")
    print(f"Kernel Herding candidates came only from {target_dir}")
    return model, learning_scores, target_scores, selected
