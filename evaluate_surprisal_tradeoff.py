from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler

import __utils__ as ut
import optimization as opt


LEARNING_DIR = ut.PROJECT_ROOT / "saved_vectors" / "learning"
TARGET_DIR = ut.PROJECT_ROOT / "saved_vectors" / "testing"
PROBABILITY_SCORES_PATH = TARGET_DIR / "subset_probability_scores.csv"
PROBABILITY_MATRIX_PATH = TARGET_DIR / "subset_probability_matrix.csv"
OUTPUT_DIR = ut.PROJECT_ROOT / "_surprisal_tradeoff_evaluation"

# An empty target list evaluates every class represented in the calibrated matrix.
TARGET_CLASSES = []
SURPRISAL_LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.025]
SELECTION_COUNT = 4
KERNEL = "rbf"
STOP_WHEN_OBJECTIVE_DECREASES = True
USE_KERNEL_HERDED_REFERENCES = True
MAX_LABELED_REFERENCE_PER_SUBSET = 30
CONTACT_SHEET_THUMB_SIZE = (220, 160)


def _safe_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _lambda_name(value):
    return f"lambda_{value:g}".replace(".", "p")


def _load_inputs():
    if not PROBABILITY_SCORES_PATH.exists() or not PROBABILITY_MATRIX_PATH.exists():
        raise FileNotFoundError(
            "Run main.py first so subset_probability_scores.csv and "
            "subset_probability_matrix.csv exist."
        )

    X_learning, learning_label_ids, _, _, learning_metadata, class_mapping = ut.load_DINO_vectors(
        LEARNING_DIR
    )
    X_target, _, _, _, target_metadata, _ = ut.load_DINO_vectors(TARGET_DIR)
    probability_scores = pd.read_csv(PROBABILITY_SCORES_PATH)
    probability_matrix = pd.read_csv(PROBABILITY_MATRIX_PATH)

    if len(probability_scores) != len(X_target):
        raise ValueError("subset_probability_scores.csv length does not match target vectors.")
    if len(probability_matrix) != len(X_target):
        raise ValueError("subset_probability_matrix.csv length does not match target vectors.")

    labeled_mask = learning_label_ids >= 0
    X_learning = X_learning[labeled_mask]
    learning_label_ids = learning_label_ids[labeled_mask]
    learning_metadata = learning_metadata.loc[labeled_mask].reset_index(drop=True)

    scaler = StandardScaler()
    X_learning_scaled = scaler.fit_transform(X_learning)
    X_target_scaled = scaler.transform(X_target)

    return {
        "X_learning": X_learning_scaled,
        "learning_label_ids": learning_label_ids,
        "learning_metadata": learning_metadata,
        "X_target": X_target_scaled,
        "target_metadata": target_metadata,
        "class_mapping": class_mapping,
        "probability_scores": probability_scores,
        "probability_matrix": probability_matrix,
        "gamma": 1.0 / X_learning_scaled.shape[1],
    }


def _selected_classes(class_mapping, probability_matrix):
    rows = []
    requested = {name.lower() for name in TARGET_CLASSES}

    for _, class_row in class_mapping.iterrows():
        class_name = class_row["label_name"]
        probability_column = f"{class_name}_positive_probability"
        if probability_column not in probability_matrix.columns:
            continue
        if TARGET_CLASSES and class_name.lower() not in requested:
            continue
        rows.append(class_row)

    return rows


def _build_contact_sheet(selection_rows, output_path):
    if not selection_rows:
        return

    thumb_width, thumb_height = CONTACT_SHEET_THUMB_SIZE
    label_height = 74
    columns = min(4, len(selection_rows))
    rows = int(np.ceil(len(selection_rows) / columns))
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, row in enumerate(selection_rows):
        source_path = Path(row["path"])
        col = index % columns
        grid_row = index // columns
        x0 = col * thumb_width
        y0 = grid_row * (thumb_height + label_height)

        try:
            image = Image.open(source_path).convert("RGB")
            image.thumbnail((thumb_width, thumb_height))
            image_x = x0 + (thumb_width - image.width) // 2
            image_y = y0 + (thumb_height - image.height) // 2
            sheet.paste(image, (image_x, image_y))
        except Exception:
            draw.rectangle([x0, y0, x0 + thumb_width, y0 + thumb_height], outline="red")
            draw.text((x0 + 6, y0 + 6), "image load failed", fill="red", font=font)

        label_y = y0 + thumb_height + 4
        label_lines = [
            f"rank {int(row['optimization_rank'])}",
            f"P+ {float(row['positive_probability']):.3f}",
            f"Hgain {float(row['von_neumann_entropy_gain']):.5f}",
            f"actual {row.get('actual_label_name', 'unlabeled')}",
        ]
        draw.multiline_text((x0 + 6, label_y), "\n".join(label_lines), fill="black", font=font)

    sheet.save(output_path)


def _copy_selected_images(selection_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in selection_rows:
        source_path = Path(row["path"])
        if not source_path.exists():
            print(f"Missing source image: {source_path}")
            continue

        destination_name = (
            f"rank_{int(row['optimization_rank']):02d}"
            f"__p_{float(row['positive_probability']):.4f}"
            f"__surprisal_{float(row['surprisal']):.4f}"
            f"__hgain_{float(row['von_neumann_entropy_gain']):.6f}"
            f"__actual_{_safe_name(row.get('actual_label_name', 'unlabeled'))}"
            f"__{source_path.name}"
        )
        shutil.copy2(source_path, output_dir / destination_name)


def _evaluate_one_class(inputs, class_row, surprisal_lambda):
    label_id = int(class_row["label_id"])
    class_name = class_row["label_name"]
    probability_column = f"{class_name}_positive_probability"
    class_mask = opt._metadata_has_label(inputs["learning_metadata"], label_id)

    if class_mask.sum() == 0:
        return [], None

    class_reference = opt._reference_subset(
        inputs["X_learning"][class_mask],
        MAX_LABELED_REFERENCE_PER_SUBSET,
        method=("kernel_herding" if USE_KERNEL_HERDED_REFERENCES else "all"),
        kernel=KERNEL,
        gamma=inputs["gamma"],
    )
    base_entropy = opt.von_neumann_entropy(
        class_reference,
        kernel=KERNEL,
        gamma=inputs["gamma"],
    )

    candidate_indices = np.arange(len(inputs["X_target"]))
    candidate_probabilities = inputs["probability_matrix"][probability_column].to_numpy()
    selected = opt.greedy_maximize_von_neumann_entropy(
        X_labeled_class=class_reference,
        X_candidates=inputs["X_target"],
        candidate_indices=candidate_indices,
        candidate_positive_probabilities=candidate_probabilities,
        selection_count=SELECTION_COUNT,
        kernel=KERNEL,
        gamma=inputs["gamma"],
        probability_lambda=surprisal_lambda,
        stop_when_objective_decreases=STOP_WHEN_OBJECTIVE_DECREASES,
    )

    rows = []
    for selected_item in selected:
        target_index = selected_item["target_index"]
        metadata_row = inputs["target_metadata"].iloc[target_index].to_dict()
        score_row = inputs["probability_scores"].loc[target_index].to_dict()
        positive_probability = float(
            inputs["probability_matrix"].loc[target_index, probability_column]
        )
        surprisal = -float(np.log(positive_probability + 1e-12))

        rows.append({
            "subset_label_id": label_id,
            "subset_label_name": class_name,
            "surprisal_lambda": surprisal_lambda,
            "path": metadata_row["path"],
            "optimization_rank": selected_item["optimization_rank"],
            "kernel": KERNEL,
            "reference_method": (
                "kernel_herding" if USE_KERNEL_HERDED_REFERENCES else "all"
            ),
            "stop_when_objective_decreases": STOP_WHEN_OBJECTIVE_DECREASES,
            "objective": selected_item["objective"],
            "objective_gain": selected_item["objective_gain"],
            "base_von_neumann_entropy": base_entropy,
            "von_neumann_entropy": selected_item["von_neumann_entropy"],
            "von_neumann_entropy_gain": selected_item["von_neumann_entropy_gain"],
            "positive_probability": positive_probability,
            "surprisal": surprisal,
            "surprisal_sum": -selected_item["log_probability_sum"],
            "best_predicted_subset_label_name": score_row["predicted_subset_label_name"],
            "best_predicted_subset_probability": score_row["positive_probability"],
            "actual_label_id": metadata_row.get("label_id", -1),
            "actual_label_name": metadata_row.get("label_name", "unlabeled"),
        })

    summary = {
        "subset_label_id": label_id,
        "subset_label_name": class_name,
        "surprisal_lambda": surprisal_lambda,
        "reference_method": (
            "kernel_herding" if USE_KERNEL_HERDED_REFERENCES else "all"
        ),
        "selected_count": len(rows),
        "labeled_reference_count": int(class_mask.sum()),
        "candidate_count": len(candidate_indices),
        "base_von_neumann_entropy": base_entropy,
        "final_von_neumann_entropy": (
            rows[-1]["von_neumann_entropy"] if rows else base_entropy
        ),
        "final_objective": rows[-1]["objective"] if rows else base_entropy,
        "surprisal_sum": rows[-1]["surprisal_sum"] if rows else 0.0,
    }
    return rows, summary


def run_surprisal_tradeoff_evaluation():
    inputs = _load_inputs()
    class_rows = _selected_classes(inputs["class_mapping"], inputs["probability_matrix"])

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Evaluating {len(class_rows)} target classes across "
        f"{len(SURPRISAL_LAMBDAS)} surprisal trade-off values"
    )
    print(
        "Objective: H(labeled subset union S) "
        "- lambda * sum_{x in S}[-log(P_hat_+(x))]"
    )

    all_rows = []
    summary_rows = []

    for class_number, class_row in enumerate(class_rows, start=1):
        class_name = class_row["label_name"]
        class_dir = OUTPUT_DIR / _safe_name(class_name)
        print(f"[Class {class_number}/{len(class_rows)}] {class_name}")

        for lambda_number, surprisal_lambda in enumerate(SURPRISAL_LAMBDAS, start=1):
            print(
                f"  [lambda {lambda_number}/{len(SURPRISAL_LAMBDAS)}] "
                f"{surprisal_lambda:g}"
            )
            rows, summary = _evaluate_one_class(inputs, class_row, surprisal_lambda)
            all_rows.extend(rows)
            if summary is not None:
                summary_rows.append(summary)

            lambda_dir = class_dir / _lambda_name(surprisal_lambda)
            _copy_selected_images(rows, lambda_dir)
            pd.DataFrame(rows).to_csv(lambda_dir / "selection.csv", index=False)
            _build_contact_sheet(rows, lambda_dir / "contact_sheet.jpg")

    selections = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summary_rows)
    selections.to_csv(OUTPUT_DIR / "all_tradeoff_selections.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "tradeoff_summary.csv", index=False)

    lambda_summary = (
        summary.groupby("surprisal_lambda", as_index=False)
        .agg(
            total_selected=("selected_count", "sum"),
            classes_with_selections=(
                "selected_count",
                lambda values: int((values > 0).sum()),
            ),
            mean_selected_per_class=("selected_count", "mean"),
            median_selected_per_class=("selected_count", "median"),
            mean_final_objective=("final_objective", "mean"),
            mean_final_entropy=("final_von_neumann_entropy", "mean"),
        )
        .sort_values("surprisal_lambda")
    )
    lambda_summary.to_csv(OUTPUT_DIR / "lambda_summary.csv", index=False)

    print(f"Saved trade-off evaluation -> {OUTPUT_DIR}")
    print(f"Selected {len(selections)} images across all runs")
    print(lambda_summary.to_string(index=False))
    return selections, summary


if __name__ == "__main__":
    run_surprisal_tradeoff_evaluation()
