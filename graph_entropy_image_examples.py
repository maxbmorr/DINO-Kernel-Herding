from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler

import __utils__ as ut
import optimization as opt


TRAIN_DIR = ut.PROJECT_ROOT / "saved_vectors" / "train"
SELECTION_PATH = (
    ut.PROJECT_ROOT / "saved_vectors" / "retrain" / "optimization_selection.csv"
)
OUTPUT_DIR = ut.PROJECT_ROOT / "_organized_calibrated_images" / "AUC_evaluation"
OUTPUT_PATH = OUTPUT_DIR / "_selection_entropy_image_examples.png"
TABLE_PATH = OUTPUT_DIR / "_selection_entropy_image_examples.csv"
EXAMPLES_PER_GROUP = 3


def _open_square(path, size=420):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def _has_label(metadata, label_id):
    token = str(int(label_id))
    return metadata["all_label_ids"].fillna("").astype(str).map(
        lambda value: token in value.split("|")
    ).to_numpy()


def create_entropy_image_examples(examples_per_group=EXAMPLES_PER_GROUP):
    vectors = np.load(TRAIN_DIR / "vectors.npy")
    metadata = pd.read_csv(TRAIN_DIR / "metadata.csv")
    selection = pd.read_csv(SELECTION_PATH)
    if len(vectors) != len(metadata):
        raise ValueError("Training vectors and metadata are not aligned.")

    selection = selection[selection["von_neumann_entropy_gain"] > 0].copy()
    class_rows = (
        selection[["subset_label_id", "subset_label_name"]]
        .drop_duplicates()
        .sort_values("subset_label_name")
    )
    if class_rows.empty:
        raise ValueError("No selected images have positive entropy gain.")

    scaled = StandardScaler().fit_transform(vectors)
    gamma = 1.0 / scaled.shape[1]
    records = []
    figure, axes = plt.subplots(
        len(class_rows), examples_per_group * 2,
        figsize=(3.2 * examples_per_group * 2, 3.65 * len(class_rows)),
        squeeze=False,
    )

    for row_index, class_row in enumerate(class_rows.itertuples(index=False)):
        label_id = int(class_row.subset_label_id)
        class_name = str(class_row.subset_label_name)
        label_positions = np.flatnonzero(_has_label(metadata, label_id))
        herd_count = min(examples_per_group, len(label_positions))
        herded = opt.kernel_herd_reference_subset(
            scaled[label_positions], herd_count, "rbf", gamma
        )
        # Map the returned vectors back to their aligned metadata positions.
        reference_positions = []
        for vector in herded:
            local = int(np.argmin(np.linalg.norm(scaled[label_positions] - vector, axis=1)))
            reference_positions.append(int(label_positions[local]))

        selected_rows = (
            selection[selection["subset_label_id"] == label_id]
            .sort_values(
                ["von_neumann_entropy_gain", "optimization_rank"],
                ascending=[False, True],
            )
            .head(examples_per_group)
        )

        for column, position in enumerate(reference_positions):
            item = metadata.iloc[position]
            axis = axes[row_index, column]
            axis.imshow(_open_square(item["path"]))
            axis.set_title(
                f"Labeled reference {column + 1}\n{item['all_label_names']}",
                fontsize=9,
            )
            records.append({
                "class_id": label_id, "class_name": class_name,
                "group": "labeled_reference", "display_order": column + 1,
                "path": item["path"], "all_label_names": item["all_label_names"],
                "optimization_rank": np.nan, "positive_probability": np.nan,
                "entropy_gain": np.nan, "objective_gain": np.nan,
            })

        for offset, selected_row in enumerate(selected_rows.itertuples(index=False)):
            column = examples_per_group + offset
            axis = axes[row_index, column]
            axis.imshow(_open_square(selected_row.path))
            axis.set_title(
                f"Selected rank {int(selected_row.optimization_rank)}\n"
                rf"$\Delta H={selected_row.von_neumann_entropy_gain:.2e}$, "
                rf"$\hat P_+={selected_row.positive_probability:.3f}$",
                fontsize=9,
            )
            records.append({
                "class_id": label_id, "class_name": class_name,
                "group": "selected_positive_entropy_gain",
                "display_order": offset + 1, "path": selected_row.path,
                "all_label_names": getattr(selected_row, "all_label_names", ""),
                "optimization_rank": selected_row.optimization_rank,
                "positive_probability": selected_row.positive_probability,
                "entropy_gain": selected_row.von_neumann_entropy_gain,
                "objective_gain": selected_row.objective_gain,
            })

        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(class_name, fontsize=12, fontweight="bold")

    for column in range(examples_per_group):
        axes[0, column].text(
            0.5, 1.22, "Known labeled references", transform=axes[0, column].transAxes,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
        axes[0, examples_per_group + column].text(
            0.5, 1.22, "Selected unlabeled candidates", transform=axes[0, examples_per_group + column].transAxes,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    figure.suptitle(
        "Labeled references contrasted with selections that increase entropy",
        fontsize=16, y=1.01,
    )
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(records).to_csv(TABLE_PATH, index=False)
    print(f"Saved entropy image examples -> {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    create_entropy_image_examples()
