from pathlib import Path
import re

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
GALLERY_PATH = OUTPUT_DIR / "_selection_entropy_top_50_gallery.png"
GALLERY_TABLE_PATH = OUTPUT_DIR / "_selection_entropy_top_50_gallery.csv"
CLASS_GALLERY_DIR = OUTPUT_DIR / "selection_entropy_by_class"
EXAMPLES_PER_GROUP = 3
GALLERY_COUNT = 50


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


def create_selected_gallery(gallery_count=GALLERY_COUNT, columns=10):
    selection = pd.read_csv(SELECTION_PATH)
    retrain_metadata = pd.read_csv(
        ut.PROJECT_ROOT / "saved_vectors" / "retrain" / "metadata.csv"
    )
    selection = selection[selection["von_neumann_entropy_gain"] > 0].copy()
    selection = selection.merge(
        retrain_metadata[["path", "all_label_ids", "all_label_names"]],
        on="path", how="left",
    )
    selection = selection.sort_values(
        ["von_neumann_entropy_gain", "optimization_rank"],
        ascending=[False, True],
    )

    # Round-robin across classes so one class cannot dominate the gallery.
    groups = {
        name: group.reset_index(drop=True)
        for name, group in selection.groupby("subset_label_name", sort=True)
    }
    chosen = []
    used_paths = set()
    depth = 0
    while len(chosen) < gallery_count:
        added = False
        for group in groups.values():
            if depth >= len(group):
                continue
            row = group.iloc[depth]
            if row["path"] not in used_paths:
                chosen.append(row)
                used_paths.add(row["path"])
                added = True
                if len(chosen) == gallery_count:
                    break
        if not added and all(depth + 1 >= len(group) for group in groups.values()):
            break
        depth += 1
    gallery = pd.DataFrame(chosen).reset_index(drop=True)
    gallery["target_class_present"] = gallery.apply(
        lambda row: str(int(row["subset_label_id"]))
        in str(row.get("all_label_ids", "")).split("|"),
        axis=1,
    )
    gallery.insert(0, "gallery_position", np.arange(1, len(gallery) + 1))
    gallery.to_csv(GALLERY_TABLE_PATH, index=False)

    rows = int(np.ceil(len(gallery) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.25 * columns, 2.65 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, item in zip(axes, gallery.itertuples(index=False)):
        axis.imshow(_open_square(item.path, size=300))
        present = bool(item.target_class_present)
        border = "#2e8b57" if present else "#d1495b"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor(border)
        axis.set_title(
            f"{item.subset_label_name} · rank {int(item.optimization_rank)}\n"
            rf"$\Delta H={item.von_neumann_entropy_gain:.1e}$ · "
            rf"$\hat P_+={item.positive_probability:.2f}$",
            fontsize=7.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes[len(gallery):]:
        axis.axis("off")

    figure.suptitle(
        f"Top {len(gallery)} unique selected images with positive entropy gain\n"
        "Green border: target present after labeling · Red border: target absent",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(GALLERY_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved selected-image gallery -> {GALLERY_PATH}")
    return GALLERY_PATH


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "class"


def create_all_selected_galleries(columns=10):
    selection = pd.read_csv(SELECTION_PATH)
    retrain_metadata = pd.read_csv(
        ut.PROJECT_ROOT / "saved_vectors" / "retrain" / "metadata.csv"
    )
    selection = selection.merge(
        retrain_metadata[["path", "all_label_ids", "all_label_names"]],
        on="path", how="left",
    )
    selection["target_class_present"] = selection.apply(
        lambda row: str(int(row["subset_label_id"]))
        in str(row.get("all_label_ids", "")).split("|"),
        axis=1,
    )
    CLASS_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []

    for class_name, class_data in selection.groupby("subset_label_name", sort=True):
        class_data = class_data.sort_values("optimization_rank").reset_index(drop=True)
        class_data.insert(0, "gallery_position", np.arange(1, len(class_data) + 1))
        safe_name = _safe_filename(class_name)
        table_path = CLASS_GALLERY_DIR / f"{safe_name}_all_selected.csv"
        image_path = CLASS_GALLERY_DIR / f"{safe_name}_all_selected.png"
        class_data.to_csv(table_path, index=False)

        rows = int(np.ceil(len(class_data) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(2.2 * columns, 2.55 * rows), squeeze=False
        )
        axes = axes.ravel()
        for axis, item in zip(axes, class_data.itertuples(index=False)):
            axis.imshow(_open_square(item.path, size=280))
            present = bool(item.target_class_present)
            border = "#2e8b57" if present else "#d1495b"
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                spine.set_edgecolor(border)
            gain_color = "#2e8b57" if item.objective_gain >= 0 else "#d1495b"
            axis.set_title(
                f"rank {int(item.optimization_rank)}\n"
                rf"$\Delta H={item.von_neumann_entropy_gain:.1e}$ · "
                rf"$\hat P_+={item.positive_probability:.2f}$",
                fontsize=7.2, color=gain_color,
            )
            axis.set_xticks([])
            axis.set_yticks([])
        for axis in axes[len(class_data):]:
            axis.axis("off")

        positives = int(class_data["target_class_present"].sum())
        figure.suptitle(
            f"{class_name}: all {len(class_data)} selected images "
            f"({positives} target-present, {len(class_data) - positives} target-absent)\n"
            "Green border: target present · Red border: target absent · "
            "Title color: positive/negative net gain",
            fontsize=14,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.965))
        figure.savefig(image_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        outputs.append(image_path)
        print(f"Saved complete class gallery -> {image_path}")
    return outputs


if __name__ == "__main__":
    create_entropy_image_examples()
    create_selected_gallery()
    create_all_selected_galleries()
