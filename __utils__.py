from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
os.environ.setdefault("U2NET_HOME", str(PROJECT_ROOT / ".u2net"))

from PIL import Image
from rembg import remove 
import numpy as np
import pandas as pd
import DINO as DN

def rmv_bckgnd(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    if not input_root.is_absolute():
        input_root = PROJECT_ROOT / input_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    saved_paths = []

    for image_path in input_root.rglob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue
        relative_path = image_path.relative_to(input_root)
        output_path = output_root / relative_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(image_path).convert("RGBA")
            no_bg = remove(img)
            no_bg.save(output_path)
            saved_paths.append(output_path)
            print("Saved:", output_path)
        except Exception as e:
            print("Failed", image_path)
            print(e)

    if not saved_paths:
        raise FileNotFoundError(
            f"No images found in {input_root}. Expected files ending in: "
            f"{', '.join(sorted(image_extensions))}"
        )

    return output_root


def save_DINO_vectors(X, Y, paths, data=None, metadata=None):
    output_dir = PROJECT_ROOT / "saved_vectors"
    output_dir.mkdir(exist_ok=True)

    label_ids = Y.cpu().numpy()
    class_names = data or []
    label_names = [
        class_names[label_id] if 0 <= label_id < len(class_names) else "unlabeled"
        for label_id in label_ids
    ]

    np.save(output_dir / "vectors.npy", X.cpu().numpy())
    np.save(output_dir / "labels.npy", label_ids)

    metadata_frame = pd.DataFrame({
        "label_id": label_ids,
        "label_name": label_names,
        "path": paths
    })

    if metadata is not None:
        metadata_frame = pd.concat(
            [metadata_frame, pd.DataFrame(metadata)],
            axis = 1
        )

    metadata_frame.to_csv(output_dir / "metadata.csv", index=False)

    class_mapping = pd.DataFrame({
        "label_id": list(range(len(class_names))),
        "label_name": class_names
    })
    class_mapping.to_csv(output_dir / "class_mapping.csv", index=False)
    return


def create_dataset(data_location, bckgnd_rmv = True, annotation_path = None):
    data_location = Path(data_location)
    if not data_location.is_absolute():
        data_location = PROJECT_ROOT / data_location

    if annotation_path is not None:
        annotation_path = Path(annotation_path)
        if not annotation_path.is_absolute():
            annotation_path = PROJECT_ROOT / annotation_path

    if bckgnd_rmv == True and annotation_path is not None:
        raise ValueError("COCO annotation labels should be used with bckgnd_rmv=False.")

    if bckgnd_rmv == True:
        no_bg_location = "training_data_no_bg"
        rmv_bckgnd(data_location, no_bg_location)
        data_location = no_bg_location 
    X, Y, paths, data, metadata = DN.DINO_Vector(
        data_location,
        annotation_path = annotation_path,
        return_metadata = True
    )
    save_DINO_vectors(X, Y, paths, data, metadata)
    return X, Y, paths, data

def load_DINO_vectors(input_dir = "saved_vectors"):
    input_dir = Path(input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    X = np.load(input_dir / "vectors.npy")
    label_ids = np.load(input_dir / "labels.npy")
    metadata = pd.read_csv(input_dir / "metadata.csv")
    class_mapping = pd.read_csv(input_dir / "class_mapping.csv")
    label_names = metadata["label_name"].values
    paths = metadata["path"].values
    return X, label_ids, label_names, paths, metadata, class_mapping

def save_vector_split(output_dir, X, label_ids, metadata, class_mapping):
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "vectors.npy", X)
    np.save(output_dir / "labels.npy", label_ids)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    class_mapping.to_csv(output_dir / "class_mapping.csv", index=False)

def split_DINO_vectors_with_multilabel_training(
    input_dir="saved_vectors",
    selected_class_names=None,
    min_samples_per_class=300,
    samples_per_class=1000,
    random_state=42,
):
    """Build a leakage-free split with a multi-label quota per training class."""
    if not selected_class_names:
        raise ValueError("selected_class_names must contain at least one class.")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")
    if not 0 < min_samples_per_class <= samples_per_class:
        raise ValueError(
            "min_samples_per_class must be positive and no greater than "
            "samples_per_class."
        )

    X, label_ids, _, _, metadata, class_mapping = load_DINO_vectors(input_dir)
    if "all_label_names" not in metadata.columns:
        raise ValueError("Multi-label sampling requires metadata.all_label_names.")

    label_sets = metadata["all_label_names"].fillna("").astype(str).map(
        lambda value: set(value.split("|")) if value else set()
    )
    rng = np.random.default_rng(random_state)
    sampled_by_class = {}
    training_indices = set()
    selected_name_set = set(selected_class_names)
    eligible_by_class = {
        class_name: np.array(
            [
                index
                for index, labels in enumerate(label_sets)
                if class_name in labels
                and len(labels & selected_name_set) == 1
            ],
            dtype=int,
        )
        for class_name in selected_class_names
    }
    empty_classes = [
        class_name
        for class_name, eligible in eligible_by_class.items()
        if len(eligible) == 0
    ]
    if empty_classes:
        raise ValueError(
            "Selected classes have no eligible images: " + ", ".join(empty_classes)
        )
    balanced_sample_count = min(
        samples_per_class,
        min(len(eligible) for eligible in eligible_by_class.values()),
    )
    if balanced_sample_count < min_samples_per_class:
        raise ValueError(
            f"The selected class set supports only {balanced_sample_count} "
            f"balanced images per class; at least {min_samples_per_class} "
            "are required."
        )

    for class_name, eligible in eligible_by_class.items():
        chosen = rng.choice(
            eligible, size=balanced_sample_count, replace=False
        )
        sampled_by_class[class_name] = chosen
        training_indices.update(chosen.tolist())

    train_indices = np.array(sorted(training_indices), dtype=int)
    remaining_mask = np.ones(len(X), dtype=bool)
    remaining_mask[train_indices] = False
    remaining_indices = np.flatnonzero(remaining_mask)

    retrain_count = min(2 * len(train_indices), len(remaining_indices))
    target_test_count = min(
        2 * retrain_count,
        len(remaining_indices) - retrain_count,
    )
    retrain_indices = np.sort(
        rng.choice(remaining_indices, size=retrain_count, replace=False)
    )
    retrain_mask = np.zeros(len(X), dtype=bool)
    retrain_mask[retrain_indices] = True
    test_pool = remaining_indices[~retrain_mask[remaining_indices]]
    test_indices = np.sort(
        rng.choice(test_pool, size=target_test_count, replace=False)
    )
    unused_count = len(test_pool) - len(test_indices)
    actual_test_positive = sum(
        bool(label_sets.iloc[index] & selected_name_set)
        for index in test_indices
    )

    input_dir = Path(input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    for split_name, indices in (
        ("train", train_indices),
        ("retrain", retrain_indices),
        ("test", test_indices),
    ):
        save_vector_split(
            input_dir / split_name,
            X[indices],
            label_ids[indices],
            metadata.iloc[indices].reset_index(drop=True),
            class_mapping,
        )

    print(
        f"Multi-label train split: {len(train_indices)} unique images "
        f"using {balanced_sample_count} images per class "
        f"(balanced, cap={samples_per_class})"
    )
    for class_name, indices in sampled_by_class.items():
        print(f"  {class_name}: {len(indices)} sampled images")
    print(f"Random retrain split: {len(retrain_indices)} images (2x training)")
    print(f"Random test split: {len(test_indices)} images (2x retraining)")
    print(
        f"Test images containing a selected class: {actual_test_positive}/"
        f"{len(test_indices)} ({actual_test_positive / len(test_indices):.1%})"
    )
    print(f"Unused images: {unused_count}")
    return train_indices, retrain_indices, test_indices
