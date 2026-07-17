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

def _three_way_stratified_indices(label_ids, test_size, random_state):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    train_indices = []
    retrain_indices = []
    test_indices = []

    for label_id in np.unique(label_ids):
        class_indices = np.where(label_ids == label_id)[0]
        rng.shuffle(class_indices)

        if len(class_indices) == 1:
            train_indices.extend(class_indices)
            continue

        if len(class_indices) == 2:
            train_indices.append(class_indices[0])
            retrain_indices.append(class_indices[1])
            continue

        test_count = max(1, int(round(len(class_indices) * test_size)))
        test_count = min(test_count, len(class_indices) - 2)
        non_test_indices = class_indices[test_count:]
        train_count = len(non_test_indices) // 2

        test_indices.extend(class_indices[:test_count])
        train_indices.extend(non_test_indices[:train_count])
        retrain_indices.extend(non_test_indices[train_count:])

    train_indices = np.array(sorted(train_indices))
    retrain_indices = np.array(sorted(retrain_indices))
    test_indices = np.array(sorted(test_indices))
    return train_indices, retrain_indices, test_indices


def split_DINO_vectors(input_dir="saved_vectors", test_size=0.2, random_state=42):
    X, label_ids, _, _, metadata, class_mapping = load_DINO_vectors(input_dir)
    train_indices, retrain_indices, test_indices = _three_way_stratified_indices(
        label_ids,
        test_size,
        random_state,
    )

    input_dir = Path(input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    train_dir = input_dir / "train"
    retrain_dir = input_dir / "retrain"
    test_dir = input_dir / "test"

    save_vector_split(
        train_dir,
        X[train_indices],
        label_ids[train_indices],
        metadata.iloc[train_indices].reset_index(drop=True),
        class_mapping
    )
    save_vector_split(
        retrain_dir,
        X[retrain_indices],
        label_ids[retrain_indices],
        metadata.iloc[retrain_indices].reset_index(drop=True),
        class_mapping
    )
    save_vector_split(
        test_dir,
        X[test_indices],
        label_ids[test_indices],
        metadata.iloc[test_indices].reset_index(drop=True),
        class_mapping
    )

    print(f"Train split: {len(train_indices)} images -> {train_dir}")
    print(f"Retrain split: {len(retrain_indices)} images -> {retrain_dir}")
    print(f"Test split: {len(test_indices)} images -> {test_dir}")

    return train_indices, retrain_indices, test_indices
