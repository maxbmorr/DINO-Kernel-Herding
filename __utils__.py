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


def save_DINO_vectors(X, Y, paths, data=None):
    output_dir = PROJECT_ROOT / "saved_vectors"
    output_dir.mkdir(exist_ok=True)

    label_ids = Y.cpu().numpy()
    class_names = data or []

    np.save(output_dir / "vectors.npy", X.cpu().numpy())
    np.save(output_dir / "labels.npy", label_ids)

    metadata = pd.DataFrame({
        "label_id": label_ids,
        "label_name": [class_names[label_id] for label_id in label_ids],
        "path": paths
    })
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    class_mapping = pd.DataFrame({
        "label_id": list(range(len(class_names))),
        "label_name": class_names
    })
    class_mapping.to_csv(output_dir / "class_mapping.csv", index=False)
    return


def create_dataset(data_location, bckgnd_rmv = True):
    if bckgnd_rmv == True:
        no_bg_location = "training_data_no_bg"
        rmv_bckgnd(data_location, no_bg_location)
        data_location = no_bg_location 
    X, Y, paths, data = DN.DINO_Vector(data_location)
    save_DINO_vectors(X, Y, paths, data)
    return X, Y, paths, data
