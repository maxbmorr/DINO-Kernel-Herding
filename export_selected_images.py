from pathlib import Path
import re
import shutil

import pandas as pd

import __utils__ as ut


SELECTION_CSV = ut.PROJECT_ROOT / "saved_vectors" / "testing" / "optimization_selection.csv"
OUTPUT_DIR = ut.PROJECT_ROOT / "organized_selected_images"


def _safe_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def export_selected_images(selection_csv=SELECTION_CSV, output_dir=OUTPUT_DIR):
    selection_csv = Path(selection_csv)
    output_dir = Path(output_dir)

    if not selection_csv.exists():
        raise FileNotFoundError(f"Run main.py first. Missing: {selection_csv}")

    selections = pd.read_csv(selection_csv)
    if len(selections) == 0:
        print("No selected images found to export.")
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    copied_count = 0
    missing_count = 0

    for _, row in selections.iterrows():
        source_path = Path(row["path"])
        if not source_path.exists():
            missing_count += 1
            print(f"Missing source image: {source_path}")
            continue

        subset_name = _safe_name(row["subset_label_name"])
        actual_name = _safe_name(row.get("actual_label_name", "unlabeled"))
        best_name = _safe_name(row.get("best_predicted_subset_label_name", "unknown"))
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(exist_ok=True)

        rank = int(row["optimization_rank"])
        probability = float(row["positive_probability"])
        entropy_gain = float(row["von_neumann_entropy_gain"])
        destination_name = (
            f"rank_{rank:02d}"
            f"__p_{probability:.4f}"
            f"__hgain_{entropy_gain:.6f}"
            f"__actual_{actual_name}"
            f"__best_{best_name}"
            f"__{source_path.name}"
        )
        destination_path = subset_dir / destination_name
        shutil.copy2(source_path, destination_path)
        copied_count += 1

    selections.to_csv(output_dir / "optimization_selection_manifest.csv", index=False)

    print(f"Copied {copied_count} selected images into {output_dir}")
    if missing_count:
        print(f"Skipped {missing_count} missing source images")
    print(f"Manifest saved -> {output_dir / 'optimization_selection_manifest.csv'}")

    return output_dir


if __name__ == "__main__":
    export_selected_images()
