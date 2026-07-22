import random
import secrets

import pandas as pd

import __utils__ as ut
import dual_model_calibration as dmc
import evaluate_dual_model_auc as dual_auc
import export_calibrated_images as calibrated_exporter
import export_selected_images as exporter
import hyperparameter_sweep as hp_sweep
import optimization as opt
import subset_probability as sp


# Retained experiment switches
CREATE_DINO_DATASET = False

# COCO input
COCO_IMAGE_DIR = "coco/train2017"
COCO_ANNOTATION_PATH = "coco/annotations/instances_train2017.json"

# Proof-of-concept class subset. Each run records its generated seed.
CLASS_COUNT = 5

# Fixed proof-of-concept sampling cap (classes with fewer images use all of them).
_MIN_TRAIN_IMAGES_PER_CLASS = 300
_MAX_TRAIN_IMAGES_PER_CLASS = 1000
_MAX_CLASS_SELECTION_ATTEMPTS = 1000

# Labeled reference representation
USE_KERNEL_HERDED_REFERENCES = True
KERNEL_HERDED_REFERENCE_COUNT = 100

# Eigenvalue calculation
# Direct LAPACK is faster at the current reference size. Secular mode is
# experimental and automatically falls back to direct decomposition on failure.
USE_SECULAR_EIGENVALUE_UPDATES = False


def _reference_method():
    return "kernel_herding" if USE_KERNEL_HERDED_REFERENCES else "all"


def _eigenvalue_method():
    return "secular" if USE_SECULAR_EIGENVALUE_UPDATES else "direct"


def _random_selected_classes():
    metadata = pd.read_csv(ut.PROJECT_ROOT / "saved_vectors" / "metadata.csv")
    label_lists = metadata["all_label_names"].fillna("").astype(str).str.split("|")
    counts = {}
    for labels in label_lists:
        for class_name in set(labels):
            if class_name:
                counts[class_name] = counts.get(class_name, 0) + 1
    available_classes = sorted(
        name
        for name, count in counts.items()
        if count >= _MIN_TRAIN_IMAGES_PER_CLASS
    )
    if CLASS_COUNT > len(available_classes):
        raise ValueError(
            f"CLASS_COUNT={CLASS_COUNT} exceeds the "
            f"{len(available_classes)} available classes."
        )
    selection_seed = secrets.randbits(32)
    rng = random.Random(selection_seed)
    for attempt in range(1, _MAX_CLASS_SELECTION_ATTEMPTS + 1):
        selected = rng.sample(available_classes, CLASS_COUNT)
        selected_set = set(selected)
        usable_counts = {
            class_name: sum(
                class_name in labels and len(set(labels) & selected_set) == 1
                for labels in label_lists
            )
            for class_name in selected
        }
        if min(usable_counts.values()) >= _MIN_TRAIN_IMAGES_PER_CLASS:
            break
    else:
        raise RuntimeError(
            "Could not find a random class set with at least "
            f"{_MIN_TRAIN_IMAGES_PER_CLASS} usable images per class after "
            f"{_MAX_CLASS_SELECTION_ATTEMPTS} attempts."
        )
    selection_manifest = pd.DataFrame({
        "random_seed": [selection_seed] * len(selected),
        "label_name": selected,
    })
    selection_manifest.to_csv(
        ut.PROJECT_ROOT / "saved_vectors" / "selected_classes.csv", index=False
    )
    print(
        f"Selected proof-of-concept classes (seed={selection_seed}, "
        f"attempt={attempt}):"
    )
    print(", ".join(selected))
    print(
        "Usable images before balancing: "
        + ", ".join(
            f"{class_name}={usable_counts[class_name]}"
            for class_name in selected
        )
    )
    return selected, selection_seed


def run_pipeline():
    if CREATE_DINO_DATASET:
        vectors, labels, paths, classes = ut.create_dataset(
            data_location=COCO_IMAGE_DIR,
            annotation_path=COCO_ANNOTATION_PATH,
            bckgnd_rmv=False,
        )
        print("Vector shape:", vectors.shape)
        print("Label shape:", labels.shape)
        print("Class count:", len(classes))
        print("Image count:", len(paths))

    selected_classes, selection_seed = _random_selected_classes()
    ut.split_DINO_vectors_with_multilabel_training(
        "saved_vectors",
        selected_class_names=selected_classes,
        min_samples_per_class=_MIN_TRAIN_IMAGES_PER_CLASS,
        samples_per_class=_MAX_TRAIN_IMAGES_PER_CLASS,
        random_state=selection_seed,
    )

    best, _, _ = hp_sweep.run_hyperparameter_sweep("saved_vectors/train")
    fixed_c = float(best["C"])
    probability_lambda = float(best["lambda"])
    selected_data_weight = float(best["selected_data_weight"])
    print(
        "Applying selected hyperparameters: "
        f"lambda={probability_lambda:g}, "
        f"trust={selected_data_weight:g}, C={fixed_c:g}"
    )

    baseline_model, _, _ = sp.score_directory(
        learning_dir="saved_vectors/train",
        target_dir="saved_vectors/retrain",
        tune_regularization=False,
        fixed_c=fixed_c,
        use_hard_negative_mining=False,
        selected_class_names=selected_classes,
    )

    opt.optimize_subset_selection(
        learning_dir="saved_vectors/train",
        target_dir="saved_vectors/retrain",
        selection_count=None,
        probability_lambda=probability_lambda,
        kernel="rbf",
        stop_when_objective_decreases=True,
        reference_method=_reference_method(),
        max_labeled_reference_per_subset=KERNEL_HERDED_REFERENCE_COUNT,
        eigenvalue_method=_eigenvalue_method(),
        selected_class_names=selected_classes,
    )

    model_0, model_1 = dmc.calibrate_baseline_and_augmented_models(
        training_dir="saved_vectors/train",
        retraining_dir="saved_vectors/retrain",
        tune_regularization=False,
        fixed_c=fixed_c,
        use_hard_negative_mining=False,
        baseline_model=baseline_model,
        selected_class_names=selected_classes,
        selected_data_weight=selected_data_weight,
    )
    calibrated_exporter.export_calibrated_test_images(
        model_0,
        model_1,
        test_dir="saved_vectors/test",
    )
    dual_auc.evaluate_dual_model_auc(
        test_dir="saved_vectors/test",
    )

    exporter.export_selected_images()


if __name__ == "__main__":
    run_pipeline()
