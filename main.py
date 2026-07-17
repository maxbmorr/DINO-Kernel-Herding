import random

import __utils__ as ut
import dual_model_calibration as dmc
import embedding_classifier as ec
import evaluate_lambda_selection_accuracy as lambda_accuracy_eval
import evaluate_surprisal_tradeoff as tradeoff_eval
import export_calibrated_images as calibrated_exporter
import export_selected_images as exporter
import optimization as opt
import subset_probability as sp


# Pipeline stages
CREATE_DINO_DATASET = True
SPLIT_DINO_DATASET = True
RUN_EMBEDDING_CLASSIFIER = True
RUN_SUBSET_PROBABILITY = True
RUN_OPTIMIZATION = True
RUN_EXPORT_SELECTED_IMAGES = True
RUN_SURPRISAL_TRADEOFF_EVALUATION = False
RUN_LAMBDA_SELECTION_ACCURACY_EVALUATION = False

# COCO input
COCO_IMAGE_DIR = "coco/val2017"
COCO_ANNOTATION_PATH = "coco/annotations/instances_val2017.json"
REMOVE_IMAGE_BACKGROUNDS = False

# Three-way split: reserve test, then divide the remainder into train/retrain
TEST_SPLIT_SIZE = 0.2

# Probability classifier
TUNE_CLASSIFIER_REGULARIZATION = False
FIXED_CLASSIFIER_C = 1.0
USE_HARD_NEGATIVE_MINING = True
HARD_NEGATIVE_FRACTION = 0.2
HARD_NEGATIVE_WEIGHT = 5.0

# Subset optimization
OPTIMIZATION_SELECTION_COUNT = 8
OPTIMIZATION_PROBABILITY_LAMBDA = 0.005
OPTIMIZATION_KERNEL = "rbf"  # "cosine" or "rbf"
OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES = True

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


def run_pipeline():
    baseline_model = None

    if CREATE_DINO_DATASET:
        vectors, labels, paths, classes = ut.create_dataset(
            data_location=COCO_IMAGE_DIR,
            annotation_path=COCO_ANNOTATION_PATH,
            bckgnd_rmv=REMOVE_IMAGE_BACKGROUNDS,
        )
        print("Vector shape:", vectors.shape)
        print("Label shape:", labels.shape)
        print("Class count:", len(classes))
        print("Image count:", len(paths))

    if SPLIT_DINO_DATASET:
        ut.split_DINO_vectors(
            "saved_vectors",
            test_size=TEST_SPLIT_SIZE,
            random_state=int(100 * random.random()),
        )

    if RUN_EMBEDDING_CLASSIFIER:
        ec.train_and_evaluate(
            learning_dir="saved_vectors/train",
            testing_dir="saved_vectors/test",
        )

    if RUN_SUBSET_PROBABILITY:
        baseline_model, _, _ = sp.score_directory(
            learning_dir="saved_vectors/train",
            target_dir="saved_vectors/retrain",
            tune_regularization=TUNE_CLASSIFIER_REGULARIZATION,
            fixed_c=FIXED_CLASSIFIER_C,
            use_hard_negative_mining=USE_HARD_NEGATIVE_MINING,
            hard_negative_fraction=HARD_NEGATIVE_FRACTION,
            hard_negative_weight=HARD_NEGATIVE_WEIGHT,
        )

    if RUN_OPTIMIZATION:
        opt.optimize_subset_selection(
            learning_dir="saved_vectors/train",
            target_dir="saved_vectors/retrain",
            selection_count=OPTIMIZATION_SELECTION_COUNT,
            probability_lambda=OPTIMIZATION_PROBABILITY_LAMBDA,
            kernel=OPTIMIZATION_KERNEL,
            stop_when_objective_decreases=(
                OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES
            ),
            reference_method=_reference_method(),
            max_labeled_reference_per_subset=KERNEL_HERDED_REFERENCE_COUNT,
            eigenvalue_method=_eigenvalue_method(),
        )

    model_0, model_1 = dmc.calibrate_baseline_and_augmented_models(
        training_dir="saved_vectors/train",
        retraining_dir="saved_vectors/retrain",
        tune_regularization=TUNE_CLASSIFIER_REGULARIZATION,
        fixed_c=FIXED_CLASSIFIER_C,
        use_hard_negative_mining=USE_HARD_NEGATIVE_MINING,
        hard_negative_fraction=HARD_NEGATIVE_FRACTION,
        hard_negative_weight=HARD_NEGATIVE_WEIGHT,
        baseline_model=baseline_model,
    )
    calibrated_exporter.export_calibrated_test_images(
        model_0,
        model_1,
        test_dir="saved_vectors/test",
    )

    if RUN_EXPORT_SELECTED_IMAGES:
        exporter.export_selected_images()

    if RUN_SURPRISAL_TRADEOFF_EVALUATION:
        tradeoff_eval.run_surprisal_tradeoff_evaluation()

    if RUN_LAMBDA_SELECTION_ACCURACY_EVALUATION:
        lambda_accuracy_eval.evaluate_lambda_selection_accuracy()


if __name__ == "__main__":
    run_pipeline()
