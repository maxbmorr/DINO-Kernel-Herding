import __utils__ as ut
import embedding_classifier as ec
import evaluate_lambda_selection_accuracy as lambda_accuracy_eval
import evaluate_surprisal_tradeoff as tradeoff_eval
import export_selected_images as exporter
import optimization as opt
import subset_probability as sp
import random

CREATE_DINO_DATASET = False
SPLIT_DINO_DATASET = True
RUN_EMBEDDING_CLASSIFIER = True
RUN_SUBSET_PROBABILITY = True
RUN_OPTIMIZATION = True
RUN_EXPORT_SELECTED_IMAGES = True
RUN_SURPRISAL_TRADEOFF_EVALUATION = False
RUN_LAMBDA_SELECTION_ACCURACY_EVALUATION = False

OPTIMIZATION_SELECTION_COUNT = 5
OPTIMIZATION_PROBABILITY_LAMBDA = 0.005
OPTIMIZATION_KERNEL = "rbf"  # "cosine" or "rbf"
OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES = True

USE_KERNEL_HERDED_REFERENCES = True
KERNEL_HERDED_REFERENCE_COUNT = 50
TUNE_CLASSIFIER_REGULARIZATION = True
FIXED_CLASSIFIER_C = 1.0

if CREATE_DINO_DATASET:
    X, Y, paths, classes = ut.create_dataset(
        data_location="coco/val2017",
        annotation_path="coco/annotations/instances_val2017.json",
        bckgnd_rmv=False
    )

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Class count:", len(classes))
    print("Image count:", len(paths))

if SPLIT_DINO_DATASET:
    ut.split_DINO_vectors("saved_vectors", test_size=0.2, random_state= int(100*random.random()))

if RUN_EMBEDDING_CLASSIFIER:
    ec.train_and_evaluate(
        learning_dir="saved_vectors/learning",
        testing_dir="saved_vectors/testing",
    )

if RUN_SUBSET_PROBABILITY:
    sp.score_directory(
        learning_dir="saved_vectors/learning",
        target_dir="saved_vectors/testing",
        tune_regularization=TUNE_CLASSIFIER_REGULARIZATION,
        fixed_c=FIXED_CLASSIFIER_C,
    )

if RUN_OPTIMIZATION:
    opt.optimize_subset_selection(
        learning_dir="saved_vectors/learning",
        target_dir="saved_vectors/testing",
        selection_count=OPTIMIZATION_SELECTION_COUNT,
        probability_lambda=OPTIMIZATION_PROBABILITY_LAMBDA,
        kernel=OPTIMIZATION_KERNEL,
        stop_when_objective_decreases=OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES,
        reference_method=(
            "kernel_herding" if USE_KERNEL_HERDED_REFERENCES else "all"
        ),
        max_labeled_reference_per_subset=KERNEL_HERDED_REFERENCE_COUNT,
    )

if RUN_EXPORT_SELECTED_IMAGES:
    exporter.export_selected_images()

if RUN_SURPRISAL_TRADEOFF_EVALUATION:
    tradeoff_eval.run_surprisal_tradeoff_evaluation()

if RUN_LAMBDA_SELECTION_ACCURACY_EVALUATION:
    lambda_accuracy_eval.evaluate_lambda_selection_accuracy()
