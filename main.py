import __utils__ as ut
import embedding_classifier as ec
import optimization as opt
import subset_probability as sp

CREATE_DINO_DATASET = False
SPLIT_DINO_DATASET = True
RUN_EMBEDDING_CLASSIFIER = True
RUN_SUBSET_PROBABILITY = True
RUN_OPTIMIZATION = True

OPTIMIZATION_SELECTION_COUNT = 4
OPTIMIZATION_PROBABILITY_LAMBDA = 0
OPTIMIZATION_KERNEL = "cosine"  # "cosine" or "rbf"
OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES = True

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
    ut.split_DINO_vectors("saved_vectors", test_size=0.2, random_state=42)

if RUN_EMBEDDING_CLASSIFIER:
    ec.train_and_evaluate(
        learning_dir="saved_vectors/learning",
        testing_dir="saved_vectors/testing",
    )

if RUN_SUBSET_PROBABILITY:
    sp.score_directory(
        learning_dir="saved_vectors/learning",
        target_dir="saved_vectors/testing",
    )

if RUN_OPTIMIZATION:
    opt.optimize_subset_selection(
        learning_dir="saved_vectors/learning",
        target_dir="saved_vectors/testing",
        selection_count=OPTIMIZATION_SELECTION_COUNT,
        probability_lambda=OPTIMIZATION_PROBABILITY_LAMBDA,
        kernel=OPTIMIZATION_KERNEL,
        stop_when_objective_decreases=OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES,
    )
