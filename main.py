import __utils__ as ut
import embedding_classifier as ec
import entropy_calculator as ent

CREATE_DINO_DATASET = False
SPLIT_DINO_DATASET = True
RUN_EMBEDDING_CLASSIFIER = True
RUN_ENTROPY_CALCULATOR = True

HERDING_SELECTION_COUNT = 4
HERDING_CANDIDATE_POOL = "all"  # "all", "interesting", or "candidate"
HERDING_PER_CLASS = True

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

if RUN_ENTROPY_CALCULATOR:
    ent.run_entropy_analysis(
        learning_dir="saved_vectors/learning",
        calibration_dir="saved_vectors/testing",
        target_dir="saved_vectors/testing",
        herding_count=HERDING_SELECTION_COUNT,
        herding_candidate_pool=HERDING_CANDIDATE_POOL,
        herding_per_class=HERDING_PER_CLASS,
        top_n=10,
    )
