# DINO-Kernel-Herding
For CNiEL research 2026 for the Dr. Austin Brockmeier.  This is my primary project for my Summer Research.

## What the Code Currently Does

The current pipeline starts in `main.py`.

The active learning classifier is a DINO-embedding classifier in `embedding_classifier.py`. It learns only from labeled images, using the labels saved in `saved_vectors/learning/labels.npy`, then tests on `saved_vectors/testing/labels.npy`.

The DINO vector pipeline is still available, but it is behind flags in `main.py`:

- `CREATE_DINO_DATASET`: creates DINO feature vectors from COCO or class-folder images.
- `SPLIT_DINO_DATASET`: splits saved DINO vectors into learning/testing folders.
- `RUN_EMBEDDING_CLASSIFIER`: trains and tests a classifier using labeled DINO vectors.
- `RUN_SUBSET_PROBABILITY`: estimates calibrated multi-label membership probability for each target image and class/subset.
- `RUN_OPTIMIZATION`: selects target images using a von Neumann entropy objective with a log-probability term.
- `RUN_EXPORT_SELECTED_IMAGES`: copies selected target images into organized review folders.
- `RUN_SURPRISAL_TRADEOFF_EVALUATION`: runs the class/lambda surprisal trade-off sweep and exports visual comparison folders.
- `RUN_LAMBDA_SELECTION_ACCURACY_EVALUATION`: uses true labels after selection to generate lambda accuracy CSVs and graphs.

## Dependencies

This project uses Python 3.13 and the following Python packages:

```powershell
python -m pip install torch torchvision pillow rembg onnxruntime numpy pandas scikit-learn matplotlib
```

Main dependencies:

- `torch`: runs the DINOv2 model and tensor operations.
- `torchvision`: loads image folders and applies image transforms.
- `pillow`: opens and converts image files.
- `rembg`: removes image backgrounds.
- `onnxruntime`: runs the background-removal model used by `rembg`.
- `numpy`: saves DINO vectors as `.npy` files.
- `pandas`: saves metadata and class mappings as `.csv` files.
- `scikit-learn`: trains and evaluates the labeled DINO-embedding classifier.
- `matplotlib`: creates distribution graphs from probability and optimization CSV files.

`rembg` also installs several supporting packages, including `pymatting`, `scipy`, `scikit-image`, `numba`, `pooch`, and `tqdm`.

## First Run Notes

The first time the project runs, it may download model files:

- DINOv2 through `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
- `u2net.onnx` for background removal through `rembg`

The code stores the `rembg` model in the local `.u2net/` folder and Numba cache files in `.numba_cache/`.

## Data Format

Input images can be organized in class folders:

```text
training_data/
  cow/
    cow1.png
    cow2.png
```

The code can also read COCO-style image folders without class subfolders:

```text
training_data/
  000000000001.jpg
  000000000002.jpg
```

If no COCO annotation file is provided or found, flat folders are treated as unlabeled. Labels are saved as `-1` and label names are saved as `unlabeled`.

Unlabeled images can be embedded, but they are not used to train the learning classifier. The classifier only learns from rows with real labels.

## COCO Labels

The code can now use COCO annotation JSON files for labels. A normal COCO layout looks like this:

```text
coco/
  train2017/
    000000000001.jpg
    000000000002.jpg
  annotations/
    instances_train2017.json
```

You can call the dataset creation with an explicit annotation path:

```python
X, Y, paths, classes = ut.create_dataset(
    data_location="coco/train2017",
    annotation_path="coco/annotations/instances_train2017.json",
    bckgnd_rmv=False
)
```

If `annotation_path` is not passed, the code will look for common COCO annotation filenames near the image folder, such as:

```text
annotations/instances_train2017.json
annotations/instances_val2017.json
```

COCO images can contain multiple object categories. To keep the existing saved format simple, each image gets one primary label in `labels.npy` and `metadata.csv`. The primary label is the category of the largest annotated object in that image.

The full multi-label COCO information is still saved in `metadata.csv`:

- `label_id`: primary label id
- `label_name`: primary label name
- `all_label_ids`: all COCO label ids found in the image
- `all_label_names`: all COCO label names found in the image
- `image_id`: the COCO image id

## Labeled Embedding Classifier

The learning classifier is in `embedding_classifier.py`.

It uses DINO vectors that were already created from labeled images. The usual flow is:

```powershell
python main.py
```

With the current flags in `main.py`, this trains on:

```text
saved_vectors/learning/
```

and tests on:

```text
saved_vectors/testing/
```

The classifier ignores unlabeled rows where `label_id` is `-1`. If all rows are unlabeled, it stops with an error because there is nothing labeled to learn from.

This classifier uses the existing saved-vector format:

```text
vectors.npy
labels.npy
metadata.csv
class_mapping.csv
```

## Calibrated Subset Probability

The subset probability code is in `subset_probability.py`.

It estimates \(\hat P_+(x)\), the calibrated probability that a target image contains each learned class/subset.

The model uses all COCO labels in `all_label_ids`, not only the primary label. An image containing both a person and a dog is therefore positive for both classes. This avoids treating secondary objects as false negatives.

For each class, the pipeline:

- creates a natural one-vs-rest target using all annotated objects
- divides the learning set into up to five stratified folds
- tunes logistic regularization `C` inside each training fold using an inner stratified cross-validation and log loss
- trains the tuned class-balanced logistic regression on the outer training folds and predicts the held-out fold
- repeats until every learning image has one out-of-fold decision score
- fits a sigmoid calibrator on all out-of-fold scores at the natural class frequency
- tunes `C` again and refits the final logistic classifier on the complete learning set
- derives a class-specific threshold targeting 80% precision

The regularization search grid is:

```python
REGULARIZATION_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
```

Smaller `C` values apply stronger regularization. Each class selects its own
value using nested cross-validated log loss, which penalizes confident incorrect
predictions. This changes the calibrated probability input but does not change
the entropy objective, candidate set, or greedy optimization algorithm.

Regularization tuning can be disabled in `main.py` when faster probability
model fitting is more important:

```python
TUNE_CLASSIFIER_REGULARIZATION = True
FIXED_CLASSIFIER_C = 1.0
```

When tuning is `True`, each class uses the nested log-loss search described
above. When it is `False`, all outer-fold classifiers and the final classifier
use `FIXED_CLASSIFIER_C`; five-fold out-of-fold calibration still runs. Fixed
mode therefore preserves probability calibration while avoiding the inner
regularization searches. Changing this switch affects probability estimates,
not the downstream entropy optimization formula.

Classes need at least six positive and six negative learning examples. Classes below that minimum are skipped because their probabilities cannot be calibrated reliably.

The classifiers use DINO embeddings from:

```text
saved_vectors/learning/
```

Then the target images from:

```text
saved_vectors/testing/
```

are scored against every calibrated class/subset model. The probability used by optimization is:

```text
P_hat_+(x) = calibrated P(class appears in image | DINO embedding)
```

The result is not normalized across classes because COCO images can contain multiple classes. Each class gets an independent probability and class-specific decision threshold.

Running `main.py` with `RUN_SUBSET_PROBABILITY = True` writes:

```text
saved_vectors/testing/subset_probability_scores.csv
saved_vectors/testing/subset_probability_matrix.csv
saved_vectors/testing/subset_probability_raw_matrix.csv
saved_vectors/testing/subset_probability_calibration_metrics.csv
saved_vectors/testing/subset_probability_evaluation_metrics.csv
```

Important columns in `subset_probability_scores.csv`:

- `path`: image path
- `predicted_subset_label_id`: class/subset with the highest calibrated probability
- `predicted_subset_label_name`: name of that class/subset
- `positive_probability`: calibrated \(\hat P_+(x)\) for the highest-scoring subset
- `subset_probability`: compatibility alias for `positive_probability`
- `raw_positive_probability`: uncalibrated logistic regression output
- `second_positive_probability`: calibrated probability of the runner-up subset
- `top_two_probability_margin`: difference between the top two probabilities
- `predicted_class_threshold`: calibrated decision threshold for the top subset
- `passes_predicted_class_threshold`: whether the top score passes that threshold
- `classes_above_threshold`: number of independently detected subsets
- `probability_method`: `calibrated_multilabel_ovr`
- `actual_label_id`: known label, if the target folder has labels
- `actual_label_name`: known label name, if the target folder has labels
- `actual_all_label_ids`: all known COCO labels for the target image

`subset_probability_matrix.csv` stores calibrated values in one `*_positive_probability` column per class. `subset_probability_raw_matrix.csv` stores the corresponding uncalibrated outputs for auditing.

The calibration metrics CSV reports the calibration method, fold count,
candidate and selected `C` values, per-outer-fold `C` values, out-of-fold sample
counts, threshold, Brier score, expected calibration error, and average
precision. The evaluation metrics CSV reports those metrics plus precision and
recall on the independent testing split. Every learning image is held out
exactly once for calibration scoring; testing images are never used to fit the
classifier, regularization search, or calibrator.

This does not draw boxes or detect multiple objects. It answers: given this image's DINO embedding, how positive/subset-like is it for each learned subset?

## Von Neumann Optimization

The optimization code is in `optimization.py`.

It uses:

```text
saved_vectors/learning/
saved_vectors/testing/subset_probability_scores.csv
saved_vectors/testing/subset_probability_matrix.csv
```

For each learned subset/class, it:

- takes every learning image containing that class in `all_label_ids` as the reference set
- considers every target/testing image for that subset
- greedily selects images that maximize normalized von Neumann entropy of the kernel matrix plus a log-probability term

The candidate set and objective are unchanged between rounds. For efficiency,
the implementation caches the reference-to-reference and
reference-to-candidate kernel blocks, then computes only the kernel rows and
entropy values that depend on the current greedy selections. Every testing
image is still eligible for every subset.

In plain terms, it asks:

```text
For this subset, which target image best increases kernel-space diversity
after the objective also charges the image by log P_hat_+(x)?
```

The objective is:

```text
H(labeled subset c + selected target images)
+ lambda * sum(log(P_hat_+(x)))
```

There is no hard rule that a target image must first be predicted as subset `c`.
Every target image can be considered for every subset. The subset membership
pressure enters through the `lambda * log(P_hat_+(x))` term.

When `OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES` is enabled, the best remaining
candidate is accepted only when its combined marginal gain is positive:

```text
delta H_vN(x | S) + lambda * log(P_hat_+(x)) > 0
```

Selection stops for the class when even the best remaining candidate has zero
or negative combined gain.

The main controls are in `main.py`:

```python
OPTIMIZATION_SELECTION_COUNT = 5
OPTIMIZATION_PROBABILITY_LAMBDA = 0.005
OPTIMIZATION_KERNEL = "rbf"
OPTIMIZATION_STOP_WHEN_OBJECTIVE_DECREASES = True
USE_KERNEL_HERDED_REFERENCES = True
KERNEL_HERDED_REFERENCE_COUNT = 50
```

When `USE_KERNEL_HERDED_REFERENCES` is enabled, known learning labels first
define each class subset. Kernel herding then compresses that subset to 50
representative DINO embeddings by greedily matching its kernel mean. All von
Neumann entropy and candidate calculations use this herded reference coreset.
True testing labels are not used.

Set `USE_KERNEL_HERDED_REFERENCES = False` to use every labeled embedding in
each class. There is no random reference sampling or cap in this mode. Classes
smaller than `KERNEL_HERDED_REFERENCE_COUNT` already use every labeled embedding
when herding is enabled.

Running `main.py` with `RUN_OPTIMIZATION = True` writes:

```text
saved_vectors/testing/optimization_selection.csv
saved_vectors/testing/optimization_summary.csv
```

Important columns in `optimization_selection.csv`:

- `subset_label_name`: subset/class being optimized
- `path`: selected target image
- `optimization_rank`: greedy selection order inside that subset
- `kernel`: kernel used for the von Neumann entropy calculation
- `stop_when_objective_decreases`: whether selection requires positive combined objective gain
- `objective`: value of `H + lambda * sum(log(P_hat_+(x)))` after this image is selected
- `objective_gain`: change in the objective from selecting this image
- `probability_lambda`: scalar lambda used in the log-probability term
- `base_von_neumann_entropy`: entropy before adding selected target images
- `von_neumann_entropy`: entropy after this selected image is added
- `von_neumann_entropy_gain`: improvement from adding this image
- `log_probability`: `log(positive_probability)` for the selected image
- `log_probability_sum`: cumulative log-probability sum for the selected subset
- `positive_probability`: \(\hat P_+(x)\) for the optimized subset
- `subset_probability`: compatibility alias for `positive_probability`
- `best_predicted_subset_label_name`: subset with the highest probability for this image
- `best_predicted_subset_probability`: probability of that best predicted subset

## Selected Image Export

The selected image exporter is in `export_selected_images.py`.

When this switch is enabled in `main.py`:

```python
RUN_EXPORT_SELECTED_IMAGES = True
```

the selected target images are copied into:

```text
organized_selected_images/
```

Each subset gets its own folder. The image filenames include selection rank,
\(\hat P_+(x)\), entropy gain, actual label if available, and the best predicted
subset. The originals are not moved or modified.

## Surprisal Trade-Off Evaluation

The systematic trade-off evaluator is in `evaluate_surprisal_tradeoff.py`.

It evaluates selected target classes across multiple values of the surprisal
trade-off:

```text
H(labeled subset c + selected target images)
- lambda * sum(-log(P_hat_+(x)))
```

This is the same as:

```text
H(labeled subset c + selected target images)
+ lambda * sum(log(P_hat_+(x)))
```

The main controls are at the top of `evaluate_surprisal_tradeoff.py`:

```python
TARGET_CLASSES = []  # every calibrated class
SURPRISAL_LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.025]
SELECTION_COUNT = 4
KERNEL = "rbf"
USE_KERNEL_HERDED_REFERENCES = True
MAX_LABELED_REFERENCE_PER_SUBSET = 30
```

Run:

```powershell
python evaluate_surprisal_tradeoff.py
```

Or turn it on from `main.py`:

```python
RUN_SURPRISAL_TRADEOFF_EVALUATION = True
```

It writes:

```text
_surprisal_tradeoff_evaluation/
```

Inside that folder, each target class has one folder per lambda value. Each
lambda folder contains the selected images, a `selection.csv`, and a
`contact_sheet.jpg` for quick visual review.

The root also contains `lambda_summary.csv`, which reports total selections,
classes with at least one selection, mean selections per class, and aggregate
objective and entropy values for each lambda.

### Label-Only Lambda Accuracy Evaluation

Ground-truth COCO labels are never used by probability scoring or optimization.
After a sweep finishes, they can be used strictly for evaluation:

```powershell
python evaluate_lambda_selection_accuracy.py
```

Or enable only the corresponding switch in `main.py` after sweep outputs exist:

```python
RUN_LAMBDA_SELECTION_ACCURACY_EVALUATION = True
```

It defaults to `False`, so normal pipeline runs do not use true labels for this
evaluation or regenerate its graphs.

A selection is correct when the optimized subset appears anywhere in the
image's `all_label_ids`. This produces:

```text
_surprisal_tradeoff_evaluation/lambda_accuracy_summary.csv
_surprisal_tradeoff_evaluation/lambda_class_accuracy.csv
_surprisal_tradeoff_evaluation/labeled_tradeoff_selections.csv
_surprisal_tradeoff_evaluation/lambda_selection_accuracy.png
_surprisal_tradeoff_evaluation/lambda_class_accuracy_heatmap.png
```

`lambda_selection_accuracy.png` compares precision, correct/incorrect counts,
and the quantity/correctness trade-off. The heatmap shows selection precision
for every class and lambda. These metrics must not be fed back into a selection
run except when choosing a global lambda on a designated validation split.

## Distribution Graphs

Distribution graphs are created by `graph_distributions.py`.

Run:

```powershell
python graph_distributions.py
```

The graphs are saved in:

```text
distribution_graphs/
```

The current graphs show calibrated multi-label probability distributions, predicted subset counts, optimization selection counts, objective gains, entropy gains, and probability vs entropy gain for selected images.

Running `main.py` creates:

```text
training_data_no_bg/
saved_vectors/
  vectors.npy
  labels.npy
  metadata.csv
  class_mapping.csv
```

## License

This project is licensed under the MIT License. See `LICENSE`.
