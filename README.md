# DINO-Kernel-Herding
For CNiEL research 2026 for the Dr. Austin Brockmeier.  This is my primary project for my Summer Research.

## What the Code Currently Does

The current pipeline starts in `main.py`.

The configured input is the complete annotated COCO 2017 training set:

```text
coco/train2017/                                      118,287 images
coco/annotations/instances_train2017.json
```

The active learning classifier is a DINO-embedding classifier in `embedding_classifier.py`. It learns only from labeled images in `saved_vectors/train/`, while `saved_vectors/test/` remains the final test fold.

`main.py` retains only three experiment switches:

- `CREATE_DINO_DATASET`: creates DINO feature vectors before the experiment.
- `USE_KERNEL_HERDED_REFERENCES`: toggles kernel-herded reference compression.
- `USE_SECULAR_EIGENVALUE_UPDATES`: toggles experimental secular updates.

Splitting, hyperparameter selection, calibrated scoring, optimization,
retraining, export, and final AUC evaluation run automatically in that order.

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

The proof-of-concept split is controlled by `CLASS_COUNT`. Each run randomly
chooses that many classes with at least 300 usable images each. Every chosen
class contributes the same number of multi-label-positive images. That balanced
quota is between 300 and 1,000 and is determined by the available count of the
least-populated selected class. To keep the final positive counts exactly equal, a training
image may contain only one of the five selected classes, although it may still
contain any number of non-selected COCO classes. Duplicate images are stored
once.

To run ten fully independent experiments and average the model metrics, use:

```powershell
py main.py --10run
```

Each run generates a new class selection, random seed, train/retrain/test split,
hyperparameter sweep, optimized acquisition, random baseline, and test
evaluation. Results are checkpointed after every completed run in:

```text
_ten_run_average/_ten_run_results.csv
_ten_run_average/_ten_run_average.csv
_ten_run_average/_ten_run_average.png
_ten_run_average/_ten_run_roc_average.csv
_ten_run_average/_ten_run_caterpillar.png
_ten_run_average/_ten_run_dumbbell.png
_ten_run_average/_ten_run_entropy_probability_tradeoff.png
```

The folder also preserves each run's individual comparison CSV and PNG. The
average graph contains separate macro-ROC and micro-ROC panels. Each plots mean
true-positive rate against false-positive rate on a shared 1,001-point grid,
with a shaded pointwise 95% confidence interval across the ten independent
runs. `_ten_run_roc_average.csv` stores every mean curve and confidence bound.
Each run also preserves its selection-quality calibration graph and its
entropy–probability tradeoff graph, plus the CSV data behind both figures, using
`run_01_...` through `run_10_...` filenames.
The caterpillar plot shows each run's macro-AUC change relative to `M_0`; the
dumbbell plot compares absolute macro AUC across models; and the pooled
entropy-probability plot combines all selected images in the common
`lambda * log(P)` objective space.

Ten percent of the final training set contains none of the selected classes.
These known neither-class images are negative examples for every one-vs-rest
classifier. With 550 positives from each of two classes, this gives 1,100
positive images plus 122 neither-class negatives: `N_train = 1,222`. Images
positive for one selected class also remain valid negatives for the other.

```text
retrain size = 2 * unique training size = 2,444 images
test size = 5,000 images
```

Retrain and test rows are sampled randomly without replacement, the three sets
are disjoint, and unused COCO images remain outside all splits. Before random
filling, each retrain and test split reserves enough known-positive images to
guarantee at least 5% prevalence for every selected class. The training quota
is reduced when necessary to leave those positives available downstream. With
a 5,000-image test set, this reserves at least 250 known positives for every
selected class:

```text
saved_vectors/train/    original labeled training data
saved_vectors/retrain/  acquisition pool used by subset selection
saved_vectors/test/     untouched final test data
```

With the current flags in `main.py`, the embedding classifier trains on:

```text
saved_vectors/train/
```

and tests on:

```text
saved_vectors/test/
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
- divides the train set into up to five stratified folds
- tunes logistic regularization `C` inside each training fold using an inner stratified cross-validation and log loss
- trains an initial class-balanced logistic regression on each outer training fold
- finds the true negatives with the highest class scores, increases their sample weight, and refits the fold classifier
- predicts the untouched held-out fold with the hard-negative-refit classifier
- repeats until every learning image has one out-of-fold decision score
- fits a sigmoid calibrator on all out-of-fold scores at the natural class frequency
- tunes `C` again, mines hard negatives, and refits the final logistic classifier on the complete train set
- derives a class-specific threshold from out-of-fold predictions by maximizing
  balanced accuracy (equivalently, Youden's J statistic)

The main pipeline selects one global `C` jointly with lambda and selected-data
trust using the original-training cross-validation sweep. Its initial grid is:

```python
C_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
```

Smaller `C` values apply stronger regularization. If the raw optimum for `C`,
lambda, or selected-data weight is on either boundary, that grid expands by
another decade in the required direction and the sweep continues.
The selected global `C` is then used by all final out-of-fold-calibrated class
models. There is no separate `TUNE_CLASSIFIER_REGULARIZATION` switch or
`FIXED_CLASSIFIER_C` override in `main.py`, preventing a second tuning mechanism
from disagreeing with the joint hyperparameter sweep.

Hard-negative mining is disabled in the fixed `main.py` pipeline, and its old
top-level controls have been removed.

For each class, `HARD_NEGATIVE_FRACTION` selects that fraction of true
negative training images with the largest initial decision scores.
`HARD_NEGATIVE_WEIGHT` is their sample weight during the refit. Mining runs
independently inside every outer training fold, so the held-out fold used for
calibration is never inspected while choosing hard negatives. The same
procedure is applied to the final classifier using the complete train set.
Disabling mining restores the previous classifier training.

Hard-negative mining changes the probability estimates, but not the entropy
objective, positivity constraint, candidate set, or greedy subset optimizer.

Classes need at least six positive and six negative learning examples. Classes below that minimum are skipped because their probabilities cannot be calibrated reliably.

The classifiers use DINO embeddings from:

```text
saved_vectors/train/
```

Then the target images from:

```text
saved_vectors/retrain/
```

are scored against every calibrated class/subset model. The probability used by optimization is:

```text
P_hat_+(x) = calibrated P(class appears in image | DINO embedding)
```

The result is not normalized across classes because COCO images can contain multiple classes. Each class gets an independent probability and class-specific decision threshold.

Running `main.py` with `RUN_SUBSET_PROBABILITY = True` writes:

```text
saved_vectors/retrain/subset_probability_scores.csv
saved_vectors/retrain/subset_probability_matrix.csv
saved_vectors/retrain/subset_probability_raw_matrix.csv
saved_vectors/retrain/subset_probability_calibration_metrics.csv
saved_vectors/retrain/subset_probability_evaluation_metrics.csv
```

## Baseline and Augmented Calibration

After subset optimization, `dual_model_calibration.py` creates three separately
calibrated probability models:

```text
M_0 = original labeled train images
M_1 = original labeled train images + newly labeled selected retrain images
M_rand = original labeled train images + an equal-size random forced-positive sample
```

Only paths present in `optimization_selection.csv` are added to `M_1`. A
selected image is added once even when it was selected for multiple subsets.
`M_rand` samples without replacement and uses exactly the same number of unique
augmentation images as `M_1`. Each image's forced-positive class is also drawn
independently and uniformly at random, with no class-balancing constraint, and
is treated as positive even when that class is absent from the COCO annotation.
Real labels for other classes are preserved. Its seed is the recorded
data-selection seed, and its training manifest records every forced-positive
assignment.
Its COCO multi-label metadata supplies the newly revealed labels. The module
raises an error if a selected path does not belong to the current retrain
split, which prevents an old selection file from being combined with a new
random split.

All three models independently fit their scaler, fold-safe hard-negative models,
out-of-fold sigmoid calibrators, and class thresholds. This stage saves the
models and calibration records but does not score or compare them.

This calibration stage always runs from `main.py` after the optional subset
optimization stage. It does not have an enable/disable switch.

Outputs are written to:

```text
saved_models/calibrated/M_0.joblib
saved_models/calibrated/M_0_calibration_metrics.csv
saved_models/calibrated/M_0_training_manifest.csv
saved_models/calibrated/M_1.joblib
saved_models/calibrated/M_1_calibration_metrics.csv
saved_models/calibrated/M_1_training_manifest.csv
saved_models/calibrated/M_rand.joblib
saved_models/calibrated/M_rand_calibration_metrics.csv
saved_models/calibrated/M_rand_training_manifest.csv
```

After saving the models, `main.py` also classifies every image in the untouched
test split with all three calibrations and organizes copies into:

```text
_organized_calibrated_images/
  M_0/
    person/
    dog/
    ...
  M_1/
    person/
    dog/
    ...
  M_rand/
    ...
```

Each model folder contains a `prediction_manifest.csv` recording only the
predicted folder assignment. This export reads test embeddings and image paths,
but does not use true test labels or calculate accuracy, calibration, or model
comparison metrics.
If no selected class exceeds its optimized decision threshold, the image is
assigned `predicted_label_id=-1`, `predicted_label_name=negative`, and exported
to the model's `negative/` folder. The model is therefore not forced to call
every image one of the selected classes.

After all three models are saved, `main.py` automatically compares them using the
true multi-label annotations of the untouched test fold. The evaluator can
also be rerun independently with:

```powershell
python evaluate_dual_model_auc.py
```

This writes completely separate macro- and micro-ROC graphs and supporting AUC
tables to:

```text
_organized_calibrated_images/AUC_evaluation/_model_comparison.png
_organized_calibrated_images/AUC_evaluation/_model_comparison.csv
```

Classes without both a positive and negative test example are excluded because
ROC AUC is undefined for those classes.

No bootstrap comparison is run. Model quality is reported directly on the
untouched test fold using per-model AUC and accuracy.

The main pipeline exposes the selection-size search grid in `main.py`:

```python
K_VALUES = tuple(range(20, 201, 20))
```

The sweep first selects lambda, `C`, and the trust weight using 40 ranked images
per class. It then freezes those three values,
generates the full 200-image ranking only for that winning configuration, tests
every nonzero 20-image prefix, and chooses one global `K` by cross-validated ROC
AUC. If the same image is selected for multiple classes, it is included only
once when training `M_1`.

`_model_comparison.csv` is the single numerical results file. Its `overall`
rows report macro and micro AUC, macro and micro thresholded accuracy, and
exact-match multi-label accuracy. Its `class` rows report AUC and binary
accuracy for every model and evaluated class, including changes from `M_0`.
`_model_comparison.png` is one dashboard containing macro ROC, micro ROC,
overall accuracy, and per-class accuracy for all three models.
Accuracy uses each model's class-specific decision thresholds. These thresholds
maximize balanced accuracy on out-of-fold training predictions and are fixed
before the untouched test set is evaluated; the test labels never tune them.

## Hyperparameter sweep and sensitivity

Run `python hyperparameter_sweep.py` to sweep lambda, the selected/re-labeled
data trust weight, logistic-regression L2 inverse regularization `C`, and the
per-class selection count K. K is searched from 20 through 200 in increments
of 20; zero is excluded.
The winning configuration is the one with the highest mean cross-validated
macro ROC AUC; mean micro ROC AUC breaks ties. Lambda, `C`, and trust are chosen
at K=40 first. They are then held fixed while one global K is selected for all
classes, and the main pipeline uses that exact-cardinality value.
Trust is searched logarithmically from `0.001` through `1.0`; `C` is searched
from `0.0001` through `10.0` to cover stronger L2 regularization after the
previous optimum landed on the lower boundary. Lambda, trust, and `C` are all
shown on logarithmic sensitivity axes. A boundary optimum adds another decade
below the minimum or above the maximum and evaluates it. Multiple boundary
parameters expand together. This repeats until all three optima are interior.
A safety guard raises an error after five consecutive expansion rounds rather
than allowing an accidental endless search.
Lambda has a hard lower bound of `1e-4`; downward expansion stops there, and a
lambda optimum at that constrained minimum is accepted.
Selection uses three-fold cross-validation entirely within the original
training set. Each fold contains labeled-base, simulated-unlabeled candidate,
and untouched validation roles. The real retrain and test labels are not used
to choose hyperparameters. The combination with the highest mean macro ROC AUC
is selected directly, with mean micro ROC AUC used as a tie-breaker.

Each `(fold, C)` baseline is fully out-of-fold calibrated so lambda always
multiplies the same calibrated `log(P)` quantity used by the final optimizer.
Trust-weight variants use lightweight logistic models because macro AUC depends
on within-class ranking; the winning final model is fully calibrated. Baseline
fits, candidate probabilities, fold transforms, and lambda selections are
reused wherever their inputs do not change. Every completed combination is
checkpointed and the console reports completion percentage and estimated time
remaining.

Results, the selected configuration, and one-at-a-time sensitivity graphs are
written under `_hyperparameter_sensitivity/`.

Run `python graph_entropy_image_examples.py` to create a qualitative panel that
contrasts kernel-herded labeled references with the selected candidates having
the largest positive marginal entropy gains for each class. Each selected image
is annotated with its optimization rank, entropy gain, and positive-class
probability. The PNG and its source table are written under
`_organized_calibrated_images/AUC_evaluation/`.
The same script also creates `_selection_entropy_top_50_gallery.png`, a balanced
gallery of 50 unique positive-entropy-gain selections. Green borders indicate
that the target class was present after labeling; red borders indicate that it
was absent. The corresponding rows are saved in
`_selection_entropy_top_50_gallery.csv`.
It additionally creates a `selection_entropy_by_class/` directory containing
one PNG and CSV per selected class. These class-specific galleries include
every selected image in optimization-rank order. Border color reports whether
the target class was revealed to be present, and title color reports whether
the total selection objective gain was positive or negative.
For a compact qualitative result, the script creates
`_qualitative_ranked_entropy_selections.png` and its matching CSV. It shows the
ten selected images with the greatest marginal entropy increase for each class,
ranked left-to-right, with their revealed COCO labels printed on every tile.

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
recall on the retrain acquisition pool using its known COCO labels for
auditing. It is not the untouched final test evaluation. Every train image is
held out exactly once for calibration scoring. Retrain images are not used to fit
`M_0`; only selected retrain images with newly revealed labels are added to
`M_1`. Final test images are not used by either model during fitting.

This does not draw boxes or detect multiple objects. It answers: given this image's DINO embedding, how positive/subset-like is it for each learned subset?

## Von Neumann Optimization

The optimization code is in `optimization.py`.

It uses:

```text
saved_vectors/train/
saved_vectors/retrain/subset_probability_scores.csv
saved_vectors/retrain/subset_probability_matrix.csv
```

For each learned subset/class, it:

- takes every train image containing that class in `all_label_ids` as the reference set
- considers every retrain image for that subset
- greedily selects images that maximize normalized von Neumann entropy of the kernel matrix plus a log-probability term

The candidate set and objective are unchanged between rounds. For efficiency,
the implementation caches the reference-to-reference and
reference-to-candidate kernel blocks, then computes only the kernel rows and
entropy values that depend on the current greedy selections. Every retrain
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

The remaining optimization controls in `main.py` are:

```python
USE_KERNEL_HERDED_REFERENCES = True
KERNEL_HERDED_REFERENCE_COUNT = 100
USE_SECULAR_EIGENVALUE_UPDATES = False
```

Selection is always open-ended with a positive-objective-gain stop, uses the
RBF kernel, and receives lambda from the cross-validated sweep.

When `USE_KERNEL_HERDED_REFERENCES` is enabled, known learning labels first
define each class subset. Kernel herding then compresses that subset to 100
representative DINO embeddings by greedily matching its kernel mean. All von
Neumann entropy and candidate calculations use this herded reference coreset.
True retrain labels are not used during selection.

Set `USE_KERNEL_HERDED_REFERENCES = False` to use every labeled embedding in
each class. There is no random reference sampling or cap in this mode. Classes
smaller than `KERNEL_HERDED_REFERENCE_COUNT` already use every labeled embedding
when herding is enabled.

`USE_SECULAR_EIGENVALUE_UPDATES` controls how candidate kernel eigenvalues are
computed:

```text
False -> direct LAPACK eigvalsh
True  -> arrowhead secular-equation update with direct fallback
```

The secular method diagonalizes the current kernel once per greedy round and
solves the bordered arrowhead eigenproblem for each candidate. It has better
asymptotic complexity, but the current SciPy/Python root solver is slower than
optimized LAPACK at the project's usual 30-50 reference sizes. It therefore
defaults to `False`. Both paths use the same objective and produce equivalent
selections within numerical tolerance.

Running `main.py` with `RUN_OPTIMIZATION = True` writes:

```text
saved_vectors/retrain/optimization_selection.csv
saved_vectors/retrain/optimization_summary.csv
```

Important columns in `optimization_selection.csv`:

- `subset_label_name`: subset/class being optimized
- `path`: selected target image
- `optimization_rank`: greedy selection order inside that subset
- `kernel`: kernel used for the von Neumann entropy calculation
- `eigenvalue_method`: `direct` or `secular`
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
SURPRISAL_LAMBDAS = [0.001, 0.005, 0.01, 0.025]
SELECTION_COUNT = None
KERNEL = "rbf"
USE_KERNEL_HERDED_REFERENCES = True
USE_SECULAR_EIGENVALUE_UPDATES = False
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
