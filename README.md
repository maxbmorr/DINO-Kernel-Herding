# DINO-Kernel-Herding
For CNiEL research 2026 for the Dr. Austin Brockmeier.  This is my primary project for my Summer Research.

## What the Code Currently Does

The current pipeline starts in `main.py`.

The default learning classifier is now a DINO-embedding classifier in `embedding_classifier.py`. It learns only from labeled images, using the labels saved in `saved_vectors/learning/labels.npy`, then tests on `saved_vectors/testing/labels.npy`.

A separate COCO-pretrained Faster R-CNN detector is also available in `coco_classifier.py`. That detector does not learn from your dataset; it is already pretrained on COCO and runs directly on images.

The DINO vector pipeline is still available, but it is behind flags in `main.py`:

- `CREATE_DINO_DATASET`: creates DINO feature vectors from COCO or class-folder images.
- `SPLIT_DINO_DATASET`: splits saved DINO vectors into learning/testing folders.
- `RUN_EMBEDDING_CLASSIFIER`: trains and tests a classifier using labeled DINO vectors.
- `RUN_ENTROPY_CALCULATOR`: fits KDE models on labeled DINO vectors, then uses Kernel Herding to select a useful subset of interesting images.
- `RUN_COCO_CLASSIFIER`: runs the COCO-pretrained object detector.

The old threshold-similarity classifier has been removed from the runnable code.

## Dependencies

This project uses Python 3.13 and the following Python packages:

```powershell
python -m pip install torch torchvision pillow rembg onnxruntime numpy pandas scikit-learn matplotlib
```

Main dependencies:

- `torch`: runs the DINOv2 model, Faster R-CNN detector, and tensor operations.
- `torchvision`: loads image folders, applies image transforms, and provides the COCO-pretrained Faster R-CNN detector.
- `pillow`: opens and converts image files.
- `rembg`: removes image backgrounds.
- `onnxruntime`: runs the background-removal model used by `rembg`.
- `numpy`: saves DINO vectors as `.npy` files.
- `pandas`: saves metadata and class mappings as `.csv` files.
- `scikit-learn`: trains and evaluates the labeled DINO-embedding classifier.
- `matplotlib`: creates graph visualizations for entropy, density, and Kernel Herding.

`rembg` also installs several supporting packages, including `pymatting`, `scipy`, `scikit-image`, `numba`, `pooch`, and `tqdm`.

## First Run Notes

The first time the project runs, it may download model files:

- DINOv2 through `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
- Faster R-CNN through `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")`
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

## Entropy / KDE / Kernel Herding Data Finder

The entropy calculator is in `entropy_calculator.py`.

It uses DINO embeddings, not raw pixels. It has two stages.

First, it fits one kernel density estimation model per labeled class using:

```text
saved_vectors/learning/
```

Then it learns thresholds from data the classifier did not train on:

```text
saved_vectors/testing/
```

It marks an image as a candidate only when all of these are true:

- entropy is low, meaning the model is confident about that class
- the image is in-distribution, meaning it is not a complete KDE outlier
- representation gap is high, meaning it is not close to the learned examples already in that class

It still marks `complete_outlier` for images with extremely low density. Those are shown in the scores and graphs, but they are not part of the candidate subset.

Second, it creates an `interesting_subset.csv` file from only the target/test/unlabeled images that are interesting but not complete outliers. This is the small side-group of the data you care about.

Thresholds are learned separately for each class. For labeled test data, the code uses the actual class label. For unlabeled data, it uses the predicted class. This means an interesting zoomed-out cow affects the cow subset, but it does not change the pig subset.

Third, it applies Kernel Herding separately inside each class-specific candidate subset. This chooses a few representative examples per class-specific poorly represented subset.

Running `main.py` with `RUN_ENTROPY_CALCULATOR = True` writes:

```text
saved_vectors/learning/entropy_scores.csv
saved_vectors/learning/learning_entropy_density_plot.png
saved_vectors/learning/learning_pca_plot.png
saved_vectors/testing/entropy_scores.csv
saved_vectors/testing/interesting_subset.csv
saved_vectors/testing/herding_selection.csv
saved_vectors/testing/class_subset_summary.csv
saved_vectors/testing/target_entropy_density_plot.png
saved_vectors/testing/target_pca_plot.png
saved_vectors/testing/target_subset_entropy_density_plot.png
saved_vectors/testing/target_subset_pca_plot.png
saved_vectors/testing/class_subset_graphs/
```

Important columns:

- `predicted_label_name`: most likely class from the KDE probabilities
- `class_probability`: probability of that class
- `entropy`: uncertainty across classes
- `best_log_density`: how strongly the image fits its best class KDE
- `low_entropy`: `True` when the image is confidently assigned to its threshold class
- `in_distribution`: `True` when the image is not a complete KDE outlier
- `representation_score`: similarity to the closest learned example in that class
- `representation_gap`: `1 - representation_score`; higher means less represented by learned examples
- `poorly_represented`: `True` when representation gap is high for that class
- `interesting`: `True` when the image is confident, in-distribution, and poorly represented
- `complete_outlier`: `True` when the image is too far outside the learned density
- `herding_candidate`: `True` when the image is in the small candidate subset
- `herding_selected`: `True` when Kernel Herding selected the image
- `herding_rank`: selection order from Kernel Herding
- `threshold_label_id`: class id used for the class-specific threshold
- `threshold_label_name`: class name used for the class-specific threshold

`interesting_subset.csv` contains the small target-only candidate subset. `herding_selection.csv` contains the final smaller Kernel Herding subset, ordered by `herding_rank`. `class_subset_summary.csv` shows how many target images, interesting candidates, and selected images each class-specific subset has. Kernel Herding only selects from the target set, normally `saved_vectors/testing/` or another unlabeled vector folder. It does not select from the learning set.

The graph files show:

- `learning_entropy_density_plot.png`: entropy vs KDE density for the learned/labeled reference set
- `learning_pca_plot.png`: a 2D PCA view of the learned/labeled reference embeddings
- `target_entropy_density_plot.png`: entropy vs KDE density for test or unlabeled candidates, with Kernel Herding selections circled
- `target_pca_plot.png`: a 2D PCA view of the target candidates, with Kernel Herding selections circled
- `target_subset_entropy_density_plot.png`: only the small candidate subset, with Kernel Herding selections circled
- `target_subset_pca_plot.png`: only the small candidate subset in 2D PCA space, with Kernel Herding selections circled
- `class_subset_graphs/<class name>/representation_gap_entropy.png`: one class-specific subset graph per class
- `class_subset_graphs/<class name>/pca_subset.png`: one class-specific PCA subset graph per class

In each class folder, `representation_gap_entropy.png` shows all target images for that class in the background, the candidate subset on top, and Kernel Herding selections circled. The dashed horizontal line is the class-specific low-entropy cutoff. The dashed vertical line is the class-specific high-representation-gap cutoff. Candidate points are the class points that fall below the entropy cutoff and to the right of the representation-gap cutoff, while also not being complete KDE outliers.

## COCO Classifier

The current classifier is in `coco_classifier.py`.

It uses:

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
```

The model is pretrained on COCO and returns object detections:

- `label_id`
- `label_name`
- `score`
- `box`

Run the demo with:

```powershell
python main.py
```

Or classify one image directly:

```python
import coco_classifier as cc

detections = cc.classify_image(
    "coco/val2017/000000000139.jpg",
    score_threshold=0.5
)
cc.print_detections(detections)
```

Running `main.py` creates:

```text
training_data_no_bg/
saved_vectors/
  vectors.npy
  labels.npy
  metadata.csv
  class_mapping.csv
```
