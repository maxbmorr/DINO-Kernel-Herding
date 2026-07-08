# DINO-Kernel-Herding
For CNiEL research 2026 for the Dr. Austin Brockmeier.  This is my primary project for my Summer Research.

## What the Code Currently Does

The current pipeline starts in `main.py`.

1. It calls `create_dataset()` from `__utils__.py` using `training_data/` as the input folder.
2. If background removal is enabled, each image is processed with `rembg`.
3. The background-removed images are saved into `training_data_no_bg/`, keeping the same class-folder structure.
4. `DINO.py` loads DINOv2 ViT-S/14 from `facebookresearch/dinov2` using `torch.hub`.
5. Images are resized, center-cropped, normalized, and passed through DINOv2.
6. The resulting feature vectors, labels, image paths, and class names are returned.
7. The vectors and metadata are saved in `saved_vectors/`.

At the moment, `main.py` prints:

- the shape of the DINO feature matrix `X`
- the label tensor `Y`
- the class names
- the image paths used to create the vectors

## Dependencies

This project uses Python 3.13 and the following Python packages:

```powershell
python -m pip install torch torchvision pillow rembg onnxruntime numpy pandas
```

Main dependencies:

- `torch`: runs the DINOv2 model and tensor operations.
- `torchvision`: loads image folders and applies image transforms.
- `pillow`: opens and converts image files.
- `rembg`: removes image backgrounds.
- `onnxruntime`: runs the background-removal model used by `rembg`.
- `numpy`: saves DINO vectors as `.npy` files.
- `pandas`: saves metadata and class mappings as `.csv` files.

`rembg` also installs several supporting packages, including `pymatting`, `scipy`, `scikit-image`, `numba`, `pooch`, and `tqdm`.

## First Run Notes

The first time the project runs, it may download model files:

- DINOv2 through `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
- `u2net.onnx` for background removal through `rembg`

The code stores the `rembg` model in the local `.u2net/` folder and Numba cache files in `.numba_cache/`.

## Data Format

Input images should be organized in class folders:

```text
training_data/
  cow/
    cow1.png
    cow2.png
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
