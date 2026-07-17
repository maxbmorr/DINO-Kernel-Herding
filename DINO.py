import torch
import json
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ImagePathDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.samples = [
            str(path)
            for path in sorted(self.dataset_path.rglob("*"))
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not self.samples:
            raise FileNotFoundError(f"No images found in {self.dataset_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, -1

class CocoImageDataset(Dataset):
    def __init__(self, dataset_path, annotation_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.annotation_path = Path(annotation_path)
        self.transform = transform

        with self.annotation_path.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        categories = sorted(coco.get("categories", []), key=lambda category: category["id"])
        self.category_id_to_label_id = {
            category["id"]: index
            for index, category in enumerate(categories)
        }
        self.classes = [category["name"] for category in categories]

        annotations_by_image_id = {}
        for annotation in coco.get("annotations", []):
            category_id = annotation.get("category_id")
            if category_id not in self.category_id_to_label_id:
                continue

            annotations_by_image_id.setdefault(annotation["image_id"], []).append(annotation)

        self.samples = []
        self.sample_metadata = []

        for image_info in sorted(coco.get("images", []), key=lambda image: image["file_name"]):
            image_path = self.dataset_path / image_info["file_name"]
            if not image_path.exists():
                continue

            annotations = annotations_by_image_id.get(image_info["id"], [])
            label_ids = sorted({
                self.category_id_to_label_id[annotation["category_id"]]
                for annotation in annotations
            })

            if annotations:
                primary_annotation = max(annotations, key=lambda annotation: annotation.get("area", 0))
                primary_label_id = self.category_id_to_label_id[primary_annotation["category_id"]]
            else:
                primary_label_id = -1

            self.samples.append((str(image_path), primary_label_id))
            self.sample_metadata.append({
                "image_id": image_info["id"],
                "all_label_ids": "|".join(str(label_id) for label_id in label_ids),
                "all_label_names": "|".join(self.classes[label_id] for label_id in label_ids),
            })

        if not self.samples:
            raise FileNotFoundError(
                f"No COCO images from {self.annotation_path} were found in {self.dataset_path}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def find_coco_annotation(dataset_path):
    dataset_path = Path(dataset_path)
    candidates = [
        dataset_path / "_annotations.coco.json",
        dataset_path / "annotations.json",
        dataset_path.parent / "annotations" / f"instances_{dataset_path.name}.json",
        dataset_path.parent / "annotations" / "instances_train2017.json",
        dataset_path.parent / "annotations" / "instances_val2017.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None

def load_DINO(): #used in the primary file to load DINO for use
    device = "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device)
    model.eval()
    print("DINOv2")
    print("Device", device)
    return device, model 

def setup():
    device, model = load_DINO()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225)
        )
    ])
    return device, transform, model

def DINO_Vector(dataset_path, annotation_path=None, return_metadata=False):
    device, transform, model = setup()
    dataset_path = Path(dataset_path)
    annotation_path = Path(annotation_path) if annotation_path is not None else find_coco_annotation(dataset_path)
    metadata = None

    if annotation_path is not None:
        data = CocoImageDataset(dataset_path, annotation_path, transform = transform)
        classes = data.classes
        paths = [path for path, label in data.samples]
        metadata = data.sample_metadata
    else:
        try:
            data = datasets.ImageFolder(dataset_path, transform = transform)
            classes = data.classes
            paths = [path for path, label in data.samples]
        except FileNotFoundError:
            data = ImagePathDataset(dataset_path, transform = transform)
            classes = ["unlabeled"]
            paths = data.samples

    loader = DataLoader(data, batch_size = 16, shuffle = False)

    embeddings_list = []
    labels_list = []

    processed_count = 0

    with torch.no_grad():
        for images, labels in loader:
            batch_size = images.size(0)
            batch_paths = paths[processed_count:processed_count + batch_size]
            print(
                f"Processing images {processed_count + 1}-"
                f"{processed_count + batch_size} of {len(paths)}: "
                f"{batch_paths[-1]}"
            )

            images = images.to(device)
            
            embeddings = model(images)

            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())
            processed_count += batch_size
    X = torch.cat(embeddings_list, dim = 0)
    Y = torch.cat(labels_list, dim = 0)
    if return_metadata:
        return X, Y, paths, classes, metadata
    return X, Y, paths, classes
