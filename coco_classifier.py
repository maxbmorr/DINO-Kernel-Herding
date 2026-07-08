from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms import functional as F


def load_classifier(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.to(device)
    model.eval()

    return model, weights, device


def classify_image(
    image_path,
    model=None,
    weights=None,
    device=None,
    score_threshold=0.5,
    max_detections=10,
):
    if model is None or weights is None:
        model, weights, device = load_classifier(device)
    elif device is None:
        device = next(model.parameters()).device

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    categories = weights.meta["categories"]
    detections = []

    for box, label_id, score in zip(
        prediction["boxes"].cpu(),
        prediction["labels"].cpu(),
        prediction["scores"].cpu(),
    ):
        score = float(score)
        if score < score_threshold:
            continue

        label_id = int(label_id)
        detections.append({
            "image_path": str(image_path),
            "label_id": label_id,
            "label_name": categories[label_id],
            "score": score,
            "box": [float(value) for value in box.tolist()],
        })

        if len(detections) >= max_detections:
            break

    return detections


def classify_images(
    image_paths,
    score_threshold=0.5,
    max_detections=10,
    device=None,
):
    model, weights, device = load_classifier(device)
    results = {}

    for image_path in image_paths:
        results[str(image_path)] = classify_image(
            image_path,
            model=model,
            weights=weights,
            device=device,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )

    return results


def print_detections(detections):
    if not detections:
        print("No detections above threshold.")
        return

    for detection in detections:
        print(
            f"{detection['label_name']} "
            f"({detection['score']:.3f}) "
            f"box={detection['box']}"
        )


def demo(
    image_path="coco/val2017/000000000139.jpg",
    score_threshold=0.5,
    max_detections=10,
):
    detections = classify_image(
        image_path,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )
    print(f"Detections for {image_path}:")
    print_detections(detections)
    return detections


if __name__ == "__main__":
    demo()
