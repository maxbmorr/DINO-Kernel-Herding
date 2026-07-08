import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import __utils__ as ut


def _filter_labeled_rows(X, label_ids, metadata):
    labeled_mask = label_ids >= 0
    return X[labeled_mask], label_ids[labeled_mask], metadata[labeled_mask]


def train_classifier(input_dir="saved_vectors/learning"):
    X, label_ids, label_names, paths, metadata, class_mapping = ut.load_DINO_vectors(input_dir)
    X, label_ids, metadata = _filter_labeled_rows(X, label_ids, metadata)

    if len(label_ids) == 0:
        raise ValueError(
            "No labeled images found. Train with COCO annotations or class-folder images first."
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    )
    model.fit(X, label_ids)
    print(f"Trained on {len(label_ids)} labeled images from {input_dir}")
    return model, class_mapping


def evaluate_classifier(
    model,
    class_mapping,
    input_dir="saved_vectors/testing",
):
    X, label_ids, label_names, paths, metadata, _ = ut.load_DINO_vectors(input_dir)
    X, label_ids, metadata = _filter_labeled_rows(X, label_ids, metadata)

    if len(label_ids) == 0:
        raise ValueError("No labeled testing images found.")

    predicted_ids = model.predict(X)
    accuracy = accuracy_score(label_ids, predicted_ids)

    label_ids_in_report = sorted(np.unique(np.concatenate([label_ids, predicted_ids])))
    label_names_in_report = [
        class_mapping.loc[class_mapping["label_id"] == label_id, "label_name"].iloc[0]
        for label_id in label_ids_in_report
    ]

    print(f"Tested on {len(label_ids)} labeled images from {input_dir}")
    print(f"Accuracy: {accuracy:.3f}")
    print(
        classification_report(
            label_ids,
            predicted_ids,
            labels=label_ids_in_report,
            target_names=label_names_in_report,
            zero_division=0,
        )
    )

    results = pd.DataFrame({
        "path": metadata["path"].values,
        "actual_label_id": label_ids,
        "predicted_label_id": predicted_ids,
    })
    results["actual_label_name"] = results["actual_label_id"].map(
        dict(zip(class_mapping["label_id"], class_mapping["label_name"]))
    )
    results["predicted_label_name"] = results["predicted_label_id"].map(
        dict(zip(class_mapping["label_id"], class_mapping["label_name"]))
    )

    return accuracy, results


def train_and_evaluate(
    learning_dir="saved_vectors/learning",
    testing_dir="saved_vectors/testing",
):
    model, class_mapping = train_classifier(learning_dir)
    return evaluate_classifier(model, class_mapping, testing_dir)
