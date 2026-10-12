from pathlib import Path
import inspect

import numpy as np
import torch
from sklearn.svm import SVC

from pointjepa.models import PointJepa
from pointjepa.datasets.ModelNet40Ply2048 import ModelNet40Ply2048


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
rng = np.random.default_rng(42)

checkpoint_path = (
    ROOT / "checkpoints"
    / "pretrain_point-jepa_epoch=499-step=40500.ckpt"
)

# Load the pretrained encoder.
checkpoint = torch.load(checkpoint_path, map_location="cpu")
valid = inspect.signature(PointJepa.__init__).parameters

config = {
    k: v for k, v in checkpoint["hyper_parameters"].items()
    if k in valid
}

model = PointJepa(**config)
state = checkpoint["state_dict"]

for prefix, module in [
    ("tokenizer.", model.tokenizer),
    ("positional_encoding.", model.positional_encoding),
    ("student.", model.student),
]:
    weights = {
        k[len(prefix):]: v
        for k, v in state.items()
        if k.startswith(prefix)
    }
    module.load_state_dict(weights, strict=True)

model = model.to(DEVICE).eval()

# Load real ModelNet40 objects.
data_dir = ROOT / "data" / "modelnet40_ply_hdf5_2048"
train = ModelNet40Ply2048(str(data_dir), split="train")
test = ModelNet40Ply2048(str(data_dir), split="test")

class_names = (data_dir / "shape_names.txt").read_text().splitlines()


def select_indices(dataset, per_class):
    indices = []
    for label in range(40):
        candidates = np.flatnonzero(dataset.labels == label)
        indices.extend(rng.choice(candidates, per_class, replace=False))
    return indices


@torch.no_grad()
def extract_features(dataset, indices, batch_size=4):
    features = []
    labels = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        xyz = np.stack([dataset[i][0] for i in batch_indices])
        y = [int(dataset[i][1]) for i in batch_indices]

        points = torch.from_numpy(xyz).float().to(DEVICE)

        # Official validation preprocessing: FPS, center, unit sphere.
        points = model.val_transformations(points)

        tokens, centers = model.tokenizer(points)
        pos = model.positional_encoding(centers)
        x = model.student(tokens, pos).last_hidden_state

        # Original Point-JEPA linear-probing pooling.
        z = torch.cat(
            [x.max(dim=1).values, x.mean(dim=1)],
            dim=-1,
        )

        features.append(z.cpu().numpy())
        labels.extend(y)

    return np.concatenate(features), np.array(labels)


# Full ModelNet40 evaluation
train_indices = np.arange(len(train))
test_indices = np.arange(len(test))

print("Extracting training features...")
X_train, y_train = extract_features(train, train_indices)

print("Extracting test features...")
X_test, y_test = extract_features(test, test_indices)

print("Train features:", X_train.shape)
print("Test features:", X_test.shape)

# Original implementation uses a linear SVM with C=0.012.
svm = SVC(C=0.012, kernel="linear")
svm.fit(X_train, y_train)

accuracy = svm.score(X_test, y_test)
predictions = svm.predict(X_test)

print(f"\nFull ModelNet40 linear-probing accuracy: {accuracy * 100:.2f}%")

for actual, predicted in zip(y_test[:5], predictions[:5]):
    print(
        f"Actual: {class_names[actual]:15s} "
        f"Predicted: {class_names[predicted]}"
    )

print("\nLinear probing completed!")