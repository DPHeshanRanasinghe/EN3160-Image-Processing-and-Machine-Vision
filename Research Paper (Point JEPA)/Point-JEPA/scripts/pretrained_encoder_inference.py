from pathlib import Path

import torch
from pointjepa.models import PointJepa


# --------------------------------------------------
# 1. Load official pretrained checkpoint
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT / "checkpoints"
    / "pretrain_point-jepa_epoch=499-step=40500.ckpt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location="cpu")

# Reconstruct the model using its saved configuration
import inspect

saved_hparams = checkpoint["hyper_parameters"]
valid_params = inspect.signature(PointJepa.__init__).parameters

model_config = {
    key: value
    for key, value in saved_hparams.items()
    if key in valid_params
}

ignored_params = sorted(set(saved_hparams) - set(model_config))

print("Ignored checkpoint parameters:", ignored_params)

model = PointJepa(**model_config)

state = checkpoint["state_dict"]


# --------------------------------------------------
# 2. Load only the reusable encoder components
# --------------------------------------------------
def load_component(module, prefix):
    component_state = {
        name[len(prefix):]: tensor
        for name, tensor in state.items()
        if name.startswith(prefix)
    }

    if not component_state:
        raise RuntimeError(f"No weights found for {prefix}")

    module.load_state_dict(component_state, strict=True)
    print(f"Loaded {prefix}: {len(component_state)} tensors")


load_component(model.tokenizer, "tokenizer.")
load_component(model.positional_encoding, "positional_encoding.")
load_component(model.student, "student.")

model.to(device)
model.eval()


# --------------------------------------------------
# 3. Load a real ModelNet40 point cloud
# --------------------------------------------------
from pointjepa.datasets.ModelNet40Ply2048 import ModelNet40Ply2048

torch.manual_seed(42)

dataset_dir = ROOT / "data" / "modelnet40_ply_hdf5_2048"

dataset = ModelNet40Ply2048(
    root=str(dataset_dir),
    split="train"
)

points_np, label = dataset[0]

class_names = (
    dataset_dir / "shape_names.txt"
).read_text().splitlines()

print("\n=== Real ModelNet40 Object ===")
print("Object category:", class_names[int(label)])
print("Original shape:", points_np.shape)

# Convert NumPy array into a batched PyTorch tensor
points = torch.from_numpy(points_np).float()
points = points.unsqueeze(0).to(device)

# Apply the official Point-JEPA validation preprocessing:
# Subsample 2048 -> 1024, center, normalize to unit sphere
points = model.val_transformations(points)

print("Preprocessed shape:", points.shape)

# --------------------------------------------------
# 4. Run pretrained encoder inference
# --------------------------------------------------
with torch.no_grad():

    tokens, centers = model.tokenizer(points)

    positions = model.positional_encoding(centers)

    features = model.student(
        tokens, positions
    ).last_hidden_state

    max_features = features.max(dim=1).values
    mean_features = features.mean(dim=1)

    global_features = torch.cat(
        [max_features, mean_features], dim=-1
    )


# --------------------------------------------------
# 5. Display output dimensions
# --------------------------------------------------
print("\n=== Point-JEPA Inference ===")
print("Device:", device)
print("Input points:", points.shape)
print("Patch embeddings:", tokens.shape)
print("Patch centers:", centers.shape)
print("Transformer output:", features.shape)
print("Global representation:", global_features.shape)

print("\nInference completed successfully!")