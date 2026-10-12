"""Shared frozen Point-JEPA loading and unpooled feature extraction helpers.

Encoder tensors load strictly; source HDF5 files use deterministic sorted order.
Preprocessing and extraction reuse the validated downstream encoder procedure.
"""

import argparse
import glob
import hashlib
import inspect
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
# Allow direct execution without requiring an editable package installation.
sys.path.insert(0, str(ROOT))

from pointjepa.datasets.ModelNet40Ply2048 import ModelNet40Ply2048
from pointjepa.models import PointJepa


CHECKPOINT = ROOT / "checkpoints/pretrain_point-jepa_epoch=499-step=40500.ckpt"
DATA_DIR = ROOT / "data/modelnet40_ply_hdf5_2048"
CACHE_DIR = ROOT / "experiments/attention_pooling/cache"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return value

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_encoder(device):
    """Use the working reference procedure, with strict component loading."""
    print(f"Loading checkpoint: {CHECKPOINT.relative_to(ROOT)}", flush=True)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    saved_hparams = checkpoint["hyper_parameters"]
    valid_params = inspect.signature(PointJepa.__init__).parameters
    config = {key: value for key, value in saved_hparams.items()
              if key in valid_params and key != "self"}
    ignored_params = sorted(set(saved_hparams) - set(config))
    print("Ignored checkpoint configuration parameters:", ignored_params, flush=True)

    # Prevent the constructor's optional non-strict secondary checkpoint load.
    if config.get("pretrained_ckpt_path") is not None:
        print("Overriding pretrained_ckpt_path=None; loading the specified checkpoint strictly.")
    config["pretrained_ckpt_path"] = None

    # These are protocol requirements, not changes to pretrained weight shapes.
    config["val_transformations"] = ["subsample", "center", "unit_sphere"]
    config["transformation_subsample_points"] = 1024
    model = PointJepa(**config)
    if model.tokenizer.grouping.num_groups != 64 or model.tokenizer.token_dim != 384:
        raise RuntimeError("Expected the original 64-patch, 384-dimensional encoder")

    state = checkpoint["state_dict"]
    loaded_tensors = {}
    for prefix, module in [
        ("tokenizer.", model.tokenizer),
        ("positional_encoding.", model.positional_encoding),
        ("student.", model.student),
    ]:
        weights = {name[len(prefix):]: tensor for name, tensor in state.items()
                   if name.startswith(prefix)}
        if not weights:
            raise RuntimeError(f"No pretrained weights found for {prefix}")
        # Missing, unexpected, or incompatible encoder tensors must raise.
        module.load_state_dict(weights, strict=True)
        loaded_tensors[prefix] = len(weights)
        print(f"Strictly loaded {prefix}: {len(weights)} tensors", flush=True)

    unused = [name for name in state if not name.startswith(
        ("tokenizer.", "positional_encoding.", "student."))]
    print(f"Excluded {len(unused)} non-encoder tensors (e.g. teacher/predictor).")
    # Only the three reusable components need GPU memory for extraction.
    for module in (model.tokenizer, model.positional_encoding, model.student):
        module.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, {
        "ignored_checkpoint_configuration_parameters": ignored_params,
        "strictly_loaded_tensor_counts": loaded_tensors,
        "excluded_non_encoder_tensor_names": unused,
        "pretrained_ckpt_path_override": None,
    }

def load_dataset(split):
    """Keep the original dataset class, but make its HDF5 ordering stable."""
    files = sorted(glob.glob(str(DATA_DIR / f"ply_data_{split}*.h5")))
    if not files:
        raise FileNotFoundError(f"No ModelNet40 {split} HDF5 files in {DATA_DIR}")
    # The repository class uses unsorted glob internally. Scope this override
    # to construction; no repository implementation or source data is changed.
    with patch.object(glob, "glob", return_value=files):
        dataset = ModelNet40Ply2048(str(DATA_DIR), split=split)
    manifest = []
    offset = 0
    for path, data in zip(files, dataset.data_list):
        manifest.append({"path": str(Path(path).relative_to(ROOT)),
                         "start_index": offset, "num_samples": len(data)})
        offset += len(data)
    print(f"Official {split} dataset: points={dataset.data.shape}, labels={dataset.labels.shape}")
    return dataset, manifest

@torch.no_grad()
def extract_features(model, dataset, indices, device, batch_size, split, seed):
    """Preserve every student patch; pool only in later research stages."""
    # Reset separately from initialization so random FPS starts are reproducible
    # for the same seed, batch size, device, and software environment.
    seed_everything(seed)
    features = torch.empty((len(indices), 64, 384), dtype=torch.float16)
    labels = torch.from_numpy(dataset.labels[indices].copy()).long()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        xyz = np.stack([dataset[int(index)][0] for index in batch_indices])
        points = torch.from_numpy(xyz).float().to(device)
        points = model.val_transformations(points)
        if tuple(points.shape[1:]) != (1024, 3):
            raise RuntimeError(f"Unexpected preprocessed points shape: {tuple(points.shape)}")
        tokens, centers = model.tokenizer(points)
        positions = model.positional_encoding(centers)
        patches = model.student(tokens, positions).last_hidden_state
        expected = (len(batch_indices), 64, 384)
        if tuple(patches.shape) != expected:
            raise RuntimeError(f"Expected {expected}, got {tuple(patches.shape)}")
        cached = patches.to(device="cpu", dtype=torch.float16)
        if not torch.isfinite(cached).all():
            raise RuntimeError(f"Non-finite cached features in {split} batch at {start}")
        features[start:start + len(batch_indices)] = cached
        if start == 0 or (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(indices):
            print(f"{split}: {min(start + batch_size, len(indices))}/{len(indices)} samples", flush=True)
    print(f"{split}: features={tuple(features.shape)}, labels={tuple(labels.shape)}, "
          f"indices={indices.shape}, dtype={features.dtype}, device={features.device}")
    return {"features": features, "labels": labels,
            "indices": torch.from_numpy(indices.copy()),
            "source_split": "test" if split == "test" else "train"}
