"""Extract unpooled full-ModelNet40 features using metadata protocol version 2.

The train/val/test entries contain features, labels, indices, and source_split.
Version 2 uses a stratified fraction of the official training split and the
entire official test split, preserving its unequal class proportions.
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.model_selection import train_test_split

if __package__:
    from . import feature_utils as extractor
else:
    import feature_utils as extractor


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_COUNTS = {"train": 9840, "test": 2468, "classes": 40}


def validation_fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("Validation fraction must be finite and in (0, 1)")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=validation_fraction, default=0.10)
    parser.add_argument("--batch-size", type=extractor.positive_int, default=4)
    args = parser.parse_args()
    if not 0 <= args.seed <= 2 ** 32 - 1:
        parser.error("--seed must be in [0, 2**32 - 1] for sklearn splitting")
    return args


def select_full_indices(train, test, seed=42, val_fraction=0.10):
    """Randomly stratify only official training objects, then retain source order."""
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be in (0, 1)")
    for split, dataset in (("train", train), ("test", test)):
        if len(dataset) != OFFICIAL_COUNTS[split]:
            raise ValueError(f"Official {split} split must contain {OFFICIAL_COUNTS[split]} objects")
        if dataset.labels.shape != (len(dataset),) or not np.array_equal(
            np.unique(dataset.labels), np.arange(40)
        ):
            raise ValueError(f"{split}: expected labels covering all 40 classes")
    official_train_indices = np.arange(len(train), dtype=np.int64)
    train_indices, val_indices = train_test_split(
        official_train_indices, test_size=val_fraction,
        random_state=seed, stratify=train.labels, shuffle=True,
    )
    # Random membership, sorted extraction order: never reorder source datasets
    # or mix official training objects with official test objects.
    indices = {"train": np.sort(train_indices), "val": np.sort(val_indices),
               "test": np.arange(len(test), dtype=np.int64)}
    expected_val = math.ceil(len(train) * val_fraction)
    if len(indices["val"]) != expected_val or len(indices["train"]) != len(train) - expected_val:
        raise RuntimeError("Incorrect stratified split sizes")
    for split, selected in indices.items():
        source = test if split == "test" else train
        if len(np.unique(selected)) != len(selected):
            raise RuntimeError(f"Duplicate {split} indices")
        if not np.array_equal(np.unique(source.labels[selected]), np.arange(40)):
            raise ValueError(f"{split}: validation fraction leaves some classes unrepresented")
    if np.intersect1d(indices["train"], indices["val"]).size:
        raise RuntimeError("Training and validation indices overlap")
    if not np.array_equal(np.sort(np.concatenate([indices["train"], indices["val"]])),
                          official_train_indices):
        raise RuntimeError("Training and validation must partition the entire official training split")
    if not np.array_equal(indices["test"], np.arange(2468)):
        raise RuntimeError("Test indices must cover the entire official test split in order")
    print("Stratified train/val indices are unique, disjoint, and cover all 9,840 training objects.")
    print("Test indices cover all 2,468 official test objects in source order.")
    for split, selected in indices.items():
        print(f"{split}: {len(selected)} objects; expected features=({len(selected)}, 64, 384)")
    return indices


def validate_extracted_entry(entry, dataset, indices, split):
    """Verify cached labels and indices refer to the exact selected source objects."""
    features = entry["features"]
    if features.shape != (len(indices), 64, 384) or features.dtype != torch.float16:
        raise RuntimeError(f"{split}: unexpected feature shape or dtype")
    if any(entry[key].device.type != "cpu" for key in ("features", "labels", "indices")):
        raise RuntimeError(f"{split}: cache tensors must reside on CPU")
    if not torch.equal(entry["indices"], torch.from_numpy(indices)):
        raise RuntimeError(f"{split}: extracted indices differ from selection")
    if not torch.equal(entry["labels"], torch.from_numpy(dataset.labels[indices]).long()):
        raise RuntimeError(f"{split}: labels differ from original dataset samples")
    if entry["source_split"] != ("test" if split == "test" else "train"):
        raise RuntimeError(f"{split}: incorrect official source split")
    # The reused extractor rejects non-finite features in every batch, including
    # after float16 conversion; no large duplicate validation tensor is needed.


def save_atomic_no_overwrite(cache, output):
    """Publish a complete temporary file atomically, refusing an existing name."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {output}")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{output.name}.",
                                         suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            torch.save(cache, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link atomically publishes the completed file and
        # fails if another process created the destination. Unlike os.replace,
        # it never overwrites an existing cache, even in a concurrent run.
        os.link(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    args = parse_args()
    percentage = format(args.val_fraction * 100, ".12g")
    output = extractor.CACHE_DIR / (
        f"full_modelnet40_seed{args.seed}_val{percentage}_bs{args.batch_size}.pt")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {output}")
    extractor.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Full ModelNet40 extraction: seed={args.seed}, validation fraction={args.val_fraction}, "
          f"batch size={args.batch_size}, device={device}", flush=True)
    class_names = (extractor.DATA_DIR / "shape_names.txt").read_text().splitlines()
    if len(class_names) != 40 or len(set(class_names)) != 40:
        raise ValueError("Expected 40 unique ModelNet40 class names")
    train, train_manifest = extractor.load_dataset("train")
    test, test_manifest = extractor.load_dataset("test")
    indices = select_full_indices(train, test, args.seed, args.val_fraction)
    model, loading_metadata = extractor.load_encoder(device)
    per_class_counts = {
        split: {name: int(count) for name, count in zip(
            class_names, np.bincount((test if split == "test" else train).labels[selected], minlength=40))}
        for split, selected in indices.items()
    }
    cache = {
        "class_names": class_names, "seed": args.seed,
        "checkpoint_identifier": {"path": str(extractor.CHECKPOINT.relative_to(ROOT)),
                                  "sha256": extractor.checkpoint_sha256(extractor.CHECKPOINT)},
        "metadata": {
            "format_version": 2, "protocol": "full_modelnet40_stratified_train_validation",
            "dataset": "ModelNet40Ply2048",
            "dataset_root": str(extractor.DATA_DIR.relative_to(ROOT)),
            "official_dataset_counts": OFFICIAL_COUNTS,
            "source_file_manifest": {"train": train_manifest, "test": test_manifest},
            "index_convention": "Zero-based index in the sorted-HDF5 official source split",
            "split_protocol": {"method": "sklearn.model_selection.train_test_split",
                               "stratify": "official training labels", "random_state": args.seed,
                               "validation_fraction": args.val_fraction,
                               "validation_count_rule": "ceil(9840 * validation_fraction)",
                               "extraction_order": "ascending original indices within each source split",
                               "test_selection": "all official test objects; no train/test mixing"},
            "subset_counts": {split: len(selected) for split, selected in indices.items()},
            "per_class_counts": per_class_counts,
            "batch_size": args.batch_size, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "sklearn_version": sklearn.__version__,
            "preprocessing": ["FPS subsample to 1024 (random start)",
                              "center", "normalize to unit sphere"],
            "feature_source": "student.last_hidden_state; no masking, sequencing, or pooling",
            "feature_shape_per_sample": [64, 384], "compute_dtype": "float32",
            "cache_dtype": "float16", "labels_dtype": "int64", "indices_dtype": "int64",
            "cache_device": "cpu", "split_order": ["train", "val", "test"],
            "extraction_seeds": {"train": args.seed, "val": args.seed + 1, "test": args.seed + 2},
            "reproducibility": "FPS depends on seed, batch size, device, and software environment",
            "encoder_loading": loading_metadata,
        },
    }
    for offset, split in enumerate(("train", "val", "test")):
        dataset = test if split == "test" else train
        cache[split] = extractor.extract_features(
            model, dataset, indices[split], device, args.batch_size, split, args.seed + offset)
        validate_extracted_entry(cache[split], dataset, indices[split], split)
    print("All features extracted and validated; saving complete cache atomically.", flush=True)
    save_atomic_no_overwrite(cache, output)
    print(f"Saved full-dataset cache: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
