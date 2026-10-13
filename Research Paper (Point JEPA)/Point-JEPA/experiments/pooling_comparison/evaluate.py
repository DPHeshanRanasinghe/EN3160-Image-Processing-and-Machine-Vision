"""Evaluate best-validation classifiers on the full official ModelNet40 test split."""

import argparse
import csv
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import h5py
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .pooling_models import MODEL_CLASSES
    from .provenance import binding_for_checkpoint, validate_cache_binding
else:
    from pooling_models import MODEL_CLASSES
    from provenance import binding_for_checkpoint, validate_cache_binding


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/pooling_comparison"
DEFAULT_CACHE = "experiments/pooling_comparison/cache/full_modelnet40_seed42_val10_bs4.pt"
FULL_DESCRIPTION = "Full official ModelNet40 test split: all 2,468 objects"


def evaluation_protocol(cache):
    if cache["metadata"]["format_version"] != 2:
        raise ValueError("Full ModelNet40 evaluation requires metadata format version 2")
    return 2468, FULL_DESCRIPTION


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.run_id in (".", "..") or not args.run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.run_id
    ):
        parser.error("--run-id must be a directory name using letters, digits, '_' or '-'")
    return args


def root_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_source_labels(cache, split):
    """Read only source HDF5 labels to verify index/label provenance."""
    manifest = cache["metadata"]["source_file_manifest"][split]
    paths = [item["path"] for item in manifest]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{split}: source manifest must be sorted and unique")
    labels = []
    for item in manifest:
        with h5py.File(root_path(item["path"]), "r") as stream:
            values = np.asarray(stream["label"][:], dtype=np.int64).reshape(-1)
        if len(values) != item["num_samples"]:
            raise ValueError(f"{split}: source HDF5 count differs from cache manifest")
        labels.append(values)
    return np.concatenate(labels)


def load_checkpoints(run_id):
    """Reconstruct the saved architectures, with no fallback or partial loading."""
    models, checkpoints, paths = {}, {}, {}
    with (EXPERIMENT / "results" / run_id / "summary.json").open() as stream:
        run_summary = json.load(stream)
    names = list(run_summary["models"])
    if not names or any(name not in MODEL_CLASSES for name in names):
        raise ValueError("Run contains no models or unsupported model names")
    for name in names:
        model_class = MODEL_CLASSES[name]
        path = EXPERIMENT / "checkpoints" / run_id / f"{name}_best.pt"
        binding = binding_for_checkpoint(path)
        checkpoint = torch.load(path, map_location="cpu")
        # In-memory relocation annotation; historical checkpoint bytes stay intact.
        checkpoint['_relocation_binding'] = binding
        config = checkpoint["configuration"]
        architecture = config["architecture"]
        if checkpoint["architecture_name"] != model_class.__name__:
            raise ValueError(f"{name}: incorrect checkpoint architecture name")
        if set(architecture) != {"feature_dim", "hidden_dim", "num_classes", "dropout"}:
            raise ValueError(f"{name}: unexpected architecture configuration")
        if architecture["feature_dim"] != 384 or architecture["num_classes"] != 40:
            raise ValueError(f"{name}: expected 384 feature dimensions and 40 classes")
        if architecture["hidden_dim"] != config["hidden_dim"] or architecture["dropout"] != config["dropout"]:
            raise ValueError(f"{name}: inconsistent saved architecture settings")
        if checkpoint["seed"] != config["seed"] or not isinstance(checkpoint["seed"], int):
            raise ValueError(f"{name}: inconsistent training seed")
        if config["checkpoint_selection"] != "minimum validation loss":
            raise ValueError(f"{name}: checkpoint was not selected by minimum validation loss")
        if not 1 <= checkpoint["best_epoch"] <= config["epochs"]:
            raise ValueError(f"{name}: invalid best epoch")
        if not math.isfinite(checkpoint["validation_loss"]) or checkpoint["validation_loss"] < 0:
            raise ValueError(f"{name}: invalid saved validation loss")
        if not 0 <= checkpoint["validation_accuracy"] <= 1:
            raise ValueError(f"{name}: invalid saved validation accuracy")
        model = model_class(**architecture)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError(f"{name}: non-finite saved model parameters")
        model.eval()
        model.requires_grad_(False)
        models[name], checkpoints[name], paths[name] = model, checkpoint, path
        print(f"{name}: {path}; strictly loaded {model_class.__name__}; "
              f"best validation epoch={checkpoint['best_epoch']}", flush=True)
    configurations = [checkpoint["configuration"] for checkpoint in checkpoints.values()]
    if any(config != configurations[0] for config in configurations[1:]):
        raise ValueError("Checkpoint training configurations differ")
    return models, checkpoints, paths


def validate_cache_metadata(cache, cache_path, checkpoints):
    """Check provenance without accessing test feature tensors."""
    metadata = cache["metadata"]
    class_names = cache["class_names"]
    if len(class_names) != 40 or not all(isinstance(name, str) and name for name in class_names):
        raise ValueError("Expected 40 nonempty cached class names")
    if len(set(class_names)) != 40:
        raise ValueError("Cached class names are not unique")
    evaluation_protocol(cache)
    if metadata["dataset"] != "ModelNet40Ply2048":
        raise ValueError("Unsupported cache format or dataset")
    if metadata["feature_shape_per_sample"] != [64, 384] or metadata["cache_dtype"] != "float16":
        raise ValueError("Unexpected cached feature shape or dtype metadata")
    if metadata["feature_source"] != "student.last_hidden_state; no masking, sequencing, or pooling":
        raise ValueError("Expected frozen contextualized student patch features")
    if metadata["preprocessing"] != ["FPS subsample to 1024 (random start)",
                                     "center", "normalize to unit sphere"]:
        raise ValueError("Unexpected preprocessing protocol")
    if not isinstance(cache["seed"], int) or metadata["extraction_seeds"]["test"] != cache["seed"] + 2:
        raise ValueError("Inconsistent cache extraction seed metadata")
    if metadata["protocol"] != "full_modelnet40_stratified_train_validation":
        raise ValueError("Unexpected full-dataset protocol")
    if metadata["official_dataset_counts"] != {"train": 9840, "test": 2468, "classes": 40}:
        raise ValueError("Incorrect official dataset counts")
    if metadata["subset_counts"] != {"train": 8856, "val": 984, "test": 2468}:
        raise ValueError("Expected the full 8,856/984 training-validation partition")
    protocol = metadata["split_protocol"]
    if (protocol["validation_fraction"] != 0.10 or protocol["random_state"] != cache["seed"]
            or protocol["stratify"] != "official training labels"):
        raise ValueError("Inconsistent stratified split metadata")
    combined = []
    for split in ("train", "val"):
        entry = cache[split]
        count = metadata["subset_counts"][split]
        if entry["features"].shape != (count, 64, 384):
            raise ValueError(f"{split}: incorrect full-cache feature dimensions")
        selected, labels = entry["indices"], entry["labels"]
        if selected.shape != (count,) or selected.dtype != torch.int64:
            raise ValueError(f"{split}: invalid original indices")
        if len(torch.unique(selected)) != count:
            raise ValueError(f"{split}: duplicate original indices")
        if labels.shape != (count,) or labels.dtype != torch.int64:
            raise ValueError(f"{split}: invalid labels")
        if not torch.equal(torch.unique(labels), torch.arange(40)):
            raise ValueError(f"{split}: expected all 40 classes")
        counts = [metadata["per_class_counts"][split][name] for name in class_names]
        if torch.bincount(labels, minlength=40).tolist() != counts:
            raise ValueError(f"{split}: labels differ from cached class-count metadata")
        combined.append(selected)
    if not torch.equal(torch.cat(combined).sort().values, torch.arange(9840)):
        raise ValueError("Train/validation must be disjoint and cover all 9,840 original indices")
    # Reproduce split membership from labels and seed without any test inference.
    from sklearn.model_selection import train_test_split
    original_labels = torch.empty(9840, dtype=torch.int64)
    for split in ("train", "val"):
        original_labels[cache[split]["indices"]] = cache[split]["labels"]
    if not np.array_equal(original_labels.numpy(), original_source_labels(cache, "train")):
        raise ValueError("Training/validation labels differ from original HDF5 objects")
    expected_train, expected_val = train_test_split(
        np.arange(9840), test_size=0.10, random_state=cache["seed"],
        stratify=original_labels.numpy(), shuffle=True)
    for split, expected in (("train", expected_train), ("val", expected_val)):
        if not np.array_equal(cache[split]["indices"].sort().values.numpy(), np.sort(expected)):
            raise ValueError(f"{split}: indices do not match the recorded stratified protocol")
    train_manifest = metadata["source_file_manifest"]["train"]
    train_offset = 0
    for item in train_manifest:
        if (not Path(item["path"]).name.startswith("ply_data_train")
                or item["start_index"] != train_offset or item["num_samples"] <= 0):
            raise ValueError("Invalid official training manifest")
        train_offset += item["num_samples"]
    if train_offset != 9840:
        raise ValueError("Training manifest must contain all 9,840 objects")
    manifest = metadata["source_file_manifest"]["test"]
    offset = 0
    for item in manifest:
        filename = Path(item["path"]).name
        if not filename.startswith("ply_data_test") or not filename.endswith(".h5"):
            raise ValueError("Test manifest contains a non-test HDF5 file")
        if item["start_index"] != offset or item["num_samples"] <= 0:
            raise ValueError("Invalid test manifest offsets")
        offset += item["num_samples"]
    if offset != 2468:
        raise ValueError("Expected an official test manifest containing 2,468 objects")
    for name, checkpoint in checkpoints.items():
        config = checkpoint["configuration"]
        validate_cache_binding(config, cache_path, checkpoint.get('_relocation_binding'))
        for split in ("train", "val"):
            expected = metadata["subset_counts"][split]
            if config["dataset_sizes"][split] != expected or len(cache[split]["labels"]) != expected:
                raise ValueError(f"{name}: {split} subset size differs from training")
            if cache[split]["source_split"] != "train":
                raise ValueError("Training/validation must originate from the official train split")
    return class_names


def load_test_data(cache):
    count, _ = evaluation_protocol(cache)
    entry = cache["test"]
    if entry["source_split"] != "test":
        raise ValueError("Held-out objects must originate from the official test split")
    features, labels, indices = entry["features"], entry["labels"], entry["indices"]
    if not all(isinstance(value, torch.Tensor) for value in (features, labels, indices)):
        raise ValueError("Expected tensor features, labels, and original indices")
    if (features.shape != (count, 64, 384) or features.dtype != torch.float16
            or features.device.type != "cpu"):
        raise ValueError(f"Expected CPU float16 test features with shape [{count}, 64, 384]")
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite test features")
    for name, values in (("labels", labels), ("indices", indices)):
        if values.shape != (count,) or values.dtype != torch.int64:
            raise ValueError(f"Expected int64 {name} with shape [{count}]")
    if labels.min().item() < 0 or labels.max().item() >= 40:
        raise ValueError("Test labels must be in [0, 39]")
    if indices.min().item() < 0 or indices.max().item() >= 2468 or len(torch.unique(indices)) != count:
        raise ValueError("Invalid or duplicate original test indices")
    if not torch.equal(indices.sort().values, torch.arange(2468)):
        raise ValueError("Full test indices must cover all 2,468 official objects")
    if not torch.equal(torch.unique(labels), torch.arange(40)):
        raise ValueError("Full test labels must represent all 40 classes")
    expected_counts = [cache["metadata"]["per_class_counts"]["test"][name]
                       for name in cache["class_names"]]
    if torch.bincount(labels, minlength=40).tolist() != expected_counts:
        raise ValueError("Full test labels differ from cached class-count metadata")
    if not np.array_equal(labels.numpy(), original_source_labels(cache, "test")[indices.numpy()]):
        raise ValueError("Test labels differ from original HDF5 objects at the cached indices")
    # The full feature dataset stays on CPU; batches alone move to the device.
    return TensorDataset(features.float(), labels), labels.numpy(), indices.numpy()


def most_confused_pairs(matrix, class_names):
    # Directed pairs: true class -> predicted class. Ties use class-index order.
    pairs = [(int(matrix[i, j]), i, j) for i in range(40) for j in range(40)
             if i != j and matrix[i, j] > 0]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [{"true_class": class_names[i], "predicted_class": class_names[j],
             "count": count} for count, i, j in pairs[:5]]


@torch.no_grad()
def evaluate_model(model, loader, device, class_names):
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, predictions, targets = 0.0, [], []
    for features, labels in loader:
        logits = model(features.to(device, non_blocking=True))
        if logits.shape != (len(labels), 40) or not torch.isfinite(logits).all():
            raise RuntimeError("Expected finite logits with shape [B, 40]")
        loss = criterion(logits, labels.to(device, non_blocking=True))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite test loss")
        total_loss += loss.item()
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    matrix = confusion_matrix(targets, predictions, labels=np.arange(40))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(40), average="macro", zero_division=0)
    metrics = {"accuracy": float((predictions == targets).mean()),
               "cross_entropy_loss": total_loss / len(targets),
               "macro_precision": float(precision), "macro_recall": float(recall),
               "macro_f1": float(f1), "num_samples": len(targets),
               "num_correct": int((predictions == targets).sum()),
               "most_confused_pairs": most_confused_pairs(matrix, class_names)}
    model.cpu()
    return metrics, predictions, matrix


def save_artifacts(output, summary, labels, indices, predictions, matrices, class_names):
    with (output / "evaluation_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "test_predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        names = list(predictions)
        writer.writerow(["original_test_index", "ground_truth_label", "ground_truth_class"] +
                         [field for name in names for field in
                          (f"{name}_label", f"{name}_prediction")])
        for row, (index, label) in enumerate(zip(indices, labels)):
            values = [int(index), int(label), class_names[label]]
            for name in names:
                prediction = int(predictions[name][row])
                values.extend([prediction, class_names[prediction]])
            writer.writerow(values)
    for name, matrix in matrices.items():
        with (output / f"{name}_confusion_matrix.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_class / predicted_class"] + class_names)
            for class_name, row in zip(class_names, matrix):
                writer.writerow([class_name] + row.tolist())


def main():
    args = parse_args()
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    print(f"Evaluating training run: {args.run_id}; device={device}", flush=True)
    models, checkpoints, checkpoint_paths = load_checkpoints(args.run_id)
    cache_path = root_path(args.cache)
    cache = torch.load(cache_path, map_location="cpu")
    class_names = validate_cache_metadata(cache, cache_path, checkpoints)
    count, scope = evaluation_protocol(cache)
    print(scope, flush=True)
    dataset, labels, indices = load_test_data(cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    summary = {"run_id": args.run_id, "evaluation_scope": scope,
               "metadata_format_version": cache["metadata"]["format_version"],
               "num_samples": count, "official_test_split_size": 2468,
               "experimental_limitations": [
                   "An exploratory evaluation previously examined part of the official test split; it was not entirely untouched throughout development.",
                   "Single training seed; MLP uses a train/validation partition, unlike the paper's SVM trained on all 9,840 objects."],
               "cache_path": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path), "cache_sha256": sha256(cache_path),
               "cache_seed": cache["seed"], "training_seed": next(iter(checkpoints.values()))["seed"],
               "encoder_checkpoint_identifier": cache["checkpoint_identifier"],
               "class_names": class_names, "batch_size": args.batch_size,
               "device": str(device), "torch_version": str(torch.__version__),
               "training_configuration": next(iter(checkpoints.values()))["configuration"],
               "cache_provenance_check": "Recorded training path and metadata; training checkpoints contain no historical cache hash",
               "confusion_matrix_convention": "rows=true class, columns=predicted class",
               "confused_pair_convention": "directed true class -> predicted class; positive off-diagonal counts",
               "macro_metric_convention": "all 40 classes; undefined precision/recall set to zero",
               "models": {}}
    del cache
    predictions, matrices = {}, {}
    for name, model in models.items():
        metrics, predictions[name], matrices[name] = evaluate_model(model, loader, device, class_names)
        checkpoint = checkpoints[name]
        summary["models"][name] = {
            **metrics, "architecture_name": checkpoint["architecture_name"],
            "checkpoint_path": str(checkpoint_paths[name].relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint_paths[name]),
            "best_validation_epoch": checkpoint["best_epoch"],
            "selection_validation_loss": checkpoint["validation_loss"],
            "selection_validation_accuracy": checkpoint["validation_accuracy"],
        }
        print(f"{name} ({count} test objects): accuracy={metrics['accuracy']:.2%}, "
              f"loss={metrics['cross_entropy_loss']:.6f}, macro precision={metrics['macro_precision']:.4f}, "
              f"recall={metrics['macro_recall']:.4f}, F1={metrics['macro_f1']:.4f}", flush=True)
        print(f"{name} most confused directed class pairs: {metrics['most_confused_pairs']}")
    if "baseline" in predictions and "attention" in predictions:
        baseline_correct = predictions["baseline"] == labels
        attention_correct = predictions["attention"] == labels
        paired = {"both_correct": int((baseline_correct & attention_correct).sum()),
                  "only_baseline_correct": int((baseline_correct & ~attention_correct).sum()),
                  "only_attention_correct": int((~baseline_correct & attention_correct).sum()),
                  "neither_correct": int((~baseline_correct & ~attention_correct).sum())}
        summary["paired_correctness_counts"] = paired
        print(f"Paired comparison ({count} test objects): {paired}")
    evaluation_id = "evaluation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    output = EXPERIMENT / "results" / args.run_id / evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    summary["evaluation_id"] = evaluation_id
    save_artifacts(output, summary, labels, indices, predictions, matrices, class_names)
    print(f"Saved evaluation artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
