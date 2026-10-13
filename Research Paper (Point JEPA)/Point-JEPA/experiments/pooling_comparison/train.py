"""Train a controlled pooling comparison using only cached train/val features."""

import argparse
import csv
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .pooling_models import (
        MaxAttentionClassifier, MaxMeanClassifier, MODEL_CLASSES, count_trainable_parameters,
    )
else:
    from pooling_models import (
        MaxAttentionClassifier, MaxMeanClassifier, MODEL_CLASSES, count_trainable_parameters,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = "experiments/pooling_comparison/cache/full_modelnet40_seed42_val10_bs4.pt"


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_CLASSES),
                        default=["baseline", "attention"],
                        help="Classifiers to train (default: baseline attention); use max_min_std alone for the third model")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", default="experiments/pooling_comparison/results")
    parser.add_argument("--checkpoints-dir", default="experiments/pooling_comparison/checkpoints")
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-dim", type=positive_int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    return args


def root_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_training_data(cache_path):
    # PyTorch 1.13-compatible loading. Only train/val entries are accessed;
    # the serialized cache also contains held-out data, which stays unused.
    cache = torch.load(cache_path, map_location="cpu")
    datasets = {}
    indices = {}
    sources = {}
    for split in ("train", "val"):
        entry = cache[split]
        features, labels, original_indices = (
            entry["features"], entry["labels"], entry["indices"]
        )
        if not all(isinstance(value, torch.Tensor)
                   for value in (features, labels, original_indices)):
            raise ValueError(f"{split}: expected tensor features, labels, and indices")
        if features.ndim != 3 or tuple(features.shape[1:]) != (64, 384) or len(features) == 0:
            raise ValueError(f"{split}: expected nonempty [N, 64, 384], got {tuple(features.shape)}")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError(f"{split}: features must be finite floating-point tensors")
        for name, values in (("labels", labels), ("indices", original_indices)):
            if values.shape != (len(features),) or values.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            ):
                raise ValueError(f"{split}: expected integer {name} with shape [N]")
        if labels.min().item() < 0 or labels.max().item() >= 40:
            raise ValueError(f"{split}: labels must be in [0, 39]")
        if not torch.equal(torch.unique(labels).long(), torch.arange(40)):
            raise ValueError(f"{split}: expected all 40 ModelNet40 classes")
        if original_indices.min().item() < 0 or len(torch.unique(original_indices)) != len(features):
            raise ValueError(f"{split}: original indices must be nonnegative and unique")
        sources[split] = entry["source_split"]
        indices[split] = set(original_indices.tolist())
        # Float32 tensors stay on CPU; only individual batches move to the GPU.
        datasets[split] = TensorDataset(features.float(), labels.long())
        print(f"{split}: features={tuple(features.shape)}, labels={tuple(labels.shape)}, "
              f"compute dtype=float32, storage device=cpu", flush=True)
    if sources != {"train": "train", "val": "train"}:
        raise ValueError("Both subsets must originate from the official training split")
    if indices["train"] & indices["val"]:
        raise ValueError("Training and validation original indices overlap")
    print("Training and validation original indices are disjoint.", flush=True)
    return datasets


def make_loaders(datasets, batch_size, seed, device):
    # A fresh generator gives both models the same permutation in each epoch.
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                       generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    val = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                     generator=torch.Generator().manual_seed(seed), num_workers=0,
                     pin_memory=device.type == "cuda")
    return train, val


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    attention_gradient_nonzero = False
    with torch.enable_grad() if training else torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if logits.shape != (len(labels), 40) or not torch.isfinite(logits).all():
                raise RuntimeError("Expected finite classification logits with shape [B, 40]")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; stopping training")
            if training:
                loss.backward()
                if isinstance(model, MaxAttentionClassifier):
                    gradient = model.pooling.attention.weight.grad
                    if gradient is None or not torch.isfinite(gradient).all():
                        raise RuntimeError("Attention score weights did not receive finite gradients")
                    attention_gradient_nonzero |= gradient.abs().sum().item() > 0
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    if training and isinstance(model, MaxAttentionClassifier) and not attention_gradient_nonzero:
        raise RuntimeError("Attention score gradients were zero throughout the epoch")
    return {"loss": total_loss / total, "accuracy": correct / total,
            "samples": total, "attention_gradient_nonzero": attention_gradient_nonzero}


def train_model(name, model, datasets, args, config, device, results_dir, checkpoints_dir,
                criterion):
    # Reset dropout RNG as well as sample order before each independent run.
    seed_everything(args.seed)
    train_loader, val_loader = make_loaders(datasets, args.batch_size, args.seed, device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    csv_path = results_dir / f"{name}_metrics.csv"
    checkpoint_path = checkpoints_dir / f"{name}_best.pt"
    best_loss, best_epoch, best_accuracy = float("inf"), 0, 0.0
    stale_epochs = 0
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
              "attention_gradient_nonzero"]
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device)
            writer.writerow({"epoch": epoch, "train_loss": train_metrics["loss"],
                             "train_accuracy": train_metrics["accuracy"],
                             "val_loss": val_metrics["loss"],
                             "val_accuracy": val_metrics["accuracy"],
                             "attention_gradient_nonzero": train_metrics["attention_gradient_nonzero"]})
            stream.flush()
            print(f"{name} epoch {epoch}/{args.epochs}: "
                  f"train loss={train_metrics['loss']:.6f}, acc={train_metrics['accuracy']:.2%}; "
                  f"val loss={val_metrics['loss']:.6f}, acc={val_metrics['accuracy']:.2%}", flush=True)
            # Strict minimum validation loss alone decides checkpoint selection.
            if val_metrics["loss"] < best_loss:
                best_loss, best_epoch, best_accuracy = val_metrics["loss"], epoch, val_metrics["accuracy"]
                stale_epochs = 0
                torch.save({
                    "model_state_dict": {key: value.detach().cpu().clone()
                                         for key, value in model.state_dict().items()},
                    "architecture_name": type(model).__name__,
                    "configuration": config, "seed": args.seed,
                    "best_epoch": best_epoch, "validation_loss": best_loss,
                    "validation_accuracy": best_accuracy,
                }, checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"{name}: early stopping after {args.patience} epochs without improvement.")
                    break
    # Restore the selected model; final training metrics refer to the last epoch.
    best = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    print(f"{name}: saved and restored best epoch {best_epoch}; checkpoint={checkpoint_path}")
    model.cpu()
    return {"architecture_name": type(model).__name__, "epochs_completed": epoch,
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "best_validation_accuracy": best_accuracy,
            "final_training_metrics": train_metrics,
            "final_validation_metrics": val_metrics,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path), "metrics_csv": str(csv_path.relative_to(ROOT)) if csv_path.is_relative_to(ROOT) else str(csv_path)}


def main():
    args = parse_args()
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    cache_path = root_path(args.cache)
    datasets = load_training_data(cache_path)
    seed_everything(args.seed)
    architecture = {"feature_dim": 384, "hidden_dim": args.hidden_dim,
                    "num_classes": 40, "dropout": args.dropout}
    models = {name: MODEL_CLASSES[name](**architecture) for name in args.models}
    identical = None
    if "baseline" in models and "attention" in models:
        models["attention"].head.load_state_dict(models["baseline"].head.state_dict(), strict=True)
        identical = all(torch.equal(value, models["attention"].head.state_dict()[key])
                        for key, value in models["baseline"].head.state_dict().items())
        if not identical:
            raise RuntimeError("Initial classifier heads differ")
        print(f"Initial MLP head parameters identical: {identical}", flush=True)
    counts = {name: count_trainable_parameters(model) for name, model in models.items()}
    print(f"Device: {device}; trainable parameters: {counts}")
    print(f"Batch inputs: [B, 64, 384]; logits: [B, 40] (B <= {args.batch_size})")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    results_dir = root_path(args.results_dir) / run_id
    checkpoints_dir = root_path(args.checkpoints_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "cache_path": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path), "resolved_device": str(device),
              "architecture": architecture, "optimizer": "AdamW", "loss": "CrossEntropyLoss",
              "checkpoint_selection": "minimum validation loss",
              "torch_version": str(torch.__version__),
              "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()}}
    summary = {"run_id": run_id, "configuration": config, "cache_path": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path),
               "initial_heads_identical": identical, "initial_parameter_counts": counts,
               "models": {}}
    criterion = nn.CrossEntropyLoss()
    for name, model in models.items():
        summary["models"][name] = train_model(
            name, model, datasets, args, config, device, results_dir, checkpoints_dir, criterion)
    summary_path = results_dir / "summary.json"
    with summary_path.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Experiment summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
