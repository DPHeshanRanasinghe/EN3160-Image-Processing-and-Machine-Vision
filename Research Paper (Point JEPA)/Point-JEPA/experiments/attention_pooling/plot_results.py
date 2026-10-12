"""Create publication figures from saved CSV/JSON artifacts; no model inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/attention_pooling"
MODELS = ("baseline", "attention")
LABELS = {"baseline": "Max/Mean + MLP", "attention": "Max/Attention + MLP"}
COLORS = {"baseline": "#0072B2", "attention": "#D55E00"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-id", default="20261012T042856_572366Z_e467f11c")
    parser.add_argument("--full-evaluation-id", default="evaluation_20261012T043112_742087Z_346dc31c")
    parser.add_argument("--output-dir", default="experiments/attention_pooling/figures/full_modelnet40")
    parser.add_argument("--overwrite", action="store_true", help="Replace only known generated files in the output directory")
    parser.add_argument("--dpi", type=int, default=300)
    # These are external reference scalars, not fabricated experimental artifacts.
    parser.add_argument("--svm-accuracy", type=float, default=93.60)
    parser.add_argument("--paper-accuracy", type=float, default=93.7)
    parser.add_argument("--paper-spread", type=float, default=0.2)
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("--dpi must be at least 150")
    if not all(np.isfinite(value) and 0 <= value <= 100
               for value in (args.svm_accuracy, args.paper_accuracy, args.paper_spread)):
        parser.error("Reference accuracy/spread values must be finite and in [0, 100]")
    return args


def read_json(path, sources):
    sources.append(path)
    return json.loads(path.read_text())


def read_csv(path, sources):
    sources.append(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_experiment(run_id, evaluation_id, sources):
    directory = EXPERIMENT / "results" / run_id
    training = read_json(directory / "summary.json", sources)
    evaluation_dir = directory / evaluation_id
    evaluation = read_json(evaluation_dir / "evaluation_summary.json", sources)
    if training["run_id"] != run_id or evaluation["run_id"] != run_id:
        raise ValueError("Artifact run ID mismatch")
    names = evaluation["class_names"]
    if len(names) != 40 or len(set(names)) != 40:
        raise ValueError("Expected 40 unique class names")
    curves, matrices = {}, {}
    predictions = read_csv(evaluation_dir / "test_predictions.csv", sources)
    if len(predictions) != evaluation["num_samples"]:
        raise ValueError("Prediction count differs from evaluation summary")
    indices = [int(row["original_test_index"]) for row in predictions]
    if sorted(indices) != list(range(2468)):
        raise ValueError("Full predictions must cover every official test index exactly once")
    targets = np.array([int(row["ground_truth_label"]) for row in predictions])
    correct = {}
    for model in MODELS:
        rows = read_csv(directory / f"{model}_metrics.csv", sources)
        curves[model] = {key: np.array([float(row[key]) for row in rows])
                         for key in ("epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy")}
        if not all(np.isfinite(values).all() for values in curves[model].values()):
            raise ValueError("Non-finite training metrics")
        best = training["models"][model]
        selected = int(curves[model]["epoch"][curves[model]["val_loss"].argmin()])
        if selected != best["best_epoch"] or selected != evaluation["models"][model]["best_validation_epoch"]:
            raise ValueError("Checkpoint epoch does not match minimum validation loss")
        path = evaluation_dir / f"{model}_confusion_matrix.csv"
        sources.append(path)
        with path.open(newline="") as stream:
            matrix_rows = list(csv.reader(stream))
        if matrix_rows[0][1:] != names or [row[0] for row in matrix_rows[1:]] != names:
            raise ValueError("Confusion matrix class-name order mismatch")
        matrix = np.array([[int(value) for value in row[1:]] for row in matrix_rows[1:]])
        predicted = np.array([int(row[f"{model}_label"]) for row in predictions])
        recomputed = np.zeros((40, 40), dtype=np.int64)
        for row, target, prediction in zip(predictions, targets, predicted):
            if not 0 <= target < 40 or not 0 <= prediction < 40:
                raise ValueError("Prediction labels outside [0, 39]")
            if row["ground_truth_class"] != names[target] or row[f"{model}_prediction"] != names[prediction]:
                raise ValueError("Prediction class names disagree with labels")
            recomputed[target, prediction] += 1
        if matrix.shape != (40, 40) or not np.array_equal(matrix, recomputed):
            raise ValueError("Confusion CSV differs from saved predictions")
        correct[model] = predicted == targets
        metrics = evaluation["models"][model]
        if not np.isclose(correct[model].mean(), metrics["accuracy"]):
            raise ValueError("Saved accuracy differs from predictions")
        true_positive = matrix.diagonal().astype(float)
        precision = np.divide(true_positive, matrix.sum(0), out=np.zeros(40), where=matrix.sum(0) != 0)
        recall = np.divide(true_positive, matrix.sum(1), out=np.zeros(40), where=matrix.sum(1) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(40), where=precision + recall != 0)
        for key, value in (("macro_precision", precision.mean()), ("macro_recall", recall.mean()), ("macro_f1", f1.mean())):
            if not np.isclose(value, metrics[key]):
                raise ValueError(f"Saved {key} differs from confusion matrix")
        matrices[model] = matrix
    baseline, attention = correct["baseline"], correct["attention"]
    paired = {"both_correct": int((baseline & attention).sum()),
              "only_baseline_correct": int((baseline & ~attention).sum()),
              "only_attention_correct": int((~baseline & attention).sum()),
              "neither_correct": int((~baseline & ~attention).sum())}
    if paired != evaluation["paired_correctness_counts"]:
        raise ValueError("Saved paired counts differ from predictions")
    return {"training": training, "evaluation": evaluation, "curves": curves, "matrices": matrices}


def save_figure(fig, name, output, dpi):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf", flush=True)


def plot_curves(experiment, protocol, output, dpi):
    sizes = experiment["training"]["configuration"]["dataset_sizes"]
    for metric, ylabel, multiplier in (("loss", "Cross-entropy loss", 1), ("accuracy", "Accuracy (%)", 100)):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for model in MODELS:
            curve = experiment["curves"][model]
            for split, style in (("train", "-"), ("val", "--")):
                ax.plot(curve["epoch"], curve[f"{split}_{metric}"] * multiplier,
                        color=COLORS[model], linestyle=style, label=f"{LABELS[model]}: {split}")
            best_epoch = experiment["training"]["models"][model]["best_epoch"]
            selected = np.flatnonzero(curve["epoch"] == best_epoch)[0]
            ax.axvline(best_epoch, color=COLORS[model], alpha=0.35, linestyle=":")
            ax.scatter(best_epoch, curve[f"val_{metric}"][selected] * multiplier,
                       color=COLORS[model], marker="*", s=170, zorder=5,
                       label=f"{model}: selected epoch {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=ylabel,
               title=f"{protocol}: {sizes['train']:,} train / {sizes['val']:,} validation")
        if metric == "accuracy":
            ax.set_ylim(0, 100)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        save_figure(fig, f"{protocol.lower()}_{metric}_curves", output, dpi)


def plot_confusions(full, output, dpi):
    names = full["evaluation"]["class_names"]
    vmax = max(matrix.max() for matrix in full["matrices"].values())
    for model in MODELS:
        fig, ax = plt.subplots(figsize=(12, 11))
        matrix = full["matrices"][model]
        # Vector cells preserve sharpness when the PDF is enlarged for print.
        im = ax.pcolormesh(matrix, cmap="Blues", vmin=0, vmax=vmax,
                           shading="flat", rasterized=False)
        ax.set(xlim=(0, 40), ylim=(40, 0), aspect="equal")
        ax.set_xticks(np.arange(40) + 0.5, labels=names, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(40) + 0.5, labels=names, fontsize=8)
        ax.set(xlabel="Predicted class", ylabel="True class",
               title=f"{LABELS[model]} — full ModelNet40 test split (N=2,468)")
        for i, j in zip(*np.nonzero(matrix)):
            ax.text(j + 0.5, i + 0.5, str(matrix[i, j]), ha="center", va="center", fontsize=5.5,
                    color="white" if matrix[i, j] > vmax / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Object count (shared scale)")
        fig.tight_layout()
        save_figure(fig, f"full_{model}_confusion_matrix", output, dpi)


def plot_metrics(full, output, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 1]})
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    x = np.arange(4)
    for offset, model in zip((-0.18, 0.18), MODELS):
        values = [100 * full["evaluation"]["models"][model][key] for key in keys]
        bars = axes[0].bar(x + offset, values, width=0.36, color=COLORS[model], label=LABELS[model])
        axes[0].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    axes[0].set_xticks(x, ["Accuracy", "Macro precision", "Macro recall", "Macro F1"])
    axes[0].set(ylabel="Percent", ylim=(0, 105), title="Full test metrics (N=2,468)")
    axes[0].legend(loc="lower left", fontsize=9)
    losses = [full["evaluation"]["models"][model]["cross_entropy_loss"] for model in MODELS]
    bars = axes[1].bar(range(2), losses, color=[COLORS[model] for model in MODELS])
    axes[1].bar_label(bars, fmt="%.6f", fontsize=9, padding=3)
    axes[1].set_xticks(range(2), ["Mean", "Attention"])
    axes[1].set(ylabel="Cross-entropy", ylim=(0, max(losses) * 1.25), title="Loss (lower is better)")
    fig.tight_layout()
    save_figure(fig, "full_test_metric_comparison", output, dpi)


def plot_paired(full, output, dpi):
    counts = full["evaluation"]["paired_correctness_counts"]
    keys = ["both_correct", "only_baseline_correct", "only_attention_correct", "neither_correct"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(4), [counts[key] for key in keys],
                   color=["#009E73", COLORS["baseline"], COLORS["attention"], "#777777"])
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_yticks(range(4), ["Both correct", "Baseline only", "Attention only", "Neither correct"])
    ax.invert_yaxis()
    ax.set(xlabel="Objects", xlim=(0, max(counts.values()) * 1.12),
           title="Paired predictions on the same 2,468 test objects")
    fig.tight_layout()
    save_figure(fig, "full_paired_correctness", output, dpi)


def plot_summary(full, args, output):
    values = [args.paper_accuracy, args.svm_accuracy,
              100 * full["evaluation"]["models"]["baseline"]["accuracy"],
              100 * full["evaluation"]["models"]["attention"]["accuracy"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Paper-reported linear SVM\n9,840 training objects",
              "Our reproduced linear SVM\n9,840 training objects",
              "Our Max/Mean + MLP\n8,856 train / 984 validation",
              "Our Max/Attention + MLP\n8,856 train / 984 validation"]
    for i, (value, color) in enumerate(zip(values, ["#555555", "#009E73", COLORS["baseline"], COLORS["attention"]])):
        ax.errorbar(value, i, xerr=args.paper_spread if i == 0 else None,
                    fmt="o", color=color, capsize=5, markersize=8)
        label = f"{value:.1f} ± {args.paper_spread:.1f}%" if i == 0 else f"{value:.2f}%"
        ax.annotate(label, (value, i), xytext=(28 if i == 0 else 12, -4), textcoords="offset points")
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set(xlabel="Full ModelNet40 test accuracy (%) — zoomed axis", xlim=(90, 96),
           title="Point-JEPA: reproduction and pooling extension (2,468 test objects)")
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.03, "Protocols differ. Paper ±0.2 is reported variability, not our confidence interval.\n"
             "Our neural results use one seed; attention has higher accuracy but worse macro F1 and loss.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, "publication_summary", output, args.dpi)


def main():
    args = parse_args()
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    sources = []
    full = load_experiment(args.full_run_id, args.full_evaluation_id, sources)
    if full["evaluation"]["num_samples"] != 2468:
        raise ValueError("Expected the saved full evaluation of all 2,468 objects")
    base = Path(args.output_dir)
    if not base.is_absolute():
        base = ROOT / base
    output = base
    expected_names = ["full_loss_curves", "full_accuracy_curves",
                      "full_baseline_confusion_matrix", "full_attention_confusion_matrix",
                      "full_test_metric_comparison", "full_paired_correctness", "publication_summary"]
    expected_files = {f"{name}.{extension}" for name in expected_names for extension in ("png", "pdf")}
    expected_files.add("figure_manifest.json")
    if output.exists():
        existing = list(output.iterdir())
        if existing and not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}; use --overwrite deliberately")
        if any(not path.is_file() or path.name not in expected_files for path in existing):
            raise ValueError("Unexpected output content: preserve it and choose another output directory")
    output.mkdir(parents=True, exist_ok=True)
    plot_curves(full, "Full", output, args.dpi)
    plot_confusions(full, output, args.dpi)
    plot_metrics(full, output, args.dpi)
    plot_paired(full, output, args.dpi)
    plot_summary(full, args, output)
    manifest = {"configuration": vars(args), "matplotlib_version": matplotlib.__version__,
                "source_artifacts": [{"path": str(path.relative_to(ROOT)),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                     for path in sources],
                "reference_scalars": {"svm_accuracy": args.svm_accuracy,
                                      "svm_source": "Project-reported reproduction; no saved SVM metrics JSON supplied",
                                      "paper_accuracy": args.paper_accuracy, "paper_spread": args.paper_spread,
                                      "paper_source": "https://arxiv.org/abs/2404.16432; upstream README model zoo"},
                "figures": sorted(path.name for path in output.glob("*.png"))}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"All artifact consistency checks passed. Figures: {output}")


if __name__ == "__main__":
    main()
