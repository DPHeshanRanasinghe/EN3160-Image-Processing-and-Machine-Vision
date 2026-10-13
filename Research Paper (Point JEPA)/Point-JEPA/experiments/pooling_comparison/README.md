# Point-JEPA pooling comparison — EN3160 research project

A frozen Point-JEPA encoder supplies contextualized patch features to one neural control baseline and two pooling extensions. Both extensions have completed full ModelNet40 training and evaluation. All results below are from retained experiment records; this reorganization required no retraining or official test inference.

## Attribution and project scope

Ayumu Saito, Prachi Kudeshia, and Jiju Poovvancheri, *Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud*, WACV 2025, [arXiv:2404.16432](https://arxiv.org/abs/2404.16432). Official implementation: [Ayumu-J-S/Point-JEPA](https://github.com/Ayumu-J-S/Point-JEPA).

The original `pointjepa/` implementation, configurations, pretrained encoder, and [MIT license](../../LICENSE) are preserved. The license credits Karim Abou Zeid (2023), reflecting the Point2Vec foundation acknowledged by Point-JEPA. Our contributions are the downstream experiment scripts, documentation, and figures.

This project is a directory within the parent EN3160 Git working tree, whose Git root is the parent `github/` directory. There is no independent nested `.git` directory in Point-JEPA. Run commands below from the **Point-JEPA project root**, not the parent Git root. The migration preserved Git metadata and unrelated parent-repository changes. Publication stages only the intended Point-JEPA research files.

## Research question and architectures

We compare two alternatives to max/mean pooling: replacing the mean vector with learned attention, and combining extrema with patchwise population dispersion. All three classifiers reuse the same frozen feature cache:

```text
2,048 XYZ points → random-start FPS to 1,024 → center → unit-sphere normalization
  → tokenizer + positional encoding + frozen student/context Transformer
  → 64 contextualized patches × 384 dimensions
  → pooling → MLP → 40 raw class logits
```

The pretrained encoder has 64 groups, 32 points per group, 384 feature dimensions, 12 Transformer layers, and 6 attention heads. Extraction uses `student.last_hidden_state` without pretraining context masking or sequencing. There is no CLS token, encoder fine-tuning, or repeated JEPA pretraining.

| Role | Classifier | Pooled representation | MLP input | Trainable parameters |
|---|---|---|---:|---:|
| Neural control | `MaxMeanClassifier` | Concatenated max and mean | 768 | 207,144 |
| Extension 1 | `MaxAttentionClassifier` | Concatenated max and learned attention | 768 | 207,528 |
| Extension 2 | `MaxMinStdClassifier` | Concatenated max, min, population std | 1,152 | 305,448 |

Every head uses `Linear(input_dim,256) → ReLU → Dropout(0.2) → Linear(256,40)`. The baseline and attention heads started with identical weights. The Max/Min/Std input is wider, so its first layer has more parameters; its head initialization is not identical to the other heads. Differences cannot be attributed to pooling independently of classifier capacity.

### Extension 1: Max + Attention + MLP

For `x` shaped `[B,64,384]`:

```python
scores = Linear(384, 1, bias=False)(x)
weights = softmax(scores, dim=1)
attention_features = sum(weights * x, dim=1)
global_features = concat(max_features, attention_features)
```

A bias-free scorer learns normalized weights across patches. A shared scalar bias would cancel in softmax. The classifier optionally returns attention weights `[B,64]` with `return_attention=True`; normal training returns logits `[B,40]`.

### Extension 2: Max + Min + Population Std + MLP

```python
x = x.float()
global_features = torch.cat([
    x.max(dim=1).values,
    x.min(dim=1).values,
    x.std(dim=1, unbiased=False),
], dim=-1)
```

Population deviation uses the divisor P=64, pooling over patch tokens (`dim=1`). **`unbiased=False` is retained.** The mean defines deviation internally; no mean vector is concatenated. Pooling has no learned parameters, produces `[B,1152]`, and gives zero deviation for a single patch.

## Original linear-SVM reproduction: separate protocol

The reference reproduction concatenates max/mean features and fits `SVC(C=0.012, kernel="linear")` on all 9,840 official training objects, then evaluates all 2,468 test objects. Its project-reported accuracy is **93.60%**; the upstream README reports **93.7 ± 0.2%** for the paper's SVM.

This SVM is scientifically distinct from the neural control. It uses the entire training split with no validation holdout, and separately extracts float32 pooled features. The MLPs use an 8,856/984 training/validation split and a shared float16 patch-feature cache. The reference scripts do not save an SVM metrics JSON; no SVM macro metrics, curves, predictions, or confusion matrix are invented. Paper variability is not a confidence interval for our single-seed experiments.

## Verified environment and required assets

| Component | Verified installation |
|---|---|
| OS | Ubuntu 24.04.4 LTS, x86_64 |
| Python | 3.10.22, conda-forge |
| Python executable | `/home/heshan-ranasinghe/miniforge3/envs/pointjepa/bin/python` |
| PyTorch | 1.13.1+cu117 |
| PyTorch3D | 0.7.5 |
| PyTorch CUDA runtime | 11.7 |
| cuDNN | 8.5.0 (reported version 8500) |
| GPU | NVIDIA GeForce GTX 1650 with Max-Q Design, nominal 4 GB |
| NVIDIA driver | 595.99.02 |
| NumPy / scikit-learn | 1.23.5 / 1.1.2 |
| Matplotlib / h5py | 3.7.1 / 3.7.0 |
| PyTorch Lightning | 1.9.5 |

These values were read from the working installation. CUDA 11.7 denotes the PyTorch build/runtime, not the system toolkit or driver's maximum supported CUDA version. Use the already configured `pointjepa` environment; no environment changes are required.

Expected assets relative to the Point-JEPA project root:

```text
checkpoints/pretrain_point-jepa_epoch=499-step=40500.ckpt
data/modelnet40_ply_hdf5_2048/shape_names.txt
data/modelnet40_ply_hdf5_2048/ply_data_train*.h5
data/modelnet40_ply_hdf5_2048/ply_data_test*.h5
experiments/pooling_comparison/cache/full_modelnet40_seed42_val10_bs4.pt
```

Pretrained encoder SHA-256 recorded by the completed evaluation:

```text
e002b66c505285d8ac66d73f5adb59d1e4fd6e5e52ba4fcd173b6fe8efcaf6bf
```

The original [DATASETS.md](../../DATASETS.md) and README model zoo provide dataset/checkpoint acquisition details. Keep datasets, feature caches, pretrained weights, classifier weights, and environments outside Git history.

## Completed neural protocol

| Protocol | Training | Validation | Official testing |
|---|---:|---:|---:|
| Reproduced linear SVM | 9,840 | None | 2,468 |
| All three neural classifiers | 8,856 | 984 | 2,468 |

The neural split is stratified `train_test_split(test_size=0.10, random_state=42)` applied only to official training labels. Sorted indices and sorted HDF5 source manifests define object identity. Training and validation are disjoint and cover all 9,840 training objects; all 40 classes are represented. Test objects are excluded from training and checkpoint selection.

Cache format/protocol version 2 stores `[8856,64,384]`, `[984,64,384]`, and `[2468,64,384]` CPU float16 features; labels and indices are int64. Computation converts features to float32. Extraction seeds are 42/43/44 for train/validation/test. The encoder stays in evaluation mode with gradients disabled.

All neural experiments used seed 42, maximum 50 epochs, batch size 32, AdamW, learning rate 0.001, weight decay 0.0001, CrossEntropyLoss, dropout 0.2, and early-stopping patience 10. Fresh seeded loaders provide the same shuffled sample order for corresponding epochs; validation is unshuffled. Only batches move to the GPU. Attention gradients are checked during training.

Checkpoints are selected strictly by **minimum validation loss**. Selected validation accuracy is the value at that epoch, not the maximum across epochs.

| Models | Training run | Evaluation |
|---|---|---|
| Baseline and attention | `20261012T042856_572366Z_e467f11c` | `evaluation_20261012T043112_742087Z_346dc31c` |
| Max/Min/Std | `20261013T141057_022304Z_917cf31b` | `evaluation_20261013T141438_951650Z_625b2d10` |

| Model | Selected epoch | Epochs completed | Selected validation loss | Selected validation accuracy |
|---|---:|---:|---:|---:|
| Max/Mean | 8 | 18 | 0.278440 | 90.85% |
| Max/Attention | 8 | 18 | 0.301432 | 90.24% |
| Max/Min/Std | 12 | 22 | 0.300756 | 90.35% |

## Recorded full-test results

All values come from saved evaluation JSON. Accuracy and macro metrics were independently checked against saved predictions and confusion matrices during migration. Loss is read from saved JSON; hard-label predictions cannot reconstruct cross-entropy.

| Metric | Max + Mean + MLP | Max + Attention + MLP | Max + Min + Std + MLP |
|---|---:|---:|---:|
| Correct / 2,468 | 2,271 | 2,275 | 2,228 |
| Accuracy | 92.017828% | 92.179903% | 90.275527% |
| Cross-entropy | 0.245712 | 0.259602 | 0.301262 |
| Macro precision | 0.894793 | 0.894227 | 0.871187 |
| Macro recall | 0.878047 | 0.873587 | **0.887093** |
| Macro F1 | **0.880583** | 0.878013 | 0.873411 |

Attention improves accuracy by four correct objects (+0.1621 percentage points), but has worse loss and macro metrics than the neural control. The saved baseline/attention paired counts are 2,236 both correct, 35 baseline only, 39 attention only, and 158 neither correct.

Max/Min/Std achieves the **highest macro recall**, 0.887093, while yielding lower overall accuracy, macro precision, and macro F1, and higher cross-entropy. Macro recall weights each class equally; accuracy weights each object equally. Higher class-averaged recall can coexist with lower accuracy on this imbalanced test set. The precision reduction also limits macro F1. It predicts 43 fewer objects correctly than the baseline and 47 fewer than attention.

Neither extension establishes a statistically significant improvement. These are single-seed observations, without repeated-seed confidence intervals or a significance claim. All three MLP accuracies are below the separately reproduced SVM, with the protocol differences above preventing a controlled SVM-versus-MLP comparison.

Metrics average all 40 classes with undefined precision/recall assigned zero. Confusion matrices use true classes as rows and predicted classes as columns. Losses are weighted by actual batch size.

## Final directory structure

```text
experiments/pooling_comparison/
  README.md
  feature_utils.py                 # shared strict encoder extraction
  extract_full_features.py         # version-2 cache extraction
  pooling_models.py                # all three pooling classifiers
  train.py                        # selected model training
  evaluate.py                     # strict full-protocol evaluation
  provenance.py                   # explicit hash-bound historical relocation
  plot_results.py                 # three-model saved-artifact figures
  validate_migration.py            # validation without official inference
  migration_manifest.json          # pre-migration inventory and hashes
  relocation_manifest.json         # explicit full-run cache/checkpoint bindings
  validation_report.json           # completed migration validation
  cache/full_modelnet40_seed42_val10_bs4.pt
  checkpoints/
    20261012T035804_764978Z_f72eb563/ # retained preliminary one-epoch run
    20261012T042856_572366Z_e467f11c/ # baseline_best.pt, attention_best.pt
    20261013T141057_022304Z_917cf31b/ # max_min_std_best.pt
  results/
    20261012T035804_764978Z_f72eb563/
    20261012T042856_572366Z_e467f11c/
      evaluation_20261012T043112_742087Z_346dc31c/
    20261013T141057_022304Z_917cf31b/
      evaluation_20261013T141438_951650Z_625b2d10/
  figures/
    full_modelnet40/                # immutable historical two-model figures
    three_model_comparison/         # final three-model PNG/PDF figures
```

Every result directory retains its original summary and epoch CSVs. Every evaluation directory retains its original evaluation summary, predictions CSV, and confusion CSVs. The preliminary run is archived separately from final scientific results and excluded from final figures.

## Commands from the Point-JEPA root

All commands in this section run from the **Point-JEPA project root**, the directory containing `pointjepa/`, `scripts/`, and this project’s root README. If starting at the parent Git root, first run:

```bash
cd "Research Paper (Point JEPA)/Point-JEPA"
```

Use the existing environment. If `conda activate` is unavailable in a fresh Bash shell, load its shell hook first:

```bash
source /home/heshan-ranasinghe/miniforge3/etc/profile.d/conda.sh
conda activate pointjepa
python --version
```

The shell-hook path is specific to the verified workstation; other installations should use their own Conda shell initialization. `python -B` suppresses generated bytecode caches; it does not change model computation.

### Quick live demonstration

These commands use artificial pooling inputs or one real **training** object. They do not train a classifier or evaluate the official test split:

```bash
python -B experiments/pooling_comparison/pooling_models.py
python -B -m scripts.pretrained_encoder_inference
```

The first command checks all three classifiers, finite logits/gradients, normalized attention, parameter counts, and population standard deviation. The second strictly loads the pretrained encoder and prints `[1,64,384]` patch features and a `[1,768]` max/mean representation. It requires the local pretrained checkpoint and ModelNet40 training files. CUDA is used when available; CPU execution is supported but may be slower.

Open the already generated `figures/three_model_comparison/publication_summary.png` and `full_test_metric_comparison.png` for results. No full SVM reproduction, feature extraction, classifier training, or official test inference is needed during the presentation.

Optional pre-presentation integrity checks read the retained cache/checkpoints and HDF5 labels, so allow more time and memory than the minimal demo:

```bash
python -B experiments/pooling_comparison/validate_migration.py
python -B experiments/pooling_comparison/train.py --help
python -B experiments/pooling_comparison/evaluate.py --help
python -B experiments/pooling_comparison/extract_full_features.py --help
python -B experiments/pooling_comparison/plot_results.py --help
```

Migration validation checks historical hashes, strictly loads all five classifier checkpoints, validates full-cache metadata and source HDF5 labels, rejects unregistered paths/cache content, and cross-checks saved figures and scientific records. Generated Python bytecode is excluded from required provenance checks because it is ignored by Git and may change on import; scientific caches, checkpoints, saved records, and figures still require their exact historical hashes. It never calls official test inference. An optional `--report PATH` writes a new JSON report and refuses to overwrite an existing report. Omit `--report` when the existing validation report should be retained.

### Complete experimental reproduction

The recorded experiments are complete. Full encoder extraction, SVM reproduction, and classifier training can take substantial time on this workstation; exact duration depends on hardware. Evaluation also performs official test inference. These workflows are provided for a deliberate future reproduction and were verified without rerunning training or official test evaluation during cleanup. Use the retained artifacts for submission and the live demonstration.

#### Figure generation from existing results

The final figures already exist. Regeneration reads only saved CSV/JSON and is safe without retraining. For a fresh output directory (choose another name if this one already exists):

```bash
python -B experiments/pooling_comparison/plot_results.py \
  --output-dir experiments/pooling_comparison/figures/presentation_copy
```

To explicitly regenerate only the new final figures:

```bash
python -B experiments/pooling_comparison/plot_results.py --overwrite
```

Defaults select the two retained final runs/evaluations. Supported options include `--full-run-id`, `--full-evaluation-id`, `--std-run-id`, `--std-evaluation-id`, `--output-dir`, and `--dpi`. The historical `figures/full_modelnet40/` directory is protected from overwrite. The script reads saved CSV/JSON only; it loads no models or feature cache.

#### Feature extraction, classifier training, and evaluation

The feature cache and completed runs are already available. The following commands document reproducibility and would perform new extraction, training, or evaluation; they were **not executed during migration**.

```bash
# Refuses an existing cache; only needed to reproduce extraction from scratch.
python -B experiments/pooling_comparison/extract_full_features.py \
  --seed 42 --val-fraction 0.10 --batch-size 4

# Historical baseline/attention training recipe; creates a new run ID.
python -B experiments/pooling_comparison/train.py --models baseline attention \
  --cache experiments/pooling_comparison/cache/full_modelnet40_seed42_val10_bs4.pt \
  --seed 42 --epochs 50 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 10

# Historical second-extension recipe; creates a new run ID.
python -B experiments/pooling_comparison/train.py --models max_min_std \
  --cache experiments/pooling_comparison/cache/full_modelnet40_seed42_val10_bs4.pt \
  --seed 42 --epochs 50 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 10

# Each invocation would create a NEW evaluation directory.
python -B experiments/pooling_comparison/evaluate.py \
  --run-id 20261012T042856_572366Z_e467f11c \
  --cache experiments/pooling_comparison/cache/full_modelnet40_seed42_val10_bs4.pt \
  --batch-size 32 --device auto
python -B experiments/pooling_comparison/evaluate.py \
  --run-id 20261013T141057_022304Z_917cf31b \
  --cache experiments/pooling_comparison/cache/full_modelnet40_seed42_val10_bs4.pt \
  --batch-size 32 --device auto

# Separate original full linear-SVM reproduction (expensive).
# Re-extracts ALL train/test features, then fits and scores the SVM.
python -B -m scripts.modelnet40_linear_probe
```

The reference SVM script has no argparse interface: importing it or running it with `--help` executes the full experiment. Its module location, syntax, dependencies, data paths, extraction code, and `SVC(C=0.012, kernel="linear")` recipe were inspected; it was not executed during cleanup. The single-object encoder script also runs at import time and has no CLI flags.

Omitting `--models` retains the original baseline/attention default. `--device auto`, `cpu`, and `cuda` are supported by training/evaluation. Historical checkpoint reconstruction remains strict, with architecture, split, manifest, source-label, and cache validation retained.

## Presentation figures and provenance

Each figure has a 300-dpi PNG and vector PDF in `figures/three_model_comparison/`:

| Stem | Content |
|---|---|
| `full_accuracy_curves` | Three aligned panels; actual train/validation histories and selected epochs 8/8/12 |
| `full_loss_curves` | Three aligned panels with minimum-validation-loss markers |
| `full_baseline_confusion_matrix` | Full Max/Mean confusion counts |
| `full_attention_confusion_matrix` | Full Max/Attention confusion counts |
| `full_max_min_std_confusion_matrix` | Full Max/Min/Std confusion counts |
| `full_test_metric_comparison` | Three-model accuracy, macro precision/recall/F1, and saved cross-entropy |
| `publication_summary` | Separate SVM-reference and neural-comparison panels |

Confusion matrices use consistent class order and a shared count scale. PDF cells remain vector geometry. Curves stop at each model's actual final epoch; stars mark the validation-loss-selected checkpoint. Summary axes explicitly identify the zoom. No inferred confidence intervals are drawn for our models.

The plotter checks complete unique test-index coverage, aligned labels across runs, matching cache/encoder provenance and training settings, selected epochs, confusion matrices, and macro metrics. The new figure manifest records both sets of CSV/JSON sources with SHA-256 hashes, configuration, reference-scalar provenance, and hashes of every PNG/PDF. Exact metrics remain in the unchanged evaluation JSON; plot annotations are rounded for readability.

## Migration, historical paths, and rollback

Historical artifact **contents were not rewritten**. Absolute paths in old checkpoints and JSON describe their original execution environment; the old figure manifest similarly keeps its historical source paths. They are historical provenance, not current executable commands.

`migration_manifest.json` inventories every pre-migration file. `relocation_manifest.json` explicitly maps each retained full run's original cache path and exact checkpoint location to current repository-relative paths, with checkpoint, cache, training-summary, and evaluation-summary hashes. A relocation is accepted only for a registered checkpoint with the recorded bytes and the exact registered cache path with matching retained evaluation hash. Arbitrary paths, basename substitutions, and modified cache content are rejected. Existing architecture and source-data checks remain intact. Future run metadata uses relative paths for assets inside the project.

The full cache is 605,185,547 bytes; its verified SHA-256 is:

```text
c5a638d3c6bfacf2d90ebe8cba554311541fa5e9270c2cff349b91f96127ddef
```

The temporary repository-local backup has been moved to a verified, dated archive outside Git:

```text
/home/heshan-ranasinghe/research_archives/point_jepa_pooling_migration_2026-10-13_150903_UTC
```

All ten backup files were copied and checked against their SHA-256 values before the repository-local backup was removed. No unique source snapshot or research data was discarded. The archive contains original source/documentation snapshots and inventory; all historical research artifacts remain under this experiment directory. `migration_manifest.json`, `relocation_manifest.json`, and `validation_report.json` remain in the project. The final command/cleanup audit is recorded in `reproducibility_audit.json`.

For rollback, first preserve any subsequent work. Restore the archived Python files and experiment README to this directory, `project.gitignore` to the project-root `.gitignore`, and `root.README.md` to the project-root README. Remove only migration-created files and the new three-model figure directory, then rename `pooling_comparison` back to `attention_pooling`. Verify original artifact hashes against the inventory. Retain the external archive; do not reset Git or discard subsequent work automatically.

Large caches/checkpoints, datasets, environments, Python caches, and temporary backups remain excluded from Git. Compact final results, provenance manifests, CSVs, and scientific figures form the research submission. The external archive is local recovery material and is not published. Scoped `.gitattributes` preserve saved result/figure bytes across checkouts and recognize Python CSV writers’ intentional CRLF endings without normalizing historical records.

## Limitations

- One neural seed was evaluated. No statistically significant improvement or multi-seed confidence interval is claimed.
- Max/Min/Std uses a wider head and more trainable parameters, so this is not a parameter-matched pooling ablation.
- Historical training checkpoints contain no cache hash. Current cache bytes match both retained evaluation hashes, but this cannot independently prove training-time cache identity.
- An exploratory evaluation previously examined part of the official test split. The full test split was not untouched throughout development; it was not used to select final checkpoints or adjust full-run training hyperparameters.
- Float16 feature caching can affect results relative to the SVM's separately extracted float32 pooled features.
- Random FPS depends on seed, batch size, source ordering, device, and software. Recorded metadata does not guarantee bitwise portability across machines.
- The preliminary one-epoch run's small feature cache is absent. Its checkpoints and records are preserved, but it is excluded from the official full-protocol evaluator and final figures.
- The SVM's 93.60% is a project-reported scalar, with no retained SVM metrics JSON; its exact prediction-level metrics cannot be independently reconstructed from available records.
- Reference SVM scripts retain the original dataset class's native HDF5 glob order; research extraction uses an explicitly sorted source manifest.
- The strict shared encoder loader records unsupported legacy configuration keys, disables secondary checkpoint loading, and strictly loads tokenizer, positional embedding, and student tensors; teacher/predictor tensors are explicitly excluded.
- Source HDF5 label files are required for provenance validation. This migration reorganizes the project directory, not the parent Git repository.
