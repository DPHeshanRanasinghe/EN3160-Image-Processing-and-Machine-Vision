# Point-JEPA reproduction and full ModelNet40 pooling extension — EN3160

This project reproduces frozen Point-JEPA linear probing and compares two neural classifiers on the complete official ModelNet40 test split. The research question is whether post-Transformer learnable attention can replace mean pooling effectively when substantially more labeled training data are available. The pretrained encoder stays frozen; only pooling/classification parameters are trained.

## Original work and attribution

Ayumu Saito, Prachi Kudeshia, and Jiju Poovvancheri, *Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud*, WACV 2025, [arXiv:2404.16432](https://arxiv.org/abs/2404.16432). Official implementation: [Ayumu-J-S/Point-JEPA](https://github.com/Ayumu-J-S/Point-JEPA).

The upstream [root README](../../README.md), [MIT license](../../LICENSE), and `pointjepa/` package are preserved. The existing license credits Karim Abou Zeid (2023); Point-JEPA acknowledges its Point2Vec foundation. Preserve both the license notice and the Point-JEPA authors' attribution. Our additions are the downstream research scripts listed below, not a replacement for the original pretraining implementation.

## Architecture and representations

During original JEPA pretraining, a PointNet-based tokenizer creates point-cloud patches. Positional embeddings encode patch centers. A sequencer and context/target samplers select spatial blocks; a student/context Transformer and predictor learn to predict representations produced by an EMA target encoder. This project uses the authors' pretrained checkpoint and does not repeat pretraining.

Downstream feature extraction follows the working SVM inference path:

```text
2,048 XYZ points
  → random-start FPS to 1,024 points → center → unit-sphere normalization
  → tokenizer → positional encoding → frozen student/context Transformer
  → 64 contextualized patches × 384 dimensions
  → max/mean OR max/attention pooling → 768-dimensional object representation
```

The saved encoder configuration has 64 groups, 32 points per group, 384 feature dimensions, 12 Transformer layers, and 6 attention heads. We extract `student.last_hidden_state` for every patch without pretraining context masking or sequencing. No CLS token or Transformer modification is introduced.

### Original linear-SVM reproduction

For `x` shaped `[B,64,384]`, concatenate `x.max(dim=1).values` and `x.mean(dim=1)` to obtain `[B,768]`. Fit `SVC(C=0.012, kernel="linear")` using all 9,840 official training objects and evaluate on all 2,468 official test objects. Our observed accuracy is **93.60%**, compared with the paper-reported **93.7 ± 0.2%**.

### Neural baseline and proposed modification

The baseline uses the same max/mean pooling and an MLP. The modification replaces the mean with learnable attention:

```text
scores = Linear(384, 1, bias=False)(x)          # [B,64,1]
weights = softmax(scores, dim=1)               # normalized across patches
attention_features = sum(weights * x, dim=1)  # [B,384]
global_features = concat(max_features, attention_features)  # [B,768]
```

The attention scoring weights learn through backpropagation. A scalar bias is unnecessary because a shared score offset cancels in softmax. Optional `return_attention=True` on the attention classifier returns `(logits, weights)` with weights shaped `[B,64]`.

Both neural models use exactly `Linear(768,256) → ReLU → Dropout(0.2) → Linear(256,40)` and return raw logits `[B,40]`. Parameter counts are 207,144 for `MaxMeanClassifier` and 207,528 for `MaxAttentionClassifier`. Their MLP heads start with identical parameters. The MLP baseline is distinct from the original linear SVM.

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

Expected assets relative to the repository root:

```text
checkpoints/pretrain_point-jepa_epoch=499-step=40500.ckpt
data/modelnet40_ply_hdf5_2048/shape_names.txt
data/modelnet40_ply_hdf5_2048/ply_data_train*.h5
data/modelnet40_ply_hdf5_2048/ply_data_test*.h5
experiments/attention_pooling/cache/full_modelnet40_seed42_val10_bs4.pt
```

Pretrained encoder SHA-256 recorded by the completed evaluation:

```text
e002b66c505285d8ac66d73f5adb59d1e4fd6e5e52ba4fcd173b6fe8efcaf6bf
```

The original [DATASETS.md](../../DATASETS.md) and README model zoo provide dataset/checkpoint acquisition details. Keep datasets, feature caches, pretrained weights, classifier weights, and environments outside Git history.

## Complete ModelNet40 protocol

| Experiment | Training | Validation | Testing |
|---|---:|---:|---:|
| Linear SVM reproduction | 9,840 | None | 2,468 |
| Neural pooling comparison | 8,856 | 984 | 2,468 |

The neural split uses `train_test_split(..., test_size=0.10, random_state=42, stratify=train.labels)` on the original 9,840 training objects. Training/validation indices are unique, disjoint, and jointly cover the entire official training split. All 40 classes remain represented. Selected indices are sorted before extraction; the complete official test split is never mixed into training. Class counts are unequal in the full test split.

Sorted HDF5 loading produces an explicit source-file manifest. Original indices refer to that manifest's concatenation order, separately within each official split. Metadata protocol/version **2** records official and partition counts, stratification, class counts, preprocessing, checkpoint path/hash, and software/device settings. Each `train`, `val`, and `test` entry contains `features`, `labels`, `indices`, and `source_split`. Expected feature shapes are `[8856,64,384]`, `[984,64,384]`, and `[2468,64,384]`. Features are cached on CPU as float16, labels/indices as int64; model computations use float32.

Training settings are seed 42, maximum 50 epochs, batch size 32, AdamW, learning rate 0.001, weight decay 0.0001, CrossEntropyLoss, and early-stopping patience 10. Both models reset relevant random seeds and receive the same shuffled sample order in corresponding epochs through fresh seeded DataLoader generators. Validation is never shuffled. Only batches move to the GPU. Attention gradients are checked during training.

Checkpoints are selected exclusively by minimum validation loss. The validation accuracy reported for a selected checkpoint is the accuracy at that epoch, not the maximum accuracy across epochs. Both models selected **epoch 8** and stopped after **epoch 18**, following ten epochs without a lower validation loss.

Final training run: `20261012T042856_572366Z_e467f11c`.

Final evaluation: `evaluation_20261012T043112_742087Z_346dc31c`.

## Exact reproducibility commands

Run from the Point-JEPA repository root in the configured environment. These commands document the completed workflow; reproducing training or evaluating again is unnecessary for the live demo. Shared research scripts resolve assets against the repository root.

### Reference inference and original SVM reproduction

The working reference scripts use module-style execution, which makes the repository package importable from the repository root:

```bash
python -m scripts.pretrained_encoder_inference
python -m scripts.modelnet40_linear_probe
```

The first command shows one real training object's `[1,64,384]` Transformer features and `[1,768]` pooled vector. The second extracts all original train/test features with batch size 4, fits the linear SVM, and prints accuracy. These scripts expose no CLI flags and do not save a metrics JSON. Their 93.60% reproduction value is explicitly identified as a project-reported scalar in the figure manifest.

### Full feature extraction

```bash
python experiments/attention_pooling/extract_full_features.py \
  --seed 42 --val-fraction 0.10 --batch-size 4
```

The completed cache already exists: skip this command for the final submission/demo. The extractor refuses an existing cache. It reuses `feature_utils.py`, strictly loads the pretrained encoder components, checks every batch for finite features, and atomically publishes a completed temporary file using a same-directory hard link. It does not pool features. The encoder is in evaluation mode with no gradients. Extraction seeds are 42/43/44 for train/validation/test.

### Train the two existing neural classifiers

```bash
python experiments/attention_pooling/train.py \
  --cache experiments/attention_pooling/cache/full_modelnet40_seed42_val10_bs4.pt \
  --seed 42 --epochs 50 --batch-size 32 \
  --lr 0.001 --weight-decay 0.0001 --patience 10
```

The default cache now points to the full dataset. Training uses only training and validation entries and creates a new unique run ID. The completed run above is retained; a fresh run would have a different ID. No architecture variants or encoder fine-tuning are included.

### Evaluate best-validation checkpoints

```bash
python experiments/attention_pooling/evaluate.py \
  --run-id 20261012T042856_572366Z_e467f11c \
  --cache experiments/attention_pooling/cache/full_modelnet40_seed42_val10_bs4.pt \
  --batch-size 32 --device auto
```

The final evaluation is already saved; do not repeat it during the demo. The evaluator supports the version-2 full 90/10 protocol only, strictly reconstructs the saved architectures, checks the cache path against both checkpoints, reproduces training split membership, verifies source HDF5 labels/indices, and checks full test coverage. Source label files must remain available. `--device auto`, `cpu`, and `cuda` are supported by training and evaluation. Each new evaluation creates a unique result directory.

### Generate final figures from retained results

```bash
python experiments/attention_pooling/plot_results.py
```

Final output directory: `experiments/attention_pooling/figures/full_modelnet40/`. After it exists, use the explicitly supported regeneration command:

```bash
python experiments/attention_pooling/plot_results.py --overwrite
```

The plotter overwrites only recognized generated filenames when explicitly requested and rejects unexpected directory contents. Supported options include `--full-run-id`, `--full-evaluation-id`, `--output-dir`, `--dpi`, `--svm-accuracy`, `--paper-accuracy`, and `--paper-spread`. Default artifact IDs are the retained final IDs.

## Saved results and scientific interpretation

| Model | Best epoch | Validation loss | Validation accuracy | Full-test accuracy |
|---|---:|---:|---:|---:|
| Max/Mean + MLP | 8 | 0.278440 | 90.85% | 92.02% |
| Max/Attention + MLP | 8 | 0.301432 | 90.24% | 92.18% |
| Reproduced Max/Mean + linear SVM | — | — | — | 93.60% |
| Paper-reported linear SVM | — | — | — | 93.7 ± 0.2% |

| Metric on all 2,468 test objects | Max/Mean + MLP | Max/Attention + MLP |
|---|---:|---:|
| Correct objects | 2,271 | 2,275 |
| Accuracy (unrounded) | 92.0178% | 92.1799% |
| Cross-entropy | 0.245712 | 0.259602 |
| Macro precision | 0.894793 | 0.894227 |
| Macro recall | 0.878047 | 0.873587 |
| Macro F1 | 0.880583 | 0.878013 |

Paired predictions: **2,236 both correct, 35 baseline only, 39 attention only, 158 neither correct**. Attention gains four correct objects, or **0.1621 percentage points**, while performing worse on cross-entropy and every macro metric. This does not establish a statistically reliable or consistent improvement.

Accuracy is total correct divided by total samples. Losses are weighted by actual batch size. Macro precision, recall, and F1 average all 40 classes with undefined divisions assigned zero. Confusion matrices use true classes as rows and predictions as columns; confused class pairs are directed true→predicted errors.

The MLP comparison is not an identical paper-reproduction protocol: our neural classifiers train on 8,856 objects and reserve 984 for validation; the reproduced SVM uses all 9,840. Both neural accuracies remain below the reproduced SVM and the paper's reported SVM result. The paper's ±0.2 is its reported variability, not a confidence interval for our single-seed runs.

Retained artifacts relative to `experiments/attention_pooling/`:

```text
checkpoints/20261012T042856_572366Z_e467f11c/
  baseline_best.pt
  attention_best.pt
results/20261012T042856_572366Z_e467f11c/
  baseline_metrics.csv
  attention_metrics.csv
  summary.json
  evaluation_20261012T043112_742087Z_346dc31c/
    evaluation_summary.json
    test_predictions.csv
    baseline_confusion_matrix.csv
    attention_confusion_matrix.csv
```

## Final scientific figures

Each stem below has a 300-dpi PNG and vector PDF in `figures/full_modelnet40/`:

| Figure stem | Content |
|---|---|
| `full_accuracy_curves` | Training/validation accuracies and selected epoch markers |
| `full_loss_curves` | Training/validation losses and selected checkpoint epochs |
| `full_baseline_confusion_matrix` | Full 40-class Max/Mean confusion counts |
| `full_attention_confusion_matrix` | Full 40-class Max/Attention confusion counts |
| `full_test_metric_comparison` | Accuracy/macro metrics and cross-entropy |
| `full_paired_correctness` | The four paired correctness categories |
| `publication_summary` | Paper SVM, reproduced SVM, and both neural classifiers |

The confusion matrices have readable class labels and a shared color scale; their cells are vector geometry in PDF. Stars on both curve types mark validation-loss-selected checkpoints. The summary clearly separates the SVM and MLP protocols and labels its zoomed accuracy axis. `figure_manifest.json` records only retained full-run CSV/JSON source artifacts, hashes, final figure files, and provenance of the external paper/SVM scalars. The plotter validates saved curves, predictions, confusion matrices, macro metrics, and paired counts without loading any model or feature cache.

## Limitations

- One neural training seed was used; no repeated-seed confidence interval or statistical significance is claimed.
- An exploratory evaluation previously examined part of the official test split. The complete test split was not untouched throughout development; it was not used to select full-run checkpoints or adjust training hyperparameters.
- Float32 embeddings were cached as float16 and converted back for computation, so quantization can affect results. Both neural models share the exact cache; the separate SVM reference extracts float32 pooled features directly.
- Random FPS depends on seed, batch size, device, source ordering, and software versions. Metadata records these details but does not promise bitwise portability across machines.
- Legacy checkpoint configuration keys unsupported by `PointJepa.__init__` are printed and recorded by the shared loader. Supported settings reconstruct the model; any secondary `pretrained_ckpt_path` is disabled. Tokenizer, positional encoding, and student tensors load strictly: missing, unexpected, or incompatible tensors raise errors. Teacher/predictor tensors are excluded explicitly. This is configuration compatibility handling, not silent weight skipping.
- Training checkpoints record an absolute cache path but no historical cache hash. Evaluation records its current SHA-256 and checks metadata and source labels; it cannot independently prove historical byte identity. Relocated projects must preserve or deliberately review provenance paths.
- Reference downstream scripts retain the original dataset class's native HDF5 glob order; the shared research loader explicitly sorts and records that order.

## Minimum live demonstration

From the repository root in the configured environment:

```bash
python experiments/attention_pooling/pooling_models.py
```

This CPU synthetic test checks `[4,64,384]` inputs, `[4,40]` finite logits, parameter counts, normalized attention, and nonzero attention gradients. Then open the already generated `publication_summary.png`, `full_test_metric_comparison.png`, and `full_paired_correctness.png`. If plots need regeneration, run `python experiments/attention_pooling/plot_results.py --overwrite`. No training or test inference is necessary for this presentation. The module-style pretrained inference command above is available separately for an optional live encoder demonstration.

## File-by-file research contributions

| File | Purpose |
|---|---|
| `scripts/pretrained_encoder_inference.py` | Working original checkpoint loading and one-object encoder demonstration |
| `scripts/modelnet40_linear_probe.py` | Original full ModelNet40 frozen max/mean linear-SVM reproduction |
| `experiments/attention_pooling/feature_utils.py` | Shared strict checkpoint loading, sorted HDF5 manifests, original validation preprocessing, and unpooled extraction |
| `experiments/attention_pooling/extract_full_features.py` | Full stratified split, version-2 metadata, source checks, and atomic cache publication |
| `experiments/attention_pooling/pooling_models.py` | Separate pooling and head modules; baseline and bias-free attention classifiers |
| `experiments/attention_pooling/train.py` | Identical head initialization, seeded loaders, metrics CSVs, early stopping |
| `experiments/attention_pooling/evaluate.py` | Full-only strict best-checkpoint loading, metrics, confusion matrices, and paired predictions |
| `experiments/attention_pooling/plot_results.py` | Full-only saved-artifact validation and seven scientific PNG/PDF figures |
| `experiments/attention_pooling/README.md` | This final reproducibility and demonstration guide |

The upstream root README and MIT license remain intact. For eventual submission, preserve attribution and distinguish these additions clearly. Point-JEPA is currently an independent nested checkout inside the EN3160 repository; publication should deliberately resolve that structure rather than blindly adding nested Git content. No Git metadata, remotes, index, or commits were altered by the cleanup. Large binary assets should remain external, while compact report data and figures can be selectively published under a reviewed ignore policy.
