# EN3160 Assignment 3

Open notebook.ipynb in Jupyter using a Python environment with torch, torchvision, numpy, pandas, matplotlib and ipykernel. Start Jupyter from this folder.

The notebook contains setup and print CSS; Q1 manual CIFAR-10 gradients; Q2 LeNet-5 and a three-convolution CNN; Q3 ResNet18 fine-tuning and feature extraction; result tables, learning curves, confusion matrices and discussion placeholders; and small smoke checks.

## Run next

1. Set RUN_Q1, RUN_Q2 and/or RUN_Q3 to True in the setup cell.
2. Keep ALLOW_DOWNLOAD=True for first-time dataset preparation, or supply datasets under data/ and set it to False. PREPARE_DATA=True prepares all datasets without training.
3. Restart the kernel and run all cells. Q1 and Q2 use exactly 10 epochs; Q3 defaults to 25. RUN_SMOKE_TESTS may be disabled after checks pass.
4. Completed runs save CSV histories, JSON metrics and PyTorch parameters in results/. Q2 also saves confusion matrices. Save notebook outputs and fill discussion placeholders before exporting the PDF. Confirm the final submission filename.

## Checks performed on 2026-10-12

- Notebook schema and every code cell's Python syntax validated.
- Default code cells executed from top to bottom on CUDA with PyTorch 2.5.1 and torchvision 0.20.1. All full-training switches remained False.
- Q1 forbidden-operation scan passed. Sampled manual gradients for W1, b1, W2 and b2 matched central finite differences with maximum absolute error 9.79e-11. Extreme-logit cross-entropy and one synthetic parameter update passed.
- Both Q2 models passed forward, one synthetic optimization batch and confusion-matrix checks. The modern CNN has exactly three convolutions. Fixed average pooling supports the strict deterministic CUDA setting.
- Both Q3 modes loaded cached ImageNet ResNet18 weights and passed synthetic forward and trainability checks. One feature-extraction classifier update left every backbone parameter and batch-normalization buffer unchanged.
- No full training, dataset downloads or final result claims were made. No datasets were present in this folder, so Q3's real-image batch check was deferred. Download/extraction code is included.

The notebook follows the PyTorch tutorial at https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html for Q3.
