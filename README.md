# MNIST Digit Classifier  
A simple, clean, and well-structured machine learning project demonstrating handwritten digit recognition using a neural network trained on the MNIST dataset.

This project shows practical skills in:
- Deep learning  
- Data preprocessing  
- Model training and evaluation  
- Python scripting  
- Using PyTorch for neural networks  

---

# ⭐ Project Overview

This model is trained to classify handwritten digits (0–9) using the MNIST dataset.  
It includes:
- A training script  
- Evaluation on the test set  
- A function to predict on custom images  
- A clean, minimal model architecture (2-layer or CNN – your choice)

This small project reflects strong fundamentals in applied AI.

---

# 🚀 Features

- Load & preprocess MNIST dataset  
- Train a fully connected or CNN neural network  
- Evaluate model accuracy  
- Save/load trained model  
- Predict on custom digit images using PIL  
- Fully reproducible codebase  

---

# 🧠 Model Architecture (Simple Version)

Default model:
Input (28×28)
→ Flatten
→ Linear (784 → 128)
→ ReLU
→ Linear (128 → 10)
→ Softmax

You can later upgrade to:
- A small CNN  
- Batch normalization  
- Dropout  
- Better augmentation  

---

# 📦 Installation

Install dependencies:

```bash
pip install torch torchvision matplotlib pillow
git clone https://github.com/<your-username>/mnist-digit-classifier.git
cd mnist-digit-classifier
```


Training now uses the same `DigitClassifier` architecture as inference, so saved state-dictionary keys load correctly. Run `python train.py --epochs 5 --output model.pth`; importing `train` no longer downloads data or starts training.

Inference applies EXIF orientation before resizing, keeping phone-camera digits aligned with their displayed orientation.

Batch inference consumes input-path iterators one chunk at a time instead of materializing the complete path list. Prediction results remain in input order.

`evaluate_model()` returns accuracy, sample count, correct count, and a 10×10 confusion matrix (rows are true labels, columns predicted labels). Malformed or non-finite model logits are rejected.

Batch inference enforces a maximum batch size of 1,024 images. Oversized values are rejected before paths are consumed or a model is loaded, preserving the bounded-memory contract.

Training writes a complete temporary checkpoint and atomically replaces the previous weights only after the save succeeds. Save failures clean up the partial file and preserve the previous checkpoint.

Transparent images are composited onto the appropriate input background before polarity inversion, so `--invert` no longer turns fully transparent background pixels into bright foreground noise.

Equal inference probabilities use ascending digit indexes as the deterministic tie-breaker for single-image and batch rankings.

`python predict.py digit.png --json` emits only a structured prediction and ranked candidates. Programmatic callers can suppress the human report with `print_report=False`.

`evaluate_model(model_path, test_loader=loader)` evaluates an existing compatible digit loader without requesting MNIST. The same label/logit validation and confusion-matrix metrics apply.

`python train.py --seed 7` seeds PyTorch initialization and loader randomness. Identical synthetic CPU runs are regression-tested; different hardware, versions and nondeterministic kernels can still change results. Supplying a seed changes the process PyTorch RNG state.

`iter_predictions(paths, batch_size=32)` yields predictions while retaining only one bounded chunk. `predict_batch` still returns a list. The iterator restores gradient mode before yielding to the caller.

Local checkpoint paths are checked against a 16 MiB file-size budget before deserialization, well above the fixed classifier’s normal checkpoint size. This catches accidental oversized inputs; it is not a guarantee against every malformed archive. The loader still uses weights-only CPU deserialization and validates tensor shapes and finite values.

Inference and evaluation require real floating-point tensor logits with the documented batch/class shape. Integer, complex, non-tensor, malformed, and non-finite outputs are rejected before ranking or softmax.

Custom evaluation loaders are validated before model execution: finite float32 digit tensors, aligned non-empty integer labels, and digit indexes 0–9. This mirrors the supported training-image shape contract and prevents accidental broadcasting or non-finite evaluations.

Seeded CPU training now restores the caller’s PyTorch CPU RNG state, including when training raises. Reproducibility is scoped to this training call; unseeded training retains normal global RNG consumption. No CUDA state-isolation guarantee is made by this CPU trainer.

Custom images are limited to 16 Mi-pixels before EXIF transposition, RGBA compositing, or model-canvas allocation. The inclusive limit bounds the large intermediate image expansion; it is separate from the final 28×28 model tensor size.

The prediction CLI validates top-k before model access (usage status 2) and reports normal image/model input failures with status 1 and a concise stderr diagnostic. Successful JSON mode stays machine-readable on stdout.

Checkpoint validation also checks the loaded destination tensors: finite float64 values that overflow float32 model parameters are rejected before inference.

`python evaluate.py --model model.pth --json` reports per-digit support, predicted count, precision and recall (ratios), plus balanced accuracy as a percentage over classes with nonzero support. Undefined precision/recall are JSON null. Evaluation without a supplied loader retains the existing MNIST download behavior.
