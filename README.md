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
