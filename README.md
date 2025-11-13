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
