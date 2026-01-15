# Emotion Classification with GRU Neural Networks

A text emotion classification model built with TensorFlow/Keras that categorizes sentences into five emotion categories: anger, fear, joy, sadness, and surprise. Achieved **97.16% validation accuracy** on a 150,000 sentence dataset.

## Overview

This project implements a Gated Recurrent Unit (GRU) based neural network for sentiment analysis. GRU was chosen over LSTM for its computational efficiency while maintaining comparable performance. The architecture uses two stacked GRU layers with dropout regularization to prevent overfitting.

## Results

| Metric | Score |
|--------|-------|
| Training Accuracy | 98.17% |
| Validation Accuracy | 97.16% |
| Training Loss | 0.0433 |
| Validation Loss | 0.0702 |

These results were achieved in just 10 epochs through careful hyperparameter tuning.

## Model Architecture

```
Embedding (vocab_size → 140 dimensions)
    ↓
GRU (128 units, 50% dropout, return sequences)
    ↓
GRU (64 units, 50% dropout)
    ↓
Dropout (50%)
    ↓
Dense (32 units, ReLU)
    ↓
Dense (5 units, Softmax)
```

## Key Hyperparameters

- **Sequence Length:** 140 tokens
- **Batch Size:** 8192
- **Learning Rate:** 0.026
- **Optimizer:** AdamW with weight decay (1e-1)
- **Loss Function:** Sparse Categorical Crossentropy

## Installation

```bash
git clone https://github.com/yourusername/emotion-classification.git
cd emotion-classification
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.10
pandas
numpy
scikit-learn
```

## Usage

### Training

```python
from emotion_classifier import train_model

# Train the model
model, history = train_model('path/to/Emotion-Dataset.csv')
```

### Inference

```python
from emotion_classifier import predict_emotion

text = "I feel so happy today!"
emotion = predict_emotion(model, text, tokenizer)
print(f"Predicted emotion: {emotion}")
```

## Dataset

The model was trained on a dataset containing 150,000 labeled sentences across five emotion categories. Preprocessing steps include:

- Removing duplicate sentences (reduced data leakage between train/test splits)
- Lowercasing emotion labels for consistency
- Tokenizing and padding sequences to uniform length
- 70/30 train/test split with shuffle

## Development Journey

### Version 1
Initial implementation with conservative parameters (batch size 500, sequence length 15). Required 15+ epochs to exceed 70% accuracy.

### Version 2
Increased batch size to 8192 and sequence length to 140. Reached 95% validation accuracy but showed signs of overfitting with training accuracy continuing to rise while validation plateaued.

### Version 3 (Final)
Key optimizations:
- Increased dropout from 30% to 50%
- Switched from Adam to AdamW optimizer
- Tuned learning rate from 0.001 to 0.026

Result: 97.16% validation accuracy in 10 epochs (previously took 30+ epochs for lower accuracy).

## Optimizer Comparison

| Optimizer | Val Accuracy | Notes |
|-----------|--------------|-------|
| AdamW | 97.16% | Best performance |
| Adam | 95.18% | Slower convergence |
| Nadam | 96.91% | Did not outperform AdamW |

## Limitations

Analysis of high-confidence misclassifications revealed that remaining errors (2-3%) typically involve sentences with emotional ambiguity. For example, text expressing both anger and sadness, or fear and sadness. These edge cases may require more sophisticated architectures or pre-trained language models to resolve.

## Example Predictions

| Text | Predicted | Actual |
|------|-----------|--------|
| "i feel sure that s is right in that the despondency follows the suppression of anger" | joy | joy ✓ |
| "i could truly feel anymore sitting petrified on that ice cream parlor bench..." | fear | fear ✓ |
| "i did what i needed to do which was to feel miserable without a time limit" | sadness | sadness ✓ |


## Full Report

[View the complete project report (PDF)](https://storage.googleapis.com/anmstorage/Emotion_classification_report.pdf)
