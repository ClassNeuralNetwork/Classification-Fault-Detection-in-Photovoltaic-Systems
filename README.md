# Photovoltaic Module Fault Detection using Convolutional Neural Networks

---

## Problem Statement

Photovoltaic (PV) solar panels are subject to various types of physical and electrical degradation over time. Defective modules generate localized heat spots that are invisible to the naked eye but can be captured through infrared thermography. If left undetected, these anomalies reduce energy output, accelerate panel degradation, and in severe cases may cause fires or complete system failure.

Manual inspection of large-scale solar farms is time-consuming, costly, and error-prone. Automating this process through computer vision and deep learning offers a scalable and reliable alternative for predictive maintenance.

---

## Objective

The goal of this project is to build and evaluate deep learning models capable of classifying individual photovoltaic modules as defective or non-defective based on infrared thermographic images captured by drones.

The project follows a complete machine learning pipeline: from raw image preprocessing and dataset balancing through to model training, evaluation, and comparison between a custom-built CNN and two transfer learning architectures.

---

## Dataset

The dataset used is the Photovoltaic System Thermography dataset, available on Kaggle. It contains 120 infrared images of solar panel arrays captured at 640×512 pixel resolution in RGB format, accompanied by JSON annotation files that describe the bounding polygon and defect label (defected_module: true/false) for each individual module within the image.

Only dataset_1 was used in this project, containing 4,107 individual module crops extracted from the original images, of which only 26 are labeled as defective — representing a severe class imbalance of approximately 99.4% non-defective vs 0.6% defective.

---

## Pipeline 

### 1. Module Extraction and Normalization

Each module was extracted from the full panel image using the corner coordinates provided in the JSON annotations. Crops were resized to 64×64 pixels and converted to grayscale (since all three RGB channels in dataset_1 carry identical thermal information). Pixel values were normalized to the range [0.0, 1.0] by dividing by 255.

### 2. Train / Validation / Test Split

The dataset was split before any data augmentation to prevent data leakage. The final proportions are 60% training, 20% validation, and 20% test — all containing only real images. The validation and test sets were kept completely free of artificially generated samples.

### 3. Data Augmentation 

To address the severe class imbalance, oversampling was applied exclusively to the defective class using ImageDataGenerator (TensorFlow/Keras), generating synthetic samples through random rotations up to 180°, horizontal and vertical flips, and zoom variations up to 10%. The augmentation brought the training set to an approximately 50/50 class balance.

### 4. Class Weighting 

In addition to augmentation, compute_class_weight('balanced') was applied to the original unbalanced training labels to assign a higher penalty to misclassifying defective modules during loss computation. The two strategies are complementary: augmentation balances sample counts while class weighting balances the cost of errors.

### 5. Model Architecture of a Custom CNN

### 6. Transfer Learning Models

Two pre-trained architectures were fine-tuned for comparison. Both used ImageNet weights with frozen base layers, a Global Average Pooling layer, and the same MLP head as the custom CNN. The base layers were kept frozen throughout training.

- ResNet50: Deep residual network with skip connections, robust to vanishing gradients.
- VGG16: Sequential deep architecture known for strong feature extraction on visual tasks.

Input images were replicated from 1 to 3 channels and preprocessed according to each architecture's original preprocessing function before being fed to the pre-trained bases.

### 7. Evaluation 

---

## Conclusions 

---

## References

Dataset: Marcos Gabriel | Photovoltaic System Thermography
Notebooks used as reference during development:

- EA MSc | Anomaly Detection in Photovoltaic Cell
- Dataset Intro | Photovoltaic System Thermography

---

## Licensed

This project is licensed under the MIT License. You are free to use, copy, modify, and distribute the software, provided you retain the original copyright notices.
