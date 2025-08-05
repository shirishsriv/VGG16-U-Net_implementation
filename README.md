# VGG16-UNet Semantic Segmentation Pipeline

This project demonstrates an end-to-end pipeline for semantic segmentation using a VGG16-based U-Net model with TensorFlow/Keras. The workflow includes dataset preparation, preprocessing, model building, training, prediction, and mask saving.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Mask Generation](#2-mask-generation)
  - [3. Data Splitting](#3-data-splitting)
  - [4. Preprocessing](#4-preprocessing)
  - [5. Model Training](#5-model-training)
  - [6. Prediction & Saving Masks](#6-prediction--saving-masks)
- [Model Architecture](#model-architecture)
- [Notes & Troubleshooting](#notes--troubleshooting)
- [References](#references)

---

## Overview

This repository provides a reproducible pipeline for semantic segmentation using transfer learning (VGG16 as encoder) and a U-Net style decoder. The code is designed to process a custom dataset of images and generate binary (or grayscale) segmentation masks.

---

## Directory Structure

```
segmentation_dataset/
    images/      # Raw input images
    masks/       # Generated masks (grayscale)
segment_dataset/
    train/
        images/
        masks/
    val/
        images/
        masks/
    test/
        images/
        masks/
```

---

## Getting Started

### 1. Dataset Preparation

Place your original images (e.g., `.jpg`, `.png`) in `segmentation_dataset/images/`.

### 2. Mask Generation

The script converts each image to a grayscale mask and saves it with a `mask_` prefix in `segmentation_dataset/masks/`.

```python
import cv2, os
for filename in os.listdir('segmentation_dataset/images'):
    img = cv2.imread(os.path.join('segmentation_dataset/images', filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join('segmentation_dataset/masks', "mask_"+filename), gray)
```

### 3. Data Splitting

Split the dataset into training, validation, and test sets:

```python
split_dataset(
    image_dir='segmentation_dataset/images',
    mask_dir='segmentation_dataset/masks',
    output_dir='segment_dataset',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### 4. Preprocessing

Both images and masks are resized and normalized for model input:

```python
X_train, Y_train = preprocess_data('./segment_dataset/train/images', './segment_dataset/train/masks')
X_val, Y_val = preprocess_data('./segment_dataset/val/images', './segment_dataset/val/masks')
```

### 5. Model Training

Build and train a VGG16-UNet on the dataset:

```python
model = build_vgg16_unet((128, 128, 3))
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=25, batch_size=8)
model.save("vgg16_unet.h5")
```

### 6. Prediction & Saving Masks

Predict on test images and save masks:

```python
X_test, test_filenames = load_test_images("segment_dataset/test/images")
preds = model.predict(X_test)
for i in range(len(preds)):
    mask_array = preds[i].squeeze()
    mask_image = Image.fromarray(mask_array * 255)
    mask_image.save(f"segment_dataset/test/masks/pred_mask_{i:03d}.png")
```

---

## Model Architecture

- **Encoder:** VGG16 pretrained on ImageNet (without the classification head).
- **Decoder:** U-Net style, using skip connections from VGG16 layers.
- **Output:** Single channel (sigmoid) for binary segmentation.

---

## Notes & Troubleshooting

- Ensure `segmentation_dataset/images` and `segmentation_dataset/masks` are **not empty** before running the split.
- Naming must be consistent: input image `cat_1.jpg` → mask `mask_cat_1.jpg`.
- For multi-class segmentation, update mask preparation and output layer activation.
- Prefer saving models in the Keras format (`.keras`), though `.h5` is still supported.
- For reproducibility, set random seeds in Python, NumPy, and TensorFlow.
- Adjust `IMG_HEIGHT` and `IMG_WIDTH` as needed for your dataset.

---

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Keras Applications - VGG16](https://keras.io/api/applications/vgg/#vgg16-function)
- [Keras Model Saving](https://www.tensorflow.org/guide/keras/save_and_serialize)

---

## License

This project is for educational purposes. Adapt and use for your datasets as needed.
