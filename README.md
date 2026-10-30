
# Milk Tea Freshness Detection

## Project Title
RGB Image-Based Freshness Estimation of Brewed Milk Tea Using Deep Learning

## Description
This project estimates the freshness level of brewed milk tea
using RGB images and a deep learning approach.

The application uses EfficientNetV2-S integrated with
Convolutional Block Attention Module (CBAM) to classify
milk tea images into three freshness categories.

## Model Architecture
- EfficientNetV2-S
- CBAM Attention Module
- Input Image Size: 224 × 224 × 3

## Freshness Categories

| Class | Time Range |
|---|---|
| FRESH | 0–3 minutes |
| MODERATE | 4–10 minutes |
| LOW | 11–20 minutes |

## Input
RGB image of brewed milk tea.

## Output
Predicted freshness category and class-wise probabilities.

## Technologies Used
- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
- Hugging Face Hub

## Application Features
- Upload milk tea images
- Capture images using a camera
- Predict freshness category
- Display prediction confidence
- Display class-wise probabilities

## Model Repository
https://huggingface.co/kanimozhi23/milk-tea-freshness-model

## Disclaimer
The application estimates visual freshness patterns
learned from the dataset. It is not a laboratory test
or a food-safety assessment.
