# COVID-19 Image Classification Model

## Introduction

This repository contains code and instructions for training and using a Convolutional Neural Network (CNN) model for COVID-19 image classification. The model is designed to classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal. It aims to aid deep learning and AI enthusiasts in contributing to improving COVID-19 detection using chest X-rays.

## Dataset

The dataset used for this model is the "COVID-19 Image Dataset - 3 Way Classification" containing around 137 cleaned images of COVID-19 and 317 images in total, which include cases of Viral Pneumonia and Normal chest X-rays. The dataset is structured into the test and train directories.

- **Number of Classes:** 3 (COVID-19, Viral Pneumonia, Normal)
- **Total Images:** 317
- **Train Images:** [Number]
- **Test Images:** [Number]
-you can get the dataset from <a> https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset </a>

## Model Architecture

The CNN model follows the following architecture:

- **Model Type:** Sequential
- **Input Layer:** Accepts input images with dimensions (500, 500, 3) representing a 500x500 pixel image with 3 color channels (RGB).
- **Convolutional Layers:**
  - Layer 1: 32 filters, kernel size (3, 3), activation function: ReLU
  - Layer 2: MaxPooling with pool size (2, 2)
  - Layer 3: 64 filters, kernel size (3, 3), activation function: ReLU
  - Layer 4: MaxPooling with pool size (2, 2)
  - Layer 5: 128 filters, kernel size (3, 3), activation function: ReLU
  - Layer 6: MaxPooling with pool size (2, 2)
- **Flatten Layer:** Flattens the feature maps for fully connected layers.
- **Fully Connected Layers:**
  - Layer 7: Dense layer with 128 units and ReLU activation function.
- **Dropout Layer:** Added with a dropout rate of 0.5 for regularization.
- **Output Layer:**
  - Classification layer with 3 units for classifying into 3 classes.
  - Activation function: Softmax

The model architecture is designed to capture and extract features from input chest X-ray images and make predictions for COVID-19 detection.

## Requirements

- TensorFlow (version)
- NumPy (version)
- Other dependencies

## Usage

- Explain how to use the model for various tasks:
  - Training: Include instructions on how to train the model using your dataset.
  - Evaluation: Describe how to evaluate the model's performance on a test dataset.
  - Inference: Explain how to use the model for making predictions on new or unseen chest X-ray images.

## How to Run

- Provide step-by-step instructions on how to run the code:
  - Clone the repository.
  - Install dependencies.
  - Run the training script.
  - Run the evaluation or inference script.

## Contact

- If you have any questions, issues, or collaboration, contact me using <a> yeshiwasdagnaw23@gmail.com"</a>
