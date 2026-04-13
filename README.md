# White Blood Cell Classifier (CNN)

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify white blood cell images into four categories:
- Eosinophils
- Lymphocytes
- Monocytes
- Neutrophils

The goal of this project is to apply deep learning techniques to biological image data and evaluate model performance on unseen samples.

---

## Overview

This pipeline includes:
- Image preprocessing and normalization
- Dataset loading using torchvision.datasets.ImageFolder
- A CNN architecture for feature extraction and classification
- Visualization of model predictions

---

## Dataset

This project uses the Blood Cell Images dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

Due to size and licensing constraints, the dataset is not included in this repository.

### Dataset Setup

After downloading, organize the dataset as follows:

data/
  TRAIN/
    eosinophil/
    lymphocyte/
    monocyte/
    neutrophil/
  TEST/
    eosinophil/
    lymphocyte/
    monocyte/
    neutrophil/

---

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/white-blood-cell-classifier.git  
cd white-blood-cell-classifier  

Install dependencies:

pip install -r requirements.txt  

---

## Usage

Run the training and evaluation script:

python white_blood_cell_classifier.py  

The script will:
- Train the CNN model
- Print training loss per epoch
- Evaluate accuracy on the test dataset
- Display sample predictions with true vs predicted labels

---

## Model Architecture

The CNN consists of:
- 3 convolutional layers with ReLU activation and max pooling
- Feature flattening into a fully connected network
- A hidden layer with dropout for regularization
- Final classification layer with output size equal to the number of cell types

---

## Results

- Test Accuracy: ~84%

The model performs reasonably well on most classes but may show bias toward more represented cell types, suggesting potential dataset imbalance.

---

## Future Improvements

- Add data augmentation to improve generalization
- Address class imbalance
- Implement transfer learning (e.g., ResNet)
- Add confusion matrix and F1 score for deeper evaluation
- Hyperparameter tuning (learning rate, batch size, architecture)

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
