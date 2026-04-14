# White Blood Cell Classifier (CNN)

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify white blood cell images into four categories:
- Eosinophils
- Lymphocytes
- Monocytes
- Neutrophils

The model is implemented in PyTorch and evaluated using accuracy, per-class performance, and confusion matrix analysis.

## Dataset

This project uses the Blood Cell Images dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

### Methods

- Image preprocessing using torchvision transforms
- CNN model for feature extraction and classification
- Training using cross-entropy loss
- Evaluation using:
    Accuracy
    Per-class accuracy
    Confusion matrix
    Error analysis

---

## Results

- Test Accuracy: ~84%

- Strong performance on lymphocytes
- Lower performance on monocytes and eosinophils
- Common confusion between similar cell types

---

## Tools Used

- Python
- PyTorch
- pandas
- Matplotlib / seaborn

---

## Notes

This notebook was developed and run using Google Colab.
Paths may need to be adjusted for local environments.
