# MediScribe – Doctor’s Prescription Identifier

A Deep Learning CRNN-based OCR System for Handwritten Medical Prescriptions

---

## Overview

MediScribe is an Optical Character Recognition (OCR) system designed to read handwritten doctor prescriptions.  
Handwritten text—especially medical handwriting—is extremely difficult due to:

- Irregular writing styles
- No fixed spacing between characters
- Overlapping strokes
- Noise & distortions in scanned images

To solve this, MediScribe uses a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC), allowing it to recognize text without needing character-level alignment.

---

## Key Features

- CRNN architecture (CNN + BiLSTM + CTC)
- Clean, modular notebook pipeline
- Simple preprocessing (resize → grayscale → normalize)
- Training and validation monitoring
- CTC decoding for label generation
- Fuzzy matching post-processing for correcting OCR mistakes
- Visualizations for loss curves & predictions
- Final accuracy and result analysis

---

## Model Architecture

1. **CNN – Feature Extraction**  
   Extracts spatial patterns (edges, curves, shapes) from prescription images and produces feature sequences.

2. **BiLSTM – Sequence Modeling**  
   Learns the temporal patterns of characters from left to right and right to left.

3. **CTC Loss + Decoder**  
   Allows OCR without aligned labels and predicts variable-length words.

---

## 📁 Project Structure

```
MediScribe/
│── model/
│── results/
│── README.md
│── requirements.txt
```

---

## Implementation

### Training Pipeline

- **Preprocessing**
  - Resize images
  - Convert to grayscale
  - Normalize pixel values

- **Dataset & Dataloader**
  - Custom dataset class
  - Train/validation split

- **Model Training**
  - Adam optimizer
  - Learning rate = 0.0002
  - CTC Loss
  - Epoch-wise sample predictions

- **Outputs**
  - Training loss
  - Validation loss
  - Character predictions
  - Saved models (`crnn_final.pth`, `corr_final.pth`)

---

## Evaluation

### Performance Measures

- CTC loss
- Character recognition accuracy
- Word-level accuracy
- Fuzzy matching accuracy

### Analysis

- Train loss decreases consistently
- Validation loss fluctuates (normal for CTC on small data)
- Accuracy improves significantly after ~35 epochs
- Fuzzy matching increases final word correctness

---

## Results

### Training & Validation Curves

- Training loss → smooth downward curve
- Validation loss → fluctuates but overall decreases
- Occasional spikes due to unstable CTC alignment

### Predictions

Reads medical words like:
- aceta
- pancef
- dolo
- ketanov
- amoxy

Fuzzy matcher corrects near matches  
Final validation accuracy: ~70%

### Hyperparameter Tuning

- LR = 0.0002 gives smooth convergence
- Batch size selected based on GPU memory
- 50 epochs optimal for this dataset

---

## Prediction Pipeline

- Preprocess input image
- Extract sequence features using CNN
- Pass through BiLSTM for temporal modeling
- Decode using CTC (greedy decoding)
- Apply fuzzy matching for correction
- Return cleaned final text

---

## Conclusion

MediScribe successfully demonstrates the effectiveness of CRNN + CTC for handwritten medical OCR.  
Despite minimal preprocessing and a small dataset, the model learns character structures well and generalizes using fuzzy correction.

### Future Work

- Add transformer-based decoder
- Larger prescription dataset
- Add denoising & thresholding
- Multi-line prescription extraction

---

## Contributors

This project was completed by:

- Sujal Mhatre : Data preprocessing & dataset pipeline
- Afreen Kazi: CRNN model design & training loop
- Arpita Singh: Evaluation, fuzzy matching, documentation, visualization

---

## Requirements

- torch
- torchvision
- numpy
- matplotlib
- opencv-python
- tqdm
- python-Levenshtein

---

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Open the notebook:
`ML_MiniProjectCRNN.ipynb`

Run all cells to train and evaluate the MediScribe OCR model.
