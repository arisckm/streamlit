# Sentiment Classification using Deep Learning (Bi-GRU)

## 📌 Overview
This project implements a sentiment analysis model to classify text into positive and negative categories. It combines classical preprocessing with a deep learning approach using **Bidirectional GRU**. To address class imbalance, oversampling was applied before training.

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- scikit-learn  
- imbalanced-learn (RandomOverSampler)  
- numpy, pandas  

## 📂 Dataset
- Format: CSV file containing labeled text data (positive/negative).  
- Preprocessing included: tokenization, padding sequences, and class balancing with oversampling.

## ⚙️ Methodology
1. Preprocessed text data (tokenization, sequence padding).  
2. Applied **RandomOverSampler** to handle class imbalance.  
3. Designed a deep learning model with:
   - Embedding layer  
   - Bidirectional GRU (with dropout and recurrent dropout)  
   - Batch Normalization + Dropout  
   - Dense output layer with sigmoid activation  
4. Compiled with **Adam optimizer** and binary crossentropy loss.  
5. Trained with early stopping to avoid overfitting.  

## 📊 Results
- Achieved **~XX% accuracy** on the test dataset.  
- Training stabilized with early stopping after N epochs.  

## 🚀 How to Run
```bash
git clone <repo-link>
cd sentiment-classification
pip install -r requirements.txt
python sentiment_gru.py
