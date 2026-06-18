# Financial Sentiment Analysis

A Natural Language Processing (NLP) project that classifies financial news and statements into sentiment categories using both traditional machine learning and deep learning approaches.

## Overview

This project uses the Financial PhraseBank dataset to predict sentiment labels:

- Positive
- Neutral
- Negative

The objective is to compare different NLP pipelines and evaluate their effectiveness on financial text classification.

## Dataset

Dataset: Financial PhraseBank

The dataset contains financial news sentences annotated with sentiment labels by human experts.


## Features

- Text preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- TF-IDF feature extraction
- Machine Learning baselines
- LSTM-based deep learning model
- Performance comparison across models
- Training and validation loss visualization

## Models Evaluated

### Traditional Machine Learning

- Naive Bayes
- Logistic Regression(TF-IDF and WordToVec)
- LSTM (WordToVec)
- FinBERT

### Deep Learning

- LSTM Classifier (PyTorch)

## Project Structure

```
.
├── finance-phrasebank/
├── models/
├── utils/
├── bulk_test/
├── eda_scikit_learn_improved.ipynb
├── bulk_evaluate.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Jupyter Notebook

## Training

Example:

```python
train_infer_lstm(
    X_train_lstm,
    y_train_lstm,
    epoch=9
)
```

## Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Results

| Model | Accuracy |
|---------|---------|
| Logistic Regression | XX% |
| SVM | XX% |
| Random Forest | XX% |
| LSTM | XX% |

Replace with your actual results.

## Key Learnings

- Compared classical NLP pipelines with neural network approaches.
- Explored the impact of sequence-based models on sentiment classification.
- Implemented batching, embedding layers, and LSTM architectures in PyTorch.
- Evaluated model performance using multiple classification metrics.