# Sentiment Analysis Using NLP & Machine Learning

## 📌 Project Overview

This project implements an **end-to-end Sentiment Analysis pipeline** using Natural Language Processing (NLP) and Machine Learning (ML) models. The goal is to classify text data (movie reviews) into **Positive** or **Negative** sentiment.  

The pipeline includes:

- Text preprocessing (cleaning, tokenization, stopword removal, lemmatization)  
- Feature engineering using **Bag of Words (BoW)** and **TF-IDF**  
- Training and evaluation of **three ML models**:  
  - Logistic Regression  
  - Naive Bayes  
  - Decision Tree  
- Comparison of model performance using **Accuracy, Precision, Recall, and F1 Score**  

---

## 📂 Dataset

- Source: [IMDb Movie Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- Format: CSV  
- Columns:
  - `review` – The raw text of the movie review  
  - `sentiment` – Label (`positive` or `negative`)  

> The notebook automatically searches for the CSV in your Google Drive, so you don’t need to manually type the path.

---

## 🛠️ Requirements

- Python 3.x  
- Libraries:
  - pandas
  - numpy
  - nltk
  - scikit-learn
- Google Colab (recommended for easy Drive mounting and GPU support)

NLTK resources used:
```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
