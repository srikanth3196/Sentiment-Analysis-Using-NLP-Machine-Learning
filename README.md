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
  - ⚙️ How to Run
Upload the notebook to Google Colab.
Ensure the CSV file is in your Google Drive.
Run all cells sequentially:
Mounts Google Drive
Finds and loads the dataset
Preprocesses the text
Vectorizes using BoW and TF-IDF
Trains and evaluates Logistic Regression, Naive Bayes, and Decision Tree
Outputs performance metrics
🧹 Preprocessing Steps
Convert text to lowercase
Remove URLs and non-alphabetic characters
Tokenize text
Remove stopwords
Lemmatize words
📈 Models & Metrics
Model	Features	Accuracy	Precision	Recall	F1 Score
Logistic Regression	BoW / TF-IDF	…	…	…	…
Naive Bayes	BoW / TF-IDF	…	…	…	…
Decision Tree	BoW / TF-IDF	…	…	…	…

Metrics are computed for each combination of model and feature vector.

🔍 Insights
TF-IDF usually outperforms Bag of Words.
Logistic Regression is generally the best model for text classification.
Naive Bayes is very fast and performs well on smaller datasets.
Decision Tree may overfit without pruning or ensemble methods.
💾 Output
sentiment_results.csv – Contains metrics for all models and feature types.
clean_text column added to the dataset after preprocessing.
📁 Project Structure
sentiment_analysis/
│
├─ sentiment_analysis.ipynb       # Full Colab-ready pipeline
├─ README.md                      # This file
├─ sentiment_results.csv          # Output metrics (generated)
└─ IMDB_Dataset.csv               # Dataset (downloaded from Kaggle / stored in Drive)
📌 References
IMDb Dataset: Kaggle IMDb Reviews
NLTK Documentation: https://www.nltk.org/
Scikit-learn Documentation: https://scikit-learn.org/
📝 Notes
This notebook is designed to run end-to-end on Google Colab.
You can extend it with advanced feature embeddings (Word2Vec, GloVe) or additional models like Random Forest, XGBoost, or deep learning models for improved performance.
  - nltk
  - scikit-learn
- Google Colab (recommended for easy Drive mounting and GPU support)

NLTK resources used:
```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
