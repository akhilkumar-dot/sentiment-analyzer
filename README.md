# Sentiment Analyzer on Twitter Dataset

This project is a **Sentiment Analysis** system built using **Natural Language Processing (NLP)** techniques and a **Logistic Regression** classifier. It analyzes tweets to determine whether their sentiment is positive or negative, based on a dataset of over 1.5 million tweets.

## 🔍 Overview

- **Model Type**: Logistic Regression
- **Vectorization**: TF-IDF
- **Dataset**: 1,599,999 tweets × 6 columns
- **Accuracy**: 79%
- **Deployment**: Streamlit Web App

## 📊 Dataset

The dataset used contains **1.6 million tweets**, each labeled with its sentiment:

- **Positive (1)** or **Negative (0)**
- Columns include: sentiment, tweet ID, date, flag, user, text

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- scikit-learn
- NLTK
- Streamlit

## ⚙️ Features

- Clean and preprocess tweet text (remove stopwords, tokenize, etc.)
- Convert text to numeric form using **TfidfVectorizer**
- Train and evaluate a Logistic Regression model
- Show accuracy, confusion matrix, and visualizations
- Streamlit interface to predict sentiment of custom input tweets

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/akhilkumar-dot/sentiment-analyzer.git
cd sentiment-analyzer
