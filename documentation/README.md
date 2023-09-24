# Project Documentation

## Project Statement

The objective of this project is to develop a Natural Language Processing (NLP) tool that can accurately analyze and detect the nuances of sentiment in climate news articles. This tool will aim to determine whether a given climate news article carries a positive, negative, or neutral sentiment.

## 1. Project Scope

The project goal is to create a tool to detect nuances in climate news articles. This project is based on Natural Language Understanding, where we will perform Sentiment Analysis on climate news articles to determine their positivity.

### Data Collection

- Gather a diverse dataset of climate news articles from reputable sources. This dataset should include a mix of positive, negative, and neutral sentiment articles.

### Data Preprocessing

- Clean and preprocess the collected data by removing irrelevant information, formatting the text, and tokenizing the articles.

### Data Visualization

- Explore and create meaningful visualizations of the dataset.

### Sentiment Analysis Model Exploration & Development

- Explore various approaches to modeling sentiment analysis, including Standard Machine Learning (Regressions, SVMs, Naive Bayes) and Deep Learning-based approaches.
- Develop a sentiment analysis model tailored to climate news. This model should be trained to detect positive, negative, and neutral sentiments accurately.

### Pipeline 
Simulate a pipeline that preprocess news and interact with one of the models

### User Interface

- Develop a user-friendly interface that allows users to input or upload climate news articles for sentiment analysis.
- Nice-to-have: The interface can provide sentiment scores and visualizations to make the analysis more accessible.

## 2. Write-Ups Your Team Developed

### 1. Data Collection

We explored Kaggle, Open Source Datasets, GitHub, and scraped data from [New York Times and The Guardian](/scripts/data_scraping.py). It was challenging to find consistent datasets that fulfilled our project scope requirements for running sentiment analysis on climate news. We proceeded to use Huggingface expert-labeled datasets. Finally, we assumed that "risk" and "opportunity" should refer to negative and positive news.

### 2. Data Preprocessing

Some of the data techniques used were:
- Conversion to lowercase
- Removal of numbers
- Removal of stop words
- Lemmatization
- Tokenization

### 3. Data Visualization

Word Clouds were used to visualize words associated with positive and negative news.

### 4. Sentiment Analysis Model Exploration & Development

We explored different techniques such as CountVectorizer, TF-IDF Vectorizer, and different models such as Support Vector Machine, Logistic Regression, Naive Bayes, and LSTM Neural Network.

### 5. Pipeline
Our pipeline is jupyter notebook called [pipeline.ipynb](/notebooks/pipeline.ipynb)
### 6. User Interface

The use interface was created using Gradio

### 7. Bag of goodies
Our exploration lead us to ideas for [future features](/notebooks/Ideas_exploration.ipynb)
