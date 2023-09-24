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

### User Interface

- Develop a user-friendly interface that allows users to input or upload climate news articles for sentiment analysis.
- Nice-to-have: The interface can provide sentiment scores and visualizations to make the analysis more accessible.

## 2. Write-Ups Your Team Developed

### 1. Data Collection

We explored Kaggle, Open Source Datasets, GitHub, and scraped data from New York Times and The Guardian((#add the link to scrappers here). Initially, we wrote scripts to web-crawl and gather headlines of climate news articles from news a, however it was challenging to find consistent datasets that fulfilled our project scope requirements for running sentiment analysis on climate news and manually review sentiment to each statement. Hence, we proceeded to use Huggingface expert-labeled datasets. Finally, we assumed that "risk" and "opportunity" should refer to negative and positive news.

### 2. Data Preprocessing

Some of the data techniques used were:
- Conversion to lowercase
- Removal of numbers
- Removal of stop words
- Lemmatization
- Tokenization

### 3. Data Visualization

- Word Clouds were used to visualize words associated with positive and negative news.
- Scatter plot was used to visualize the average sentiments in relation to their frequencies for the commonly-used words.
- Pie Chart was used to visualize the ratio of keywords with positive, neutral and negative average sentiments.
- Bar Charts were used to see the word counts and frequency distribution for each sentiment, positive, neutral and negative.

### 4. Sentiment Analysis Model Exploration & Development

We explored different techniques and models such as 
- CountVectorizer
- TF-IDF Vectorizer 
- Support Vector Machine (SVM)
- Logistic Regression
- Naive Bayes
- LSTM Neural Network.

### 5. User Interface

#To be completed
