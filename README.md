Movie Review Sentiment Analysis
This project demonstrates a simple yet effective approach to performing sentiment analysis on movie reviews. It utilizes common Natural Language Processing (NLP) techniques and a Logistic Regression model to classify reviews as either "Positive" or "Negative".

Code Overview
The Google Colab notebook AnalysisStentimentFilm.ipynb walks through the following steps:

Library Imports and NLTK Downloads:

Imports essential libraries such as pandas for data manipulation, numpy for numerical operations, matplotlib and seaborn for visualization, re for regular expressions, nltk for NLP tasks, and sklearn for machine learning models and metrics.

Ensures necessary NLTK resources like stopwords and punkt are downloaded for text processing.

Dataset Loading and Initial Exploration:

Loads a movie review dataset from a raw CSV file hosted on GitHub. This dataset contains review texts and their corresponding sentiment labels (positive/negative).

Prints the first 5 rows, basic information (df.info()), and the distribution of sentiment labels to provide an initial understanding of the data.

Text Preprocessing:

Initializes PorterStemmer and loads English stop words from NLTK.

Defines a clean_text function that performs the following transformations on each review:

Removes HTML tags.

Removes non-alphabetic characters and converts text to lowercase.

Tokenizes the text into individual words.

Removes stop words and applies stemming (reducing words to their root form).

Joins the processed words back into a single string.

Applies this clean_text function to the 'review' column, creating a new 'cleaned_review' column.

Displays examples of original vs. cleaned reviews to illustrate the effect of preprocessing.

Label Encoding:

Converts the categorical 'sentiment' labels ('positive', 'negative') into numerical representations ('1' for positive, '0' for negative).

Displays the updated DataFrame head and the distribution of the new numerical sentiment labels.

Data Splitting:

Divides the cleaned_review (features, X) and sentiment_numeric (target, y) into training and testing sets using train_test_split.

A test_size of 20% is used, and stratify=y ensures that the sentiment distribution is maintained across both sets.

Prints the sizes and sentiment distributions of the training and testing sets.

Text Vectorization (TF-IDF):

Initializes a TfidfVectorizer to convert cleaned text into numerical TF-IDF (Term Frequency-Inverse Document Frequency) features. max_features=5000 is used to limit the vocabulary size.

fit_transform is applied to the training data (X_train) to learn the vocabulary and transform the text.

transform is applied to the testing data (X_test) using the vocabulary learned from the training data.

Prints the shape of the resulting TF-IDF matrices.

Model Training (Logistic Regression):

An instance of LogisticRegression is created. max_iter=1000 is set to help with convergence for potentially large datasets.

The model is trained (fit) using the TF-IDF transformed training data (X_train_tfidf) and their corresponding labels (y_train).

Model Prediction:

The trained model makes predictions on the TF-IDF transformed test data (X_test_tfidf).

Model Evaluation:

Calculates and prints the accuracy_score of the model on the test set.

Displays a classification_report, which includes precision, recall, and F1-score for both 'Negative' and 'Positive' classes.

Generates and visualizes a confusion_matrix using seaborn.heatmap to show the counts of true positives, true negatives, false positives, and false negatives.

Sentiment Prediction Function:

Defines a predict_sentiment function that takes raw text as input.

This function cleans the input text, transforms it using the trained tfidf_vectorizer, and then uses the trained LogisticRegression model to predict the sentiment.

Based on the prediction (1 or 0), it returns "Positif" or "Negatif".

Provides several example reviews to demonstrate how the function can be used to predict sentiment for new text inputs.

Dataset
The dataset used in this project is sourced from:
https://raw.githubusercontent.com/Ankit152/IMDB-Sentiment-Analysis/master/IMDB-Dataset.csv

It contains two columns:

review: The text content of the movie review.

sentiment: The corresponding sentiment label, either 'positive' or 'negative'.

Libraries Used
pandas

numpy

matplotlib

seaborn

re

nltk

scikit-learn

How to Use (in Google Colab)
Open the AnalysisStentimentFilm.ipynb file in Google Colab.

Run all cells sequentially.

Ensure you have an active internet connection to download the dataset and NLTK resources.

The output will display the data exploration, preprocessing steps, model training progress, evaluation metrics, and example sentiment predictions.
