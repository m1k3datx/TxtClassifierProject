import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import re

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

from time import time

import warnings
warnings.filterwarnings("ignore")

from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('US-Economic-News.csv', encoding='ISO-8859-1')

# Display the shape of the dataset
print(data.shape)

# Display the value counts of the 'relevance' column
data["relevance"].value_counts()

# Convert class labels into binary outcome variables
# 1 for Yes (relevant), and 0 for No (not relevant), and ignore "Not sure"
data = data[data.relevance != "not sure"]
data["relevance"] = data.relevance.map({'yes': 1, 'no': 0})
data = data[["text", "relevance"]]  # taking text input and output variable as relevance

# Define SEED variable
SEED = 123

# Address class imbalance using RandomUnderSampler
rus = RandomUnderSampler(random_state=SEED)
X_resampled, y_resampled = rus.fit_resample(data[['text']], data['relevance'])
data = pd.DataFrame({'text': X_resampled['text'], 'relevance': y_resampled})

# Display the shape of the preprocessed dataset
print(data.shape)

data.head()

# Text Cleaning
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text Cleaning

def clean(doc):
    doc = doc.lower().strip()
    doc = doc.replace('</br>', ' ')
    doc = doc.replace('-', ' ')
    doc = ''.join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    tokens = word_tokenize(doc)
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Apply text cleaning to the dataset
data['text'] = data['text'].apply(clean)
data.head()

# Split the data into training and test sets
SEED = 123
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['relevance'], test_size=0.2, random_state=SEED, stratify=data['relevance'])

# Extract features using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Convert to array
X_train_tfidf = X_train_tfidf.toarray()
X_test_tfidf = X_test_tfidf.toarray()

# Display the shape of the feature vectors
print(X_train_tfidf.shape, y_train.shape)
print(X_test_tfidf.shape, y_test.shape)

# Train and evaluate classifiers

# Naive Bayes Classifier
print("\nNaive Bayes Classifier")
gnb = GaussianNB()
gnb.fit(X_train_tfidf, y_train)
y_pred_train = gnb.predict(X_train_tfidf)
y_pred_test = gnb.predict(X_test_tfidf)
print("Training Accuracy score:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy score:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, target_names=['not relevant', 'relevant']))

# Multinomial Naive Bayes
print("\nMultinomial Naive Bayes")
mnb = MultinomialNB()
mnb.fit(X_train_tfidf, y_train)
y_pred_train = mnb.predict(X_train_tfidf)
y_pred_test = mnb.predict(X_test_tfidf)
print("Training Accuracy score:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy score:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, target_names=['not relevant', 'relevant']))

# Logistic Regression Classifier
print("\nLogistic Regression Classifier")
lr = LogisticRegression(random_state=SEED)
lr.fit(X_train_tfidf, y_train)
y_pred_train = lr.predict(X_train_tfidf)
y_pred_test = lr.predict(X_test_tfidf)
print("Training Accuracy score:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy score:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, target_names=['not relevant', 'relevant']))

# Support Vector Machines
print("\nSupport Vector Machines")
svc = LinearSVC(class_weight='balanced')
svc.fit(X_train_tfidf, y_train)
y_pred_train = svc.predict(X_train_tfidf)
y_pred_test = svc.predict(X_test_tfidf)
print("Training Accuracy score:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy score:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, target_names=['not relevant', 'relevant']))

# Decision Tree Classifier
print("\nDecision Tree Classifier")
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train_tfidf, y_train)
y_pred_train = dt.predict(X_train_tfidf)
y_pred_test = dt.predict(X_test_tfidf)
print("Training Accuracy score:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy score:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, target_names=['not relevant', 'relevant']))

# Random Forest Classifier
print("\nRandom Forest Classifier")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=SEED)
rf.fit(X_train_tfidf, y_train)
y_pred_train = rf.predict(X_train_tfidf)
y_pred_test = rf.predict(X_test_tfidf)
print("Training Accuracy score:", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy score:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, target_names=['not relevant', 'relevant']))