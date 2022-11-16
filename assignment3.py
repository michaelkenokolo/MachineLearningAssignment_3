#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import scipy.stats
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords

nltk.download('stopwords')

__author__ = 'Jacob Hajjar, Michael-Ken Okolo'
__email__ = 'hajjarj@csu.fullerton.edu, michaelken.okolo1@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar, michaelkenokolo'


def txt_process(text):
    remove_punc = [character for character in text if character not in string.punctuation]
    remove_punc = ''.join(remove_punc)
    cl_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]
    return cl_words


def main():
    """the main function"""
    df = pd.read_csv('emails.csv')
    df.drop_duplicates(inplace=True)
    e_messages = CountVectorizer(analyzer=txt_process).fit_transform((df['text']))
    y = df['spam']

    x_train, x_test, y_train, y_test = train_test_split(e_messages, y, test_size=0.33, random_state=42)
    naive_bayes(x_train, x_test, y_train, y_test)
    knn_classifier(x_train, x_test, y_train, y_test)
    svm_classifier(x_train, x_test, y_train, y_test)


def analyze_classifier(model_name, x_predicted, y_test):
    """Output the classification report and confusion matrices for Naive Bayes, KNN, and SVM classifiers"""
    print(model_name)
    print("Classification Report", classification_report(y_test, x_predicted))
    print("\n Confusion Matrix: \n", confusion_matrix(y_test, x_predicted))


def roc_curves(y_test, pred):
    """Calculate and plot the roc curves for Naive Bayes, KNN, and SVM classifiers"""
    metrics.RocCurveDisplay.from_predictions(y_test, pred)
    # plt.show()


def find_95_confidence_interval(acc_test, y_test):
    """Calculate and output the 95% confidence interval for Naive Bayes, KNN, and SVM classifiers"""
    z_value = scipy.stats.norm.ppf((1 + 0.95) / 2.0)
    ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / y_test.shape[0])
    ci_lower = acc_test - ci_length
    ci_upper = acc_test + ci_length
    print(ci_lower, ci_upper)


def naive_bayes(x_train, x_test, y_train, y_test):
    """Classify emails using Naïve Bayes"""
    NB_classifier = MultinomialNB().fit(x_train, y_train)
    pred = NB_classifier.predict(x_test)
    analyze_classifier("Naive Bayes", pred, y_test)
    roc_curves(y_test, pred)
    find_95_confidence_interval(NB_classifier.score(x_test, y_test), y_test)


def knn_classifier(x_train, x_test, y_train, y_test):
    """Classify emails using KNN"""
    KNN_classifier = KNeighborsClassifier().fit(x_train, y_train)
    pred = KNN_classifier.predict(x_test)
    analyze_classifier("KNN", pred, y_test)
    roc_curves(y_test, pred)
    find_95_confidence_interval(KNN_classifier.score(x_test, y_test), y_test)


def svm_classifier(x_train, x_test, y_train, y_test):
    """Classify emails using SVM"""
    SVM_classifier = SVC(kernel='linear').fit(x_train, y_train)
    pred = SVM_classifier.predict(x_test)
    analyze_classifier("SVM", pred, y_test)
    roc_curves(y_test, pred)
    find_95_confidence_interval(SVM_classifier.score(x_test, y_test), y_test)


if __name__ == '__main__':
    main()
