#!/usr/bin/env python3
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import string

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
    y=df['spam']

    x_train, x_test, y_train, y_test = train_test_split(e_messages, y, test_size=0.33, random_state=42)
    naive_bayes(x_train, x_test, y_train, y_test)
    knn_classifier(x_train, x_test, y_train, y_test)
    sv_classifier(x_train, x_test, y_train, y_test)

def analyze_classifier(model_name, x_predicted, y_test):
    print(model_name)
    print("Classification Report",classification_report(y_test, x_predicted))
    print("\n Confusion Matrix: \n",confusion_matrix(y_test, x_predicted))

def naive_bayes(x_train, x_test, y_train, y_test):
    NB_classifier = MultinomialNB().fit(x_train, y_train)
    pred = NB_classifier.predict(x_test)
    analyze_classifier("Naive Bayes", pred, y_test)

def knn_classifier(x_train, x_test, y_train, y_test):
    neigh_classifier = KNeighborsClassifier().fit(x_train, y_train)
    pred = neigh_classifier.predict(x_test)
    analyze_classifier("KNN", pred, y_test)

def sv_classifier(x_train, x_test, y_train, y_test):
    svclassifier = SVC(kernel='linear').fit(x_train, y_train)
    pred = svclassifier.predict(x_test)
    analyze_classifier("SVM", pred, y_test)


if __name__ == '__main__':
    main()
