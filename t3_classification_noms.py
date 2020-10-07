# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import os
import unicodedata

import glob
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test-names-t3.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms

classifierBayes = [[None, None, None, None], [None, None, None, None]]
classifierRegression = [[None, None, None, None], [None, None, None, None]]
vectorizer = [[None, None, None, None], [None, None, None, None]]


def load_names():
    """Lecture des noms et langues d'origine d'un fichier. Par la suite,
       sauvegarde des noms pour chaque origine dans le dictionnaire names_by_origin."""
    for filename in find_files(datafiles):
        origin = get_origin_from_filename(filename)
        all_origins.append(origin)
        names = read_names(filename)
        names_by_origin[origin] = names


def find_files(path):
    """Retourne le nom des fichiers contenus dans un répertoire.
       glob fait le matching du nom de fichier avec un pattern - par ex. *.txt"""
    return glob.glob(path)


def get_origin_from_filename(filename):
    """Passe-passe qui retourne la langue d'origine d'un nom de fichier.
       Par ex. cette fonction retourne Arabic pour "./data/names/Arabic.txt". """
    return os.path.splitext(os.path.basename(filename))[0]


def read_names(filename):
    """Retourne une liste de tous les noms contenus dans un fichier."""
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def unicode_to_ascii(s):
    """Convertion des caractères spéciaux en ascii. Par exemple, Hélène devient Helene.
       Tiré d'un exemple de Pytorch. """
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def load_test_names(filename):
    """Retourne un dictionnaire contenant les données à utiliser pour évaluer vos modèles.
       Le dictionnaire contient une liste de noms (valeurs) et leur origine (clé)."""
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data


# ---------------------------------------------------------------------------
# Fonctions à développer pour ce travail - Ne pas modifier les signatures et les valeurs de retour


def train_classifiers():
    load_names()

    # Divide training data into X_train and Y_train
    y_train = []
    X_train = []
    for key in names_by_origin:
        for value in names_by_origin[key]:
            y_train.append(key)
            X_train.append(value)

    print("___ Model performance testing ___ n=1 unigramn, n=2 bigram, n=3 trigram, n=4 multigram, weight=0 tf, weight=1 tfidf")

    # Create the 8 vectorizers and the 16 models
    for i in range(2):
        for y in range(4):

            if i == 0:
                if y == 0:
                    vectorizer[i][y] = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 1))
                elif y == 1:
                    vectorizer[i][y] = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(2, 2))
                elif y == 2:
                    vectorizer[i][y] = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(3, 3))
                elif y == 3:
                    vectorizer[i][y] = CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 3))
            elif i == 1:
                if y == 0:
                    vectorizer[i][y] = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 1))
                elif y == 1:
                    vectorizer[i][y] = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(2, 2))
                elif y == 2:
                    vectorizer[i][y] = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(3, 3))
                elif y == 3:
                    vectorizer[i][y] = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 3))

            vectorizer[i][y].fit(X_train)
            X_train_vectorized = vectorizer[i][y].transform(X_train)

            for classifier_type in range(2):
                if classifier_type == 0:
                    classifierBayes[i][y] = MultinomialNB()
                    classifierBayes[i][y].fit(X_train_vectorized, y_train)
                    # ___ Print cross evaluation score ___
                    scores = cross_val_score(classifierBayes[i][y], X_train_vectorized, y_train, cv=5)
                    print("Cross Evaluation (model - Bayes, n={}, weight={}): Exactitude moyenne: {:0.2f} {:0.2f}".format(y+1, i, scores.mean(), scores.std() * 2))
                if classifier_type == 1:
                    classifierRegression[i][y] = LogisticRegression(max_iter=5000)
                    classifierRegression[i][y].fit(X_train_vectorized, y_train)
                    # ___ Print cross evaluation score ___
                    scores = cross_val_score(classifierRegression[i][y], X_train_vectorized, y_train, cv=5)
                    print("Cross Evaluation (model - Regression, n={}, weight={}): Exactitude moyenne: {:0.2f} +-{:0.2f}".format(y+1, i, scores.mean(), scores.std() * 2))


def get_classifier(type, n=3, weight='tf'):
    if type == 'naive_bayes':
        if n == 1:
            if weight == 'tf':
                return classifierBayes[0][0]
            elif weight == 'tfidf':
                return classifierBayes[1][0]
        elif n == 2:
            if weight == 'tf':
                return classifierBayes[0][1]
            elif weight == 'tfidf':
                return classifierBayes[1][1]
        elif n == 3:
            if weight == 'tf':
                return classifierBayes[0][2]
            elif weight == 'tfidf':
                return classifierBayes[1][2]
        elif n == "multi":
            if weight == 'tf':
                return classifierBayes[0][3]
            elif weight == 'tfidf':
                return classifierBayes[1][3]
    elif type == 'logistic_regresion':
        if n == 1:
            if weight == 'tf':
                return classifierRegression[0][0]
            elif weight == 'tfidf':
                return classifierRegression[1][0]
        elif n == 2:
            if weight == 'tf':
                return classifierRegression[0][1]
            elif weight == 'tfidf':
                return classifierRegression[1][1]
        elif n == 3:
            if weight == 'tf':
                return classifierRegression[0][2]
            elif weight == 'tfidf':
                return classifierRegression[1][2]
        elif n == "multi":
            if weight == 'tf':
                return classifierRegression[0][3]
            elif weight == 'tfidf':
                return classifierRegression[1][3]

    raise Exception("Not found")


def get_vectorizer(type, n=3, weight='tf'):
    if n == 1:
        if weight == 'tf':
            return vectorizer[0][0]
        elif weight == 'tfidf':
            return vectorizer[1][0]
    elif n == 2:
        if weight == 'tf':
            return vectorizer[0][1]
        elif weight == 'tfidf':
            return vectorizer[1][1]
    elif n == 3:
        if weight == 'tf':
            return vectorizer[0][2]
        elif weight == 'tfidf':
            return vectorizer[1][2]
    elif n == "multi":
        if weight == 'tf':
            return vectorizer[0][3]
        elif weight == 'tfidf':
            return vectorizer[1][3]

    raise Exception("Not found")


def origin(name, type, n=3, weight='tf'):
    classifier = get_classifier(type, n, weight)
    vectorizer = get_vectorizer(type, n, weight)

    result = classifier.predict(vectorizer.transform([name]))

    return result


def test_classifier(test_fn, type, n=3, weight='tf'):
    classifier = get_classifier(type, n, weight)
    vectorizer = get_vectorizer(type, n, weight)

    test_data = load_test_names(test_fn)

    # Divide testing data into X_test and Y_test
    Y_test = []
    X_test = []
    for org, name_list in test_data.items():
        for value in name_list:
            Y_test.append(org)
            X_test.append(value)

    y_pred = classifier.predict(vectorizer.transform(X_test))
    score = accuracy_score(Y_test, y_pred)
    return score


if __name__ == '__main__':
    train_classifiers()

    some_name = "Lamontagne"
    some_origin = origin(some_name, 'logistic_regresion', n='multi', weight='tfidf')
    print("\nLangue d'origine de {}: {}".format(some_name, some_origin))

    some_name = "Nguyen"
    some_origin = origin(some_name, 'logistic_regresion', n='multi', weight='tfidf')
    print("\nLangue d'origine de {}: {}".format(some_name, some_origin))

    some_name = "Yassine"
    some_origin = origin(some_name, 'logistic_regresion', n='multi', weight='tfidf')
    print("\nLangue d'origine de {}: {}".format(some_name, some_origin))


    accuracy_score = test_classifier(test_filename, 'logistic_regresion', n='multi', weight='tfidf')
    print("\nAccuracy: {}".format(accuracy_score))
