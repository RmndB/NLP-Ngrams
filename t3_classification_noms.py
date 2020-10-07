# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import os
import unicodedata

import glob
import string
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test-names-t3.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms

classifier = None
vectorizer = None


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

"""
def evaluate_classifiers(filename):

    test_data = load_test_names(filename)
    # À compléter - Fonction pour l'évaluation des modèles N-grammes.
    # ...
    print("\nFonction evaluate_models - À compléter si ça peut vous être utile")
"""


def cross_validation(classifierToTest, X, y):
    scores = cross_val_score(classifierToTest, X, y, cv=5)
    print("\nÉvaluation par validation croisée (en entraînement) : ")
    print("   Exactitude (accuracy) sur chaque partition", scores)
    print("   Exactitude moyenne: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def train_classifiers():
    global classifier, vectorizer

    load_names()

    y_train = []
    X_train = []
    for key in names_by_origin:
        for value in names_by_origin[key]:
            y_train.append(key)
            X_train.append(value)

    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 3))
    vectorizer.fit(X_train)

    X_train_vectorized = vectorizer.transform(X_train)

    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # ___ Print data ___
    cross_validation(classifier, X_train_vectorized, y_train)

    """
    class_probs = list(zip(classifier.classes_, classifier.class_log_prior_))
    for x, prob in class_probs:
        print("logprob({}) = {}".format(x, round(prob, 2)))
    """


def get_classifier(type, n=3, weight='tf'):
    global classifier

    # Add condition
    return classifier


def origin(name, type, n=3, weight='tf'):
    #   - name = le nom qu'on veut classifier
    #   - type = 'naive_bayes' ou 'logistic_regresion'
    #   - n désigne la longueur des N-grammes. Choix possible = 1, 2, 3, 'multi'
    #   - weight désigne le poids des attributs. Soit tf (comptes) ou tfidf.

    global classifier, vectorizer

    score = None
    df = pd.DataFrame(vectorizer.get_feature_names(), columns=['Mots'])
    for i in range(len(classifier.classes_)):
        df[classifier.classes_[i]] = list(classifier.feature_log_prob_[i])

    question_words = [name.lower()]
    qw_probs = df[df['Mots'].isin(question_words)]
    for key in qw_probs:
        if key != 'Mots':
            newScore = qw_probs[key].values[0]
            if score is None or newScore > score:
                score = newScore
                result = key

    # result = classifier.predict(vectorizer.transform([name]))

    return result


def test_classifier(test_fn, type, n=3, weight='tf'):
    test_data = load_test_names(test_fn)
    Y_test = []
    X_test = []

    for org, name_list in test_data.items():
        for value in name_list:
            Y_test.append(org)
            X_test.append(value)

    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vectorized)

    print(y_pred)

    test_accuracy = accuracy_score(Y_test, y_pred)
    return test_accuracy


if __name__ == '__main__':
    load_names()
    print("Les {} langues d'origine sont: \n{}".format(len(all_origins), all_origins))
    chinese_names = names_by_origin["Chinese"]
    print("\nQuelques noms chinois : \n", chinese_names[:20])

    train_classifiers()

    some_name = "Travert"

    classifier = get_classifier('logistic_regresion', n=3, weight='tf')
    print("\nType de classificateur: ", classifier)

    some_origin = origin(some_name, 'naive_bayes', n='multi', weight='tfidf')
    print("\nLangue d'origine de {}: {}".format(some_name, some_origin))

    """
    test_names = load_test_names(test_filename)
    print("\nLes données pour tester vos modèles sont:")
    for org, name_list in test_names.items():
        print("\t{} : {}".format(org, name_list))
    """

    accuracy_score = test_classifier(test_filename, 1, n=3, weight='tf')
    print("\tAccuracy: {}".format(accuracy_score))