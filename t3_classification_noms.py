# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import glob
import os
import string
import unicodedata
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test-names-t3.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms 

# Fonctions utilitaires pour lire les données d'entraînement et de test - NE PAS MODIFIER

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

#---------------------------------------------------------------------------
# Fonctions à développer pour ce travail - Ne pas modifier les signatures et les valeurs de retour

def train_classifiers():
	load_names()
    # Vous ajoutez à partir d'ici tout le code dont vous avez besoin
    # pour construire les différentes versions de classificateurs de langues d'origines.
    # Voir les consignes de l'énoncé du travail pratique pour déterminer les différents modèles à entraîner.
    #
    # On suppose que les données d'entraînement ont été lues (load_names) et sont disponibles (names_by_origin).
    #
    # Vous pouvez ajouter au fichier toutes les fonctions que vous jugerez nécessaire.
    # Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    #
    # Votre code à partir d'ici...
    #
    
    
def get_classifier(type, n=3, weight='tf'):
    # Retourne le classificateur correspondant. On peut appeler cette fonction
    # après que les modèles aient été entraînés avec la fonction train_classifiers
    #
    classifier = MultinomialNB() # À modifier
    return classifier  
    
    
def origin(name, type, n=3, weight='tf'):
    # Retourne la langue d'origine prédite pour le nom.
    #   - name = le nom qu'on veut classifier
    #   - type = 'naive_bayes' ou 'logistic_regresion'
    #   - n désigne la longueur des N-grammes. Choix possible = 1, 2, 3, 'multi'
    #   - weight désigne le poids des attributs. Soit tf (comptes) ou tfidf.
    #
    # Votre code à partir d'ici...
    # À compléter...
    #
    name_origin = "French"  # À modifier
    return name_origin 
    
    
def test_classifier(test_fn, type, n=3, weight='tf'):
    test_data = load_test_names(test_fn)
    print("Nb de noms de test:", len(test_data))

    # Insérer ici votre code pour la classification des questions.
    # Votre code...

    test_accuracy = 0.8  # A modifier
    return test_accuracy


def evaluate_classifiers(filename):
    """Fonction utilitaire pour évaluer vos modèles. Aucune contrainte particulière.
       Nous n'utiliserons pas cette fonction pour l'évaluation de votre travail.
       Vous pouvez modifier le nom ou les arguments.
       """
    test_data = load_test_names(filename)
    # À compléter - Fonction pour l'évaluation des modèles N-grammes.
    # ...
    print("\nFonction evaluate_models - À compléter si ça peut vous être utile")


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    load_names()
    print("Les {} langues d'origine sont: \n{}".format(len(all_origins), all_origins))
    chinese_names = names_by_origin["Chinese"]
    print("\nQuelques noms chinois : \n", chinese_names[:20])

    train_classifiers()
    some_name = "Lamontagne"

    classifier = get_classifier('logistic_regresion', n=3, weight='tf')
    print("\nType de classificateur: ", classifier)

    some_origin = origin(some_name, 'naive_bayes', n='multi', weight='tfidf')
    print("\nLangue d'origine de {}: {}".format(some_name, some_origin))

    test_names = load_test_names(test_filename)
    print("\nLes données pour tester vos modèles sont:")
    for org, name_list in test_names.items():
        print("\t{} : {}".format(org, name_list))
    evaluate_classifiers(test_filename)
