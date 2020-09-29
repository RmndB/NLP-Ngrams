import json
from nltk import word_tokenize, bigrams, trigrams
from nltk.util import pad_sequence
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.lm.models import Laplace

import re

BOS = '<BOS>'
EOS = '<EOS>'

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes1.txt"

text = ["le cours ift 7022 est offert à distance cette année .",
        "ce cours n est habituellement pas à distance .",
        "le cours est habituellement donnée à l automne ."]

def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

def build_vocabulary(text_list):
    all_unigrams = list()
    for sentence in text_list:
        word_list = word_tokenize(sentence.lower())
        all_unigrams = all_unigrams + word_list
    voc = set(all_unigrams)
    voc.add(BOS)
    return list(voc)

def get_ngrams(text_list, n): #HEIIIIIIIIIIIIIIIIIN
    all_ngrams = list()
    for sentence in text_list:
        tokens = word_tokenize(sentence.lower())
        padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=n))
        all_ngrams = all_ngrams + list(ngrams(padded_sent, n=n))
    return all_ngrams

def train_models(filename):
    #proverbs = load_proverbs(filename)
    global modelA, modelB, modelC

    vocabulary = build_vocabulary(text)

    order = 1
    modelA = Laplace(order)
    modelA.fit([get_ngrams(text, n=order)], vocabulary_text=vocabulary)

    order = 2
    modelB = Laplace(order)
    modelB.fit([get_ngrams(text, n=order)], vocabulary_text=vocabulary)

    order = 3
    modelC = Laplace(order)
    modelC.fit([get_ngrams(text, n=order)], vocabulary_text=vocabulary)

    """
    print(len(model.vocab))
    print(model.score("ift", ["cours"]))
    print(model.logscore("ift", ["cours"]))
    test_sequence = [("le", "cours"), ("cours", "ift")]
    print(model.perplexity(test_sequence))
    print(model.generate(text_seed=['cours']))
    """

def highest_score(model, previous_word, choices):
    score = 0
    guess = None

    for choice in choices:
        if model.score(choice, previous_word) > score:
            guess = choice
            score = model.score(choice, previous_word)

    return guess

def cloze_test(incomplete_proverb, choices, n):

    incomplete_corpus = word_tokenize(incomplete_proverb)
    first_x = incomplete_corpus.index('*')

    previous_words = []
    for x in range(1, n):
        previous_words.insert(0, incomplete_corpus[first_x - x])
    previous_word_further = incomplete_corpus[first_x - n]

    print(previous_words)
    print(previous_word_further)

    if n == 1:
        result_return = highest_score(modelA, previous_words, choices)
        if result_return is not None:
            listA = previous_words.copy()
            listA.insert(0, previous_word_further)
            listB = previous_words.copy()
            listB.append(result_return)
            perplexity_return = modelA.perplexity([listA, listB])
        else:
            return 0, 0
    elif n == 2:
        result_return = highest_score(modelB, previous_words, choices)
        if result_return is not None:
            listA = previous_words.copy()
            listA.insert(0, previous_word_further)
            listB = previous_words.copy()
            listB.append(result_return)
            perplexity_return = modelB.perplexity([listA, listB])
        else:
            return 0, 0
    elif n == 3:
        result_return = highest_score(modelC, previous_words, choices)
        if result_return is not None:
            listA = previous_words.copy()
            listA.insert(0, previous_word_further)
            listB = previous_words.copy()
            listB.append(result_return)
            perplexity_return = modelC.perplexity([listA, listB])
        else:
            return 0, 0
            
    return result_return, perplexity_return

if __name__ == '__main__':
    partial_proverb = "le cours ift 7022 est *** à distance cette année ."
    options = ['est', 'offert', 'cours', 'distance']

    print(partial_proverb)
    train_models(proverbs_fn)

    solution, perplexity = cloze_test(partial_proverb, options, n=1)
    print("\n\t1 Proverbe incomplet: {} , Options: {}".format(partial_proverb, options))
    print("\tSolution = {} , Perplexité = {}\n".format(solution, perplexity))

    solution, perplexity = cloze_test(partial_proverb, options, n=2)
    print("\n\t2 Proverbe incomplet: {} , Options: {}".format(partial_proverb, options))
    print("\tSolution = {} , Perplexité = {}\n".format(solution, perplexity))

    solution, perplexity = cloze_test(partial_proverb, options, n=3)
    print("\n\t3 Proverbe incomplet: {} , Options: {}".format(partial_proverb, options))
    print("\tSolution = {} , Perplexité = {}".format(solution, perplexity))

"""
if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    
    proverbs = load_proverbs(proverbs_fn)
    print("\nNombre de proverbes : ", len(proverbs))
    train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    for partial_proverb, options in test_proverbs.items():
        solution, perplexity = cloze_test(partial_proverb, options, n=20)
        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Perplexité = {}".format(solution, perplexity))
    """