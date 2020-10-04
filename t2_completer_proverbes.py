import json

from nltk.util import pad_sequence
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.lm.models import Laplace

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes1.txt"

corpus = ["le cours ift 7022 est offert à distance cette année .",
          "ce cours n est habituellement pas à distance .",
          "le cours est habituellement donnée à l automne ."]

BOS = '<BOS>'
EOS = '<EOS>'

model = dict()
mode = 0


def build_vocabulary(text_list):
    all_unigrams = list()
    for sentence in text_list:
        word_list = word_tokenize(sentence.lower())
        all_unigrams = all_unigrams + word_list
    voc = set(all_unigrams)
    voc.add(BOS)
    return list(voc)


def get_ngrams(text_list, n=2):
    all_ngrams = list()
    for sentence in text_list:
        tokens = word_tokenize(sentence.lower())
        padded_sent = list(
            pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=n))
        all_ngrams = all_ngrams + list(ngrams(padded_sent, n=n))
    return all_ngrams


def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
    return test_data


def train_models(filename):
    # proverbs = load_proverbs(filename)

    proverbs = corpus
    vocabulary = build_vocabulary(proverbs)

    global model

    for order in range(1, 4):
        corpus_ngrams = get_ngrams(proverbs, n=order)
        newModel = Laplace(order)
        newModel.fit([corpus_ngrams], vocabulary_text=vocabulary)
        model[order] = newModel


def find_with_score(choices, previous_words, n):
    score = None
    result = None

    for choice in choices:
        if n == 1:
            context = tuple()
        elif n == 2:
            context = tuple(previous_words)
        elif n == 3:
            context = tuple([previous_words[0], previous_words[1]])

        newScore = model[n].logscore(choice, context)
        if result is None or newScore > score:
            result = choice
            score = newScore

    return result


def find_with_perplexity(choices, previous_words, words_after, n):
    perplexity = None
    result = None

    for choice in choices:
        if n == 1:
            test_sequence = choice
        elif n == 2:
            test_sequence = [(previous_words[0], choice), (choice, words_after[0])]
        elif n == 3:
            test_sequence = [(previous_words[0], previous_words[1], choice), (choice, words_after[0], words_after[1],)]

        newPerplexity = model[n].perplexity(test_sequence)

        if result is None or newPerplexity < perplexity:
            result = choice
            perplexity = newPerplexity

    return result


def cloze_test(incomplete_proverb, choices, n=3):
    incomplete_corpus = word_tokenize(incomplete_proverb.lower())
    first_x = incomplete_corpus.index('*')

    previous_words = []
    words_after = []

    for i in range(1, n):
        if first_x - i >= 0:
            previous_words.insert(0, incomplete_corpus[first_x - i])
        else:
            previous_words.insert(0, BOS)
        if first_x + 2 + i < len(incomplete_corpus):
            words_after.append(incomplete_corpus[first_x + 2 + i])
        else:
            words_after.append(EOS)

    # result = find_with_score(choices, previous_words, n)
    result = find_with_perplexity(choices, previous_words, words_after, n)

    if n == 1:
        test_sequence = result
    elif n == 2:
        test_sequence = [(previous_words[0], result), (result, words_after[0])]
    elif n == 3:
        test_sequence = [(previous_words[0], previous_words[1], result), (result, words_after[0], words_after[1],)]

    perplexity = model[n].perplexity(test_sequence)

    return result, perplexity


if __name__ == '__main__':
    train_models(proverbs_fn)

    partial_proverb = "Le cours IFT-7022 est offert à *** cette année."
    options = ['on', 'distance', 'offert', 'rien']

    solution, perplexity_result = cloze_test(partial_proverb, options, n=2)
    print("\tSolution = {} , Perplexité = {}".format(solution, perplexity_result))

if __name__ == '__main__':
    """
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
