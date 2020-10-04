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

    global model
    order = 2

    vocabulary = build_vocabulary(proverbs)
    corpus_ngrams = get_ngrams(proverbs, n=order)

    model = Laplace(order)
    model.fit([corpus_ngrams], vocabulary_text=vocabulary)


def cloze_test(incomplete_proverb, choices, n=3):
    # Find previous_word and word_after by going through incomplete_proverb
    incomplete_corpus = word_tokenize(incomplete_proverb)
    first_x = incomplete_corpus.index('*')

    if first_x - 1 >= 0:
        previous_word = incomplete_corpus[first_x - 1]
    else:
        previous_word = BOS

    if first_x + 3 < len(incomplete_corpus):
        word_after = incomplete_corpus[first_x + 3]
    else:
        word_after = EOS

    score = None
    result = None

    if n == 2:
        for choice in choices:
            print(choice, [previous_word])
            newScore = model.logscore(choice, [previous_word])
            if result is None or newScore > score:
                result = choice
                score = newScore

    test_sequence = [(previous_word, result), (result, word_after)]
    print(test_sequence)
    perplexity_result = model.perplexity(test_sequence)

    print(score)

    return result, perplexity_result


if __name__ == '__main__':
    train_models(proverbs_fn)

    partial_proverb = "Le cours IFT-7022 est *** à distance cette année."
    options = ['on', 'qui', 'offert', 'rien']

    solution, perplexity = cloze_test(partial_proverb, options, n=2)
    print("\tSolution = {} , Perplexité = {}".format(solution, perplexity))

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
