import json
from nltk import word_tokenize, bigrams, trigrams
from nltk.util import pad_sequence
from nltk.util import ngrams
from nltk.lm.models import Laplace

BOS = '<BOS>'
EOS = '<EOS>'

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes1.txt"


def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
    return test_data


def build_vocabulary(text_list):
    all_unigrams = list()
    for sentence in text_list:
        word_list = word_tokenize(sentence.lower())
        all_unigrams = all_unigrams + word_list
    voc = set(all_unigrams)
    voc.add(BOS)
    voc.add(EOS)
    return list(voc)


def get_ngrams(text_list, n):
    all_ngrams = list()
    for sentence in text_list:
        tokens = word_tokenize(sentence.lower())
        padded_sent = list(
            pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=n))
        all_ngrams = all_ngrams + list(ngrams(padded_sent, n=n))

    return all_ngrams


def train_models(filename):
    global modelA, modelB, modelC

    proverbs = load_proverbs(filename)

    vocabulary = build_vocabulary(proverbs)

    order = 1
    modelA = Laplace(order)
    modelA.fit([get_ngrams(proverbs, n=order)], vocabulary_text=vocabulary)

    order = 2
    modelB = Laplace(order)
    modelB.fit([get_ngrams(proverbs, n=order)], vocabulary_text=vocabulary)

    order = 3
    modelC = Laplace(order)
    modelC.fit([get_ngrams(proverbs, n=order)], vocabulary_text=vocabulary)


def highest_score(model, previous_word, choices):
    score = None
    guess = None

    for choice in choices:
        print(choice)
        print(previous_word)
        if score is None or model.logscore(choice, [previous_word]) > score:
            guess = choice
            score = model.logscore(choice, [previous_word])

    print(score)
    return guess


def cloze_test(incomplete_proverb, choices, n):
    incomplete_corpus = word_tokenize(incomplete_proverb)
    first_x = incomplete_corpus.index('*')

    if first_x - 1 >= 0:
        previous_words = incomplete_corpus[first_x - 1]
    else:
        previous_words = BOS

    if first_x + 3 < len(incomplete_corpus):
        word_after = incomplete_corpus[first_x + 3]
    else:
        word_after = EOS

    if n == 2:
        result_return = highest_score(modelB, previous_words, choices)
        if result_return is not None:
            sequence = [tuple([previous_words, result_return]), tuple([result_return, word_after])]
            perplexity_return = modelB.perplexity(sequence)
        else:
            return 0, 0
    else:
        return 0, 0

    """
    previous_words = []
    for x in range(1, n):
        previous_words.insert(0, incomplete_corpus[first_x - x])

    word_after = []
    for x in range(first_x + 3, first_x + 3 + n - 1):
        if x >= len(incomplete_corpus):
            word_after.append('<EOS>')
        else:
            word_after.append(incomplete_corpus[x])
    
    if n == 1:
        result_return = highest_score(modelA, [''], choices)
        if result_return is not None:
            sequence = result_return
            perplexity_return = modelA.perplexity(sequence)
        else:
            return 0, 0
    elif n == 2:
        result_return = highest_score(modelB, [incomplete_corpus[first_x - 1]], choices)
        if result_return is not None:
            sequence = [tuple([previous_words.copy().pop(), result_return]), tuple([result_return, word_after.copy().pop()])]
            perplexity_return = modelB.perplexity(sequence)
        else:
            return 0, 0
    elif n == 3:
        result_return = highest_score(modelC, [incomplete_corpus[first_x - 1]], choices)
        if result_return is not None:
            sequence = [tuple([previous_words.copy().pop(0), previous_words.copy().pop(1), result_return]), tuple([result_return, word_after.copy().pop(0), word_after.copy().pop(1)])]
            perplexity_return = modelC.perplexity(sequence)
        else:
            return 0, 0
    """
    print(sequence)
    return result_return, perplexity_return


if __name__ == '__main__':

    ORDER = 2

    train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")

    for partial_proverb, options in test_proverbs.items():
        solution, perplexity = cloze_test(partial_proverb, options, n=ORDER)
        print("\n\t1 Proverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Perplexité = {}\n".format(solution, perplexity))