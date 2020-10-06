import json

from nltk.util import pad_sequence
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.lm.models import Laplace
from collections import defaultdict
import math

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes1.txt"

BOS = '<BOS>'
EOS = '<EOS>'

model = defaultdict(lambda: defaultdict(lambda: 0))
mode = None
order = None
counter = defaultdict(lambda: 0)
total = 0

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


# TODO: Edit/Refactor this function
def create_custom_bigram(proverbs):
    global model, counter, total

    corpus_ngrams = get_ngrams(proverbs, n=2)
    corpus_unigram = get_ngrams(proverbs, n=1)

    for w1, w2 in corpus_ngrams:
        model[w1][w2] += 1

    for w in corpus_unigram:
        counter[w[0]] += 1
        total += 1
    counter[EOS] = len(load_proverbs(proverbs_fn))
    counter[BOS] = counter[EOS]

    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count

    return model


def train_models(filename):
    proverbs = load_proverbs(filename)

    vocabulary = build_vocabulary(proverbs)

    global model

    if order == 1 or order == 2 or order == 3:
        corpus_ngrams = get_ngrams(proverbs, n=order)
        model = Laplace(order)
        model.fit([corpus_ngrams], vocabulary_text=vocabulary)
    elif order == 20:
        model = create_custom_bigram(proverbs)
    else:
        raise Exception("Value must be either 1, 2, 3 or 20.")


def find_with_score(choices, previous_words, n):
    score = None
    result = None

    for choice in choices:
        if n == 1:
            context = tuple()
            newScore = model.logscore(choice, context)
        elif n == 2:
            context = tuple(previous_words)
            newScore = model.logscore(choice, context)
        elif n == 3:
            context = tuple([previous_words[0], previous_words[1]])
            newScore = model.logscore(choice, context)
        elif n == 20:
            newScore = dict(model[previous_words[0]]).get(choice)
            if newScore is not None:
                newScore = math.log(dict(model[previous_words[0]]).get(choice))
            else:
                # TODO: Better way to handle unseen values
                newScore = -99999999

        if result is None or newScore > score:
            result = choice
            score = newScore

    # TODO: Remove this line once post-debugging
    print(score)

    return result, score


def find_with_perplexity(choices, previous_words, words_after, n):
    perplexity = None
    result = None

    for choice in choices:
        if n == 1:
            test_sequence = choice
            newPerplexity = model.perplexity(test_sequence)
        elif n == 2:
            test_sequence = [(previous_words[0], choice), (choice, words_after[0])]
            newPerplexity = model.perplexity(test_sequence)
        elif n == 3:
            test_sequence = [(previous_words[0], previous_words[1], choice), (choice, words_after[0], words_after[1],)]
            newPerplexity = model.perplexity(test_sequence)
        elif n == 20:
            raise Exception("TODO")

        if result is None or newPerplexity < perplexity:
            result = choice
            perplexity = newPerplexity

    return result


def cloze_test(incomplete_proverb, choices, n=3):
    global model, counter

    incomplete_corpus = word_tokenize(incomplete_proverb.lower())
    first_x = incomplete_corpus.index('*')

    previous_words = []
    words_after = []

    if order == 1 or order == 2 or order == 3:
        for i in range(1, n):
            if first_x - i >= 0:
                previous_words.insert(0, incomplete_corpus[first_x - i])
            else:
                previous_words.insert(0, BOS)
            if first_x + 2 + i < len(incomplete_corpus):
                words_after.append(incomplete_corpus[first_x + 2 + i])
            else:
                words_after.append(EOS)
    else:
        if first_x - 1 >= 0:
            previous_words.insert(0, incomplete_corpus[first_x - 1])
        else:
            previous_words.insert(0, BOS)
        if first_x + 3 < len(incomplete_corpus):
            words_after.append(incomplete_corpus[first_x + 3])
        else:
            words_after.append(EOS)

    if mode == 0:
        result, score = find_with_score(choices, previous_words, n)
    elif mode == 1:
        result = find_with_perplexity(choices, previous_words, words_after, n)

    if n == 1:
        test_sequence = result
        perplexity = model.perplexity(test_sequence)
    elif n == 2:
        test_sequence = [(previous_words[0], result), (result, words_after[0])]
        perplexity = model.perplexity(test_sequence)
    elif n == 3:
        test_sequence = [(previous_words[0], previous_words[1], result), (result, words_after[0], words_after[1],)]
        perplexity = model.perplexity(test_sequence)
    elif n == 20:
        # print(total, previous_words[0], counter[tuple(previous_words[0])], result, counter[result], words_after[0], counter[words_after[0]])
        perplexity = 2**(-(1/3)*(math.log2(counter[previous_words[0]]/total)+math.log2(counter[result]/total)+math.log2(counter[words_after[0]]/total)))

    return result, perplexity


if __name__ == '__main__':

    order = 20
    mode = 0

    print("\nNombre de proverbes : ", len(load_proverbs(proverbs_fn)))
    train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    for partial_proverb, options in test_proverbs.items():
        solution, perplexity_result = cloze_test(partial_proverb, options, n=order)
        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Perplexité = {}".format(solution, perplexity_result))
