
from nltk.metrics.distance import *


def demo():
    edit_distance_examples = [
        ("rain", "shine"), ("abcdef", "acbdef"), ("language", "lnaguaeg"),
        ("language", "lnaugage"), ("language", "lngauage")]
    for s1, s2 in edit_distance_examples:
        print("Edit distance between '%s' and '%s':" % (s1, s2), edit_distance(s1, s2))
    for s1, s2 in edit_distance_examples:
        print("Edit distance with transpositions between '%s' and '%s':" % (s1, s2), edit_distance(s1, s2, transpositions=True))

    s1 = set('abc') #set([1, 2, 3, 4])
    s2 = set('bacd') # set([3, 4, 5])
    print("s1:", s1)
    print("s2:", s2)
    print("Binary distance:", binary_distance(s1, s2))
    print("Jaccard distance:", jaccard_distance(s1, s2))
    print("MASI distance:", masi_distance(s1, s2))
    from nltk import ngrams
    sentence = 'this is a foo bar sentences and i want to ngramize it'
    n = 3
    sixgrams = ngrams(sentence.split(), n)
    for grams in sixgrams:
        print(grams)


if __name__ == '__main__':
    demo()