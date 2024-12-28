import nltk
from nltk import PorterStemmer
from nltk import word_tokenize
import numpy as np

'''The following function returns a NumPy array where 1 means
 the word is present in the sentence, and 0 means it's not.'''

def bag_of_words(s, words):
    stemmer = PorterStemmer()
    bag = [0 for _ in range(len(words))]

    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)