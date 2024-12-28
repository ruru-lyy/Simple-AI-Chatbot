import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def preprocess_data(data):
    stemmer = PorterStemmer()
    words  = []
    labels = []
    docs_x = []
    docs_y = []
#docs_x = "What the user says" (split into words).
#docs_y = "What category or intent that sentence belongs to."
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])
# This for loop organizes sentences into words, 
# matches them to their tags, and collects unique tags.
    words = [stemmer.stem(w.lower()) for w in words if w !="?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    return np.array(training), np.array(output), words, labels

''' The training input forms the bag of words where each input sentence
is converted to a vector over the bag of words where 1 would indicate 
the word exists and 0 means not. The output forms a vector of the tags.
Convert sentences into word-presence vectors (training).
Convert tags into one-hot vectors (output).'''




