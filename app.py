
from src.intents import load_intents
from src.preprocess import preprocess_data
from src.model import build_model
from src.utils import bag_of_words
import pickle
import tensorflow as tf
import numpy as np
import random

import nltk
nltk.download('punkt_tab')

# Load data
intents = load_intents("C:\\Users\\nirul\\repos\\Simple_AI_ChatBot\\data\\intents.json")

'''
The code loads preprocessed data from a file or preprocesses and saves it if not found. 
It tries to load a pre-trained model, or if not available, it builds, trains, and saves a new model.'''

# Preprocess data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    training, output, words, labels = preprocess_data(intents)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Build and train model
try:
    model = tf.keras.models.load_model("chatbot_model.keras")
except:
    model = build_model(len(training[0]), len(output[0]))
    model.fit(training, output, epochs=1000, batch_size=8)
    model.save("chatbot_model.keras")


# Chat function
def chat():
    print("Start talking with the bot (type 'quit' to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(np.array([bag_of_words(inp, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in intents["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()


