import json

def load_intents(file_path):
    with open(file_path) as file:
        return json.load(file)



