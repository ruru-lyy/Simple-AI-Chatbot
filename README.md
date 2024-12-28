# Simple AI Chatbot

## Overview
This project implements a simple AI chatbot using deep learning techniques. The bot is designed to respond to user inputs based on the dataset "intents.json" . It contains intents and patterns in context of a shop that the bot will use to answer customer queries. Intents are the classes which the questions/queries belong to. Patterns are the sentences.

## Technologies Used
- **Python**: Programming language used for the project.
- **TensorFlow**: For building and training the neural network model.
- **NLTK**: For text tokenization and stemming.
- **NumPy**: For array manipulation.
- **Keras**: A high-level API for building and training deep learning models.

## Features
- Tokenizes and stems user input.
- Trains a deep learning model using the provided intents dataset.
- Responds to user queries with relevant answers based on intent classification.
- Saves the trained model and data as pickle files for reuse.

## How to Run
1. Clone the repository.
2. Install the necessary dependencies:
   ```bash
   pip install tensorflow nltk numpy
   ```
3. Download any required NLTK datasets:
   ```python
   import nltk
   nltk.download('punkt')
   ```
4. Run the `app.py` file:
   ```bash
   python app.py
   ```
5. Start chatting with the bot! Type 'quit' to exit.

## License
MIT License. See [LICENSE](LICENSE) for more details.
