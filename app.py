import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
import os
import json
import numpy as np

# Download NLTK components
nltk.download('wordnet')
nltk.download('punkt')

# Initialize SpaCy and NLTK components
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

# Example dictionary mapping author numerical IDs to names
dictOfAuthors = {
    0: 'AaronPressman', 1: 'AlanCrosby', 2: 'AlexanderSmith', 3: 'BenjaminKangLim', 4: 'BernardHickey',
    5: 'BradDorfman', 6: 'DarrenSchuettler', 7: 'DavidLawder', 8: 'EdnaFernandes', 9: 'EricAuchard',
    10: 'FumikoFujisaki', 11: 'GrahamEarnshaw', 12: 'HeatherScoffield', 13: 'JanLopatka', 14: 'JaneMacartney',
    15: 'JimGilchrist', 16: 'JoWinterbottom', 17: 'JoeOrtiz', 18: 'JohnMastrini', 19: 'JonathanBirt',
    20: 'KarlPenhaul', 21: 'KeithWeir', 22: 'KevinDrawbaugh', 23: 'KevinMorrison', 24: 'KirstinRidley',
    25: 'KouroshKarimkhany', 26: 'LydiaZajc', 27: 'LynneODonnell', 28: 'LynnleyBrowning', 29: 'MarcelMichelson',
    30: 'MarkBendeich', 31: 'MartinWolk', 32: 'MatthewBunce', 33: 'MichaelConnor', 34: 'MureDickie',
    35: 'NickLouth', 36: 'PatriciaCommins', 37: 'PeterHumphrey', 38: 'PierreTran', 39: 'RobinSidel',
    40: 'RogerFillion', 41: 'SamuelPerry', 42: 'SarahDavison', 43: 'ScottHillis', 44: 'SimonCowell',
    45: 'TanEeLyn', 46: 'TheresePoletti', 47: 'TimFarrand', 48: 'ToddNissen', 49: 'WilliamKazer'
}

# Load the LSTM model
def load_lstm_model():
    model_path = '/mount/src/nlp-project/author_lstm_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        raise FileNotFoundError(f"File not found: {model_path}")

# Load the tokenizer
def load_tokenizer():
    tokenizer_path = '/mount/src/nlp-project/tokenizer.json'
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return tokenizer
    else:
        raise FileNotFoundError(f"File not found: {tokenizer_path}")

# Function to load test data
def load_test_data():
    test_data_path = '/mount/src/nlp-project/mega_test.csv'
    if os.path.exists(test_data_path):
        test_data = pd.read_csv(test_data_path)
        return test_data
    else:
        raise FileNotFoundError(f"File not found: {test_data_path}")

# Load the LSTM model and tokenizer
model = load_lstm_model()
tokenizer = load_tokenizer()

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [lemmatizer.lemmatize(token.text.lower()) for token in doc if token.is_alpha]
    return ' '.join(tokens)

# Function to preprocess the text and obtain predictions
def preprocess_and_predict(text):
    preprocessed_text = preprocess_text(text)
    tokenized_input = tokenizer.texts_to_sequences([preprocessed_text])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(tokenized_input, maxlen=512, padding='post')
    
    prediction_probs = model.predict(padded_input)
    predicted_author_index = np.argmax(prediction_probs, axis=1)[0]
    predicted_author = dictOfAuthors.get(predicted_author_index, "Unknown Author")
    return predicted_author

# Function to evaluate the model
def evaluate_model(test_data):
    y_true = test_data['Author_num'].tolist()
    y_pred = []

    for text in test_data['text']:
        predicted_author = preprocess_and_predict(text)
        predicted_class = list(dictOfAuthors.keys())[list(dictOfAuthors.values()).index(predicted_author)]
        y_pred.append(predicted_class)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, f1, cf_matrix

# Streamlit UI
st.title("Author Classifier")
st.write("Enter some text and the model will predict the author.")

user_input = st.text_area("Text input")
if st.button("Predict"):
    prediction = preprocess_and_predict(user_input)
    st.write(f"The predicted author is: {prediction}")

# Button to show performance metrics
if st.button("Show Performance Metrics"):
    test_data = load_test_data()
    test_data['text'] = test_data['text'].apply(preprocess_text)
    swap_dict = {value: key for key, value in dictOfAuthors.items()}
    test_data['Author_num'] = test_data['Author'].map(swap_dict)
    
    accuracy, f1, cf_matrix = evaluate_model(test_data)

    st.subheader("Performance Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)