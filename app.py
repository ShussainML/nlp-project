import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import gdown
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
import os

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

# Define the AuthorClassifier model class
class AuthorClassifier(nn.Module):
    def __init__(self):
        super(AuthorClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, len(dictOfAuthors))  # Output size based on number of authors

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]  # [CLS] token
        output = self.fc(cls_output)
        return output

# Function to download model from Google Drive
@st.cache_resource
def download_model():
    url = 'https://drive.google.com/uc?id=1xPBuaagEXFIMRyH3iaJ8Pfvho3sgBUP-'  # Google Drive file ID for exact download
    output_path = '/mount/src/nlp-project/author_classifier_model.pth'  # Save to specified directory

    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    model = AuthorClassifier()
    model.load_state_dict(torch.load(output_path, map_location=torch.device('cpu')))  # Adjust map_location as needed
    model.eval()
    return model

# Model and Tokenizer loading
model = download_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to load test data from Google Drive
@st.cache_data
def load_test_data():
    url = 'https://drive.google.com/uc?id=1u2IoTNAbUVQdOvxo7URrxixoM4g8lOMA'  # Make sure drive download URL is correct
    output_path = '/mount/src/nlp-project/mega_test.csv'  # Save to specified directory

    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    test_data = pd.read_csv(output_path)
    return test_data

# Load test data
test_data = load_test_data()

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [lemmatizer.lemmatize(token.text.lower()) for token in doc if token.is_alpha]
    return ' '.join(tokens)

# Apply pre-processing on the text in the test data
test_data['text'] = test_data['text'].apply(preprocess_text)

# Assuming the Author column is present in the test data
swap_dict = {value: key for key, value in dictOfAuthors.items()}
test_data['Author_num'] = test_data['Author'].map(swap_dict)

# Function to preprocess the text and obtain predictions
def preprocess_and_predict(text):
    inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_author = dictOfAuthors.get(predicted_class, "Unknown Author")
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