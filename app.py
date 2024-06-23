import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

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

class AuthorClassifier(nn.Module):
    def __init__(self, output_size):
        super(AuthorClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]  # [CLS] token
        output = self.fc(cls_output)
        return output

# Assuming model file path is available and correct
model_path = 'author_classifier_model.pth'
model = AuthorClassifier(output_size=50)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_and_predict(text):
    inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_author = dictOfAuthors.get(predicted_class, "Unknown Author")
    return predicted_author

st.title("Author Classifier")
st.write("Enter some text and the model will predict the author.")

user_input = st.text_area("Text input")
if st.button("Predict"):
    prediction = preprocess_and_predict(user_input)
    st.write(f"The predicted author is: {prediction}")

# Dummy performance metrics for demonstration purposes
def load_dummy_performance_metrics():
    return 0.95, 0.92, [[100, 20], [30, 150]]  # Example confusion matrix data

if st.button("Show Performance Metrics"):
    accuracy, f1_score_val, cf_matrix = load_dummy_performance_metrics()

    st.subheader("Performance Metrics")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"F1 Score: {f1_score_val}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)