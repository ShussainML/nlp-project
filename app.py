import streamlit as st
import torch
from transformers import BartForSequenceClassification, BartTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Dictionary mapping class index to author names
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

# Function to load model and tokenizer from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "facebook/bart-large-cnn"  # Example model name, replace with your correct model name
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Load model
    model = BartForSequenceClassification.from_pretrained('sajid227/nlp-project-author-identifcation')
    
    return tokenizer, model

# Function to predict author name
def predict_author_name(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    predicted_author = dictOfAuthors.get(predicted_class, "Unknown")  # Get author name from dictionary
    return predicted_author

# Function to evaluate model performance (dummy implementation)
def evaluate_model_performance():
    # Dummy implementation: Replace with actual model evaluation metrics
    y_true = np.random.randint(0, 50, size=100)  # Example true labels
    y_pred = np.random.randint(0, 50, size=100)  # Example predicted labels
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=list(dictOfAuthors.values()), output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)
    
    return report, confusion

# Streamlit app
def main():
    st.title("Author Identification")
    tokenizer, model = load_model_and_tokenizer()
    st.write("Model and tokenizer loaded successfully!")
    
    text_input = st.text_input("Enter text to classify:", "")
    
    if st.button("Predict"):
        if text_input:
            st.write("Predicting author...")
            predicted_author = predict_author_name(model, tokenizer, text_input)
            st.write(f"Predicted author: {predicted_author}")
    
    if st.button("Evaluate Model Performance"):
        st.write("Evaluating model performance...")
        report, confusion_matrix = evaluate_model_performance()
        st.write("Model performance evaluation:")
        st.write("Classification Report:")
        st.write(report)
        st.write("Confusion Matrix:")
        st.write(confusion_matrix)

if __name__ == "__main__":
    main()
