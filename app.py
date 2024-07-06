import streamlit as st
import torch
from transformers import BartForSequenceClassification, BartTokenizer

# Function to load model and tokenizer from Hugging Face
@st.cache(hash_funcs={BartTokenizer: lambda _: None, BartForSequenceClassification: lambda _: None})
def load_model_and_tokenizer():
    model_name = "facebook/bart-large-cnn"  # Example model name, replace with your correct model name
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Load model
    model = BartForSequenceClassification.from_pretrained('sajid227/nlp-project-author-identifcation')
    
    return tokenizer, model

# Function to predict
def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    return predicted_class

# Streamlit app
def main():
    st.title("BART Model Deployment")
    
    # Load model and tokenizer using st.cache_data
    tokenizer, model = load_model_and_tokenizer()
    
    text_input = st.text_input("Enter text to classify:", "")
    if st.button("Predict"):
        if text_input:
            prediction = predict(model, tokenizer, text_input)
            st.write(f"Predicted class index: {prediction}")

if __name__ == "__main__":
    main()
