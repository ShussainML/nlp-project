import streamlit as st
import torch
from transformers import BartForSequenceClassification, BartTokenizer

# Function to load model and tokenizer
def load_model_and_tokenizer():
    # Replace with your actual path or URL where your model is stored
    model_path = "https://drive.google.com/drive/folders/1sWO4oKdh1jSsZGEIyZdTRED6p8NZS-Qr"
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)
    
    # Load model
    model = BartForSequenceClassification.from_pretrained(model_path)
    
    return model, tokenizer

# Function to predict
def predict(model, tokenizer, text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    return predicted_class

# Streamlit app
def main():
    st.title("BART Model Deployment")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    text_input = st.text_input("Enter text to classify:", "")
    if st.button("Predict"):
        if text_input:
            prediction = predict(model, tokenizer, text_input)
            st.write(f"Predicted class index: {prediction}")

if __name__ == "__main__":
    main()
