import os
import streamlit as st
import torch
from transformers import BartForSequenceClassification, BartTokenizer

# Function to download model and tokenizer from Google Drive
def download_from_drive(file_id, output):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# Function to load model and tokenizer
def load_model_and_tokenizer(model_dir):
    # Ensure the model directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Download files from Google Drive
    model_file_id = "1sg5Bh9mNzFTPLQ6lJ7sdLysGfvsc_QCk"
    config_file_id = "1fG6ipwD1beP0T3Sg3a8mWZMS2RPhuV5I"
    vocab_file_id = "1pM8C97I8SH3UVfPsUTBJsA_9xv8FeCJi"
    
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    config_path = os.path.join(model_dir, "config.json")
    vocab_path = os.path.join(model_dir, "vocab.json")
    
    download_from_drive(model_file_id, model_path)
    download_from_drive(config_file_id, config_path)
    download_from_drive(vocab_file_id, vocab_path)
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    
    # Load model
    model = BartForSequenceClassification.from_pretrained(model_dir)
    
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
    
    model_dir = "model"  # Directory to store the downloaded model
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    text_input = st.text_input("Enter text to classify:", "")
    if st.button("Predict"):
        if text_input:
            prediction = predict(model, tokenizer, text_input)
            st.write(f"Predicted class index: {prediction}")

if __name__ == "__main__":
    main()
