import streamlit as st
import torch
import os
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Set environment variable to avoid CUDA initialization errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure necessary NLTK downloads
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load SpaCy model with error handling
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.error("SpaCy model not found. Installing now...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

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

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert text to lowercase
    tokens = [
        lemmatizer.lemmatize(token.text) 
        for token in doc 
        if token.is_alpha and token.text not in stop_words
    ]
    return ' '.join(tokens)

# Function to load model and tokenizer from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "sajid227/nlp-project-author-identifcation"
    
    try:
        # Load tokenizer and model using Auto classes for better compatibility
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from HuggingFace: {str(e)}")
        st.info("Trying to load from local path...")
        
        try:
            # Try loading from local path if available
            local_model_path = "./my_bart_model"  # Adjust path as needed
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            return tokenizer, model
        except Exception as e2:
            st.error(f"Local model loading failed: {str(e2)}")
            
            # Final fallback - try with base BART tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                return tokenizer, model
            except Exception as e3:
                st.error(f"All fallback attempts failed: {str(e3)}")
                raise

# Function to predict author name
def predict_author_name(model, tokenizer, text):
    try:
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        
        # Tokenize and prepare for model input
        inputs = tokenizer(
            preprocessed_text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Run prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get prediction results
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        predicted_author = dictOfAuthors.get(predicted_class, "Unknown")
        
        # Calculate confidence
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class].item() * 100
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        top_authors = [(dictOfAuthors.get(idx.item(), "Unknown"), prob.item() * 100) 
                      for idx, prob in zip(top_indices, top_probs)]
        
        return predicted_author, confidence, top_authors
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error", 0.0, []

# Streamlit app
def main():
    st.set_page_config(page_title="Author Identification", page_icon="✍️")
    
    st.title("Author Identification")
    st.write("This app identifies the author of a given text using a fine-tuned BART model.")
    
    # Show loading spinner while loading the model
    with st.spinner("Loading model and tokenizer... This may take a minute."):
        try:
            tokenizer, model = load_model_and_tokenizer()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()
    
    # Text input area
    text_input = st.text_area("Enter text to classify:", height=200)
    
    if st.button("Predict"):
        if text_input:
            with st.spinner("Predicting author..."):
                predicted_author, confidence, top_authors = predict_author_name(model, tokenizer, text_input)
                
                # Display results
                st.write("### Results")
                st.write(f"**Predicted author:** {predicted_author}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
                # Show top predictions
                st.write("### Top 3 Predictions")
                for author, conf in top_authors:
                    st.write(f"- {author}: {conf:.2f}%")
        else:
            st.warning("Please enter some text to classify.")
    
    # About section
    with st.expander("About the Model"):
        st.write("""
        This model was trained to identify authors based on their writing style. 
        It uses a fine-tuned BART model available on Hugging Face: 
        [sajid227/nlp-project-author-identifcation](https://huggingface.co/sajid227/nlp-project-author-identifcation)
        
        The model can identify 50 different authors from the Reuters corpus.
        """)

if __name__ == "__main__":
    main()
