import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import os

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
        # Load tokenizer - using AutoTokenizer instead of BartTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model - using AutoModelForSequenceClassification instead of BartForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Trying fallback method...")
        
        # Fallback: try loading with specific base model but custom weights
        try:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return tokenizer, model
        except Exception as e2:
            st.error(f"Fallback also failed: {str(e2)}")
            raise

# Function to predict author name
def predict_author_name(model, tokenizer, text):
    try:
        preprocessed_text = preprocess_text(text)
        inputs = tokenizer(preprocessed_text, return_tensors="pt", max_length=512, padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        predicted_author = dictOfAuthors.get(predicted_class, "Unknown")  # Get author name from dictionary
        
        # Get confidence score
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class].item() * 100
        
        return predicted_author, confidence
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error", 0.0

# Function to evaluate model performance using actual data
def evaluate_model_performance(model, tokenizer, test_data):
    y_true = []
    y_pred = []
    
    try:
        for text, true_label in test_data:
            preprocessed_text = preprocess_text(text)
            inputs = tokenizer(preprocessed_text, return_tensors="pt", max_length=512, padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1).item()
            y_true.append(true_label)
            y_pred.append(predicted_class)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, target_names=[dictOfAuthors[i] for i in sorted(set(y_true))], output_dict=True)
        confusion = confusion_matrix(y_true, y_pred)
        
        return report, confusion
    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
        return None, None

# Streamlit app
def main():
    st.set_page_config(page_title="Author Identification", page_icon="✍️")
    
    st.title("Author Identification")
    st.write("This app identifies the author of a given text using a fine-tuned BART model.")
    
    with st.spinner("Loading model and tokenizer... This may take a minute."):
        try:
            tokenizer, model = load_model_and_tokenizer()
            st.success("Model and tokenizer loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()
    
    text_input = st.text_area("Enter text to classify:", height=200)
    
    if st.button("Predict"):
        if text_input:
            with st.spinner("Predicting author..."):
                predicted_author, confidence = predict_author_name(model, tokenizer, text_input)
                
                st.write("### Results")
                st.write(f"**Predicted author:** {predicted_author}")
                st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            st.warning("Please enter some text to classify.")
    
    with st.expander("About the Model"):
        st.write("""
        This model was trained to identify authors based on their writing style. 
        It uses a fine-tuned BART model available on Hugging Face: 
        [sajid227/nlp-project-author-identifcation](https://huggingface.co/sajid227/nlp-project-author-identifcation)
        
        The model can identify 50 different authors from the Reuters corpus.
        """)
    
    # Example test data section
    with st.expander("Test with Example Data"):
        st.write("You can evaluate the model on some example texts:")
        
        # Example test data
        test_data = [
            ("Britain's Ladbroke Group Plc Monday concluded a long-awaited global alliance with Hilton Hotels Corp., reuniting the Hilton brand worldwide for the first time in 32 years. Despite the tie-up, which covers 400 hotels in 49 countries, the two companies denied there was a hidden agenda to progress toward a full merger of the two groups.", 44),
            ("One of the hottest topics at a recent Internet trade show was so-called 'push technology,' which directly broadcasts customised news to computer users hooked up to the Net -- but is also seen as the next area ripe for a shakeout. Internet broadcasters are led by companies like privately held PointCast Inc., which announced a major deal with software giant Microsoft Corp. early last month.", 46),
        ]
        
        example_index = st.selectbox("Choose an example:", ["Example 1", "Example 2"])
        example_index = 0 if example_index == "Example 1" else 1
        
        st.text_area("Example Text:", value=test_data[example_index][0], height=150, key="example_text")
        true_author = dictOfAuthors[test_data[example_index][1]]
        st.write(f"True author: **{true_author}**")
        
        if st.button("Test with this Example"):
            with st.spinner("Predicting author for example..."):
                predicted_author, confidence = predict_author_name(model, tokenizer, test_data[example_index][0])
                st.write(f"Predicted author: **{predicted_author}**")
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                if predicted_author == true_author:
                    st.success("Correct prediction! ✅")
                else:
                    st.error("Incorrect prediction ❌")

if __name__ == "__main__":
    main()
