import streamlit as st
import torch
import os
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# Set environment variable to avoid CUDA initialization errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure necessary NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Load SpaCy model with error handling
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert text to lowercase
    tokens = [
        lemmatizer.lemmatize(token.text)
        for token in doc
        if token.is_alpha and token.text not in stop_words
    ]
    return ' '.join(tokens)

# Load model/tokenizer/config from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "sajid227/nlp-project-author-identifcation"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        id2label = config.id2label
        return tokenizer, model, id2label
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Predict author name
def predict_author_name(model, tokenizer, id2label, text):
    try:
        preprocessed_text = preprocess_text(text)
        inputs = tokenizer(
            preprocessed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class = logits.argmax().item()
        predicted_author = id2label.get(str(predicted_class), "Unknown")

        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class].item() * 100

        top_probs, top_indices = torch.topk(probabilities[0], 3)
        top_authors = [(id2label.get(str(idx.item()), "Unknown"), prob.item() * 100) for idx, prob in zip(top_indices, top_probs)]

        return predicted_author, confidence, top_authors

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error", 0.0, []

# Streamlit app
def main():
    st.set_page_config(page_title="Author Identification", page_icon="✍️")

    st.title("Author Identification")
    st.write("This app identifies the author of a given text using a fine-tuned BART model.")

    with st.spinner("Loading model and tokenizer..."):
        tokenizer, model, id2label = load_model_and_tokenizer()

    text_input = st.text_area("Enter text to classify:", height=200)

    if st.button("Predict"):
        if text_input.strip():
            with st.spinner("Predicting author..."):
                predicted_author, confidence, top_authors = predict_author_name(model, tokenizer, id2label, text_input)

                st.subheader("Results")
                st.metric("Predicted Author", predicted_author, f"{confidence:.1f}% confidence")

                st.subheader("Top 3 Predictions")
                cols = st.columns(3)
                for i, (author, conf) in enumerate(top_authors):
                    cols[i].metric(f"#{i+1}", author, f"{conf:.1f}%")
        else:
            st.warning("Please enter some text to classify.")

    with st.expander("About the Model"):
        st.write("""
        This model was trained to identify authors based on their writing style. 
        It uses a fine-tuned BART model available on Hugging Face: 
        [sajid227/nlp-project-author-identifcation](https://huggingface.co/sajid227/nlp-project-author-identifcation)
        """)

if __name__ == "__main__":
    main()
