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
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    logger.info(f"Attempting to load model from {model_path}")
    start_time = time.time()
    
    try:
        # Load tokenizer - using AutoTokenizer with trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load model - using AutoModelForSequenceClassification with trust_remote_code
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            trust_remote_code=True,
            # Set device map to auto for hardware acceleration if available
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        return tokenizer, model, model_path
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        st.info("Trying fallback method...")
        
        # Fallback 1: try with specific base model but custom weights
        try:
            logger.info("Attempting fallback 1: Using base BART tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
            return tokenizer, model, model_path
        except Exception as e2:
            logger.error(f"Fallback 1 failed: {str(e2)}")
            
            # Fallback 2: try with revision specified
            try:
                logger.info("Attempting fallback 2: Using revision specification")
                tokenizer = AutoTokenizer.from_pretrained(model_path, revision="main")
                model = AutoModelForSequenceClassification.from_pretrained(model_path, revision="main")
                return tokenizer, model, model_path
            except Exception as e3:
                logger.error(f"Fallback 2 failed: {str(e3)}")
                st.error(f"All fallback methods failed. Last error: {str(e3)}")
                raise

# Function to predict author name
def predict_author_name(model, tokenizer, text, debug_mode=False):
    try:
        # Start time for performance measurement
        start_time = time.time()
        
        # Preprocess text
        preprocessed_text = preprocess_text(text)
        if debug_mode:
            st.write("**Preprocessed text:**")
            st.text(preprocessed_text[:500] + ("..." if len(preprocessed_text) > 500 else ""))
            st.write(f"Preprocessed text length: {len(preprocessed_text.split())} words")
        
        # Tokenize text
        tokenize_start = time.time()
        inputs = tokenizer(preprocessed_text, return_tensors="pt", max_length=512, padding=True, truncation=True)
        tokenize_time = time.time() - tokenize_start
        
        if debug_mode:
            st.write("**Tokenizer inputs:**")
            st.write({k: v.shape for k, v in inputs.items()})
            st.write(f"Tokenization time: {tokenize_time:.4f} seconds")
        
        # Move to appropriate device if available
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            model.to('cuda')
        
        # Run inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - inference_start
        
        # Get logits and predict class
        logits = outputs.logits
        
        # Calculate probabilities with softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get top 5 predictions
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Convert to list
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Get predicted class (top 1)
        predicted_class = top_indices[0]
        predicted_author = dictOfAuthors.get(predicted_class, "Unknown")
        confidence = top_probs[0] * 100
        
        # Prepare top 5 results
        top5_results = [(dictOfAuthors.get(idx, "Unknown"), prob * 100) for idx, prob in zip(top_indices, top_probs)]
        
        # Total processing time
        total_time = time.time() - start_time
        
        if debug_mode:
            st.write("**Raw logits:**")
            st.write(logits.cpu().numpy())
            st.write(f"Inference time: {inference_time:.4f} seconds")
            st.write(f"Total processing time: {total_time:.4f} seconds")
        
        return predicted_author, confidence, top5_results, total_time
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        st.error(f"Error in prediction: {str(e)}")
        return "Error", 0.0, [], 0.0

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

# Function to create confidence bar chart for top 5 authors
def plot_confidence_chart(top5_results):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    authors = [author for author, _ in top5_results]
    confidences = [conf for _, conf in top5_results]
    
    # Plot horizontal bar chart
    bars = ax.barh(authors, confidences, color='skyblue')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', fontsize=10)
    
    # Add labels and title
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top 5 Author Predictions')
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

# Function to handle file upload
def process_uploaded_file(model, tokenizer, file_content, debug_mode):
    try:
        # Split the file content into lines
        lines = file_content.strip().split('\n')
        
        # Process each line
        results = []
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                predicted_author, confidence, _, _ = predict_author_name(
                    model, tokenizer, line, debug_mode
                )
                results.append({
                    'Text Snippet': line[:100] + ('...' if len(line) > 100 else ''),
                    'Predicted Author': predicted_author,
                    'Confidence': f"{confidence:.2f}%"
                })
        
        # Return results as DataFrame
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Streamlit app
def main():
    st.set_page_config(
        page_title="Author Identification", 
        page_icon="✍️",
        layout="wide"
    )
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Debug mode toggle in sidebar
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    if debug_mode:
        st.sidebar.info("Debug mode enabled. Extra information will be shown during processing.")
    
    # Hardware info in sidebar
    st.sidebar.subheader("Hardware Info")
    if torch.cuda.is_available():
        st.sidebar.success(f"GPU Available: {torch.cuda.get_device_name(0)}")
        st.sidebar.write(f"CUDA Version: {torch.version.cuda}")
    else:
        st.sidebar.warning("No GPU detected. Running on CPU only.")
    
    # Main content
    st.title("Author Identification")
    st.write("This app identifies the author of a given text using a fine-tuned BART model.")
    
    # Load model and tokenizer
    with st.spinner("Loading model and tokenizer... This may take a minute."):
        try:
            tokenizer, model, model_path = load_model_and_tokenizer()
            st.success("Model and tokenizer loaded successfully!")
            
            if debug_mode:
                st.write(f"Model Path: {model_path}")
                st.write(f"Model Type: {model.__class__.__name__}")
                st.write(f"Tokenizer Type: {tokenizer.__class__.__name__}")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()
    
    # Tabs for different analysis methods
    tab1, tab2, tab3 = st.tabs(["Text Input", "File Upload", "Example Testing"])
    
    # Tab 1: Text Input
    with tab1:
        text_input = st.text_area("Enter text to classify:", height=200)
        
        if st.button("Predict", key="predict_text"):
            if text_input:
                with st.spinner("Predicting author..."):
                    predicted_author, confidence, top5_results, processing_time = predict_author_name(
                        model, tokenizer, text_input, debug_mode
                    )
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("### Results")
                        st.write(f"**Predicted author:** {predicted_author}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        st.write(f"Processing time: {processing_time:.4f} seconds")
                    
                    with col2:
                        # Visualize confidence for top 5 authors
                        st.write("### Top 5 Predictions")
                        fig = plot_confidence_chart(top5_results)
                        st.pyplot(fig)
            else:
                st.warning("Please enter some text to classify.")
    
    # Tab 2: File Upload
    with tab2:
        st.write("""
        Upload a text file with multiple samples to classify. Each line will be treated as a separate text sample.
        """)
        
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        
        if uploaded_file is not None:
            # Read file content
            file_content = uploaded_file.getvalue().decode("utf-8")
            
            st.write(f"File uploaded: {uploaded_file.name}")
            st.write(f"File size: {len(file_content)} characters")
            
            if st.button("Process File", key="process_file"):
                with st.spinner("Processing file..."):
                    results_df = process_uploaded_file(model, tokenizer, file_content, debug_mode)
                    
                    if results_df is not None:
                        st.write("### Results")
                        st.dataframe(results_df)
                        
                        # Download option
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="author_identification_results.csv",
                            mime="text/csv",
                        )
    
    # Tab 3: Example Testing
    with tab3:
        st.write("You can evaluate the model on some example texts:")
        
        # Example test data with longer samples
        test_data = [
            ("Britain's Ladbroke Group Plc Monday concluded a long-awaited global alliance with Hilton Hotels Corp., reuniting the Hilton brand worldwide for the first time in 32 years. Despite the tie-up, which covers 400 hotels in 49 countries, the two companies denied there was a hidden agenda to progress toward a full merger of the two groups.", 44),
            ("One of the hottest topics at a recent Internet trade show was so-called 'push technology,' which directly broadcasts customised news to computer users hooked up to the Net -- but is also seen as the next area ripe for a shakeout. Internet broadcasters are led by companies like privately held PointCast Inc., which announced a major deal with software giant Microsoft Corp. early last month.", 46),
            ("Investors have always known that small companies are riskier than big ones. In fact, there is a widely used measure for the risk of the smallest publicly traded companies. It's called the small cap premium, and right now it is sending strange signals that have some investors nervous. Simply put, the small cap premium is the extra return needed to compensate investors for the risk of investing in volatile small stocks instead of bigger, sturdier ones.", 28),
        ]
        
        example_index = st.selectbox(
            "Choose an example:", 
            ["Example 1", "Example 2", "Example 3"]
        )
        if example_index == "Example 1":
            example_index = 0
        elif example_index == "Example 2":
            example_index = 1
        else:
            example_index = 2
        
        st.text_area("Example Text:", value=test_data[example_index][0], height=150, key="example_text")
        true_author = dictOfAuthors[test_data[example_index][1]]
        st.write(f"True author: **{true_author}**")
        
        if st.button("Test with this Example", key="test_example"):
            with st.spinner("Predicting author for example..."):
                predicted_author, confidence, top5_results, processing_time = predict_author_name(
                    model, tokenizer, test_data[example_index][0], debug_mode
                )
                
                # Display results in columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"Predicted author: **{predicted_author}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.write(f"Processing time: {processing_time:.4f} seconds")
                    
                    if predicted_author == true_author:
                        st.success("Correct prediction! ✅")
                    else:
                        st.error("Incorrect prediction ❌")
                
                with col2:
                    # Visualize confidence for top 5 authors
                    fig = plot_confidence_chart(top5_results)
                    st.pyplot(fig)
    
    with st.expander("About the Model"):
        st.write("""
        This model was trained to identify authors based on their writing style. 
        It uses a fine-tuned BART model available on Hugging Face: 
        [sajid227/nlp-project-author-identifcation](https://huggingface.co/sajid227/nlp-project-author-identifcation)
        
        The model can identify 50 different authors from the Reuters corpus.
        """)
        
        # Author list
        st.write("### Supported Authors")
        # Create a 5-column layout for author names
        cols = st.columns(5)
        for i, (idx, author) in enumerate(sorted(dictOfAuthors.items())):
            cols[i % 5].write(f"{idx}: {author}")
        
        # Model implementation details
        if debug_mode:
            st.write("### Model Implementation Details")
            st.code("""
            # Model was fine-tuned using the following approach:
            # - Base model: facebook/bart-large-cnn
            # - Training dataset: Reuters C50 corpus
            # - Fine-tuning approach: Sequence classification with author labels
            # - Training hyperparameters: 
            #   - Learning rate: 2e-5
            #   - Batch size: 8
            #   - Epochs: 5
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center">
            <p>Created with ❤️ using Streamlit and Hugging Face</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
