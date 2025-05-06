"""
Streamlit Web Application for ChatGPT Reviews Sentiment Analysis
---------------------------------------------------------------
This module provides a web interface for the sentiment analysis project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import base64
from io import BytesIO
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We don't need these imports anymore since we're using our SimplePredictor
# from src.data_preprocessing import DataPreprocessor
# from src.prediction import SentimentPredictor

# Set page configuration
st.set_page_config(
    page_title="ChatGPT Reviews Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to load model and vectorizer
@st.cache_resource
def load_predictor():
    try:
        # Use pickle instead of joblib for compatibility
        model_path = 'models/logistic_regression_model.pkl'
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Create a simplified predictor that uses our model and vectorizer
        class SimplePredictor:
            def __init__(self, model, vectorizer):
                self.model = model
                self.vectorizer = vectorizer
                self.preprocessor = None
                self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                
            def predict_sentiment(self, text):
                # Clean text (simplified version)
                if not isinstance(text, str):
                    return "unknown", [0, 0, 0]
                
                # Download NLTK resources if needed
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
                
                # Clean text
                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                
                # Convert to lowercase
                cleaned_text = text.lower()
                
                # Remove special characters and numbers
                cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
                
                # Tokenize
                tokens = word_tokenize(cleaned_text)
                
                # Remove stopwords and lemmatize
                cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
                
                # Join tokens back into a string
                cleaned_text = ' '.join(cleaned_tokens)
                
                # Vectorize
                X = self.vectorizer.transform([cleaned_text])
                
                # Predict
                if hasattr(self.model, 'predict_proba'):
                    # For models that provide probability estimates
                    proba = self.model.predict_proba(X)[0]
                    label = self.model.predict(X)[0]
                else:
                    # For models that don't provide probability estimates
                    label = self.model.predict(X)[0]
                    # Create a dummy probability distribution
                    proba = np.zeros(3)
                    proba[label] = 1.0
                
                # Convert numerical label to category
                sentiment = self.label_map.get(label, 'unknown')
                
                return sentiment, proba
                
            def predict_batch(self, texts):
                sentiments = []
                probabilities = []
                
                for text in texts:
                    sentiment, proba = self.predict_sentiment(text)
                    sentiments.append(sentiment)
                    probabilities.append(proba)
                
                return sentiments, probabilities
        
        return SimplePredictor(model, vectorizer)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to create a download link for plots
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">{text}</a>'
    return href

# Function to save a DataFrame as CSV for download
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Main function
def main():
    # Title and intro
    st.title("ChatGPT Reviews Sentiment Analysis")
    st.markdown("""
    This application demonstrates sentiment analysis on ChatGPT reviews. 
    You can analyze individual texts or upload a CSV file with reviews for batch processing.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    pages = [
        "Home", 
        "Single Text Analysis", 
        "Batch Analysis", 
        "Data Exploration", 
        "Model Performance"
    ]
    page = st.sidebar.radio("Go to", pages)
    
    # Load predictor (model and vectorizer)
    predictor = load_predictor()
    
    if page == "Home":
        display_home_page()
    
    elif page == "Single Text Analysis":
        if predictor:
            display_single_text_analysis(predictor)
        else:
            st.error("Error: Model not loaded. Please ensure models are available in the 'models' directory.")
    
    elif page == "Batch Analysis":
        if predictor:
            display_batch_analysis(predictor)
        else:
            st.error("Error: Model not loaded. Please ensure models are available in the 'models' directory.")
    
    elif page == "Data Exploration":
        display_data_exploration()
    
    elif page == "Model Performance":
        display_model_performance()

def display_home_page():
    st.markdown("""
    ## About This Project
    
    This project provides sentiment analysis for ChatGPT reviews, classifying them as positive, neutral, or negative.
    
    ### Project Features:
    
    - **Single Text Analysis**: Analyze the sentiment of individual texts
    - **Batch Analysis**: Upload a CSV file to analyze multiple reviews at once
    - **Data Exploration**: Explore the dataset distribution and characteristics
    - **Model Performance**: View the trained model's performance metrics
    
    ### Dataset Information:
    
    The model was trained on a dataset of ChatGPT reviews with the following characteristics:
    - Reviews from various platforms (Web, Mobile)
    - Multiple languages
    - Ratings from 1 to 5 stars
    
    ### Technologies Used:
    
    - Python for data processing and model training
    - NLTK and Scikit-learn for NLP tasks
    - Various ML models (Logistic Regression, SVM, Random Forest, etc.)
    - Streamlit for the web interface
    
    ### How to Use:
    
    Use the sidebar to navigate between different sections of the application.
    """)
    
    # Display images if available
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image("results/data_distribution.png", caption="Data Distribution", use_column_width=True)
        with col2:
            st.image("results/model_comparison.png", caption="Model Comparison", use_column_width=True)
    except Exception as e:
        st.info("Distribution and comparison visualizations not available yet. Run the data preprocessing and model training scripts to generate them.")

def display_single_text_analysis(predictor):
    st.header("Single Text Analysis")
    st.markdown("Enter a text to analyze its sentiment.")
    
    # Text input
    text = st.text_area("Enter text for sentiment analysis:", height=150)
    
    if st.button("Analyze Sentiment"):
        if text and len(text.strip()) > 0:
            with st.spinner("Analyzing..."):
                # Predict sentiment
                sentiment, probabilities = predictor.predict_sentiment(text)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display sentiment
                    sentiment_color = {
                        "positive": "green",
                        "neutral": "blue",
                        "negative": "red"
                    }.get(sentiment, "black")
                    
                    st.markdown(f"### Sentiment: <span style='color:{sentiment_color}'>{sentiment.upper()}</span>", unsafe_allow_html=True)
                    
                    # Display confidence
                    if hasattr(probabilities, "__len__"):
                        confidence = max(probabilities) * 100
                        st.markdown(f"**Confidence**: {confidence:.2f}%")
                
                with col2:
                    # Display probability distribution if available
                    if hasattr(probabilities, "__len__") and len(probabilities) == 3:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sentiments = ["Negative", "Neutral", "Positive"]
                        colors = ["red", "blue", "green"]
                        
                        ax.bar(sentiments, probabilities, color=colors)
                        ax.set_ylabel("Probability")
                        ax.set_title("Sentiment Probabilities")
                        
                        st.pyplot(fig)
                        
                        # Display actual probabilities
                        for sentiment_label, prob in zip(sentiments, probabilities):
                            st.text(f"{sentiment_label}: {prob:.4f}")
        else:
            st.warning("Please enter some text to analyze.")

def display_batch_analysis(predictor):
    st.header("Batch Analysis")
    st.markdown("Upload a CSV file with reviews to analyze in batch.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Select text column
            text_columns = list(df.columns)
            text_column = st.selectbox("Select the column containing the review text:", text_columns)
            
            if st.button("Analyze Sentiments"):
                if text_column:
                    with st.spinner("Analyzing sentiments... This may take a while depending on the size of your dataset."):
                        # Get texts
                        texts = df[text_column].fillna("").tolist()
                        
                        # Predict sentiments
                        sentiments, probabilities = predictor.predict_batch(texts)
                        
                        # Add results to dataframe
                        df["predicted_sentiment"] = sentiments
                        
                        # Calculate confidence if probabilities are available
                        if probabilities and len(probabilities) > 0 and hasattr(probabilities[0], "__len__"):
                            df["confidence"] = [max(prob) * 100 for prob in probabilities]
                        
                        # Display results
                        st.subheader("Analysis Results:")
                        st.dataframe(df)
                        
                        # Display sentiment distribution
                        st.subheader("Sentiment Distribution:")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sentiment_counts = df["predicted_sentiment"].value_counts().sort_index()
                        colors = ["red", "blue", "green"]
                        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
                        ax.set_title("Sentiment Distribution")
                        ax.set_xlabel("Sentiment")
                        ax.set_ylabel("Count")
                        
                        st.pyplot(fig)
                        
                        # Provide download links
                        st.markdown(get_csv_download_link(df, "sentiment_analysis_results", "Download results as CSV"), unsafe_allow_html=True)
                        st.markdown(get_image_download_link(fig, "sentiment_distribution", "Download sentiment distribution chart"), unsafe_allow_html=True)
                else:
                    st.warning("Please select a text column.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

def display_data_exploration():
    st.header("Data Exploration")
    
    # Try to load processed data
    processed_data_path = "data/processed_chatgpt_reviews.csv"
    
    if os.path.exists(processed_data_path):
        df = load_data(processed_data_path)
        
        if df is not None:
            st.write("Preview of processed data:")
            st.dataframe(df.head())
            
            # Basic statistics
            st.subheader("Basic Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Dataset Shape:", df.shape)
                st.write("Rating Distribution:")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x="rating", data=df, ax=ax)
                ax.set_title("Rating Distribution")
                st.pyplot(fig)
            
            with col2:
                st.write("Sentiment Distribution:")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x="sentiment", data=df, ax=ax)
                ax.set_title("Sentiment Distribution")
                st.pyplot(fig)
            
            # Platform and language distribution
            st.subheader("Platform and Language Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                platform_counts = df["platform"].value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                platform_counts.plot(kind="bar", ax=ax)
                ax.set_title("Platform Distribution")
                st.pyplot(fig)
            
            with col2:
                language_counts = df["language"].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(8, 5))
                language_counts.plot(kind="bar", ax=ax)
                ax.set_title("Top 10 Languages")
                st.pyplot(fig)
            
            # Word clouds if available
            st.subheader("Word Clouds")
            
            try:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image("results/positive_reviews_wordcloud.png", caption="Positive Reviews", use_column_width=True)
                
                with col2:
                    st.image("results/neutral_reviews_wordcloud.png", caption="Neutral Reviews", use_column_width=True)
                
                with col3:
                    st.image("results/negative_reviews_wordcloud.png", caption="Negative Reviews", use_column_width=True)
            except Exception as e:
                st.info("Word clouds not available yet. Run the prediction script to generate them.")
    else:
        st.warning(f"Processed data not found at {processed_data_path}. Please run the data preprocessing script first.")

def display_model_performance():
    st.header("Model Performance")
    
    # Check if model comparison is available
    if os.path.exists("results/model_comparison.png"):
        st.subheader("Model Comparison")
        st.image("results/model_comparison.png", caption="Model Comparison", use_column_width=True)
    else:
        st.info("Model comparison visualization not available yet. Run the model training script to generate it.")
    
    # Check if confusion matrix is available
    if os.path.exists("results/prediction_confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("results/prediction_confusion_matrix.png", caption="Confusion Matrix", use_column_width=True)
    else:
        st.info("Confusion matrix visualization not available yet. Run the prediction script to generate it.")
    
    # Top words by sentiment if available
    st.subheader("Top Words by Sentiment")
    
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image("results/top_words_positive.png", caption="Positive Reviews", use_column_width=True)
        
        with col2:
            st.image("results/top_words_neutral.png", caption="Neutral Reviews", use_column_width=True)
        
        with col3:
            st.image("results/top_words_negative.png", caption="Negative Reviews", use_column_width=True)
    except Exception as e:
        st.info("Top words visualizations not available yet. Run the prediction script to generate them.")

if __name__ == "__main__":
    main()
