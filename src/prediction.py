"""
Prediction Module for ChatGPT Reviews Sentiment Analysis
-------------------------------------------------------
This module handles predictions using trained models and also provides
evaluation metrics and visualizations.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
import wordcloud
from wordcloud import WordCloud
from collections import Counter

from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor

class SentimentPredictor:
    """Class for making sentiment predictions and evaluating models."""
    
    def __init__(self, model_path='../models/best_model.joblib', 
                 vectorizer_path='../models/tfidf_vectorizer.joblib',
                 preprocessor=None):
        """
        Initialize the SentimentPredictor.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file.
        vectorizer_path : str
            Path to the trained vectorizer file.
        preprocessor : DataPreprocessor, optional
            Preprocessor instance for text cleaning.
        """
        # Load model
        self.model = self._load_model(model_path)
        
        # Load vectorizer
        self.vectorizer = self._load_vectorizer(vectorizer_path)
        
        # Initialize or use provided preprocessor
        if preprocessor is None:
            self.preprocessor = DataPreprocessor(data_path=None)
        else:
            self.preprocessor = preprocessor
        
        # Dictionary to map numerical labels to sentiment categories
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Store prediction results
        self.predictions = None
        self.true_labels = None
    
    def _load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the model file.
            
        Returns:
        --------
        object
            Loaded model.
        """
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _load_vectorizer(self, vectorizer_path):
        """
        Load a trained vectorizer from disk.
        
        Parameters:
        -----------
        vectorizer_path : str
            Path to the vectorizer file.
            
        Returns:
        --------
        object
            Loaded vectorizer.
        """
        try:
            vectorizer = joblib.load(vectorizer_path)
            print(f"Vectorizer loaded from {vectorizer_path}")
            return vectorizer
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            return None
    
    def predict_sentiment(self, text):
        """
        Predict the sentiment of a given text.
        
        Parameters:
        -----------
        text : str
            Text to predict sentiment for.
            
        Returns:
        --------
        tuple
            (predicted_label, predicted_probabilities)
        """
        if self.model is None or self.vectorizer is None:
            print("Model or vectorizer not loaded properly.")
            return None, None
        
        # Clean the text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Vectorize the text
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            # For models that provide probability estimates
            proba = self.model.predict_proba(X)[0]
            label = self.model.predict(X)[0]
        else:
            # For models that don't provide probability estimates (e.g., SVM)
            label = self.model.predict(X)[0]
            # Create a dummy probability distribution
            proba = np.zeros(3)
            proba[label] = 1.0
        
        # Convert numerical label to category
        sentiment = self.label_map.get(label, 'unknown')
        
        return sentiment, proba
    
    def predict_batch(self, texts):
        """
        Predict sentiments for a batch of texts.
        
        Parameters:
        -----------
        texts : list
            List of texts to predict sentiment for.
            
        Returns:
        --------
        list
            List of predicted sentiments.
        """
        sentiments = []
        probabilities = []
        
        for text in texts:
            sentiment, proba = self.predict_sentiment(text)
            sentiments.append(sentiment)
            probabilities.append(proba)
        
        return sentiments, probabilities
    
    def evaluate_predictions(self, texts, true_labels):
        """
        Evaluate predictions on a set of texts with known labels.
        
        Parameters:
        -----------
        texts : list
            List of texts to predict sentiment for.
        true_labels : list
            List of true labels.
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics.
        """
        # Predict sentiments
        predicted_sentiments, _ = self.predict_batch(texts)
        
        # Convert string labels to numerical if needed
        if isinstance(true_labels[0], str):
            # Create reverse mapping
            reverse_map = {v: k for k, v in self.label_map.items()}
            true_labels_num = [reverse_map.get(label.lower(), 0) for label in true_labels]
        else:
            true_labels_num = true_labels
        
        # Convert predicted sentiments to numerical
        predicted_labels_num = [reverse_map.get(sentiment, 0) for sentiment in predicted_sentiments]
        
        # Store for later use
        self.predictions = predicted_labels_num
        self.true_labels = true_labels_num
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels_num, predicted_labels_num)
        precision = precision_score(true_labels_num, predicted_labels_num, average='weighted')
        recall = recall_score(true_labels_num, predicted_labels_num, average='weighted')
        f1 = f1_score(true_labels_num, predicted_labels_num, average='weighted')
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels_num, predicted_labels_num)
        
        # Detailed classification report
        report = classification_report(true_labels_num, predicted_labels_num, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # Print results
        print("\nEvaluation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels_num, predicted_labels_num))
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix from previously evaluated predictions.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot.
        """
        if self.predictions is None or self.true_labels is None:
            print("No predictions available. Run evaluate_predictions first.")
            return
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        
        plt.title("Confusion Matrix")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def generate_wordcloud(self, texts, sentiment_filter=None, save_path=None):
        """
        Generate a word cloud from texts, optionally filtered by sentiment.
        
        Parameters:
        -----------
        texts : list
            List of texts to include in the word cloud.
        sentiment_filter : str, optional
            If provided, only include texts with this sentiment.
        save_path : str, optional
            Path to save the word cloud image.
        """
        # If sentiment filter is provided, predict sentiments and filter texts
        if sentiment_filter is not None:
            # Predict sentiments
            sentiments, _ = self.predict_batch(texts)
            
            # Filter texts by sentiment
            filtered_texts = [text for text, sent in zip(texts, sentiments) if sent == sentiment_filter]
            
            if not filtered_texts:
                print(f"No texts found with sentiment '{sentiment_filter}'.")
                return
            
            # Join all filtered texts
            all_text = ' '.join(filtered_texts)
        else:
            # Join all texts
            all_text = ' '.join(texts)
        
        # Generate word cloud
        wc = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(all_text)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        
        title = f"Word Cloud - {sentiment_filter.capitalize()} Reviews" if sentiment_filter else "Word Cloud - All Reviews"
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Word cloud saved to {save_path}")
        
        plt.show()
    
    def get_top_words(self, texts, sentiment_filter=None, n=20):
        """
        Get the top n most frequent words from texts, optionally filtered by sentiment.
        
        Parameters:
        -----------
        texts : list
            List of texts to analyze.
        sentiment_filter : str, optional
            If provided, only include texts with this sentiment.
        n : int
            Number of top words to return.
            
        Returns:
        --------
        list
            List of (word, count) tuples.
        """
        # If sentiment filter is provided, predict sentiments and filter texts
        if sentiment_filter is not None:
            # Predict sentiments
            sentiments, _ = self.predict_batch(texts)
            
            # Filter texts by sentiment
            filtered_texts = [text for text, sent in zip(texts, sentiments) if sent == sentiment_filter]
            
            if not filtered_texts:
                print(f"No texts found with sentiment '{sentiment_filter}'.")
                return []
            
            # Process filtered texts
            texts_to_process = filtered_texts
        else:
            # Process all texts
            texts_to_process = texts
        
        # Clean texts and tokenize
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts_to_process]
        all_words = []
        
        for text in cleaned_texts:
            all_words.extend(text.split())
        
        # Count words
        word_counts = Counter(all_words)
        
        # Get top words
        top_words = word_counts.most_common(n)
        
        return top_words
    
    def plot_top_words(self, texts, sentiment_filter=None, n=20, save_path=None):
        """
        Plot the top n most frequent words from texts, optionally filtered by sentiment.
        
        Parameters:
        -----------
        texts : list
            List of texts to analyze.
        sentiment_filter : str, optional
            If provided, only include texts with this sentiment.
        n : int
            Number of top words to plot.
        save_path : str, optional
            Path to save the plot.
        """
        top_words = self.get_top_words(texts, sentiment_filter, n)
        
        if not top_words:
            return
        
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(counts), y=list(words))
        
        title = f"Top {n} Words - {sentiment_filter.capitalize()} Reviews" if sentiment_filter else f"Top {n} Words - All Reviews"
        plt.title(title)
        plt.xlabel('Count')
        plt.ylabel('Word')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Top words plot saved to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load preprocessed data
    processed_data_path = '../data/processed_chatgpt_reviews.csv'
    try:
        processed_df = pd.read_csv(processed_data_path)
        
        # Initialize predictor
        predictor = SentimentPredictor()
        
        # Example text for prediction
        example_text = "This app is amazing and very helpful. I love it!"
        sentiment, probabilities = predictor.predict_sentiment(example_text)
        
        print(f"\nPredicted sentiment for '{example_text}': {sentiment}")
        print(f"Prediction probabilities: {probabilities}")
        
        # Evaluate on a sample of the dataset
        sample_size = 100
        sample_df = processed_df.sample(sample_size, random_state=42)
        
        metrics = predictor.evaluate_predictions(
            sample_df['review'].tolist(),
            sample_df['sentiment_label'].tolist()
        )
        
        # Plot confusion matrix
        predictor.plot_confusion_matrix(save_path='../results/prediction_confusion_matrix.png')
        
        # Generate word clouds
        predictor.generate_wordcloud(
            sample_df['review'].tolist(),
            save_path='../results/all_reviews_wordcloud.png'
        )
        
        for sentiment in ['positive', 'neutral', 'negative']:
            predictor.generate_wordcloud(
                sample_df['review'].tolist(),
                sentiment_filter=sentiment,
                save_path=f'../results/{sentiment}_reviews_wordcloud.png'
            )
        
        # Plot top words
        predictor.plot_top_words(
            sample_df['review'].tolist(),
            n=15,
            save_path='../results/top_words_all.png'
        )
        
        for sentiment in ['positive', 'neutral', 'negative']:
            predictor.plot_top_words(
                sample_df['review'].tolist(),
                sentiment_filter=sentiment,
                n=15,
                save_path=f'../results/top_words_{sentiment}.png'
            )
            
    except FileNotFoundError:
        print(f"File not found: {processed_data_path}")
        print("Please run data_preprocessing.py first to generate the processed data file.")
