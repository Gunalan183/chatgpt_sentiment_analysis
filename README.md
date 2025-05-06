# ChatGPT Reviews Sentiment Analysis

A comprehensive Natural Language Processing (NLP) project for sentiment analysis of ChatGPT reviews, classifying them as positive, neutral, or negative based on textual content.

## Project Overview

This project implements a complete sentiment analysis pipeline for user reviews of ChatGPT. It covers the entire machine learning lifecycle, from data preprocessing to model deployment with an interactive web interface. The system analyzes review text to determine sentiment polarity, providing valuable insights into user opinions and experiences.

### Key Features

- **Data Preprocessing**: Robust text cleaning, tokenization, lemmatization, and vectorization
- **Machine Learning Models**: Implementation of multiple classification algorithms
- **Interactive Web Interface**: User-friendly Streamlit application for real-time sentiment analysis
- **Batch Processing**: Support for analyzing multiple reviews simultaneously
- **Visualization Tools**: Data exploration and model performance visualizations
- **Modular Design**: Well-structured, maintainable code organization

## Dataset

The project uses a dataset of ChatGPT user reviews with the following characteristics:

- **Size**: 10,000+ reviews
- **Features**: Date, title, review text, rating, username, helpful votes, review length, platform, language, location, version
- **Rating Scale**: 1-5 stars
- **Platforms**: Web and Mobile
- **Languages**: Multiple languages (en, fr, de, es, hi, etc.)
- **Geographical Distribution**: Reviews from various countries (USA, UK, India, Canada, Australia, Germany, etc.)

## Technical Implementation

### Data Preprocessing

1. **Text Cleaning**:
   - Conversion to lowercase
   - Removal of special characters and numbers
   - Tokenization of text

2. **Natural Language Processing**:
   - Removal of stopwords (common words that don't add significant meaning)
   - Lemmatization (reducing words to their base form)
   - N-gram extraction (capturing multi-word phrases)

3. **Feature Engineering**:
   - TF-IDF Vectorization (Term Frequency-Inverse Document Frequency)
   - Dimensionality reduction with SVD (Singular Value Decomposition)
   - Sentiment mapping (converting numerical ratings to sentiment categories)

### Machine Learning Models

The project implements and compares several classification algorithms:

1. **Logistic Regression**: Primary model used for its interpretability and efficiency
2. **Naive Bayes**: Probabilistic approach that works well with text data
3. **Support Vector Machine (SVM)**: Effective for high-dimensional feature spaces
4. **Random Forest**: Ensemble method that reduces overfitting
5. **XGBoost**: Advanced gradient boosting for improved performance

Model selection is based on:
- F1 score (balance of precision and recall)
- Cross-validation results
- Computational efficiency

### Evaluation Metrics

The model performance is evaluated using:

- **Accuracy**: Overall correct predictions
- **Precision**: Proportion of true positives among positive predictions
- **Recall**: Proportion of true positives identified correctly
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of prediction types

### Web Application

The Streamlit web application provides:

1. **Single Text Analysis**:
   - Input field for entering review text
   - Sentiment prediction with confidence levels
   - Visualization of sentiment probabilities

2. **Batch Analysis**:
   - CSV file upload capability
   - Bulk sentiment prediction
   - Results export functionality
   - Sentiment distribution visualization

3. **Data Exploration**:
   - Rating and sentiment distributions
   - Platform and language analysis
   - Word clouds for different sentiment categories
   - Top words by sentiment

4. **Model Performance**:
   - Confusion matrix visualization
   - Detailed classification metrics
   - Model comparison charts

## Project Structure

```
chatgpt_sentiment_analysis/
├── data/                       # Data storage
│   ├── chatgpt_reviews.csv     # Original dataset
│   └── processed_data.csv      # Preprocessed dataset
│
├── models/                     # Trained models and vectorizers
│   ├── logistic_regression_model.pkl  # Main sentiment classifier
│   └── tfidf_vectorizer.pkl    # Text vectorizer
│
├── results/                    # Visualizations and analysis outputs
│   ├── confusion_matrix.png    # Model evaluation visualization
│   └── data_distribution.png   # Dataset characteristic plots
│
├── src/                        # Source code modules
│   ├── data_preprocessing.py   # Data cleaning and preparation
│   ├── feature_extraction.py   # Text vectorization
│   ├── model_training.py       # Model implementation and training
│   └── prediction.py           # Prediction functionality
│
├── app.py                      # Streamlit web application
├── run_analysis.py             # Simplified pipeline execution
├── run_pipeline.py             # Complete pipeline runner
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Implementation Process

The sentiment analysis pipeline follows these steps:

1. **Data Loading**: Import raw reviews from CSV file
2. **Exploratory Data Analysis**: Understand dataset characteristics
3. **Text Preprocessing**: Clean and normalize text data
4. **Feature Extraction**: Convert text to numerical features using TF-IDF
5. **Data Splitting**: Divide data into training, validation, and test sets
6. **Model Training**: Fit models on training data and tune hyperparameters
7. **Model Evaluation**: Assess performance on validation and test sets
8. **Model Selection**: Choose the best performing model
9. **Deployment**: Implement web application for user interaction
10. **Documentation**: Provide comprehensive project documentation

## Results and Insights

The sentiment analysis model achieved the following results:

- **Accuracy**: ~85% on test data
- **F1 Score**: ~0.83 weighted average across sentiment classes
- **Class Performance**:
  - Positive sentiment: Highest precision and recall
  - Neutral sentiment: Most challenging to classify correctly
  - Negative sentiment: Good precision but lower recall

Key insights from the analysis:

- **Language Patterns**: Different sentiment categories show distinct linguistic patterns
- **Review Length**: Longer reviews tend to contain more mixed sentiments
- **Platform Differences**: Mobile and web users express sentiments differently
- **Common Themes**: Each sentiment category has unique keywords and topics

## Usage Guide

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Execute the complete analysis pipeline:

```bash
python run_analysis.py
```

Or run the pipeline in stages:

```bash
# Run just data preprocessing
python src/data_preprocessing.py

# Run feature extraction
python src/feature_extraction.py

# Run model training
python src/model_training.py

# Run prediction and visualization
python src/prediction.py
```

### Using the Web Application

Launch the Streamlit web application:

```bash
python -m streamlit run app.py
```

Then navigate to:
- Local URL: http://localhost:8501

The application provides four main sections:
1. **Home**: Overview and project information
2. **Single Text Analysis**: Analyze individual reviews
3. **Batch Analysis**: Process multiple reviews at once
4. **Data Exploration**: Explore dataset characteristics
5. **Model Performance**: Examine model evaluation metrics

## Future Improvements

Potential enhancements to the project:

1. **Advanced NLP Techniques**:
   - Word embeddings (Word2Vec, GloVe, BERT)
   - Attention mechanisms
   - Transfer learning with pre-trained language models

2. **Model Enhancements**:
   - Fine-tuning of hyperparameters
   - Deep learning models (LSTM, Transformers)
   - Ensemble methods

3. **Feature Expansion**:
   - Aspect-based sentiment analysis
   - Emotion detection beyond polarity
   - Topic modeling integration

4. **UI Improvements**:
   - Customizable visualizations
   - Real-time sentiment monitoring
   - Dashboard for trend analysis

5. **Deployment Optimizations**:
   - API development for integration
   - Performance optimization for large-scale analysis
   - Cloud deployment

## Dependencies

- Python 3.8+
- NumPy: Numerical computations
- Pandas: Data manipulation
- NLTK: Natural language processing
- Scikit-learn: Machine learning algorithms
- Matplotlib & Seaborn: Visualization
- Streamlit: Web application framework
- WordCloud: Word cloud visualization
- XGBoost: Gradient boosting

## License

This project is licensed under the MIT License.

## Acknowledgments

- The ChatGPT reviews dataset
- NLTK and Scikit-learn communities for excellent NLP and ML tools
- Streamlit for the intuitive web application framework
