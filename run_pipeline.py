"""
Pipeline Runner for ChatGPT Reviews Sentiment Analysis
-----------------------------------------------------
This script runs the complete pipeline for the project:
1. Data preprocessing
2. Feature extraction
3. Model training
4. Model evaluation and visualization
"""

import os
import sys
import time
import argparse

def run_module(module_path, description):
    """
    Run a Python module and track execution time
    
    Parameters:
    -----------
    module_path : str
        Path to the module to run
    description : str
        Description of the step for printing
    """
    print(f"\n{'-'*80}")
    print(f"Running: {description}")
    print(f"{'-'*80}")
    
    start_time = time.time()
    
    # Run the module
    os.system(f'python {module_path}')
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nCompleted: {description}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"{'-'*80}\n")

def run_streamlit():
    """
    Run the Streamlit app
    """
    print(f"\n{'-'*80}")
    print(f"Starting Streamlit Web Application")
    print(f"{'-'*80}")
    
    # Run Streamlit app
    os.system('streamlit run app.py')

def main():
    parser = argparse.ArgumentParser(description='Run the ChatGPT Reviews Sentiment Analysis pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing step')
    parser.add_argument('--skip-feature-extraction', action='store_true', help='Skip feature extraction step')
    parser.add_argument('--skip-model-training', action='store_true', help='Skip model training step')
    parser.add_argument('--skip-prediction', action='store_true', help='Skip prediction and visualization step')
    parser.add_argument('--run-app', action='store_true', help='Run the Streamlit web application')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    for directory in ['data', 'models', 'results']:
        os.makedirs(directory, exist_ok=True)
    
    # Run pipeline steps
    if not args.skip_preprocessing:
        run_module('src/data_preprocessing.py', 'Data Preprocessing')
    
    if not args.skip_feature_extraction:
        run_module('src/feature_extraction.py', 'Feature Extraction')
    
    if not args.skip_model_training:
        run_module('src/model_training.py', 'Model Training')
    
    if not args.skip_prediction:
        run_module('src/prediction.py', 'Prediction and Visualization')
    
    # Run Streamlit app if requested
    if args.run_app:
        run_streamlit()
    else:
        print("\nPipeline completed! To run the web application, use: python run_pipeline.py --run-app")

if __name__ == "__main__":
    main()
