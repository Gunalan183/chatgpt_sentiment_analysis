import pandas as pd
import os

# Print current directory for debugging
print(f"Current directory: {os.getcwd()}")

# Find the CSV file
csv_path = os.path.join("..", "chatgpt_reviews - chatgpt_reviews.csv")
data_dir_path = os.path.join("data", "chatgpt_reviews.csv")

print(f"Checking path 1: {os.path.abspath(csv_path)}")
print(f"Exists? {os.path.exists(os.path.abspath(csv_path))}")

print(f"Checking path 2: {os.path.abspath(data_dir_path)}")
print(f"Exists? {os.path.exists(os.path.abspath(data_dir_path))}")

# Try to load the data from both paths
try:
    if os.path.exists(os.path.abspath(csv_path)):
        df = pd.read_csv(os.path.abspath(csv_path))
        print(f"Successfully loaded from path 1. Shape: {df.shape}")
    elif os.path.exists(os.path.abspath(data_dir_path)):
        df = pd.read_csv(os.path.abspath(data_dir_path))
        print(f"Successfully loaded from path 2. Shape: {df.shape}")
    else:
        print("Could not find the CSV file in either location.")
except Exception as e:
    print(f"Error loading CSV: {e}")
