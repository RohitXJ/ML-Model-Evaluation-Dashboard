import pandas as pd
from preprocessing import prep_data
from pipeline import design_pipeline

print("Hi there, Welcome to ML Model Evaluation Tool!")
print("This tool will help you evaluate your tabular dataset using various metrics and Classifier ML models.")
print("Please follow the instructions to get started")
file_path = input("Enter the path to your dataset (CSV file): ")
target_column = input("Enter the target column name (the column you want to predict): ")
print("Which models do you want to compare? (separate by comma)\n1. Logistic Regression\n2. Random Forest\n3. SVM\n4. Naive Bayes\n5. Gradient Boosting")
choices = list(map(int,input().split(',')))
try:
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Validate if target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")
    
    print("Dataset loaded successfully!")
    
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path and try again.")

print("Holding on, I am processing your request...")

X,y = prep_data(data, target_column)
design_pipeline(X,y,choices)