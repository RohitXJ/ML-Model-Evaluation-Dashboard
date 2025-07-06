import pandas as pd
from preprocessing import prep_data
from pipeline import design_pipeline

def main():
    print("Welcome to the ML Model Evaluation Tool.")
    print("This tool will help you evaluate your dataset using various ML classifiers.")
    print("Follow the prompts to get started.\n")

    file_path = input("Enter the path to your dataset (CSV file): ").strip()
    target_column = input("Enter the target column name (the column to predict): ").strip()

    print("\nSelect the models you want to evaluate (comma-separated):")
    print("1. Logistic Regression")
    print("2. Random Forest")
    print("3. SVM")
    print("4. Naive Bayes")
    print("5. Gradient Boosting")

    try:
        choices = list(map(int, input("Enter model numbers (e.g. 1,3,5): ").split(',')))
        valid_choices = [i for i in choices if i in [1, 2, 3, 4, 5]]
        if not valid_choices:
            raise ValueError("No valid model numbers selected.")

        data = pd.read_csv(file_path)
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        print("Dataset loaded successfully.")
        columns_to_drop = input("Enter any columns to drop (comma-separated, or leave blank to skip): ").strip()
        print("Processing the data and preparing models...")

        X, y = prep_data(data, target_column, columns_to_drop.split(',') if columns_to_drop else None)
        design_pipeline(X, y, valid_choices)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the path and try again.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
