import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE,SMOTENC,RandomOverSampler
from evaluation.plot import imp_plot

def prep_data(data, target_column, columns_to_drop=None):
    total_rows, _ = data.shape
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Manual column drop (external input)
    if columns_to_drop:
        for col in columns_to_drop:
            if col in X.columns:
                print(f"Dropping column as specified: {col}")
                X.drop(columns=[col], inplace=True)

    # Drop non-informative columns based on keywords
    keywords = ["id", "name", "serial", "timestamp", "date", "uuid"]
    for col in X.columns:
        if any(key in col.lower() for key in keywords):
            print(f"Dropping non-informative column: {col}")
            X.drop(columns=[col], inplace=True)

    # Drop high-missing columns
    for col in X.columns:
        if X[col].isnull().sum() > 0.3 * total_rows:
            print(f"Dropping column with high missing values: {col}")
            X.drop(columns=[col], inplace=True)

    num_col = X.select_dtypes(include=['int', 'float']).columns.tolist()
    obj_col = X.select_dtypes(include=['object']).columns.tolist()

    # Encode categorical
    for col in obj_col:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scale numerical
    for col in num_col:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())
        scaler = StandardScaler()
        X[col] = scaler.fit_transform(X[[col]])

    # Estimate feature importance
    print("Estimating feature importance from balanced sample...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=min(0.3, 500 / len(X)), random_state=42)
    for train_idx, _ in sss.split(X, y):
        sample_X = X.iloc[train_idx].copy()
        sample_y = y.iloc[train_idx].copy()

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(sample_X, sample_y)

    importances = np.abs(model.coef_[0])
    sorted_idx = np.argsort(importances)[::-1]
    sorted_cols = sample_X.columns[sorted_idx]
    sorted_scores = importances[sorted_idx]

    imp_plot(sorted_cols, sorted_scores)

    # Get user input
    top_n_features = input("Select number of features to keep based on importance scores,\nor enter 'auto' for automatic selection:\n")

    # Feature selection logic
    if top_n_features.isdigit():
        top_n_features = int(top_n_features)
        X = X[sorted_cols[:top_n_features]]
        print(f"Using top {top_n_features} user-selected features.")
    elif top_n_features.strip().lower() == 'auto':
        diffs = np.diff(sorted_scores)
        elbow = np.argmax(diffs < 0.01) + 1
        X = X[sorted_cols[:elbow]]
        print(f"'Auto' selected top {elbow} features using elbow method.")
    else:
        print("Invalid input. No feature filtering applied.")

    # Drop low variance features
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    # --- Imbalance handling ---
    balance = y.value_counts(normalize=True) * 100
    max_pct = balance.max()

    if any(abs(max_pct - pct) > 7 for pct in balance):
        print("Dataset is imbalanced.")

        # Get categorical column indices based on pre-encoding list
        categorical_indices = [X.columns.get_loc(col) for col in obj_col if col in X.columns]
        num_features = X.shape[1] - len(categorical_indices)

        if len(categorical_indices) == X.shape[1]:
            print("Only categorical features found — applying RandomOverSampler.")
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)

        elif num_features == X.shape[1]:
            print("Only numerical features found — applying SMOTE.")
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X, y = smote.fit_resample(X, y)

        else:
            print("Mixed features — applying SMOTENC.")
            smote_nc = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy='auto',
                random_state=42
            )
            X, y = smote_nc.fit_resample(X, y)

        X = pd.DataFrame(X, columns=X.columns)
        y = pd.Series(y).reset_index(drop=True)

    # Final PCA fallback
    if X.shape[1] > 20:
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=0.92)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X)

    print("✅ Preprocessing complete.")
    return X, y




#print(prep_data(pd.read_csv('tested.csv'), 'Survived'))  # Example usage, replace 'test.csv' and 'target_column' with actual values