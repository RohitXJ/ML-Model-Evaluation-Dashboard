import shap
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.decomposition import PCA

def prep_data(data, target_column):
    """
    Prepares the dataset for model training and evaluation.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    target_column (str): The name of the target column to predict.
    
    Returns:
    X and y (pd.DataFrame, pd.DataFrame): Processed features and target variable.
    """
    total_col = len(data.columns.tolist())
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    num_col = X.select_dtypes(include=['int', 'float']).columns.tolist()
    obj_col = X.select_dtypes(include=['object']).columns.tolist()

    # Encoding categorical features
    for col in obj_col:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])
        
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Scaling numerical features
    for col in num_col:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())
        
        scaler = StandardScaler()
        X[col] = scaler.fit_transform(X[[col]])

    # Applying SMOTENC for imbalanced datasets
    balance = y.value_counts(normalize=True) * 100
    max_pct = balance.max()
    if any(abs(max_pct - pct) > 7 for pct in balance):
        print("Dataset is imbalanced, applying SMOTENC...")
        smote = SMOTENC(
            categorical_features=obj_col,
            sampling_strategy='auto',
            random_state=42,
        )
        X, y = smote.fit_resample(X, y)
        X = pd.DataFrame(X, columns=X.columns)
        y = pd.DataFrame(y, columns=[target_column])

    # Applying PCA if the number of features is greater than 10
    if total_col > 6:
        print("Applying PCA for dimensionality reduction...")
        pca  = PCA(n_components=0.92)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X)
    # Extracting top 5 features using SHAP if the number of features is still large
    if total_col > 7 and X.shape[1] > 5:
        print("Using SHAP (KernelExplainer + SVM) to extract 5 most important features...")

        sample_X = X.sample(n=min(300, len(X)), random_state=42)
        sample_y = y.loc[sample_X.index]

        model = SVC(kernel='rbf', probability=True)
        model.fit(sample_X, sample_y)

        # SHAP with KernelExplainer — use a background sample
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(sample_X, 10))
        shap_values = explainer.shap_values(sample_X, nsamples=100)  # fast approx

        # For binary classification, use class 1's SHAP
        shap_df = pd.DataFrame(shap_values[1], columns=[f"F{i}" for i in range(sample_X.shape[1])])
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

        top5_indices = mean_abs_shap.head(5).index.tolist()

        X = sample_X[top5_indices].copy().reset_index(drop=True)
        y = sample_y.reset_index(drop=True)
        print("Reduced dataset to top 5 SHAP-ranked features.")
    print("Data set preprocessing complete.")

    return X, y # End of prep_data function and returning processed features and target variable



#print(prep_data(pd.read_csv('tested.csv'), 'Survived'))  # Example usage, replace 'test.csv' and 'target_column' with actual values