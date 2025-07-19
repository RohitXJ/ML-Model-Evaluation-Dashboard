import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE,SMOTENC,RandomOverSampler

def prep_data(data, target_column, columns_to_drop=None):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_selection import VarianceThreshold
    from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit

    total_rows, _ = data.shape
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Drop manually specified columns
    if columns_to_drop:
        for col in columns_to_drop:
            if col in X.columns:
                print(f"Dropping column as specified: {col}")
                X.drop(columns=[col], inplace=True)

    # Drop keyword-based columns
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

    for col in obj_col:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    for col in num_col:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())
        scaler = StandardScaler()
        X[col] = scaler.fit_transform(X[[col]])

    # Drop low variance
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    # Imbalance handling
    balance = y.value_counts(normalize=True) * 100
    max_pct = balance.max()

    if any(abs(max_pct - pct) > 7 for pct in balance):
        print("Dataset is imbalanced.")
        categorical_indices = [X.columns.get_loc(col) for col in obj_col if col in X.columns]
        num_features = X.shape[1] - len(categorical_indices)

        if len(categorical_indices) == X.shape[1]:
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)
        elif num_features == X.shape[1]:
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X, y = smote.fit_resample(X, y)
        else:
            smote_nc = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy='auto',
                random_state=42
            )
            X, y = smote_nc.fit_resample(X, y)

        X = pd.DataFrame(X, columns=X.columns)
        y = pd.Series(y).reset_index(drop=True)

    # PCA fallback
    if X.shape[1] > 10:
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=0.92)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X)

    print("✅ Preprocessing complete.")
    return X, y





#print(prep_data(pd.read_csv('tested.csv'), 'Survived'))  # Example usage, replace 'test.csv' and 'target_column' with actual values