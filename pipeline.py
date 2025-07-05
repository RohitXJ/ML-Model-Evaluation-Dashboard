from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from evaluation.eval import evaluate_model
from evaluation.plot import plotting

def design_pipeline(X, y, choices):
    """
    Designs and runs ML pipelines based on selected models.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
        choices (list): List of integers representing selected model keys.
    """

    # Available model options
    models = {
        1: ("Logistic Regression", LogisticRegression(max_iter=3000)),
        2: ("Random Forest", RandomForestClassifier(n_estimators=150)),
        3: ("SVM", SVC(kernel='rbf', probability=True)),
        4: ("Naive Bayes", GaussianNB()),
        5: ("Gradient Boosting", GradientBoostingClassifier())
    }

    # Store model pipelines
    pipeline_hub = {}

    for choice in models.keys():
        if choice in choices:
            model_name, model_obj = models[choice]
            model_pipeline = Pipeline([
                ("model", model_obj)  # Only model step; preprocessing already done
            ])
            pipeline_hub[model_name] = model_pipeline

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Train, evaluate, and plot for each selected model
    for name, pipeline in pipeline_hub.items():
        print(f"\nTraining: {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"Evaluating: {name}")
        evaluate_model(y_test, y_pred, name)
        plotting(y_test, y_pred, name)
