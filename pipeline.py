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
    Function to design and execute the ML pipeline based on user choices.
    
    Args:
        X (pd.DataFrame): Features for the model.
        y (pd.Series): Target variable.
        choices (list): List of model choices selected by the user.
    """
    models = {
        1: ("Logistic Regression", LogisticRegression(max_iter=300)),
        2: ("Random Forest", RandomForestClassifier(n_estimators=100)),
        3: ("SVM", SVC(kernel='linear', probability=True)),
        4: ("Naive Bayes", GaussianNB()),
        5: ("Gradient Boosting", GradientBoostingClassifier())
    }
    pipeline_hub = {}
    for choice in models.keys():
        if choice in choices:
            model_name, model_obj = models[choice]
            model_pipeline = Pipeline([
                ("model", model_obj)
            ])
            pipeline_hub[model_name] = model_pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for name,pipeline in pipeline_hub.items():
        print(f"Training {name}...")
        pipeline.fit(X_train,y_train)
        y_pred = pipeline.predict(X_test)
        print(f"Evaluating {name}...")
        evaluate_model(y_test, y_pred, name)
        plotting(y_test, y_pred, name)
        