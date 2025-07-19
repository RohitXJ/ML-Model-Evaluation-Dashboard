from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from evaluation.eval import evaluate_model
from evaluation.plot import plotting

def design_pipeline(X, y, choices):
    import matplotlib.pyplot as plt

    models = {
        1: ("Logistic Regression", LogisticRegression(max_iter=3000)),
        2: ("Random Forest", RandomForestClassifier(n_estimators=150)),
        3: ("SVM", SVC(kernel='rbf', probability=True)),
        4: ("Naive Bayes", GaussianNB()),
        5: ("Gradient Boosting", GradientBoostingClassifier())
    }

    pipeline_hub = {}
    for choice in models.keys():
        if choice in choices:
            model_name, model_obj = models[choice]
            pipeline_hub[model_name] = Pipeline([
                ("model", model_obj)
            ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    results = []  # Collect result dictionaries

    for name, pipeline in pipeline_hub.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        eval_text = evaluate_model(y_test, y_pred, name)  # should return markdown string
        fig = plotting(y_test, y_pred, name)

        result_dict = {
            "name": name,
            "evaluation": eval_text,
            "fig": fig
        }

        # 🧠 Include feature importance if available
        try:
            model_core = pipeline.named_steps["model"]
            if hasattr(model_core, "feature_importances_"):
                result_dict["feature_names"] = X.columns.tolist()
                result_dict["importance_scores"] = model_core.feature_importances_
        except:
            pass  # Silently skip if not available

        results.append(result_dict)

    return results
