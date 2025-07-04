from sklearn.pipeline import Pipeline

def design_pipeline(X, y, choices):
    """
    Function to design and execute the ML pipeline based on user choices.
    
    Args:
        X (pd.DataFrame): Features for the model.
        y (pd.Series): Target variable.
        choices (list): List of model choices selected by the user.
    """
    models = {
        1: "Logistic Regression",
        2: "Random Forest",
        3: "SVM",
        4: "Naive Bayes",
        5: "Gradient Boosting"
        }
    for choice in models.keys():
        if choice in choices:
            model_name, model_obj = models[choice]
            model_pipeline = Pipeline([
                ("preprocess", "preprocessor"),
                ("model", model_obj)
            ])