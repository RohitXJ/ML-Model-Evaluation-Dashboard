import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def plotting(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_test)))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {name}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[1])
    axes[1].set_title(f'Classification Report - {name}')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def imp_plot(columns, scores):
    plt.figure(figsize=(10, 5))
    plt.barh(columns[::-1], scores[::-1], color="teal")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()