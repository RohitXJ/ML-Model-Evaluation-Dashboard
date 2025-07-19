from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    labels = list(report.keys())[:-3]  # skip accuracy/avg metrics
    avg_metrics = ['macro avg', 'weighted avg']

    markdown = f"### 🧪 Evaluation Report: {model_name}\n"
    markdown += f"**Accuracy:** `{acc:.4f}`\n\n"

    # Confusion matrix
    markdown += "**Confusion Matrix:**\n\n"
    markdown += "| | " + " | ".join(str(i) for i in range(len(cm))) + " |\n"
    markdown += "|---" * (len(cm) + 1) + "|\n"
    for i, row in enumerate(cm):
        markdown += f"| **{i}** | " + " | ".join(str(val) for val in row) + " |\n"

    # Classification Report
    markdown += "\n**Classification Report:**\n\n"
    markdown += "| Class | Precision | Recall | F1-score | Support |\n"
    markdown += "|-------|-----------|--------|----------|---------|\n"
    for label in labels + avg_metrics:
        if label not in report:
            continue
        row = report[label]
        markdown += f"| **{label}** | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1-score']:.2f} | {row['support']} |\n"

    return markdown
