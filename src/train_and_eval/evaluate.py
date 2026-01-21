from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, avg_option="macro"):
    """Compute accuracy, precision, recall, and F1 score."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg_option)
    recall = recall_score(y_true, y_pred, average=avg_option)
    f1 = f1_score(y_true, y_pred, average=avg_option)

    return {
        "avg_option": avg_option,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
