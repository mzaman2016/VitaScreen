from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, avg_option="macro", y_pred_proba=None):
    """Compute accuracy, precision, recall, F1 score, PR-AUC, and ROC-AUC.

    Args:
        y_true: True labels
        y_pred: Predicted class labels
        avg_option: Averaging option for multi-class metrics
        y_pred_proba: Predicted probabilities for positive class (required for PR-AUC and ROC-AUC)
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg_option)
    recall = recall_score(y_true, y_pred, average=avg_option)
    f1 = f1_score(y_true, y_pred, average=avg_option)

    # Use probabilities for PR-AUC and ROC-AUC if provided, otherwise use predictions
    proba_input = y_pred_proba if y_pred_proba is not None else y_pred
    pr_auc = average_precision_score(y_true, proba_input, average=avg_option)
    roc_auc = roc_auc_score(y_true, proba_input, average=avg_option)

    return {
        "avg_option": avg_option,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }
