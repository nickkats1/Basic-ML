from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def eval_metrics(y_true, y_pred, y_pred_prob):
    return {
        "roc auc": roc_auc_score(y_true, y_pred_prob),
        "acc": accuracy_score(y_true, y_pred),
        "f1 score": f1_score(y_true, y_pred, average="macro"),
    }