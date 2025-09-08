from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def compute_cls_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}
