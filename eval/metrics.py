from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def compute_cls_metrics(labels, preds, probs=None):
    """Enhanced metrics computation with proper AUC calculation.
    
    Args:
        labels: True labels
        preds: Predicted labels (binary)
        probs: Predicted probabilities (optional, for proper AUC calculation)
        
    Returns:
        Dictionary with computed metrics
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    
    # Add precision and recall
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    recall = recall_score(labels, preds, average='binary', zero_division=0)
    
    # Proper AUC calculation using probabilities when available
    try:
        if probs is not None:
            # Use probabilities for proper AUC calculation
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                # Multi-class probabilities, use positive class
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                # Binary probabilities
                auc = roc_auc_score(labels, probs)
        else:
            # Fallback to using predictions (less accurate but still valid)
            auc = roc_auc_score(labels, preds)
    except Exception as e:
        # Log warning but don't fail
        print(f"Warning: AUC calculation failed: {e}")
        auc = float("nan")
    
    return {
        "accuracy": acc, 
        "f1": f1, 
        "auc": auc,
        "precision": precision,
        "recall": recall
    }
