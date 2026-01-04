import numpy as np

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    return cm
def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp =np.sum(cm,axis=0)-tp
    precision = tp / (tp + fp + 1e-10)
    return precision

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    recall = tp / (tp + fn + 1e-10)
    return recall

def f1_score(y_true, y_pred):
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Positive, False Positive, False Negative
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # F1 Score
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
def f1_score_multiclass(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(y_true)
    f1_list = []
    
    for c in classes:
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        f1_list.append(f1)
    
    return np.mean(f1_list)


