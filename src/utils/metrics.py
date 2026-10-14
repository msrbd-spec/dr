# File: src/utils/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes standard 5-class metrics AND Binary (Referable vs Non-Referable) metrics.
    """
    # --- 5-Class Metrics ---
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    # --- Binary Metrics (Referable DR) ---
    # Map Grade 0,1 -> 0 (Non-Referable) | Grade 2,3,4 -> 1 (Referable)
    y_true_bin = np.where(y_true >= 2, 1, 0)
    y_pred_bin = np.where(y_pred >= 2, 1, 0)

    bin_acc = accuracy_score(y_true_bin, y_pred_bin) * 100.0
    bin_prec = precision_score(y_true_bin, y_pred_bin, zero_division=0) * 100.0
    bin_rec = recall_score(y_true_bin, y_pred_bin, zero_division=0) * 100.0
    bin_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0) * 100.0

    return {
        'Accuracy': float(f"{acc:.2f}"),
        'QWK': float(f"{qwk:.2f}"),
        'Precision': float(f"{prec:.2f}"),
        'Recall': float(f"{rec:.2f}"),
        'F1_Score': float(f"{f1:.2f}"),
        'Bin_Acc': float(f"{bin_acc:.2f}"),
        'Bin_Prec': float(f"{bin_prec:.2f}"),
        'Bin_Rec': float(f"{bin_rec:.2f}"),
        'Bin_F1': float(f"{bin_f1:.2f}")
    }

def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    Generates and saves a detailed per-class report to CSV.
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report_dict).transpose()
    df = df.round(2)
    df.to_csv(save_path, index=True)

def aggregate_metrics(metrics_list: list) -> dict:
    """
    Takes a list of metric dictionaries from 5 folds and computes Mean ± Std.
    """
    aggregated = {}
    for key in metrics_list[0].keys():
        # Safely cast to float before computing mean/std
        values = [float(m[key]) for m in metrics_list]
        mean_val = np.mean(values)
        std_val = np.std(values)
        aggregated[key] = f"{mean_val:.2f} ± {std_val:.2f}"
    return aggregated