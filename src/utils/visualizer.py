# File: src/utils/visualizer.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for server/cluster compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

def plot_curves(train_acc: list, val_acc: list, train_loss: list, val_loss: list, save_dir: str):
    """Plots and saves Accuracy and Loss curves (Train=Blue, Val=Red)."""
    epochs = range(1, len(train_acc) + 1)
    
    # Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'train_val_acc.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'train_val_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str, num_classes: int = 5):
    """Plots Confusion Matrix with raw counts and percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cm_perc = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100

    labels = [f"{v1}\n({v2:.1f}%)" for v1, v2 in zip(cm.flatten(), cm_perc.flatten())]
    labels = np.asarray(labels).reshape(num_classes, num_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'cm.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_auroc(y_true: np.ndarray, y_prob: np.ndarray, save_dir: str, num_classes: int = 5):
    """Plots Multi-class One-vs-Rest AUROC."""
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    plt.figure(figsize=(8, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve (OvR)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'auroc.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(features: np.ndarray, labels: np.ndarray, save_dir: str):
    """Plots t-SNE feature clustering to prove Grade Boundary Separation."""
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('t-SNE Feature Space Visualization')
    plt.savefig(os.path.join(save_dir, 'tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()