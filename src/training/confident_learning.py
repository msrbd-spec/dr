# File: src/training/confident_learning.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import copy

class ConfidentLearningFilter:
    """
    Implements Confident Learning to identify and filter label errors in the dataset.
    It computes out-of-fold probabilities and uses class-specific thresholds to 
    find samples where the human annotator likely made a mistake.
    """
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes

    def get_out_of_fold_probabilities(self, model_builder_fn, dataset, device, k_folds=5, epochs=10, batch_size=16):
        """
        Trains K models on K-1 folds to get unbiased probability predictions for the 1 held-out fold.
        
        Args:
            model_builder_fn: Function that returns a fresh, compiled PyTorch model.
            dataset: The PyTorch Dataset (APTOS training set).
            device: torch.device.
            k_folds: Number of cross-validation folds.
            
        Returns:
            np.ndarray: Out-of-fold probabilities of shape (N, num_classes).
            np.ndarray: Original noisy labels of shape (N,).
        """
        n_samples = len(dataset)
        oof_probs = np.zeros((n_samples, self.num_classes), dtype=np.float32)
        labels = np.array([dataset[i][1] for i in range(n_samples)])
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), labels)):
            print(f"\n--- Confident Learning: Out-of-Fold Phase {fold + 1}/{k_folds} ---")
            
            train_sub = Subset(dataset, train_idx)
            val_sub = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=4)
            
            model = model_builder_fn().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Train for a few epochs to get reasonable feature representations
            model.train()
            for epoch in range(epochs):
                for imgs, tgts in tqdm(train_loader, desc=f"OOF Train Epoch {epoch+1}/{epochs}", leave=False):
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    optimizer.zero_grad()
                    logits = model(imgs)
                    loss = criterion(logits, tgts)
                    loss.backward()
                    optimizer.step()
            
            # Predict on the held-out fold
            model.eval()
            fold_probs = []
            with torch.no_grad():
                for imgs, _ in tqdm(val_loader, desc="OOF Predicting", leave=False):
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    probs = torch.softmax(logits, dim=1)
                    fold_probs.append(probs.cpu().numpy())
                    
            oof_probs[val_idx] = np.vstack(fold_probs)
            
        return oof_probs, labels

    def find_label_errors(self, oof_probs: np.ndarray, labels: np.ndarray) -> list:
        """
        Identifies indices of samples that are mathematically likely to be mislabeled.
        """
        # 1. Calculate class-specific confidence thresholds
        # Threshold for class C is the average predicted probability of class C 
        # for all samples that were labeled as class C.
        thresholds = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if np.sum(class_mask) > 0:
                thresholds[c] = np.mean(oof_probs[class_mask, c])
            else:
                thresholds[c] = 1.0 / self.num_classes # Fallback
                
        print(f"Confident Learning Thresholds per class: {np.round(thresholds, 3)}")

        # 2. Identify errors
        error_indices = []
        for i in range(len(labels)):
            given_label = labels[i]
            pred_label = np.argmax(oof_probs[i])
            pred_prob = oof_probs[i, pred_label]
            
            # If the model confidently predicts a DIFFERENT class than the human annotator
            if pred_label != given_label and pred_prob >= thresholds[pred_label]:
                error_indices.append(i)
                
        print(f"Identified {len(error_indices)} label errors out of {len(labels)} samples.")
        return error_indices