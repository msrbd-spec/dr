# File: src/training/trainer.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.training.losses import OrdinalContrastiveLoss, LDAMLoss

class StandardTrainer:
    def __init__(self, model: nn.Module, device: torch.device, epochs: int = 25, lr: float = 1e-4, 
                 loss_type: str = 'ce', cls_num_list: list = None, use_drw: bool = False):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.use_drw = use_drw
        self.drw_epoch = int(epochs * 0.8) # Apply weights after 80% of training
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        self.loss_type = loss_type
        self.cls_num_list = cls_num_list
        
        # Calculate Class Balanced Weights (Effective Number of Samples)
        if cls_num_list is not None:
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * len(cls_num_list)
            self.cb_weights = torch.FloatTensor(weights).to(device)
        else:
            self.cb_weights = None

        if self.loss_type == 'ldam':
            self.criterion = LDAMLoss(cls_num_list=self.cls_num_list, weight=None)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_and_evaluate(self, train_loader, val_loader):
        history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(1, self.epochs + 1):
            # DRW Logic: Apply class weights after drw_epoch
            if self.use_drw and epoch >= self.drw_epoch and self.cb_weights is not None:
                if self.loss_type == 'ldam':
                    self.criterion.weight = self.cb_weights
                else:
                    self.criterion = nn.CrossEntropyLoss(weight=self.cb_weights)

            self.model.train()
            train_loss, correct, total = 0.0, 0, 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]"):
                imgs, labels = batch[0].to(self.device), batch[1].to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            history['train_loss'].append(train_loss / total)
            history['train_acc'].append((correct / total) * 100.0)

            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]"):
                    imgs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    logits = self.model(imgs)
                    loss = self.criterion(logits, labels)
                    
                    val_loss += loss.item() * imgs.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
            val_acc = (correct / total) * 100.0
            history['val_loss'].append(val_loss / total)
            history['val_acc'].append(val_acc)
            
            self.scheduler.step()
            print(f"Epoch {epoch} | Train Acc: {history['train_acc'][-1]:.2f}% | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()

        print(f"Training complete. Restoring best weights (Val Acc: {best_val_acc:.2f}%)")
        self.model.load_state_dict(best_model_state)
        return history, self.model


class CODANetTrainer:
    def __init__(self, model: nn.Module, device: torch.device, epochs: int = 25, lr: float = 1e-4, 
                 use_ordinal: bool = False, alpha: float = 0.5):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.use_ordinal = use_ordinal
        self.alpha = alpha 
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.ordinal_criterion = OrdinalContrastiveLoss(base_margin=1.0).to(device)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def train_and_evaluate(self, train_loader, val_loader):
        history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]"):
                imgs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                style_imgs = batch[2].to(self.device) if len(batch) == 3 else None
                
                self.optimizer.zero_grad()
                logits, projections = self.model(imgs, style_imgs)
                
                loss = self.ce_criterion(logits, labels)
                if self.use_ordinal:
                    loss += self.alpha * self.ordinal_criterion(projections, labels)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            history['train_loss'].append(train_loss / total)
            history['train_acc'].append((correct / total) * 100.0)

            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]"):
                    imgs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    logits, projections = self.model(imgs, None) 
                    
                    loss = self.ce_criterion(logits, labels)
                    if self.use_ordinal:
                        loss += self.alpha * self.ordinal_criterion(projections, labels)
                    
                    val_loss += loss.item() * imgs.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
            val_acc = (correct / total) * 100.0
            history['val_loss'].append(val_loss / total)
            history['val_acc'].append(val_acc)
            
            self.scheduler.step()
            print(f"Epoch {epoch} | Train Acc: {history['train_acc'][-1]:.2f}% | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()

        print(f"Training complete. Restoring best weights (Val Acc: {best_val_acc:.2f}%)")
        self.model.load_state_dict(best_model_state)
        return history, self.model