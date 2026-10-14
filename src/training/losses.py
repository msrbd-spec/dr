# File: src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OrdinalContrastiveLoss(nn.Module):
    """
    Enforces ordinal geometry in the feature space.
    Pulls features of the same grade together.
    Pushes features of different grades apart by a margin strictly proportional 
    to their clinical severity difference (e.g., |y_i - y_j|).
    """
    def __init__(self, base_margin: float = 1.0):
        super().__init__()
        self.base_margin = base_margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        
        dist_matrix = torch.cdist(features, features, p=2) + 1e-8
        
        labels_float = labels.float().unsqueeze(1)
        ordinal_dist_matrix = torch.abs(labels_float - labels_float.t())
        
        pos_mask = (ordinal_dist_matrix == 0).float()
        neg_mask = (ordinal_dist_matrix > 0).float()
        
        eye = torch.eye(batch_size, device=features.device)
        pos_mask = pos_mask - eye
        
        pos_loss = (pos_mask * (dist_matrix ** 2)).sum() / (pos_mask.sum() + 1e-8)
        
        dynamic_margin = self.base_margin * ordinal_dist_matrix
        neg_distances = F.relu(dynamic_margin - dist_matrix)
        neg_loss = (neg_mask * (neg_distances ** 2)).sum() / (neg_mask.sum() + 1e-8)
        
        total_loss = pos_loss + neg_loss
        return total_loss

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss.
    Used as a strong baseline for class imbalance.
    """
    def __init__(self, cls_num_list: list, max_m: float = 0.5, s: float = 30.0, weight: torch.Tensor = None):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list.to(x.device)[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        
        return F.cross_entropy(self.s * output, target, weight=self.weight)