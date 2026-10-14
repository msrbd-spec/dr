# File: src/models/mixstyle.py
import torch
import torch.nn as nn
import random

class MixStyle(nn.Module):
    """
    Feature-Space Domain Generalization.
    Mixes the channel-wise mean and variance of the source domain (APTOS) 
    with a target domain (Messidor-2) to make the network invariant to camera styles.
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x_src: torch.Tensor, x_tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_src (torch.Tensor): Feature maps from the primary dataset (B, C, H, W).
            x_tgt (torch.Tensor): Feature maps from the external style pool (B, C, H, W).
        """
        if not self.training or random.random() > self.p:
            return x_src
            
        if x_tgt is None:
            # If no external target is provided, mix styles within the source batch
            # This still acts as a strong regularizer
            x_tgt = x_src[torch.randperm(x_src.size(0))]

        B, C, H, W = x_src.size()

        # Compute channel-wise mean and std for Source
        mu_src = x_src.mean(dim=[2, 3], keepdim=True)
        var_src = x_src.var(dim=[2, 3], keepdim=True)
        sig_src = torch.sqrt(var_src + self.eps)

        # Compute channel-wise mean and std for Target
        mu_tgt = x_tgt.mean(dim=[2, 3], keepdim=True)
        var_tgt = x_tgt.var(dim=[2, 3], keepdim=True)
        sig_tgt = torch.sqrt(var_tgt + self.eps)

        # Sample mixing weight from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x_src.device)

        # Mix statistics
        mu_mix = lam * mu_src + (1 - lam) * mu_tgt
        sig_mix = lam * sig_src + (1 - lam) * sig_tgt

        # Normalize source features and apply mixed statistics
        x_src_normalized = (x_src - mu_src) / sig_src
        x_adapted = x_src_normalized * sig_mix + mu_mix

        return x_adapted