import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.models.mixstyle import MixStyle

class CODANet(nn.Module):
    def __init__(self, backbone_name: str = 'convnextv2_large.fcmae_ft_in22k_in1k_384', 
                 num_classes: int = 5, pretrained: bool = True, 
                 use_mixstyle: bool = False, mixstyle_p: float = 0.5,
                 ssl_weights_path: str = None):
        super().__init__()
        
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        
        # Load SSL Weights if provided
        if ssl_weights_path is not None and os.path.exists(ssl_weights_path):
            print(f"Loading SSL Pretrained Weights from {ssl_weights_path}...")
            self.backbone.load_state_dict(torch.load(ssl_weights_path, map_location='cpu'), strict=False)
        elif ssl_weights_path is not None:
            print(f"WARNING: SSL weights file not found at {ssl_weights_path}. Using ImageNet weights.")
            
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy_input)
            self.stage_dims = [f.shape[1] for f in features]
            self.final_dim = self.stage_dims[-1]
            
        self.use_mixstyle = use_mixstyle
        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=0.1)
            
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.final_dim, num_classes)
        )
        
        self.projector = nn.Sequential(
            nn.Linear(self.final_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward(self, x_src: torch.Tensor, x_tgt: torch.Tensor = None):
        src_feats = self.backbone(x_src)
        
        if self.use_mixstyle and self.training:
            if x_tgt is not None:
                tgt_feats = self.backbone(x_tgt)
                src_feats[1] = self.mixstyle(src_feats[1], tgt_feats[1])
            else:
                src_feats[1] = self.mixstyle(src_feats[1], None)
                
        final_map = src_feats[-1]
        pooled = self.pool(final_map).flatten(1)
        
        logits = self.classifier(pooled)
        projections = self.projector(pooled)
        projections = F.normalize(projections, p=2, dim=1)
        
        return logits, projections