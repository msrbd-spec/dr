# File: src/models/baselines.py
import torch
import torch.nn as nn
import timm

class BaseModelWrapper(nn.Module):
    """
    Wraps timm models to provide a unified interface for feature extraction 
    and standard classification.
    """
    def __init__(self, model_name: str, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        
        # Create model without the classification head (num_classes=0 returns pooled features)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Determine the feature dimension dynamically
        with torch.no_grad():
            # Use a standard 224x224 input just to trace the feature dimension
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pooled feature vector (B, F)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the classification logits (B, C)."""
        features = self.extract_features(x)
        return self.classifier(features)

def get_baseline_model(model_name: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Factory function for the 10 baseline architectures.
    These represent a diverse set of paradigms (Transformers, Dense, Lightweight, Compound).
    """
    valid_models = {
        'swinv2_l': 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
        'convnextv2_l': 'convnextv2_large.fcmae_ft_in22k_in1k_384',
        'resnet50': 'resnet50.tv_in1k',
        'efficientnetv2_m': 'efficientnetv2_rw_m.agc_in1k',
        'mobilenetv3_l': 'mobilenetv3_large_100',
        'maxvit_l': 'maxvit_large_tf_512.in1k',
        'densenet121': 'densenet121',
        'regnety_120': 'regnety_120',
        'vgg16_bn': 'vgg16_bn',
        'squeezenet1_1': 'squeezenet1_1'
    }
    
    if model_name not in valid_models:
        raise ValueError(f"Invalid model_name. Choose from: {list(valid_models.keys())}")
        
    timm_name = valid_models[model_name]
    return BaseModelWrapper(timm_name, num_classes, pretrained)