"""Hybrid EfficientNet-Lite + Swin Transformer with attention-based feature fusion (Eqs. 1-4 of the paper),
and the baselines of the ablation study (Table 10)."""
import timm
import torch
import torch.nn as nn
from torchvision import models

from .config import CNN_BACKBONE, SWIN_BACKBONE, FUSED_DIM


def make_cnn(backbone=CNN_BACKBONE, pretrained=True):
    """Returns (module producing a pooled feature vector, feature dimension)."""
    if backbone == "mobilenet_v3_small":                       # torchvision MobileNetV3-Small (576-d)
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        return nn.Sequential(models.mobilenet_v3_small(weights=w).features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1)), 576
    m = timm.create_model(backbone, pretrained=pretrained, num_classes=0)   # e.g. tf_efficientnet_lite0 (1,280-d)
    return m, m.num_features


def make_swin(pretrained=True):
    # num_classes=0 -> the 7x7 token map average-pooled to a 768-d vector
    m = timm.create_model(SWIN_BACKBONE, pretrained=pretrained, num_classes=0)
    return m, m.num_features


def head(in_dim, num_classes):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_classes))


class AttentionFusion(nn.Module):
    """Projects both feature vectors to `fused_dim` (Eq. 1), computes element-wise attention
    weights a in (0, 1) from their concatenation (Eq. 2) and returns a * p_c + (1 - a) * p_s (Eq. 3)."""

    def __init__(self, cnn_dim, swin_dim, fused_dim=FUSED_DIM):
        super().__init__()
        self.cnn_fc = nn.Linear(cnn_dim, fused_dim)
        self.swin_fc = nn.Linear(swin_dim, fused_dim)
        self.attn = nn.Sequential(nn.Linear(fused_dim * 2, fused_dim), nn.ReLU(),
                                  nn.Linear(fused_dim, fused_dim), nn.Sigmoid())

    def forward(self, cnn_feat, swin_feat):
        c, s = self.cnn_fc(cnn_feat), self.swin_fc(swin_feat)
        a = self.attn(torch.cat((c, s), dim=1))
        return a * c + (1 - a) * s


class FruitDiseaseModel(nn.Module):
    """Proposed model: CNN branch (local features) + Swin branch (global features) -> attention fusion -> MLP (Eq. 4)."""

    def __init__(self, num_classes, backbone=CNN_BACKBONE, pretrained=True):
        super().__init__()
        self.cnn, cnn_dim = make_cnn(backbone, pretrained)
        self.swin, swin_dim = make_swin(pretrained)
        self.fusion = AttentionFusion(cnn_dim, swin_dim)
        self.classifier = head(FUSED_DIM, num_classes)

    def forward(self, x):
        return self.classifier(self.fusion(self.cnn(x), self.swin(x)))


class SingleBranch(nn.Module):
    """Ablation: one backbone + the same 512-d projection and classifier head as the proposed model."""

    def __init__(self, num_classes, branch="cnn", pretrained=True):
        super().__init__()
        self.enc, dim = make_cnn(CNN_BACKBONE, pretrained) if branch == "cnn" else make_swin(pretrained)
        self.proj = nn.Linear(dim, FUSED_DIM)
        self.classifier = head(FUSED_DIM, num_classes)

    def forward(self, x):
        return self.classifier(self.proj(self.enc(x)))


class ConcatFusion(nn.Module):
    """Ablation: both backbones, plain concatenation of the two 512-d projections (no attention weights)."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.cnn, cnn_dim = make_cnn(CNN_BACKBONE, pretrained)
        self.swin, swin_dim = make_swin(pretrained)
        self.cnn_fc, self.swin_fc = nn.Linear(cnn_dim, FUSED_DIM), nn.Linear(swin_dim, FUSED_DIM)
        self.classifier = head(2 * FUSED_DIM, num_classes)

    def forward(self, x):
        return self.classifier(torch.cat([self.cnn_fc(self.cnn(x)), self.swin_fc(self.swin(x))], dim=1))


MODELS = {   # name used on the command line -> factory(num_classes)
    "proposed": lambda K: FruitDiseaseModel(K),
    "effnet_only": lambda K: SingleBranch(K, "cnn"),
    "swin_only": lambda K: SingleBranch(K, "swin"),
    "concat": lambda K: ConcatFusion(K),
    "mobilenet_attention": lambda K: FruitDiseaseModel(K, backbone="mobilenet_v3_small"),
}
MODEL_LABELS = {
    "effnet_only": "EfficientNet-Lite only",
    "swin_only": "Swin Transformer only",
    "concat": "EfficientNet-Lite + Swin (concatenation)",
    "mobilenet_attention": "MobileNetV3-Small + Swin (attention fusion)",
    "proposed": "Proposed: EfficientNet-Lite + Swin (attention fusion)",
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
