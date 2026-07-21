import torch
import torch.nn as nn
import torchvision.models as models


class BrainTumorResNet18(nn.Module):
    """
    ResNet-18 model adapted for 4-class brain tumor classification.
    Inputs are 3-channel axial slices: (FLAIR, T1gd, T2w).

    Improvements over baseline:
    - Dual/Hybrid Global Pooling (GAP + GMP) to capture diffuse edema and localized focal enhancements.
    - Flexible classification head with Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear.
    - Clean extract_features() interface for Grad-CAM / representation analysis.
    - Exposes get_parameter_groups() for layer-wise learning rate decay (LLRD).
    """
    def __init__(self, num_classes=4, pretrained=True, head_dropout=0.2, use_mixed_pooling=True):
        super(BrainTumorResNet18, self).__init__()
        self.use_mixed_pooling = use_mixed_pooling
        
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18()

        in_features = self.model.fc.in_features # 512

        if use_mixed_pooling:
            # Custom dual pooling: Global Avg + Global Max
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            # Removing original fc to handle pooling manually in forward/features
            self.model.fc = nn.Identity()
            head_in_features = in_features * 2
        else:
            head_in_features = in_features

        # 2-layer perceptron head with BatchNorm and Dropout for regularized domain adaptation
        self.head = nn.Sequential(
            nn.Linear(head_in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(256, num_classes)
        )
        if not use_mixed_pooling:
            self.model.fc = self.head

    def extract_features(self, x):
        """Returns conv bottleneck feature maps before final classification head."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        if self.use_mixed_pooling:
            feat_map = self.extract_features(x)
            avg_feat = torch.flatten(self.avg_pool(feat_map), 1)
            max_feat = torch.flatten(self.max_pool(feat_map), 1)
            concat_feat = torch.cat([avg_feat, max_feat], dim=1)
            return self.head(concat_feat)
        else:
            return self.model(x)

    def get_parameter_groups(self, base_lr: float):
        """
        Returns parameter groups with Layer-wise Learning Rate Decay (LLRD).
        Deeper layers (closer to head) receive a higher LR; early conv layers
        receive a lower LR to preserve pretrained ImageNet features.
        """
        layer_groups = {
            "head":   (1.0,  []),
            "layer4": (0.1,  []),
            "layer3": (0.05, []),
            "layer2": (0.02, []),
            "rest":   (0.01, []),   # layer1, conv1, bn1
        }
        for name, param in self.named_parameters():
            if "head" in name or "fc" in name:
                layer_groups["head"][1].append(param)
            elif "layer4" in name:
                layer_groups["layer4"][1].append(param)
            elif "layer3" in name:
                layer_groups["layer3"][1].append(param)
            elif "layer2" in name:
                layer_groups["layer2"][1].append(param)
            else:
                layer_groups["rest"][1].append(param)

        param_groups = [
            {"params": params, "lr": mult * base_lr}
            for _, (mult, params) in layer_groups.items()
            if len(params) > 0
        ]
        return param_groups


class BrainTumorEnsemble(nn.Module):
    """
    Ensemble of multiple classification models.
    Combines predictions by averaging output probabilities (softmax).
    """
    def __init__(self, models_list):
        super(BrainTumorEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x):
        logits_list = [model(x) for model in self.models]
        probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return avg_probs


def get_model(num_classes=4, pretrained=True, backbone="resnet18", head_dropout=0.2, use_mixed_pooling=False):
    """
    Factory function to initialize model architectures.
    Default settings remain strictly backward-compatible with original checkpoints.
    """
    if backbone == "resnet18":
        return BrainTumorResNet18(
            num_classes=num_classes, 
            pretrained=pretrained, 
            head_dropout=head_dropout, 
            use_mixed_pooling=use_mixed_pooling
        )
    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(256, num_classes)
        )
        return model
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        return model
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from ['resnet18', 'resnet34', 'efficientnet_b0'].")


