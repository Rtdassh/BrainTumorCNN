import torch
import torch.nn as nn
import torchvision.models as models


class BrainTumorResNet18(nn.Module):
    """
    ResNet-18 model adapted for 4-class brain tumor classification.
    Inputs are 3-channel axial slices: (FLAIR, T1gd, T2w).

    Improvements over baseline:
    - Deeper classification head with BN + Dropout for better regularization
      and calibration on small medical imaging datasets.
    - Exposes get_parameter_groups() for layer-wise learning rate decay (LLRD)
      during fine-tuning.
    """
    def __init__(self, num_classes=4, pretrained=True, head_dropout=0.4):
        super(BrainTumorResNet18, self).__init__()
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18()

        in_features = self.model.fc.in_features

        # Improved head: Linear → BN → ReLU → Dropout → Linear
        # - BatchNorm1d stabilizes training when backbone is partially frozen.
        # - Dropout(0.4) is a strong regularizer for small medical datasets.
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        return self.model(x)

    def get_parameter_groups(self, base_lr: float):
        """
        Returns parameter groups with Layer-wise Learning Rate Decay (LLRD).
        Deeper layers (closer to head) receive a higher LR; early conv layers
        receive a much lower LR to preserve pretrained ImageNet features.

        Decay multipliers (relative to base_lr):
          - fc (head)   : 1.0  × base_lr
          - layer4      : 0.1  × base_lr
          - layer3      : 0.05 × base_lr
          - layer2      : 0.02 × base_lr
          - layer1+stem : 0.01 × base_lr  (earliest layers, most conservative)
        """
        layer_groups = {
            "fc":     (1.0,  []),
            "layer4": (0.1,  []),
            "layer3": (0.05, []),
            "layer2": (0.02, []),
            "rest":   (0.01, []),   # layer1, conv1, bn1
        }
        for name, param in self.model.named_parameters():
            if name.startswith("fc"):
                layer_groups["fc"][1].append(param)
            elif name.startswith("layer4"):
                layer_groups["layer4"][1].append(param)
            elif name.startswith("layer3"):
                layer_groups["layer3"][1].append(param)
            elif name.startswith("layer2"):
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
    Ensemble of multiple ResNet-18 models.
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


def get_model(num_classes=4, pretrained=True):
    return BrainTumorResNet18(num_classes=num_classes, pretrained=pretrained)

