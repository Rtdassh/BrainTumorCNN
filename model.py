import torch
import torch.nn as nn
import torchvision.models as models


class BrainTumorResNet18(nn.Module):
    """
    ResNet-18 model adapted for multi-channel & multi-label brain tumor classification.
    Inputs are 4-channel (FLAIR, T1w, T1gd, T2w) or 3-channel axial slices.
    """
    def __init__(self, num_classes=3, pretrained=True, in_channels=4, head_dropout=0.2, use_mixed_pooling=False, use_mlp_head=True):
        super(BrainTumorResNet18, self).__init__()
        self.use_mixed_pooling = use_mixed_pooling
        self.in_channels = in_channels
        self.use_mlp_head = use_mlp_head
        
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18()

        # Adapt first conv layer if in_channels != 3
        if in_channels != 3:
            old_conv = self.model.conv1
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                padding=old_conv.padding, bias=old_conv.bias
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c] = old_conv.weight.mean(dim=1)
            self.model.conv1 = new_conv

        in_features = self.model.fc.in_features # 512

        if use_mixed_pooling:
            # Custom dual pooling: Global Avg + Global Max
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.model.fc = nn.Identity()
            head_in_features = in_features * 2
        else:
            head_in_features = in_features

        if use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(head_in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=head_dropout),
                nn.Linear(256, num_classes)
            )
            if not use_mixed_pooling:
                self.model.fc = self.head
        else:
            self.model.fc = nn.Linear(head_in_features, num_classes)

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
        # x shape: (batch_size, in_channels, H, W)
        if self.use_mixed_pooling:
            feat_map = self.extract_features(x)
            avg_feat = torch.flatten(self.avg_pool(feat_map), 1)
            max_feat = torch.flatten(self.max_pool(feat_map), 1)
            concat_feat = torch.cat([avg_feat, max_feat], dim=1)
            if self.use_mlp_head:
                return self.head(concat_feat)
            else:
                return self.model.fc(concat_feat)
        else:
            return self.model(x)

    def get_parameter_groups(self, base_lr: float):
        """
        Returns parameter groups with Layer-wise Learning Rate Decay (LLRD).
        """
        layer_groups = {
            "head":   (1.0,  []),
            "layer4": (0.1,  []),
            "layer3": (0.05, []),
            "layer2": (0.02, []),
            "rest":   (0.01, []),
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
    Combines predictions by averaging output probabilities.
    """
    def __init__(self, models_list):
        super(BrainTumorEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x):
        logits_list = [model(x) for model in self.models]
        if logits_list[0].shape[1] == 3:
            probs_list = [torch.sigmoid(logits) for logits in logits_list]
        else:
            probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return avg_probs


def get_model(num_classes=3, pretrained=True, in_channels=4, backbone="resnet18", head_dropout=0.2, use_mixed_pooling=False, use_mlp_head=True):
    """
    Factory function to initialize model architectures.
    """
    if backbone == "resnet18":
        return BrainTumorResNet18(
            num_classes=num_classes, 
            pretrained=pretrained,
            in_channels=in_channels,
            head_dropout=head_dropout, 
            use_mixed_pooling=use_mixed_pooling,
            use_mlp_head=use_mlp_head
        )
    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            old_conv = model.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c] = old_conv.weight.mean(dim=1)
            model.conv1 = new_conv
            
        in_features = model.fc.in_features
        if use_mlp_head:
            model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=head_dropout),
                nn.Linear(256, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)
        return model
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            old_conv = model.features[0][0]
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c] = old_conv.weight.mean(dim=1)
            model.features[0][0] = new_conv
            
        in_features = model.classifier[1].in_features
        if use_mlp_head:
            model.classifier = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            )
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_features, num_classes)
            )
        return model
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from ['resnet18', 'resnet34', 'efficientnet_b0'].")




