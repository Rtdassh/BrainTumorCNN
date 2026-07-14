import torch
import torch.nn as nn
import torchvision.models as models

class BrainTumorResNet18(nn.Module):
    """
    ResNet-18 model adapted for 4-class brain tumor classification.
    Inputs are 3-channel axial slices: (FLAIR, T1gd, T2w).
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(BrainTumorResNet18, self).__init__()
        if pretrained:
            # Using torchvision's latest weights interface
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18()
            
        # Modify the fully connected layer to output class logits
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        return self.model(x)

class BrainTumorEnsemble(nn.Module):
    """
    Ensemble of multiple ResNet-18 models.
    Combines predictions by averaging output probabilities (softmax).
    """
    def __init__(self, models_list):
        super(BrainTumorEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)
        
    def forward(self, x):
        # Forward pass through all models
        logits_list = [model(x) for model in self.models]
        
        # Convert logits to probabilities
        probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
        
        # Average the probabilities
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        
        # Return average probabilities (or log probabilities)
        return avg_probs

def get_model(num_classes=4, pretrained=True):
    return BrainTumorResNet18(num_classes=num_classes, pretrained=pretrained)
