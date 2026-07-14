import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score, f1_score
)

from model import get_model
from train import BrainTumorSliceDataset

# Grad-CAM implementation in pure PyTorch
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_score = logits[0, class_idx]
        class_score.backward()
        
        # Gradients and activations
        grads = self.gradients
        acts = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Interpolate to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.squeeze().detach().cpu().numpy(), class_idx

def get_center_id(patient_name):
    """
    Simulates center/scanner grouping based on patient ID ranges:
    - Center A: BRATS_001 to BRATS_199
    - Center B: BRATS_200 to BRATS_299
    - Center C: BRATS_300 to BRATS_484
    """
    try:
        num = int(patient_name.split("_")[1])
        if num <= 199:
            return "Center A (GBM study 1)"
        elif num <= 299:
            return "Center B (GBM study 2)"
        else:
            return "Center C (LGG/Other study)"
    except Exception:
        return "Unknown Center"

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_curves(all_labels, all_probs, class_names, save_dir):
    n_classes = len(class_names)
    
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        # Convert label array to one-hot for class i
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "roc_curves.png"))
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        ap = average_precision_score(y_true_binary, y_score)
        
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.3f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, "precision_recall_curves.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Brain Tumor ResNet-18 Model")
    parser.add_argument("--fold", type=int, default=1, help="Which fold model to load (1-5)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed_2d")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    metadata_path = os.path.join(processed_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Processed dataset metadata not found. Please run preprocess_data.py first.")
        
    # Get fold val patients
    from crossval import get_patient_folds
    folds = get_patient_folds(metadata_path, n_splits=5)
    _, val_patients = folds[args.fold - 1]
    
    # Load dataset
    val_dataset = BrainTumorSliceDataset(val_patients, processed_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    class_names = ["Healthy/Bg", "Edema", "Non-Enhancing Tumor", "Enhancing Tumor"]
    
    # Load model
    model = get_model(num_classes=4, pretrained=False).to(args.device)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{args.fold}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
        
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']:.4f})")
    
    # Run evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    
    # Gather slice metadata
    # The dataset class preserves sample order
    for idx, (img_path, label) in enumerate(val_dataset.samples):
        # Extract patient ID from path
        p_id = os.path.basename(img_path).split("_slice_")[0]
        all_patient_ids.append(p_id)
        
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(args.device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Print metrics
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n=== Classification Report ===")
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(results_dir, "confusion_matrix.png"))
    plot_curves(all_labels, all_probs, class_names, results_dir)
    print(f"Saved confusion matrix and ROC/PR curves to {results_dir}")
    
    # Scanner Cohort / Center Performance Analysis
    print("\n=== Robustness Analysis: Performance by Center/Scanner ===")
    centers = [get_center_id(pid) for pid in all_patient_ids]
    unique_centers = sorted(list(set(centers)))
    
    center_reports = {}
    for center in unique_centers:
        indices = [i for i, c in enumerate(centers) if c == center]
        if len(indices) == 0:
            continue
            
        c_labels = all_labels[indices]
        c_preds = all_preds[indices]
        
        c_acc = np.mean(c_labels == c_preds)
        c_f1 = f1_score(c_labels, c_preds, average='macro')
        
        print(f"{center}:")
        print(f"  Slices: {len(indices)}")
        print(f"  Accuracy: {c_acc:.4f}")
        print(f"  Macro F1: {c_f1:.4f}")
        
    # Generate Grad-CAM sample visualizations
    print("\n=== Generating Grad-CAM Visualizations ===")
    # Target the last convolutional layer of ResNet-18: model.layer4[1].conv2
    # In torchvision's resnet18, model is self.model
    target_layer = model.model.layer4[1].conv2
    grad_cam = GradCAM(model, target_layer)
    
    # Find a sample of each class
    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
    for cls in range(4):
        # Find index of first sample belonging to class cls that was correctly predicted
        idx = next((i for i in range(len(val_dataset)) if val_dataset.samples[i][1] == cls and all_preds[i] == cls), None)
        if idx is None:
            # Fallback to any sample of class cls
            idx = next((i for i in range(len(val_dataset)) if val_dataset.samples[i][1] == cls), None)
            
        if idx is not None:
            img_path, label = val_dataset.samples[idx]
            image_np = np.load(img_path)
            
            # Form input batch (1, 3, H, W)
            input_tensor = torch.tensor(image_np).unsqueeze(0).to(args.device)
            
            # Generate CAM
            cam, predicted_cls = grad_cam(input_tensor, class_idx=cls)
            
            # Display FLAIR channel as grayscale background
            flair_bg = image_np[0] # Channel 0: FLAIR
            
            axes[cls, 0].imshow(flair_bg, cmap='gray')
            axes[cls, 0].set_title(f"Original Slice (True Class: {class_names[cls]})")
            axes[cls, 0].axis('off')
            
            axes[cls, 1].imshow(flair_bg, cmap='gray')
            axes[cls, 1].imshow(cam, cmap='jet', alpha=0.5)
            axes[cls, 1].set_title(f"Grad-CAM (Target: {class_names[cls]})")
            axes[cls, 1].axis('off')
            
    plt.tight_layout()
    gradcam_save_path = os.path.join(results_dir, "gradcam_samples.png")
    plt.savefig(gradcam_save_path)
    plt.close()
    print(f"Saved Grad-CAM visualizations to {gradcam_save_path}")
    
if __name__ == "__main__":
    main()
