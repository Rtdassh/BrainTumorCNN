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
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)

from model import get_model
from train import BrainTumorSliceDataset
from crossval import get_patient_folds
from tqdm import tqdm

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(self.save_gradient)
        else:
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
        
        grads = self.gradients
        acts = self.activations
        
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.squeeze().detach().cpu().numpy(), class_idx

def get_center_id(patient_name):
    """
    Groups scanner cohorts:
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

def plot_confusion_matrices(cm, class_names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Raw Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Class")
    axes[0].set_xlabel("Predicted Class")
    
    # 2. Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                cbar_kws={'label': 'Proportion'})
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Class")
    axes[1].set_xlabel("Predicted Class")
    
    plt.suptitle("Slice-Level Model Performance Confusion Matrices", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_pr_curves(all_labels, all_probs, class_names, save_path):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. ROC Curves
    for i in range(n_classes):
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color=colors[i], lw=2.5,
                     label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC)')
    axes[0].legend(loc="lower right")
    
    # 2. Precision-Recall Curves
    for i in range(n_classes):
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        ap = average_precision_score(y_true_binary, y_score)
        
        axes[1].plot(recall, precision, color=colors[i], lw=2.5,
                     label=f'{class_names[i]} (AP = {ap:.3f})')
        
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall (PR) Curves')
    axes[1].legend(loc="lower left")
    
    plt.suptitle("Model Evaluation Curves", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_per_class_metrics(all_labels, all_preds, class_names, save_path):
    precisions = precision_score(all_labels, all_preds, average=None)
    recalls = recall_score(all_labels, all_preds, average=None)
    f1s = f1_score(all_labels, all_preds, average=None)
    
    # Calculate accuracy per class: (TP + TN) / Total
    accuracies = []
    for i in range(len(class_names)):
        tp = np.sum((all_labels == i) & (all_preds == i))
        tn = np.sum((all_labels != i) & (all_preds != i))
        accuracies.append((tp + tn) / len(all_labels))
        
    x = np.arange(len(class_names))
    width = 0.2
    
    plt.figure(figsize=(12, 7))
    plt.bar(x - 1.5 * width, accuracies, width, label='Accuracy', color='#34495e')
    plt.bar(x - 0.5 * width, precisions, width, label='Precision', color='#3498db')
    plt.bar(x + 0.5 * width, recalls, width, label='Recall', color='#2ecc71')
    plt.bar(x + 1.5 * width, f1s, width, label='F1-Score', color='#e74c3c')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Classification Performance Metrics per Class')
    plt.xticks(x, class_names)
    plt.ylim([0, 1.05])
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_center_performance(centers, all_labels, all_preds, save_path):
    unique_centers = sorted(list(set(centers)))
    accs = []
    f1s = []
    slice_counts = []
    
    for center in unique_centers:
        indices = [i for i, c in enumerate(centers) if c == center]
        c_labels = all_labels[indices]
        c_preds = all_preds[indices]
        
        accs.append(accuracy_score(c_labels, c_preds))
        f1s.append(f1_score(c_labels, c_preds, average='macro'))
        slice_counts.append(len(indices))
        
    x = np.arange(len(unique_centers))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    rects1 = ax1.bar(x - width/2, accs, width, label='Accuracy', color='#2980b9')
    rects2 = ax1.bar(x + width/2, f1s, width, label='Macro F1', color='#e67e22')
    
    ax1.set_xlabel('Clinical Center / Cohort')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Generalization Breakdown by Clinical Cohort')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n({n} slices)" for c, n in zip(unique_centers, slice_counts)])
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc='lower left')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confidence_distributions(all_labels, all_preds, all_probs, class_names, save_path):
    plt.figure(figsize=(14, 7))
    
    data_to_plot = []
    labels = []
    
    for i, name in enumerate(class_names):
        # Confidence is probability of the predicted class
        indices_correct = np.where((all_labels == i) & (all_preds == i))[0]
        indices_incorrect = np.where((all_labels == i) & (all_preds != i))[0]
        
        if len(indices_correct) > 0:
            probs_correct = all_probs[indices_correct, i]
            data_to_plot.append(probs_correct)
            labels.append(f"{name}\nCorrect\n(N={len(probs_correct)})")
            
        if len(indices_incorrect) > 0:
            # Probability model assigned to the true class even though it misclassified it
            probs_incorrect = all_probs[indices_incorrect, i]
            data_to_plot.append(probs_incorrect)
            labels.append(f"{name}\nIncorrect\n(N={len(probs_incorrect)})")
            
    # Boxplot of probabilities
    box = plt.boxplot(data_to_plot, patch_artist=True)
    
    # Color coding
    colors = []
    for label in labels:
        if "Correct" in label:
            colors.append('#2ecc71') # Green for correct
        else:
            colors.append('#e74c3c') # Red for incorrect
            
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    plt.ylabel('Model Probability Assigned to True Class')
    plt.title('Distribution of True Class Probability for Correct vs. Incorrect Predictions')
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_accuracy_vs_confidence(all_labels, all_preds, all_probs, save_path):
    # Confidence is max predicted probability
    max_probs = np.max(all_probs, axis=1)
    
    thresholds = np.linspace(0.0, 0.95, 20)
    accuracies = []
    retained_fractions = []
    
    for t in thresholds:
        retained_idx = np.where(max_probs >= t)[0]
        if len(retained_idx) > 0:
            acc = accuracy_score(all_labels[retained_idx], all_preds[retained_idx])
            frac = len(retained_idx) / len(all_labels)
            accuracies.append(acc)
            retained_fractions.append(frac)
        else:
            accuracies.append(1.0)
            retained_fractions.append(0.0)
            
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = '#1abc9c'
    ax1.set_xlabel('Confidence Threshold (Minimum Probability)')
    ax1.set_ylabel('Validation Accuracy', color=color)
    line1 = ax1.plot(thresholds, accuracies, color=color, marker='o', lw=2.5, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.4, 1.02])
    
    ax2 = ax1.twinx()  
    color = '#9b59b6'
    ax2.set_ylabel('Percentage of Slices Retained', color=color)
    line2 = ax2.plot(thresholds, retained_fractions, color=color, marker='x', lw=2, linestyle='--', label='% Retained')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([-0.05, 1.05])
    
    # added these lines
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    
    plt.title('Validation Accuracy & Sample Retention rate vs. Confidence Threshold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_patient_accuracy_distribution(patient_accuracies, save_path):
    plt.figure(figsize=(12, 7))
    
    # Histogram
    plt.hist(list(patient_accuracies.values()), bins=10, range=(0.0, 1.0), 
             color='#16a085', edgecolor='white', rwidth=0.9, alpha=0.8)
    
    plt.xlabel('Patient Slice-Level Accuracy (Proportion of Slices Correct)')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of Model Prediction Accuracy Across Validation Patients')
    plt.axvline(np.mean(list(patient_accuracies.values())), color='#c0392b', linestyle='dashed', linewidth=2.5, 
                label=f'Mean Accuracy = {np.mean(list(patient_accuracies.values())):.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_patient_confusion_matrix(y_true, y_pred, class_names, save_path):
    plt.figure(figsize=(8, 7))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Patients'})
    plt.title("Patient-Level Diagnosis Confusion Matrix\n(Average Probability Aggregation)")
    plt.ylabel("True Patient Diagnosis")
    plt.xlabel("Predicted Patient Diagnosis")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_gradcam_visualizations(model, val_dataset, all_preds, class_names, device, save_path):
    print("Generating Grad-CAM Visualizations...")
    # Last conv layer of ResNet-18
    target_layer = model.model.layer4[1].conv2
    grad_cam = GradCAM(model, target_layer)
    
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
            
            input_tensor = torch.tensor(image_np).unsqueeze(0).to(device)
            
            # Generate CAM
            cam, predicted_cls = grad_cam(input_tensor, class_idx=cls)
            
            # FLAIR sequence is channel 0
            flair_bg = image_np[0]
            
            # Left: Original image
            axes[cls, 0].imshow(flair_bg, cmap='gray')
            axes[cls, 0].set_title(f"Original (True: {class_names[cls]})")
            axes[cls, 0].axis('off')
            
            # Right: Grad-CAM Overlay
            axes[cls, 1].imshow(flair_bg, cmap='gray')
            axes[cls, 1].imshow(cam, cmap='jet', alpha=0.5)
            axes[cls, 1].set_title(f"Grad-CAM (Attention Map: {class_names[cls]})")
            axes[cls, 1].axis('off')
            
    plt.suptitle("Grad-CAM Localization Saliency Maps", y=0.99, fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Grad-CAM saliency maps to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Detailed Model Diagnostics and Visualization Suite")
    parser.add_argument("--checkpoint", type=str, 
                        default="c:/Users/JDan/Documents/NeuroMatch/Dataset/best_model/best_model_fold_1.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed_2d")
    results_dir = os.path.join(base_dir, "analysis_results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        # Fallback to local checkpoints
        fallback = os.path.join(base_dir, "checkpoints", "best_model_fold_1.pth")
        print(f"Primary checkpoint not found. Trying fallback: {fallback}")
        if os.path.exists(fallback):
            args.checkpoint = fallback
        else:
            raise FileNotFoundError(f"No checkpoint file found at {args.checkpoint} or {fallback}")
            
    # Load dataset
    metadata_path = os.path.join(processed_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Processed dataset metadata not found. Please run preprocess_data.py first.")
        
    folds = get_patient_folds(metadata_path, n_splits=5)
    _, val_patients = folds[0] # Using fold 1 corresponding to the checkpoint
    
    print(f"Loading validation slices for {len(val_patients)} patients...")
    val_dataset = BrainTumorSliceDataset(val_patients, processed_dir, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(f"Total validation slices: {len(val_dataset)}")
    
    # Inspect and load Model
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    conv1_w = state_dict.get("model.conv1.weight", state_dict.get("conv1.weight", None))
    in_channels = conv1_w.shape[1] if conv1_w is not None else 4
    
    if "model.fc.weight" in state_dict:
        num_classes = state_dict["model.fc.weight"].shape[0]
        use_mlp_head = False
    elif "fc.weight" in state_dict:
        num_classes = state_dict["fc.weight"].shape[0]
        use_mlp_head = False
    else:
        head_w = state_dict.get("head.4.weight", state_dict.get("model.fc.3.weight", None))
        num_classes = head_w.shape[0] if head_w is not None else 3
        use_mlp_head = True

    is_multilabel = (num_classes == 3)
    class_names = ["Edema", "Non-Enhancing Core", "Enhancing Core"] if is_multilabel else ["Healthy/Bg", "Edema", "Non-Enhancing", "Enhancing"]

    model = get_model(num_classes=num_classes, in_channels=in_channels, pretrained=False, use_mlp_head=use_mlp_head).to(args.device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint (Epoch {checkpoint.get('epoch', 'N/A')}, Val F1: {checkpoint.get('val_f1', 0.0):.4f})")
    print(f"Model Configuration: in_channels={in_channels}, num_classes={num_classes}, is_multilabel={is_multilabel}")

    # Run Inference
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    
    for idx, (img_path, label) in enumerate(val_dataset.samples):
        p_id = os.path.basename(img_path).split("_slice_")[0]
        all_patient_ids.append(p_id)
        
    print("Running validation inference...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Inference"):
            if in_channels == 3 and images.shape[1] == 4:
                images = images[:, [0, 2, 3]]
            images = images.to(args.device)
            
            outputs = model(images)
            if is_multilabel:
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
            else:
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_preds = np.vstack(all_preds) if is_multilabel else np.concatenate(all_preds)
    all_labels = np.vstack(all_labels) if len(all_labels[0].shape) > 0 else np.concatenate(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Harmonize label format with model output type
    if not is_multilabel and len(all_labels.shape) > 1 and all_labels.shape[1] == 3:
        true_scalar = []
        for vec in all_labels:
            if vec[2] == 1:
                true_scalar.append(3) # Enhancing
            elif vec[1] == 1:
                true_scalar.append(2) # Non-enhancing
            elif vec[0] == 1:
                true_scalar.append(1) # Edema
            else:
                true_scalar.append(0) # Healthy/Bg
        all_labels = np.array(true_scalar)
    elif is_multilabel and len(all_labels.shape) == 1:
        true_vec = []
        for cls in all_labels:
            if cls == 3:
                true_vec.append([0, 0, 1])
            elif cls == 2:
                true_vec.append([0, 1, 0])
            elif cls == 1:
                true_vec.append([1, 0, 0])
            else:
                true_vec.append([0, 0, 0])
        all_labels = np.array(true_vec)

    
    # ------------------ SLICE LEVEL ANALYTICS ------------------
    print("\n--- Generating Slice-Level Visualizations ---")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrices(cm, class_names, os.path.join(results_dir, "confusion_matrix_detailed.png"))
    print("Saved confusion_matrix_detailed.png")
    
    # ROC and PR Curves
    plot_roc_pr_curves(all_labels, all_probs, class_names, os.path.join(results_dir, "roc_pr_curves.png"))
    print("Saved roc_pr_curves.png")
    
    # Per-Class Metrics Bar Chart
    plot_per_class_metrics(all_labels, all_preds, class_names, os.path.join(results_dir, "per_class_metrics.png"))
    print("Saved per_class_metrics.png")
    
    # Center Generalization
    centers = [get_center_id(pid) for pid in all_patient_ids]
    plot_center_performance(centers, all_labels, all_preds, os.path.join(results_dir, "center_performance.png"))
    print("Saved center_performance.png")
    
    # Confidence distribution
    plot_confidence_distributions(all_labels, all_preds, all_probs, class_names, 
                                  os.path.join(results_dir, "confidence_distributions.png"))
    print("Saved confidence_distributions.png")
    
    # Accuracy vs Confidence Threshold Curve
    plot_accuracy_vs_confidence(all_labels, all_preds, all_probs, os.path.join(results_dir, "accuracy_vs_confidence.png"))
    print("Saved accuracy_vs_confidence.png")
    
    # ------------------ PATIENT LEVEL AGGREGATION ------------------
    print("\n--- Performing Patient-Level Aggregation ---")
    
    patient_slices = {}
    for idx, pid in enumerate(all_patient_ids):
        if pid not in patient_slices:
            patient_slices[pid] = []
        patient_slices[pid].append({
            'true_label': all_labels[idx],
            'pred_label': all_preds[idx],
            'probs': all_probs[idx]
        })
        
    patient_true = []
    patient_pred_voting = []
    patient_pred_prob = []
    patient_pred_max = [] # Method C
    patient_accuracies = {}
    
    # Diagnostic detailed patient info
    patient_summary = []
    
    for pid, slices in patient_slices.items():
        # Get true patient label (aggregate category: typically patients have slices across classes.
        # But let's define the overall true patient diagnosis as the maximum severity class:
        # 0: Healthy, 1: Edema, 2: Non-enhancing, 3: Enhancing
        # This mirrors clinical staging where the most severe tissue defines the classification.
        true_labels = [s['true_label'] for s in slices]
        overall_true = int(np.max(true_labels))
        
        # Calculate slice accuracy for this patient
        preds = [s['pred_label'] for s in slices]
        correct = sum([1 for s in slices if s['pred_label'] == s['true_label']])
        patient_acc = correct / len(slices)
        patient_accuracies[pid] = patient_acc
        
        # Method 1: Majority Vote (most common predicted slice class)
        # In clinical settings, we exclude healthy background slices (class 0) from the voting
        # to ensure small enhancing cores are not washed out by healthy tissues.
        # So we vote among tumor classes if any tumor is predicted, otherwise default to healthy.
        tumor_preds = [p for p in preds if p > 0]
        if len(tumor_preds) > 0:
            overall_pred_vote = int(np.bincount(tumor_preds).argmax())
        else:
            overall_pred_vote = int(np.bincount(preds).argmax())
            
        # Method 2: Mean Probability Aggregation (sum and argmax)
        all_probs_arr = np.array([s['probs'] for s in slices])
        # Average probability across all slices
        mean_probs = np.mean(all_probs_arr, axis=0)
        # Argmax of the averaged probabilities
        overall_pred_prob = int(np.argmax(mean_probs))
        
        # Method 3: Maximum Severity Aggregation (highest class predicted among slices)
        overall_pred_max = int(np.max(preds))
        
        patient_true.append(overall_true)
        patient_pred_voting.append(overall_pred_vote)
        patient_pred_prob.append(overall_pred_prob)
        patient_pred_max.append(overall_pred_max)
        
        patient_summary.append({
            'patient_id': pid,
            'center': get_center_id(pid),
            'num_slices': len(slices),
            'slice_accuracy': patient_acc,
            'true_class': class_names[overall_true],
            'pred_vote': class_names[overall_pred_vote],
            'pred_prob': class_names[overall_pred_prob],
            'pred_max': class_names[overall_pred_max],
            'mean_probs': [round(float(p), 4) for p in mean_probs]
        })
        
    patient_true = np.array(patient_true)
    patient_pred_voting = np.array(patient_pred_voting)
    patient_pred_prob = np.array(patient_pred_prob)
    patient_pred_max = np.array(patient_pred_max)
    
    # Plot Patient Accuracy Distribution
    plot_patient_accuracy_distribution(patient_accuracies, os.path.join(results_dir, "patient_slice_accuracy_distribution.png"))
    print("Saved patient_slice_accuracy_distribution.png")
    
    # Plot Patient Confusion Matrix (Using recommended Method C)
    plot_patient_confusion_matrix(patient_true, patient_pred_max, class_names, 
                                  os.path.join(results_dir, "patient_confusion_matrix.png"))
    print("Saved patient_confusion_matrix.png")
    
    # Generate Grad-CAM sample visualizations
    generate_gradcam_visualizations(model, val_dataset, all_preds, class_names, args.device,
                                    os.path.join(results_dir, "gradcam_saliency.png"))
    
    # ------------------ TEXT REPORT GENERATION ------------------
    report_path = os.path.join(results_dir, "patient_analysis_report.txt")
    print(f"Writing detailed text analysis report to {report_path}...")
    
    # Compute metrics
    slice_acc = accuracy_score(all_labels, all_preds)
    slice_f1 = f1_score(all_labels, all_preds, average='macro')
    
    pat_vote_acc = accuracy_score(patient_true, patient_pred_voting)
    pat_vote_f1 = f1_score(patient_true, patient_pred_voting, average='macro')
    
    pat_prob_acc = accuracy_score(patient_true, patient_pred_prob)
    pat_prob_f1 = f1_score(patient_true, patient_pred_prob, average='macro')
    
    pat_max_acc = accuracy_score(patient_true, patient_pred_max)
    pat_max_f1 = f1_score(patient_true, patient_pred_max, average='macro')
    
    # Sort patients by slice accuracy to find hard cases
    hard_patients = sorted(patient_summary, key=lambda x: x['slice_accuracy'])[:10]
    
    with open(report_path, "w") as f:
        f.write("======================================================================\n")
        f.write("             CLINICAL BRAIN TUMOR MODEL ANALYSIS REPORT               \n")
        f.write("======================================================================\n\n")
        f.write(f"Checkpoint evaluated: {args.checkpoint}\n")
        f.write(f"Total Validation Patients: {len(patient_slices)}\n")
        f.write(f"Total Validation Slices: {len(all_labels)}\n\n")
        
        f.write("----------------------------------------------------------------------\n")
        f.write(" 1. SLICE-LEVEL DIAGNOSTIC SUMMARY\n")
        f.write("----------------------------------------------------------------------\n")
        f.write(f"Overall Slice Accuracy: {slice_acc:.4f}\n")
        f.write(f"Overall Slice Macro F1: {slice_f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
        f.write("\n")
        
        f.write("----------------------------------------------------------------------\n")
        f.write(" 2. PATIENT-LEVEL AGGREGATION DIAGNOSTIC SUMMARY\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("Clinically, 2D slice predictions are aggregated to make a single diagnosis\n")
        f.write("decision for the patient.\n\n")
        
        f.write("Method A: Majority Voting among slice predictions\n")
        f.write(f"  - Patient-Level Accuracy: {pat_vote_acc:.4f}\n")
        f.write(f"  - Patient-Level Macro F1: {pat_vote_f1:.4f}\n\n")
        f.write("Method B: Average Probabilities across slices\n")
        f.write(f"  - Patient-Level Accuracy: {pat_prob_acc:.4f}\n")
        f.write(f"  - Patient-Level Macro F1: {pat_prob_f1:.4f}\n\n")
        f.write("Method C: Maximum Severity Aggregation (Recommended for staging)\n")
        f.write(f"  - Patient-Level Accuracy: {pat_max_acc:.4f}\n")
        f.write(f"  - Patient-Level Macro F1: {pat_max_f1:.4f}\n\n")
        
        f.write("Patient-Level Classification Report (Average Probabilities):\n")
        f.write(classification_report(patient_true, patient_pred_prob, target_names=class_names, labels=list(range(len(class_names))), zero_division=0))
        f.write("\n")
        
        f.write("Patient-Level Classification Report (Maximum Severity):\n")
        f.write(classification_report(patient_true, patient_pred_max, target_names=class_names, labels=list(range(len(class_names))), zero_division=0))
        f.write("\n")

        
        f.write("----------------------------------------------------------------------\n")
        f.write(" 3. CLINICAL SUBPOPULATION ANALYSIS (COHORT BIAS / ROBUSTNESS)\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("Performance breakdown across different MRI scanners and clinical cohorts:\n\n")
        
        unique_centers = sorted(list(set(centers)))
        for center in unique_centers:
            indices = [i for i, c in enumerate(centers) if c == center]
            c_labels = all_labels[indices]
            c_preds = all_preds[indices]
            c_acc = accuracy_score(c_labels, c_preds)
            c_f1 = f1_score(c_labels, c_preds, average='macro')
            
            # Count of patients in this center
            c_patients = set([pid for pid in all_patient_ids if get_center_id(pid) == center])
            
            f.write(f"{center}:\n")
            f.write(f"  Patients: {len(c_patients)} | Slices: {len(indices)}\n")
            f.write(f"  Slice Accuracy: {c_acc:.4f} | Slice Macro F1: {c_f1:.4f}\n\n")
            
        f.write("----------------------------------------------------------------------\n")
        f.write(" 4. HARDEST PATIENTS (LOWEST ACCURACY CASES)\n")
        f.write("----------------------------------------------------------------------\n")
        f.write("These patients have the lowest percentage of correct slice predictions,\n")
        f.write("suggesting challenging boundary conditions or atypical imaging features:\n\n")
        
        f.write(f"{'Patient ID':<15} {'Center':<25} {'Slices':<8} {'Accuracy':<10} {'True':<20} {'Predicted (Prob)':<20} {'Predicted (Max)':<20}\n")
        f.write("-" * 125 + "\n")
        for hp in hard_patients:
            f.write(f"{hp['patient_id']:<15} {hp['center']:<25} {hp['num_slices']:<8} {hp['slice_accuracy']:<10.4f} {hp['true_class']:<20} {hp['pred_prob']:<20} {hp['pred_max']:<20}\n")
            
    print(f"Analysis successfully completed! Report written to {report_path}")
    print("All diagnostic plots are saved in directory: analysis_results/")

if __name__ == "__main__":
    main()
