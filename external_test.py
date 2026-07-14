import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model import get_model, BrainTumorEnsemble

class ExternalTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load 'no' (healthy) images
        no_dir = os.path.join(dataset_dir, "no")
        for img_path in glob.glob(os.path.join(no_dir, "*.*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.samples.append((img_path, 0)) # Class 0: Healthy
                
        # Load 'yes' (tumor) images
        yes_dir = os.path.join(dataset_dir, "yes")
        for img_path in glob.glob(os.path.join(yes_dir, "*.*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.samples.append((img_path, 1)) # Class 1: Tumor (aggregate label for evaluation)
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as grayscale
        img = Image.open(img_path).convert('L')
        
        # We need to output a 3-channel tensor of size 240x240
        # Replicating the grayscale channel across all 3 channels
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # Default transform if none specified
            default_tf = T.Compose([
                T.Resize((240, 240)),
                T.ToTensor(),
            ])
            img_tensor = default_tf(img)
            
        # Replicate to 3 channels: (3, H, W)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
            
        return img_tensor, label, img_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model on External 2D Brain Tumor Dataset")
    parser.add_argument("--folds", type=str, default="1", help="Folds to use for ensemble (e.g. 1 or 1,2,3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "brain_tumor_dataset"))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"External dataset not found at {dataset_dir}")
        
    # Load checkpoints
    fold_list = [int(f.strip()) for f in args.folds.split(",")]
    checkpoint_paths = [os.path.join(checkpoint_dir, f"best_model_fold_{f}.pth") for f in fold_list]
    
    models = []
    for path in checkpoint_paths:
        if os.path.exists(path):
            model = get_model(num_classes=4, pretrained=False)
            checkpoint = torch.load(path, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models.append(model.to(args.device))
            
    if len(models) == 0:
        raise FileNotFoundError("No checkpoints found. Please train models first.")
        
    if len(models) > 1:
        print(f"Using ensemble of {len(models)} models for external validation.")
        eval_model = BrainTumorEnsemble(models)
    else:
        eval_model = models[0]
        
    eval_model.eval()
    
    # Dataset preparation
    transform = T.Compose([
        T.Resize((240, 240)),
        T.ToTensor(),
    ])
    
    dataset = ExternalTestDataset(dataset_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    print(f"Loaded {len(dataset)} external test images.")
    
    true_labels = []
    predicted_binary = []
    predicted_4class = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(args.device)
            outputs = eval_model(images)
            
            if isinstance(eval_model, BrainTumorEnsemble):
                probs = outputs
            else:
                probs = torch.softmax(outputs, dim=1)
                
            preds = probs.argmax(dim=1).cpu().numpy()
            
            # Map 4-class predictions to binary tumor prediction:
            # Class 0 -> 0 (No Tumor)
            # Class 1, 2, 3 -> 1 (Tumor present)
            binary_preds = (preds > 0).astype(int)
            
            true_labels.extend(labels.numpy())
            predicted_binary.extend(binary_preds)
            predicted_4class.extend(preds)
            
    true_labels = np.array(true_labels)
    predicted_binary = np.array(predicted_binary)
    predicted_4class = np.array(predicted_4class)
    
    # Calculate performance metrics
    acc = accuracy_score(true_labels, predicted_binary)
    cm = confusion_matrix(true_labels, predicted_binary)
    report = classification_report(true_labels, predicted_binary, target_names=["No Tumor", "Tumor"])
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Save report
    report_content = (
        "==================================================\n"
        " EXTERNAL DATASET GENERALIZATION REPORT\n"
        "==================================================\n"
        f"Dataset path: {dataset_dir}\n"
        f"Total images: {len(dataset)}\n"
        f"Ensemble models used: Folds {args.folds}\n"
        f"Overall Accuracy: {acc:.4f}\n"
        f"Sensitivity (Recall for Tumor): {sensitivity:.4f}\n"
        f"Specificity (Recall for Healthy): {specificity:.4f}\n\n"
        "Confusion Matrix:\n"
        f"              Predicted No    Predicted Yes\n"
        f"Actual No     {tn:<15} {fp:<15}\n"
        f"Actual Yes    {fn:<15} {tp:<15}\n\n"
        "Detailed Classification Report:\n"
        f"{report}\n"
        "==================================================\n"
    )
    
    print(report_content)
    
    report_path = os.path.join(results_dir, "external_test_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Report successfully saved to {report_path}")

if __name__ == "__main__":
    main()
