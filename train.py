import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from model import get_model
from crossval import get_patient_folds

# Custom Dataset
class BrainTumorSliceDataset(torch.utils.data.Dataset):
    def __init__(self, patient_list, processed_dir, transform=None, cache=True, image_size=128):
        self.processed_dir = processed_dir
        self.transform = transform
        self.cache = cache
        self.image_size = image_size
        self.samples = []
        
        # Load metadata
        with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        for p_id in patient_list:
            if p_id in metadata:
                slices = metadata[p_id]["slices"]
                for cls_str, slice_list in slices.items():
                    cls = int(cls_str)
                    for z in slice_list:
                        slice_path = os.path.join(processed_dir, f"class_{cls}", f"{p_id}_slice_{z}.npy")
                        if os.path.exists(slice_path):
                            self.samples.append((slice_path, cls))
                            
        # Cache samples in memory if enabled
        if self.cache:
            print(f"Caching {len(self.samples)} samples in memory (resized to {self.image_size}x{self.image_size})...")
            self.cached_images = []
            for slice_path, _ in tqdm(self.samples, desc="Caching dataset", leave=False):
                image = np.load(slice_path)
                image_tensor = torch.tensor(image, dtype=torch.float32)
                if self.image_size is not None:
                    image_tensor = TF.resize(image_tensor, [self.image_size, self.image_size])
                self.cached_images.append(image_tensor)
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        slice_path, label = self.samples[idx]
        
        if self.cache:
            image = self.cached_images[idx].clone() # Clone to avoid modifying cache in-place
        else:
            image = np.load(slice_path)
            image = torch.tensor(image, dtype=torch.float32)
            if self.image_size is not None:
                image = TF.resize(image, [self.image_size, self.image_size])
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ── Custom transforms (operate on float tensors before Normalize) ─────────────

# ImageNet statistics used by all pretrained torchvision models.
# MUST be the LAST transform applied to training images, and the ONLY
# transform applied to validation images.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class RandomAddNoise:
    """Additive Gaussian noise — simulates MRI acquisition noise."""
    def __init__(self, std=0.03):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


class RandomBiasField:
    """
    Simulates MRI bias field (smooth low-frequency intensity inhomogeneity).
    Multiplies each channel by a random scalar in [1-strength, 1+strength].
    This is the lightweight approximation of the spatial bias field artefact.
    """
    def __init__(self, strength=0.15):
        self.strength = strength
    def __call__(self, tensor):
        # Independent per-channel multiplicative factor
        factors = 1.0 + (torch.rand(tensor.shape[0], 1, 1) * 2 - 1) * self.strength
        return tensor * factors


class RandomModalityReplicate:
    """
    Randomly (with probability p) selects one of the 3 channels (FLAIR, T1gd, T2w)
    and replicates it across all 3 channels to simulate single-channel sequence inputs.
    This helps the model generalize to single-sequence grayscale images (like external test sets).
    """
    def __init__(self, p=0.3):
        self.p = p
    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            idx = torch.randint(0, 3, (1,)).item()
            return tensor[idx:idx+1].repeat(3, 1, 1)
        return tensor

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    val_loss = running_loss / len(loader.dataset)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Calculate macro F1
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Calculate AUC
    try:
        val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception:
        val_auc = 0.0
        
    return val_loss, val_acc, val_f1, val_auc, all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on Brain Tumor Slices")
    parser.add_argument("--fold", type=int, default=1, help="Which fold to train (1-5)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--quick", action="store_true", help="Run a quick training for testing/CPU execution")
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed_2d")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    metadata_path = os.path.join(processed_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Processed dataset metadata not found at {metadata_path}. Please run preprocess_data.py first.")
        
    # Get patient splits for cross validation
    folds = get_patient_folds(metadata_path, n_splits=5)
    train_patients, val_patients = folds[args.fold - 1]
    
    print(f"Fold {args.fold}: Training on {len(train_patients)} patients, validating on {len(val_patients)} patients")
    
    # ── Transform definitions ──────────────────────────────────────────────────
    # IMPORTANT: Order matters.
    #   1. Geometric augmentations (flip, rotate) — applied to spatial layout.
    #   2. Intensity augmentations (blur, noise, bias field) — applied to values.
    #   3. RandomErasing — drops patches to force the model not to rely on them.
    #   4. RandomModalityReplicate — helps generalize to external single-sequence data.
    #   5. T.Normalize(ImageNet) — MUST be LAST; backbone expects this range.
    #
    # NOTE: T.ColorJitter was removed. It expects uint8 or values in [0,1] before
    #       normalization. Its effect is covered by RandomBiasField + RandomAddNoise,
    #       which are MRI-appropriate equivalents.
    train_transform = T.Compose([
        # --- Geometric ---
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20, interpolation=T.InterpolationMode.BILINEAR),
        # --- Intensity / MRI-specific ---
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        RandomBiasField(strength=0.15),
        RandomAddNoise(std=0.03),
        # --- Structural dropout ---
        T.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
        # --- Modality simulation ---
        RandomModalityReplicate(p=0.3),
        # --- ImageNet normalization (MUST be last) ---
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Validation: only normalize — no augmentation.
    val_transform = T.Compose([
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Create datasets
    train_dataset = BrainTumorSliceDataset(train_patients, processed_dir, transform=train_transform)
    val_dataset   = BrainTumorSliceDataset(val_patients,   processed_dir, transform=val_transform)
    
    if args.quick:
        print("--> Quick mode active: subsetting datasets for fast test run.")
        train_dataset.samples = train_dataset.samples[:128]
        val_dataset.samples = val_dataset.samples[:32]
        
    print(f"Dataset sizes: Train = {len(train_dataset)} slices, Val = {len(val_dataset)} slices")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Compute class weights to handle class imbalance
    class_counts = [0] * 4
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    total_samples = sum(class_counts)
    
    # Weighted Cross Entropy weights = total / (num_classes * class_count)
    class_weights = []
    for count in class_counts:
        weight = total_samples / (4.0 * count) if count > 0 else 1.0
        class_weights.append(weight)
        
    print(f"Class counts in training set: {class_counts}")
    print(f"Calculated class weights: {class_weights}")
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
    # label_smoothing=0.1 prevents overconfident predictions and improves calibration.
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    # Initialize model
    model = get_model(num_classes=4, pretrained=True).to(args.device)

    # ── Two-Stage Fine-Tuning with LLRD ───────────────────────────────────────
    # Stage 1 (epochs 1..WARMUP_FREEZE_EPOCHS):
    #   Backbone is FROZEN. Only the new classification head is trained at a
    #   high LR (1e-3). This lets the head stabilize before the pretrained
    #   backbone weights are touched.
    #
    # Stage 2 (epochs WARMUP_FREEZE_EPOCHS+1..end):
    #   All layers are unfrozen with LLRD:
    #     - Head:   args.lr        (highest — newly initialized)
    #     - layer4: 0.1  × args.lr (fine-tune last residual block)
    #     - layer3: 0.05 × args.lr
    #     - layer2: 0.02 × args.lr
    #     - stem:   0.01 × args.lr (preserve early feature detectors)
    WARMUP_FREEZE_EPOCHS = 5

    # Start with backbone frozen
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

    # Cosine annealing over the full training run (restarts are handled by the epoch logic)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - WARMUP_FREEZE_EPOCHS, eta_min=1e-6
    )

    print(f"Stage 1: Training head only for {WARMUP_FREEZE_EPOCHS} epochs (backbone frozen).")
    
    # Optional W&B logger setup
    if args.wandb:
        import wandb
        wandb.init(
            project="brain-tumor-cnn",
            name=f"resnet18-fold-{args.fold}",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "fold": args.fold,
            }
        )
        
    best_val_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # ── Stage transition: unfreeze backbone at epoch WARMUP_FREEZE_EPOCHS+1 ──
        if epoch == WARMUP_FREEZE_EPOCHS + 1:
            print(f"\nStage 2: Unfreezing all layers with LLRD (base_lr={args.lr}).")
            for param in model.parameters():
                param.requires_grad = True
            param_groups = model.get_parameter_groups(base_lr=args.lr)
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            # Re-initialise scheduler for the remaining epochs
            remaining = args.epochs - WARMUP_FREEZE_EPOCHS
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-6
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc, val_f1, val_auc, val_labels, val_preds = validate(model, val_loader, criterion, args.device)

        # Only step cosine scheduler during Stage 2
        if epoch > WARMUP_FREEZE_EPOCHS:
            scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
              
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{args.fold}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"--> Saved best model checkpoint to {checkpoint_path}")
            
    print(f"\nTraining for Fold {args.fold} finished. Best Val F1: {best_val_f1:.4f}")
    if args.wandb:
        wandb.finish()
        
if __name__ == "__main__":
    main()
