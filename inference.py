import os
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from model import get_model, BrainTumorEnsemble

class BrainTumorInference:
    def __init__(self, checkpoint_paths, device="cpu"):
        self.device = device
        self.models = []
        
        # Load models
        for path in checkpoint_paths:
            if os.path.exists(path):
                print(f"Loading checkpoint from {path}")
                model = get_model(num_classes=4, pretrained=False)
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models.append(model.to(device))
            else:
                print(f"Warning: Checkpoint not found at {path}")
                
        if len(self.models) == 0:
            raise FileNotFoundError("No valid checkpoints loaded.")
            
        if len(self.models) > 1:
            print(f"Loaded ensemble of {len(self.models)} models.")
            self.model = BrainTumorEnsemble(self.models)
        else:
            self.model = self.models[0]
            
    def predict_slice(self, flair, t1gd, t2w):
        """
        Predicts the class of a single 2D axial slice.
        Channels: FLAIR, T1gd, T2w.
        """
        # Normalize each channel individually to [0, 1]
        channels = []
        for ch in [flair, t1gd, t2w]:
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                ch = (ch - ch_min) / (ch_max - ch_min)
            else:
                ch = np.zeros_like(ch)
            channels.append(ch)
            
        slice_3ch = np.stack(channels, axis=0).astype(np.float32)
        input_tensor = torch.tensor(slice_3ch).unsqueeze(0).to(self.device) # (1, 3, H, W)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            if isinstance(self.model, BrainTumorEnsemble):
                # Ensemble returns probabilities
                probs = outputs
            else:
                # Single model returns logits
                probs = torch.softmax(outputs, dim=1)
                
            prob_np = probs.cpu().squeeze().numpy()
            pred_class = int(prob_np.argmax())
            
        return pred_class, prob_np

    def predict_volume(self, nifti_image_path):
        """
        Loads a 3D/4D NIfTI MRI volume, predicts the tumor class for each axial slice,
        and returns slice indices, predicted classes, and probabilities.
        """
        img_nii = nib.load(nifti_image_path)
        img_data = img_nii.get_fdata() # Shape: (H, W, D, 4)
        
        h, w, d = img_data.shape[:3]
        
        predictions = []
        probabilities = []
        slice_indices = []
        
        for z in range(d):
            # FLAIR (0), T1gd (2), T2w (3)
            flair = img_data[:, :, z, 0]
            t1gd = img_data[:, :, z, 2]
            t2w = img_data[:, :, z, 3]
            
            # Skip empty slices
            if flair.max() == 0 or (flair > flair.mean()).sum() < 500:
                predictions.append(0) # Default to background
                probabilities.append(np.array([1.0, 0.0, 0.0, 0.0]))
                slice_indices.append(z)
                continue
                
            pred_cls, probs = self.predict_slice(flair, t1gd, t2w)
            predictions.append(pred_cls)
            probabilities.append(probs)
            slice_indices.append(z)
            
        return np.array(slice_indices), np.array(predictions), np.array(probabilities)

def main():
    parser = argparse.ArgumentParser(description="Predict Brain Tumor Classes from NIfTI Volume")
    parser.add_argument("--image_path", type=str, required=True, help="Path to 3D NIfTI image file (.nii.gz)")
    parser.add_argument("--folds", type=str, default="1", help="Comma-separated folds to use as ensemble (e.g. 1 or 1,2,3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get checkpoint paths
    fold_list = [int(f.strip()) for f in args.folds.split(",")]
    checkpoint_paths = [os.path.join(checkpoint_dir, f"best_model_fold_{f}.pth") for f in fold_list]
    
    inferer = BrainTumorInference(checkpoint_paths, device=args.device)
    
    print(f"Running inference on {args.image_path} ...")
    slices, preds, probs = inferer.predict_volume(args.image_path)
    
    # Class names
    class_names = ["Healthy/Bg", "Edema", "Non-Enhancing Tumor", "Enhancing Tumor"]
    
    # Report summary
    unique_preds, counts = np.unique(preds, return_counts=True)
    print("\n=== Volume Prediction Summary ===")
    for cls, count in zip(unique_preds, counts):
        print(f"  {class_names[cls]}: {count} slices")
        
    overall_class = int(preds.max())
    print(f"\nOverall Volume Classification: {class_names[overall_class]} (Maximum severity class detected)")
    
    # Plot classification profile along slice axis
    plt.figure(figsize=(12, 6))
    
    # Plot probability curves for each class
    for i, name in enumerate(class_names):
        plt.plot(slices, probs[:, i], label=name, alpha=0.8)
        
    plt.title(f"Tumor Class Probability Profile across Axial Slices\nOverall: {class_names[overall_class]}")
    plt.xlabel("Axial Slice Index")
    plt.ylabel("Probability")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    
    save_path = os.path.join(results_dir, f"volume_prediction_profile.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction profile chart to {save_path}")

if __name__ == "__main__":
    main()
