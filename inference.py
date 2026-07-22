import os
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from model import get_model, BrainTumorEnsemble

class BrainTumorInference:
    def __init__(self, checkpoint_paths, device="cpu"):
        self.device = device
        self.models = []
        
        # Load models
        for path in checkpoint_paths:
            if os.path.exists(path):
                print(f"Loading checkpoint from {path}")
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                state_dict = checkpoint['model_state_dict']
                
                # Dynamically infer in_channels, num_classes, and head type from state_dict
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
                
                model = get_model(num_classes=num_classes, in_channels=in_channels, pretrained=False, use_mlp_head=use_mlp_head)
                model.load_state_dict(state_dict)
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


            
    def predict_slice(self, slice_4ch_data, pre_normalized=True):
        """
        Predicts probabilities for a single 2D axial slice across 4 modalities:
        (FLAIR, T1w, T1gd, T2w) or 3 modalities if model is in_channels=3.
        """
        model_ref = self.models[0]
        model_in_channels = getattr(model_ref, "in_channels", 4)
        
        if model_in_channels == 3:
            # Extract FLAIR (0), T1gd (2), T2w (3)
            slice_data = slice_4ch_data[[0, 2, 3]]
        else:
            slice_data = slice_4ch_data
            
        if not pre_normalized:
            channels = []
            for ch_idx in range(slice_data.shape[0]):
                ch = slice_data[ch_idx]
                ch_min, ch_max = ch.min(), ch.max()
                if ch_max > ch_min:
                    ch = (ch - ch_min) / (ch_max - ch_min)
                else:
                    ch = np.zeros_like(ch)
                channels.append(ch)
            slice_tensor_data = np.stack(channels, axis=0).astype(np.float32)
        else:
            slice_tensor_data = slice_data.astype(np.float32)
            
        input_tensor = torch.tensor(slice_tensor_data)
        input_tensor = TF.resize(input_tensor, [128, 128])
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            if isinstance(self.model, BrainTumorEnsemble):
                probs = outputs
            else:
                if outputs.shape[1] == 3:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = torch.softmax(outputs, dim=1)
                
            prob_np = probs.cpu().squeeze().numpy()
            if len(prob_np.shape) == 0:
                prob_np = np.array([prob_np])
            pred_res = prob_np.argmax() if outputs.shape[1] != 3 else (prob_np >= 0.5).astype(int)
            
        return pred_res, prob_np


    def predict_volume(self, nifti_image_path):
        """
        Loads a 4D NIfTI MRI volume, predicts tumor component probabilities for each axial slice,
        and returns slice indices, predicted classes/vectors, and probabilities.
        """
        img_nii = nib.load(nifti_image_path)
        img_data = img_nii.get_fdata(dtype=np.float32) # Shape: (H, W, D, 4)
        
        # Determine output mode from first model in ensemble
        first_model = self.models[0]
        if hasattr(first_model, "head") and hasattr(first_model.head[-1], "out_features"):
            num_outputs = first_model.head[-1].out_features
        elif hasattr(first_model.model, "fc") and hasattr(first_model.model.fc, "out_features"):
            num_outputs = first_model.model.fc.out_features
        else:
            num_outputs = 4

        is_multilabel = (num_outputs == 3)

        # Normalize 4D volume per modality independently
        for mod_idx in range(4):
            vol = img_data[:, :, :, mod_idx]
            vol_min, vol_max = vol.min(), vol.max()
            if vol_max > vol_min:
                img_data[:, :, :, mod_idx] = (vol - vol_min) / (vol_max - vol_min)
            else:
                img_data[:, :, :, mod_idx] = 0.0
                
        h, w, d = img_data.shape[:3]
        
        predictions = []
        probabilities = []
        slice_indices = []
        
        for z in range(d):
            # Extract 4 modalities: FLAIR (0), T1w (1), T1gd (2), T2w (3) -> (4, H, W)
            slice_4ch = np.stack([img_data[:, :, z, mod] for mod in range(4)], axis=0)
            flair = slice_4ch[0]
            
            # Skip empty slices
            if flair.max() == 0 or (flair > flair.mean()).sum() < 500:
                if is_multilabel:
                    predictions.append(np.array([0, 0, 0]))
                    probabilities.append(np.array([0.0, 0.0, 0.0]))
                else:
                    predictions.append(0)
                    probabilities.append(np.array([1.0, 0.0, 0.0, 0.0]))
                slice_indices.append(z)
                continue
                
            pred_res, probs = self.predict_slice(slice_4ch, pre_normalized=True)
            predictions.append(pred_res)
            probabilities.append(probs)
            slice_indices.append(z)
            
        return np.array(slice_indices), np.array(predictions), np.array(probabilities)

def main():
    parser = argparse.ArgumentParser(description="Predict Brain Tumor Components from 4D NIfTI Volume")
    parser.add_argument("--image_path", type=str, required=True, help="Path to 4D NIfTI image file (.nii.gz)")
    parser.add_argument("--folds", type=str, default="1", help="Comma-separated folds to use as ensemble (e.g. 1 or 1,2,3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    fold_list = [int(f.strip()) for f in args.folds.split(",")]
    checkpoint_paths = []
    for f in fold_list:
        local_p = os.path.join(checkpoint_dir, f"best_model_fold_{f}.pth")
        parent_p = os.path.join(base_dir, "..", "best_model", f"best_model_fold_{f}.pth")
        if os.path.exists(local_p):
            checkpoint_paths.append(local_p)
        elif os.path.exists(parent_p):
            checkpoint_paths.append(parent_p)
        else:
            checkpoint_paths.append(local_p)
    
    inferer = BrainTumorInference(checkpoint_paths, device=args.device)
    
    print(f"Running inference on {args.image_path} ...")
    slices, preds, probs = inferer.predict_volume(args.image_path)
    
    if probs.shape[1] == 3:
        target_names = ["Edema", "Non-Enhancing Tumor", "Enhancing Tumor"]
        print("\n=== Volume Multi-Label Detection Summary ===")
        for i, name in enumerate(target_names):
            count = (preds[:, i] == 1).sum() if len(preds.shape) > 1 else 0
            print(f"  {name}: detected in {count} slices")
            
        plt.figure(figsize=(12, 6))
        for i, name in enumerate(target_names):
            plt.plot(slices, probs[:, i], label=f"Prob({name})", alpha=0.8, lw=2)
            
        plt.title("Multi-Label Tumor Component Probability Profiles across Axial Slices")
        plt.xlabel("Axial Slice Index")
        plt.ylabel("Sigmoid Probability")
    else:
        class_names = ["Healthy/Bg", "Edema", "Non-Enhancing Tumor", "Enhancing Tumor"]
        unique_preds, counts = np.unique(preds, return_counts=True)
        print("\n=== Volume Prediction Summary ===")
        for cls, count in zip(unique_preds, counts):
            print(f"  {class_names[cls]}: {count} slices")
            
        overall_class = int(preds.max())
        print(f"\nOverall Volume Classification: {class_names[overall_class]} (Maximum severity class detected)")
        
        plt.figure(figsize=(12, 6))
        for i, name in enumerate(class_names):
            plt.plot(slices, probs[:, i], label=name, alpha=0.8, lw=2)
            
        plt.title(f"Tumor Class Probability Profile across Axial Slices\nOverall: {class_names[overall_class]}")
        plt.xlabel("Axial Slice Index")
        plt.ylabel("Probability")

    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    
    save_path = os.path.join(results_dir, "volume_prediction_profile.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction profile chart to {save_path}")

if __name__ == "__main__":
    main()
