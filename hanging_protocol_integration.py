import os
import json
import argparse
import torch
from inference import BrainTumorInference

def generate_hanging_protocol(nifti_path, model_checkpoints, device="cpu"):
    """
    Simulates a clinical PACS (Picture Archiving and Communication System) hanging protocol.
    Automatically categorizes the volume, checks for tumor presence, and generates
    a display layout configuration for radiologists.
    """
    # 1. Run inference to detect tumor slices
    print(f"[PACS] Analyzing volume: {os.path.basename(nifti_path)}...")
    inferer = BrainTumorInference(model_checkpoints, device=device)
    slices, preds, probs = inferer.predict_volume(nifti_path)
    
    # 2. Extract findings
    class_names = ["Healthy/Bg", "Edema", "Non-Enhancing Tumor", "Enhancing Tumor"]
    overall_class = int(preds.max())
    
    # Identify key slices of interest (e.g., maximum probability slice for the tumor classes)
    tumor_slices = {}
    for i in [1, 2, 3]: # Edema, Non-Enhancing, Enhancing
        class_probs = probs[:, i]
        max_prob_idx = int(class_probs.argmax())
        max_prob = float(class_probs[max_prob_idx])
        if max_prob > 0.5:
            tumor_slices[class_names[i]] = {
                "slice_index": max_prob_idx,
                "confidence": round(max_prob, 4)
            }
            
    # 3. Create Hanging Protocol configuration
    # A standard clinical layout uses a 3-monitor setup or a grid layout
    protocol = {
        "protocol_name": "Brain Glioma MRI Workflow Layout",
        "patient_id": os.path.basename(nifti_path).split(".")[0],
        "findings": {
            "overall_severity": class_names[overall_class],
            "severity_code": overall_class,
            "key_slices_detected": tumor_slices
        },
        "monitors": [
            {
                "monitor_id": 1,
                "role": "Edema Localization (T2-FLAIR)",
                "display_series": "FLAIR (Channel 0)",
                "default_slice": tumor_slices.get("Edema", {}).get("slice_index", len(slices) // 2),
                "zoom_level": 1.0,
                "colormap": "grayscale",
                "overlay_active": True
            },
            {
                "monitor_id": 2,
                "role": "Active Tumor Enhancement (T1w-CE / T1gd)",
                "display_series": "T1gd (Channel 2)",
                "default_slice": tumor_slices.get("Enhancing Tumor", {}).get("slice_index", len(slices) // 2),
                "zoom_level": 1.2, # Zoom into enhancing core
                "colormap": "grayscale",
                "overlay_active": True
            },
            {
                "monitor_id": 3,
                "role": "Anatomical Reference & Necrosis (T2w / T1w)",
                "display_series": "T2w (Channel 3) & T1w (Channel 1)",
                "default_slice": tumor_slices.get("Non-Enhancing Tumor", {}).get("slice_index", len(slices) // 2),
                "zoom_level": 1.0,
                "colormap": "grayscale",
                "overlay_active": False
            }
        ],
        "system_actions": [
            "Highlight T1gd and FLAIR contrast mismatch",
            "Auto-scroll Monitor 2 to slice " + str(tumor_slices.get("Enhancing Tumor", {}).get("slice_index", len(slices) // 2)),
            "Prioritize reading queue flag: HIGH PRIORITY" if overall_class >= 2 else "Prioritize reading queue flag: NORMAL"
        ]
    }
    
    return protocol

def main():
    parser = argparse.ArgumentParser(description="Clinical Hanging Protocol Integration Demo")
    parser.add_argument("--image_path", type=str, required=True, help="Path to 3D NIfTI volume")
    parser.add_argument("--folds", type=str, default="1", help="Folds to use (e.g. 1 or 1,2,3)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    fold_list = [int(f.strip()) for f in args.folds.split(",")]
    checkpoints = [os.path.join(checkpoint_dir, f"best_model_fold_{f}.pth") for f in fold_list]
    
    try:
        protocol = generate_hanging_protocol(args.image_path, checkpoints, device=args.device)
        print("\n=== Generated PACS Hanging Protocol JSON ===")
        print(json.dumps(protocol, indent=4))
        
        # Save config
        save_path = os.path.join(results_dir, "hanging_protocol_config.json")
        with open(save_path, "w") as f:
            json.dump(protocol, f, indent=4)
        print(f"\nHanging protocol layout saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating hanging protocol: {e}")

if __name__ == "__main__":
    main()
