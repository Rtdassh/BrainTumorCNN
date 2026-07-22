import os
import glob
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Path setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "Task01_BrainTumour")
images_dir = os.path.join(DATA_DIR, "imagesTr")
labels_dir = os.path.join(DATA_DIR, "labelsTr")
output_dir = os.path.join(BASE_DIR, "data", "processed_2d")

def preprocess_dataset(max_slices_per_class=5, step=2):
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    
    print(f"Found {len(image_paths)} images and {len(label_paths)} labels.")
    
    # Create output directories for organizing slice files
    for i in range(4):
        os.makedirs(os.path.join(output_dir, f"class_{i}"), exist_ok=True)
        
    patient_metadata = {}
    label_counts = {"edema": 0, "non_enhancing": 0, "enhancing": 0, "healthy": 0}

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        patient_id = os.path.basename(img_path).replace(".nii.gz", "")
        
        # Load NIfTI files
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        
        lbl_data = np.asanyarray(lbl_nii.dataobj) # Shape: (H, W, D)
        h, w, d = lbl_data.shape
        
        patient_slices = []
        saved_classes = {i: 0 for i in range(4)}
        
        # Determine slice indices to extract for this patient
        needed_slices = {}
        for z in range(0, d, step):
            lbl_slice = lbl_data[:, :, z]
            
            # Determine primary class for storage categorization
            if np.any(lbl_slice == 3):
                primary_cls = 3
            elif np.any(lbl_slice == 2):
                primary_cls = 2
            elif np.any(lbl_slice == 1):
                primary_cls = 1
            else:
                primary_cls = 0
                
            if saved_classes[primary_cls] + len(needed_slices.get(primary_cls, [])) < max_slices_per_class:
                needed_slices.setdefault(primary_cls, []).append(z)
                
        if any(len(indices) > 0 for indices in needed_slices.values()):
            img_data = img_nii.get_fdata(dtype=np.float32) # Decompress 4D volume once in memory: (H, W, D, 4)
            
            # Normalize all 4 3D modalities per volume independently: FLAIR (0), T1w (1), T1gd (2), T2w (3)
            for mod_idx in range(4):
                vol = img_data[:, :, :, mod_idx]
                vol_min, vol_max = vol.min(), vol.max()
                if vol_max > vol_min:
                    img_data[:, :, :, mod_idx] = (vol - vol_min) / (vol_max - vol_min)
                else:
                    img_data[:, :, :, mod_idx] = 0.0
            
            for primary_cls, z_indices in needed_slices.items():
                for z in z_indices:
                    img_slice = img_data[:, :, z, :] # Shape: (H, W, 4)
                    lbl_slice = lbl_data[:, :, z]    # Shape: (H, W)
                    
                    # Check if brain tissue is present (FLAIR is modality 0)
                    flair_slice = img_slice[:, :, 0]
                    if flair_slice.max() == 0 or (flair_slice > flair_slice.mean()).sum() < 500:
                        continue
                        
                    # Stack ALL 4 modalities: FLAIR (0), T1w (1), T1gd (2), T2w (3) -> (4, H, W)
                    slice_4ch = np.stack([img_slice[:, :, mod] for mod in range(4)], axis=0).astype(np.float32)
                    
                    # Compute multi-label binary indicators: [has_edema, has_non_enhancing, has_enhancing]
                    has_edema = int(np.any(lbl_slice == 1))
                    has_non_enhancing = int(np.any(lbl_slice == 2))
                    has_enhancing = int(np.any(lbl_slice == 3))
                    labels_vec = [has_edema, has_non_enhancing, has_enhancing]
                    
                    if sum(labels_vec) == 0:
                        label_counts["healthy"] += 1
                    else:
                        if has_edema: label_counts["edema"] += 1
                        if has_non_enhancing: label_counts["non_enhancing"] += 1
                        if has_enhancing: label_counts["enhancing"] += 1
                        
                    # Save slice as .npy file
                    rel_path = os.path.join(f"class_{primary_cls}", f"{patient_id}_slice_{z}.npy")
                    full_path = os.path.join(output_dir, rel_path)
                    np.save(full_path, slice_4ch)
                    
                    patient_slices.append({
                        "z": z,
                        "rel_path": rel_path,
                        "labels": labels_vec,
                        "primary_class": primary_cls
                    })
                    saved_classes[primary_cls] += 1
                
        patient_metadata[patient_id] = {
            "slices": patient_slices
        }
        
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(patient_metadata, f, indent=4)
        
    print("\nPreprocessing complete!")
    print("Multi-label occurrences across saved slices:")
    for label_name, count in label_counts.items():
        print(f"  {label_name.capitalize()}: {count} occurrences")
        
if __name__ == "__main__":
    preprocess_dataset(max_slices_per_class=5, step=2)

