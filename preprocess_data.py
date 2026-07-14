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
    
    # Create output directories
    for i in range(4):
        os.makedirs(os.path.join(output_dir, f"class_{i}"), exist_ok=True)
        
    slice_counts = {i: 0 for i in range(4)}
    patient_metadata = {}

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        patient_id = os.path.basename(img_path).replace(".nii.gz", "")
        
        # Load NIfTI files
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        
        # Load only label data as it is much smaller
        lbl_data = np.asanyarray(lbl_nii.dataobj) # Shape: (H, W, D)
        
        h, w, d = lbl_data.shape
        
        # We will keep track of slices saved for this patient to limit count
        saved_classes = {i: [] for i in range(4)}
        
        # Determine slices we need to extract for this patient
        needed_slices = {}
        for z in range(0, d, step):
            lbl_slice = lbl_data[:, :, z]
            
            if np.any(lbl_slice == 3):
                cls = 3
            elif np.any(lbl_slice == 2):
                cls = 2
            elif np.any(lbl_slice == 1):
                cls = 1
            else:
                cls = 0
                
            # If we haven't reached the limit for this class, plan to extract it
            if len(saved_classes[cls]) + len(needed_slices.get(cls, [])) < max_slices_per_class:
                needed_slices.setdefault(cls, []).append(z)
                
        # If we need any slices from this patient, load image once into memory as float32
        if any(len(indices) > 0 for indices in needed_slices.values()):
            img_data = img_nii.get_fdata(dtype=np.float32) # Decompress once in memory (fast!)
            
            for cls, z_indices in needed_slices.items():
                for z in z_indices:
                    img_slice = img_data[:, :, z, :]
                    
                    # Check if there is brain tissue (non-zero background)
                    # FLAIR is modality 0
                    flair_slice = img_slice[:, :, 0]
                    if flair_slice.max() == 0 or (flair_slice > flair_slice.mean()).sum() < 500:
                        continue # Skip slices with almost no brain tissue
                        
                    # Prepare 3-channel input: FLAIR (0), T1gd (2), T2w (3)
                    channels = []
                    for mod_idx in [0, 2, 3]:
                        ch = img_slice[:, :, mod_idx]
                        ch_min, ch_max = ch.min(), ch.max()
                        if ch_max > ch_min:
                            ch = (ch - ch_min) / (ch_max - ch_min)
                        else:
                            ch = np.zeros_like(ch)
                        channels.append(ch)
                        
                    # Stack to form (3, H, W)
                    slice_3ch = np.stack(channels, axis=0).astype(np.float32)
                    
                    # Save slice as .npy
                    filename = f"{patient_id}_slice_{z}.npy"
                    filepath = os.path.join(output_dir, f"class_{cls}", filename)
                    np.save(filepath, slice_3ch)
                    
                    saved_classes[cls].append(z)
                    slice_counts[cls] += 1
                
        patient_metadata[patient_id] = {
            "slices": {cls: saved_classes[cls] for cls in range(4)}
        }
        
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(patient_metadata, f, indent=4)
        
    print("\nPreprocessing complete!")
    print("Slice counts by class:")
    for cls, count in slice_counts.items():
        print(f"  Class {cls}: {count} slices")
        
if __name__ == "__main__":
    preprocess_dataset(max_slices_per_class=5, step=2)
