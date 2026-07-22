import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_patient_folds(metadata_path, n_splits=5, random_state=42):
    """
    Groups slices by patient and performs Stratified K-Fold at the patient level.
    This prevents data leakage (slices from the same patient in both train and val).
    
    Stratification is based on the maximum class label present for each patient.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    patient_ids = list(metadata.keys())
    patient_labels = []
    
    # Determine the stratification label for each patient (maximum class present)
    for p_id in patient_ids:
        slices = metadata[p_id]["slices"]
        max_class = 0
        if isinstance(slices, list):
            for item in slices:
                max_class = max(max_class, item.get("primary_class", 0))
        elif isinstance(slices, dict):
            for cls in range(4):
                if str(cls) in slices and len(slices[str(cls)]) > 0:
                    max_class = max(max_class, cls)
        patient_labels.append(max_class)

        
    patient_ids = np.array(patient_ids)
    patient_labels = np.array(patient_labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(patient_ids, patient_labels):
        train_patients = patient_ids[train_idx].tolist()
        val_patients = patient_ids[val_idx].tolist()
        folds.append((train_patients, val_patients))
        
    return folds

if __name__ == "__main__":
    # Test script if executed directly
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(base_dir, "data", "processed_2d", "metadata.json")
    if os.path.exists(meta_path):
        folds = get_patient_folds(meta_path)
        print(f"Successfully generated {len(folds)} folds.")
        for idx, (train_p, val_p) in enumerate(folds):
            print(f"Fold {idx+1}: Train patients = {len(train_p)}, Val patients = {len(val_p)}")
    else:
        print("Metadata file not found. Run preprocess_data.py first.")
