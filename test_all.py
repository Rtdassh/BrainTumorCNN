import os
import unittest
import torch
import numpy as np
import json

from model import get_model
from crossval import get_patient_folds
from train import BrainTumorSliceDataset

class TestBrainTumorCNN(unittest.TestCase):
    
    def test_model_architecture(self):
        """Test if the 4-channel multi-label ResNet-18 model compiles and returns correct output shapes."""
        model = get_model(num_classes=3, in_channels=4, pretrained=False)
        dummy_input = torch.randn(2, 4, 240, 240)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 3))
        
    def test_cross_validation_leakage(self):
        """Verify that patient-level K-fold has no leakage (no intersection between train and val)."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(base_dir, "data", "processed_2d", "metadata.json")
        
        if os.path.exists(metadata_path):
            folds = get_patient_folds(metadata_path, n_splits=5)
            self.assertEqual(len(folds), 5)
            
            for fold_idx, (train_patients, val_patients) in enumerate(folds):
                train_set = set(train_patients)
                val_set = set(val_patients)
                
                # Check zero intersection
                intersection = train_set.intersection(val_set)
                self.assertEqual(len(intersection), 0, f"Leakage detected in fold {fold_idx+1}: {intersection}")
        else:
            print("Skipping cross-validation test (metadata.json not generated yet).")
            
    def test_dataset_loading(self):
        """Verify dataset loader handles 4-channel shapes and 3-class target vectors correctly."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        processed_dir = os.path.join(base_dir, "data", "processed_2d")
        metadata_path = os.path.join(processed_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            patient_list = list(metadata.keys())[:2] # Take first 2 patients
            
            dataset = BrainTumorSliceDataset(patient_list, processed_dir, transform=None)
            
            if len(dataset) > 0:
                img, label = dataset[0]
                self.assertEqual(img.shape[0], 4) # Should have 4 channels (FLAIR, T1w, T1gd, T2w)
                self.assertEqual(label.shape[0], 3) # Target vector should have 3 binary indicators
            else:
                print("Dataset contains no slices for the selected patients.")
        else:
            print("Skipping dataset loading test (metadata.json not generated yet).")

if __name__ == "__main__":
    unittest.main()

