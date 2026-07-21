# Brain Tumor CNN Classification Project

This project implements a robust system for 4-class brain tumor classification from 2D axial magnetic resonance imaging (MRI) slices, using the MONAI **Task01_BrainTumour (BraTS)** dataset.

The architecture is built upon a pre-trained **ResNet-18** network (with factory options for **EfficientNet-B0** and **ResNet-34**), and the training methodology incorporates best practices from the research paper *"Deep Learning–based Identification of Brain MRI Sequences"* (2023) to ensure robustness against scanner variability and strict data validation without data leakage.

---

## 🛠️ Project Structure

The repository is organized as follows:

- **`requirements.txt`**: List of required Python dependencies.
- **`setup_env.ps1`** / **`setup_env.sh`**: Environment setup scripts for Windows and Linux/Colab.
- **`preprocess_data.py`**: Converts 3D NIfTI volumes into 3-channel 2D axial slices (FLAIR, T1gd, T2w) and categorizes them by target class.
- **`model.py`**: Defines the neural network architectures (ResNet-18, EfficientNet-B0, ResNet-34), hybrid pooling, and multi-model ensembling.
- **`crossval.py`**: Splits patients using a **Patient-Stratified K-Fold** strategy to strictly prevent data leakage.
- **`train.py`**: Main training script featuring class-weighted loss, custom data augmentations, and Weights & Biases logging.
- **`evaluate.py`**: Evaluates trained fold models, generates confusion matrices, ROC/PR curves, per-center scanner robustness analysis, and **Grad-CAM** saliency maps.
- **`inference.py`**: Runs inference on new 3D NIfTI volumes and generates axial slice probability evolution profiles.
- **`external_test.py`**: Validates the model against an independent external 2D image dataset (`brain_tumor_dataset`).
- **`hanging_protocol_integration.py`**: Clinical PACS / Hanging Protocol integration demo based on automated model predictions.

---

## 🧠 Neural Network Architecture & Structural Options

The model architecture is optimized for multi-sequence feature extraction from magnetic resonance imaging (MRI):

### 1. Current Network Structure
* **Input**: 3-channel 2D axial slices `(FLAIR, T1gd, T2w)` resized to $240 \times 240$ or $128 \times 128$.
* **Feature Extractor**: **ResNet-18** pre-trained on ImageNet (captures transferrable morphological and contrast patterns).
* **Pooling**: Global Average Pooling (GAP) with optional **Dual/Hybrid Global Pooling (GAP + GMP)** to capture both broad diffuse regions (edema) and small high-intensity focal points (enhancing tumor core).
* **Classification Head**: 2-layer dense Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout:
  $$\text{Input} \xrightarrow{\text{Linear(in, 256)}} \text{BatchNorm1d} \xrightarrow{\text{ReLU}} \text{Dropout}(p=0.2) \xrightarrow{\text{Linear(256, 4)}} \text{Logits}$$
* **Ensemble Support (`BrainTumorEnsemble`)**: Combines predictions across multiple fold models by averaging softmax probability vectors $\sigma(z)$, enhancing clinical stability and calibration.
* **Layer-wise Learning Rate Decay (LLRD)**: Exposed via `get_parameter_groups()`, applying decay multipliers to initial convolutional layers to preserve pre-trained ImageNet features while adapting the head to the MRI domain.

### 2. Configurable Improvement Options

1. **Flexible Backbone Selection (`get_model`)**:
   - `resnet18` (Default): Lightweight (~11.2M parameters), fast execution, resistant to overfitting.
   - `resnet34`: Deeper convolutional capacity for complex feature representations.
   - `efficientnet_b0`: Features depthwise separable convolutions and built-in Squeeze-and-Excitation (SE) channel attention blocks (~5.3M parameters).
2. **Channel-wise Attention (Squeeze-and-Excitation)**:
   - Dynamically re-weights input sequence channels (FLAIR vs. T1gd vs. T2w) based on the tissue type present in the slice (e.g., boosting FLAIR for Edema, T1gd for Enhancing core).
3. **2.5D Spatial Context (Adjacent Slice Stacking)**:
   - Expand input channels from 3 to 9 `[z-1, z, z+1]` to supply vertical spatial continuity along the z-axis.
4. **Head Regularization & Normalization**:
   - Internal activation normalization via `BatchNorm1d` combined with $20\%$ `Dropout` to balance domain adaptation without underfitting.

---

## 🚀 Step-by-Step Usage Guide

### 1. Environment Setup

On Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

On Linux / Google Colab / macOS (Bash):
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### 2. Data Preprocessing

Preprocessing loads 3D `.nii.gz` files, filters out non-brain background regions, and extracts 2D axial slices containing the target target classes:
- **Class 0**: Healthy / Background (Healthy/Bg)
- **Class 1**: Edema (Edema)
- **Class 2**: Non-enhancing tumor (Non-enhancing tumor)
- **Class 3**: Enhancing tumor (Enhancing tumor)

To execute preprocessing:
```bash
# Activate virtual environment if necessary: .\venv\Scripts\activate or source venv/bin/activate
python preprocess_data.py
```
This generates `data/processed_2d/` organizing slices by class subfolders and creating `metadata.json` with patient distributions.

### 3. Model Training (with Cross-Validation)

To train the model on Fold 1 (default: 15 epochs):
```bash
python train.py --fold 1 --epochs 15 --batch_size 32
```
Optional flags:
- `--device cpu` or `--device cuda`
- `--wandb` to enable real-time experiment tracking on Weights & Biases.
- `--fold 2` through `--fold 5` to train alternate cross-validation folds.

Best model checkpoints based on validation Macro F1 score are automatically saved to `checkpoints/`.

### 4. Full Evaluation & Grad-CAM Visualization

After training a fold model, evaluate its overall performance and simulated clinical center breakdown:
```bash
python evaluate.py --fold 1
```
This generates the following artifacts in `results/`:
1. **`confusion_matrix.png`**: Detailed confusion matrix.
2. **`roc_curves.png`** and **`precision_recall_curves.png`**: Class-wise ROC and Precision-Recall performance curves.
3. **`gradcam_samples.png`**: Visual **Grad-CAM** saliency maps overlaid on the FLAIR sequence, highlighting decision-critical regions.
4. **Scanner Center Robustness Analysis**: Detailed Accuracy and Macro F1 breakdowns by origin scanner study.

### 5. Inference on New 3D NIfTI Volumes

To classify all axial slices of a full 3D patient volume and plot probability evolution along the axial axis:
```bash
python inference.py --image_path data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --folds 1
```
*(Pass multiple comma-separated folds to use a prediction **Ensemble**, e.g., `--folds 1,2,3`)*

This saves a profile chart to `results/volume_prediction_profile.png`, enabling radiologists to pinpoint axial tumor slice distributions.

### 6. External Validation (Generalization Test)

Validate model generalization against an independent external 2D image dataset (`brain_tumor_dataset`):
```bash
python external_test.py --folds 1
```
This script maps 4-class predictions into a binary classification (Healthy vs. Tumor) and measures **Sensitivity** and **Specificity** on non-BraTS distribution images. Results are saved to `results/external_test_report.txt`.

### 7. Clinical Integration (Hanging Protocol)

Simulates a clinical PACS system that intelligently arranges specialist displays based on automated model findings:
```bash
python hanging_protocol_integration.py --image_path data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz --folds 1
```
Generates a PACS display configuration `results/hanging_protocol_config.json` detailing monitor sequence assignments, focus slices, and initial zoom levels.

---

## 📊 Paper Best Practices Incorporated

1. **Scanner Cohort Variability Handling**: Evaluates and breaks down performance across independent center cohorts to detect scanner domain shift.
2. **Patient-Stratified K-Fold Validation**: Strictly prevents data leakage by ensuring no slice from a validation patient is present in the training set.
3. **Per-Channel Min-Max Normalization**: Scales each modality (FLAIR, T1gd, T2w) independently to $[0, 1]$ per slice to reduce relative contrast variance between scanners.
4. **Class-Weighted Cross Entropy Loss**: Adjusts gradients based on training set class frequencies to handle tumor region class imbalances.
