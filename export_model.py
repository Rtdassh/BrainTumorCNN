import os
import argparse
import torch
from model import get_model

def export_model(checkpoint_path, output_dir, device="cpu"):
    """
    Loads a trained checkpoint and exports it to TorchScript (.pt) and ONNX (.onnx) formats.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(checkpoint_path).replace(".pth", "")
    
    # 1. Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = get_model(num_classes=4, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)
    
    # Dummy input representing one batch: (1, 3, 240, 240)
    dummy_input = torch.randn(1, 3, 240, 240, device=device)
    
    # 2. Export to TorchScript
    print("Exporting model to TorchScript...")
    ts_path = os.path.join(output_dir, f"{basename}_torchscript.pt")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(ts_path)
    print(f"TorchScript model saved to {ts_path}")
    
    # 3. Export to ONNX
    try:
        import onnx
        print("Exporting model to ONNX...")
        onnx_path = os.path.join(output_dir, f"{basename}.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to {onnx_path}")
    except ImportError:
        print("Warning: 'onnx' package not installed. Skipping ONNX export.")
        
    print("Export complete!")

def main():
    parser = argparse.ArgumentParser(description="Export Trained Model to ONNX and TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save exported files")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "exported_models")
        
    export_model(args.checkpoint, args.output_dir)

if __name__ == "__main__":
    main()
