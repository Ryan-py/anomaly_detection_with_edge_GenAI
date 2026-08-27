import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# The ExecuTorch Python Runtime API
from executorch.extension.pybindings.portable_lib import _load_for_executorch

def analyze_edge_image(image_path, pte_model_path):
    print(f"Loading ExecuTorch model: {pte_model_path}...")
    # Load the compiled ExecuTorch binary
    edge_module = _load_for_executorch(pte_model_path)
    
    print(f"Loading image: {image_path}...")
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        return

    # 1. Preprocess the image (must match training exactly)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Apply transform and add batch dimension [1, 3, 224, 224]
    img_tensor = test_transform(img_pil).unsqueeze(0)
    
    print("Running INT8 XNNPACK Inference...")
    # 2. Run Inference
    # ExecuTorch requires a list/tuple of tensors as input and returns a list/tuple
    output = edge_module.forward([img_tensor])
    recon_tensor = output[0]  # Extract the reconstructed tensor
    
    # 3. Calculate Error Metrics
    # Percentage Error (Mean Absolute Error * 100)
    mae_error = torch.mean(torch.abs(img_tensor - recon_tensor)).item()
    error_percentage = mae_error * 100
    
    # Raw MSE sum for detailed logging
    mse_score = F.mse_loss(recon_tensor, img_tensor, reduction='sum').item()
    
    # Calculate Pixel-wise Error for the Heatmap (mean across RGB channels)
    error_map = torch.abs(img_tensor - recon_tensor).mean(dim=1).squeeze().numpy()

    print("Generating visualization dashboard...")
    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # -- Original Image --
    img_display = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # -- Reconstructed Image --
    recon_display = recon_tensor.squeeze(0).permute(1, 2, 0).numpy()
    axes[1].imshow(recon_display)
    axes[1].set_title("INT8 XNNPACK Reconstruction")
    axes[1].axis('off')
    
    # -- Error Heatmap --
    im = axes[2].imshow(error_map, cmap='jet', vmin=0, vmax=0.5)
    axes[2].set_title(f"Anomaly Heatmap\nError: {error_percentage:.2f}%")
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # 5. Save the result
    file_name = os.path.basename(image_path)
    save_name = f"edge_result_{file_name}"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    
    print(f"------------------------------------------------")
    print(f"Execution Complete!")
    print(f"Reconstruction Error (MSE Sum): {mse_score:.4f}")
    print(f"Anomaly Percentage: {error_percentage:.2f}%")
    print(f"Saved dashboard to: {save_name}")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    # Define your paths here
    MODEL_PATH = "vae_anomaly_xnnpack_int8.pte"
    
    # Make sure you have transferred a test image to the board!
    TEST_IMAGE_PATH = "rust_3.png" 
    
    analyze_edge_image(TEST_IMAGE_PATH, MODEL_PATH)