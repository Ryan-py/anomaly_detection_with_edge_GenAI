import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from executorch.extension.pybindings.portable_lib import _load_for_executorch

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "models" / "vae_anomaly_xnnpack_int8.pte"
DEFAULT_CRITICAL_THRESHOLD = 15.0

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ModelLoadError(Exception):
    pass


class InferenceError(Exception):
    pass


_edge_module = None
_load_error: Exception | None = None


def _get_model():
    global _edge_module, _load_error
    if _edge_module is not None:
        return _edge_module
    if _load_error is not None:
        raise _load_error
    if not MODEL_PATH.exists():
        _load_error = ModelLoadError(
            f"Model file not found at {MODEL_PATH} — copy vae_anomaly_xnnpack_int8.pte "
            "into assets/models/"
        )
        raise _load_error
    try:
        _edge_module = _load_for_executorch(str(MODEL_PATH))
    except Exception as e:
        _load_error = ModelLoadError(f"Failed to load model at {MODEL_PATH}: {e}")
        raise _load_error
    return _edge_module


def warm_up() -> None:
    try:
        _get_model()
    except ModelLoadError as e:
        print(f"[inference_engine] warm-up skipped: {e}")


def model_status() -> dict:
    try:
        _get_model()
        return {"loaded": True, "error": None}
    except ModelLoadError as e:
        return {"loaded": False, "error": str(e)}


def run_inspection(
    pil_image: Image.Image,
    heatmap_output_path: str,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> dict:
    edge_module = _get_model()
    try:
        img_tensor = _TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)
        output = edge_module.forward([img_tensor])
        recon_tensor = output[0]

        mae_error = torch.mean(torch.abs(img_tensor - recon_tensor)).item()
        error_percentage = mae_error * 100

        error_map = torch.abs(img_tensor - recon_tensor).mean(dim=1).squeeze().numpy()
        heatmap_u8 = np.uint8(np.clip(error_map / 0.5, 0, 1) * 255)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_output_path, heatmap_bgr)
    except Exception as e:
        raise InferenceError(str(e))

    return {
        "error_percentage": error_percentage,
        "is_critical": error_percentage > critical_threshold,
    }
