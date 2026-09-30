import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_PATH, CLASS_NAMES_PATH
from src.data_preprocessing import preprocess_single_image
from src.gradcam import make_gradcam_heatmap, display_gradcam_overlay
from src.utils import generate_human_readable_explanation

# Global model cache to avoid reloading model on every single prediction request
_MODEL_CACHE = None
_CLASS_NAMES_CACHE = None

def load_inference_model():
    """Loads and caches model and class mapping."""
    global _MODEL_CACHE, _CLASS_NAMES_CACHE
    if _MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file missing at {MODEL_PATH}. Run training first.")
        _MODEL_CACHE = tf.keras.models.load_model(str(MODEL_PATH))
    
    if _CLASS_NAMES_CACHE is None:
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names mapping missing at {CLASS_NAMES_PATH}.")
        with open(CLASS_NAMES_PATH, "r") as f:
            _CLASS_NAMES_CACHE = json.load(f)
            
    return _MODEL_CACHE, _CLASS_NAMES_CACHE

def predict_crop_disease(image_input, compute_gradcam=True):
    """
    Executes full crop disease prediction and Grad-CAM explainability pipeline on an input leaf image.
    
    Args:
        image_input (str, Path, PIL.Image, or np.ndarray): Leaf image input.
        compute_gradcam (bool): Whether to compute Grad-CAM heatmap and overlay.
        
    Returns:
        dict: Complete prediction dictionary.
    """
    model, class_names = load_inference_model()
    
    # 1. Preprocess input image
    img_batch = preprocess_single_image(image_input) # Shape: (1, 224, 224, 3)
    
    # Original image array in uint8 (224, 224, 3)
    orig_img_uint8 = np.uint8(img_batch[0] * 255.0)

    # 2. Model Prediction
    raw_probs = model.predict(img_batch, verbose=0)[0]
    
    # 3. Top Prediction & Confidence
    top_class_idx = int(np.argmax(raw_probs))
    predicted_class = class_names[top_class_idx]
    confidence_val = float(raw_probs[top_class_idx])

    # 4. Top 3 Predictions
    top3_indices = np.argsort(raw_probs)[::-1][:3]
    top3_predictions = [
        {
            "class_name": class_names[idx],
            "probability": float(raw_probs[idx]),
            "percentage_str": f"{float(raw_probs[idx]) * 100:.1f}%"
        }
        for idx in top3_indices
    ]

    # 5. Compute Grad-CAM
    heatmap_2d = None
    heatmap_rgb = None
    overlay_rgb = None

    if compute_gradcam:
        try:
            heatmap_2d, _ = make_gradcam_heatmap(img_batch, model, pred_index=top_class_idx)
            heatmap_rgb, overlay_rgb = display_gradcam_overlay(orig_img_uint8, heatmap_2d)
        except Exception as e:
            print(f"[WARNING] Grad-CAM computation failed: {e}")

    # 6. Generate Narrative Explanation
    explanation = generate_human_readable_explanation(predicted_class, confidence_val, top3_predictions)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_val,
        "confidence_str": f"{confidence_val * 100:.1f}%",
        "top3_predictions": top3_predictions,
        "original_image": orig_img_uint8,
        "heatmap": heatmap_2d,
        "heatmap_rgb": heatmap_rgb,
        "overlay_rgb": overlay_rgb,
        "explanation": explanation
    }

if __name__ == "__main__":
    from src.config import EXAMPLES_DIR
    sample_files = list(EXAMPLES_DIR.glob("*.jpg"))
    if sample_files:
        print(f"Testing prediction pipeline on sample image: {sample_files[0].name}")
        res = predict_crop_disease(sample_files[0])
        print("Prediction Result:")
        print(f"  Class: {res['predicted_class']}")
        print(f"  Confidence: {res['confidence_str']}")
        print(f"  Top 3: {res['top3_predictions']}")
    else:
        print("No sample image found in examples folder.")
