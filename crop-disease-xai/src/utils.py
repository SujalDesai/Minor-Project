import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CONFIDENCE_THRESHOLDS

def generate_confidence_assessment(confidence_val):
    """
    Returns qualitative confidence rating and UI color code.
    
    Args:
        confidence_val (float): Confidence value between 0.0 and 1.0.
        
    Returns:
        tuple: (rating_str, color_hex, status_badge)
    """
    if confidence_val >= CONFIDENCE_THRESHOLDS["HIGH"]:
        return "High Confidence", "#2E7D32", "🟢 HIGH CONFIDENCE"
    elif confidence_val >= CONFIDENCE_THRESHOLDS["MODERATE"]:
        return "Moderate Confidence", "#F57F17", "🟡 MODERATE CONFIDENCE"
    else:
        return "Low Confidence", "#D32F2F", "🔴 LOW CONFIDENCE"

def generate_human_readable_explanation(predicted_class, confidence_val, top3_predictions):
    """
    Generates human-understandable XAI textual explanation of prediction and Grad-CAM.
    
    Returns:
        dict: Detailed narrative explanations.
    """
    rating_str, color_hex, badge = generate_confidence_assessment(confidence_val)
    conf_pct = f"{confidence_val * 100:.1f}%"
    
    narrative_summary = (
        f"The model diagnosed the leaf image as **{predicted_class}** with **{conf_pct}** confidence "
        f"({rating_str}). The classification is based on feature representations extracted via transfer learning."
    )
    
    gradcam_explanation = (
        f"The Grad-CAM heatmaps highlight key visual pixels (shown in bright yellow/red) that exerted "
        f"the highest influence on MobileNetV2's activation score for **{predicted_class}**. "
        f"These regions indicate where the neural network focused its internal visual attention."
    )
    
    scientific_disclaimer = (
        "⚠️ **Academic & Diagnostic Disclaimer**: Grad-CAM visualizes feature attribution within the "
        "convolutional neural network layers. Highlighted regions indicate visual correlation with trained "
        "class patterns, but do NOT prove biological causation or replace field diagnosis by a certified agronomist."
    )

    what_gradcam_does_not_mean = [
        "It does not prove biological pathogen presence by itself.",
        "It does not guarantee 100% real-world diagnostic correctness under all outdoor lighting conditions.",
        "It reflects feature correlation in the ML model, not expert lab pathology tests."
    ]

    return {
        "summary": narrative_summary,
        "gradcam_explanation": gradcam_explanation,
        "disclaimer": scientific_disclaimer,
        "confidence_badge": badge,
        "confidence_color": color_hex,
        "confidence_rating": rating_str,
        "limitations": what_gradcam_does_not_mean
    }
