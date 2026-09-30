import numpy as np
import tensorflow as tf
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import GRADCAM_TARGET_LAYER
from src.model import get_gradcam_target_layer

def make_gradcam_heatmap(img_array, model, pred_index=None, target_layer_name=GRADCAM_TARGET_LAYER):
    """
    Computes genuine Grad-CAM heatmap for a target class using TensorFlow GradientTape.
    
    Args:
        img_array (np.ndarray): Preprocessed image array of shape (1, 224, 224, 3) in [0, 1].
        model (tf.keras.Model): Trained Keras classification model.
        pred_index (int, optional): Index of target class. Defaults to top predicted class.
        target_layer_name (str): Name of convolutional feature map layer.
        
    Returns:
        tuple: (heatmap_2d, pred_index) where heatmap_2d is np.ndarray of shape (224, 224) in [0, 1].
    """
    # Create sub-model that outputs both target layer activations and top model predictions
    try:
        target_layer = model.get_layer(target_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[target_layer.output, model.output]
        )
    except ValueError:
        # If target layer is inside nested backbone model (e.g. MobileNetV2)
        base_model = None
        for l in model.layers:
            if isinstance(l, tf.keras.Model) or hasattr(l, 'get_layer'):
                try:
                    target_layer = l.get_layer(target_layer_name)
                    base_model = l
                    break
                except ValueError:
                    pass

        if base_model is None:
            # Fallback search using get_gradcam_target_layer utility
            target_layer, is_base, target_layer_name = get_gradcam_target_layer(model, target_layer_name)
            base_model = model.layers[1] if is_base else None

        if base_model is not None:
            # Construct nested feature extractor
            base_sub_model = tf.keras.models.Model(
                inputs=[base_model.inputs],
                outputs=[target_layer.output, base_model.output]
            )
            
            # Construct complete gradient model
            inputs = model.inputs
            conv_outputs, base_out = base_sub_model(inputs)
            
            # Pass base_out through remaining top layers (GAP -> Dropout -> Dense)
            x = base_out
            for layer in model.layers:
                if layer != model.layers[0] and layer != base_model:
                    x = layer(x)
            preds = x
            grad_model = tf.keras.models.Model(inputs=inputs, outputs=[conv_outputs, preds])
        else:
            raise RuntimeError(f"Could not build gradient model for target layer '{target_layer_name}'.")

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
            
        class_channel = predictions[:, pred_index]

    # Compute gradients of target class score w.r.t. target layer feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        raise RuntimeError("Gradients could not be computed. Ensure target layer outputs tensor with gradient flow.")

    # Global Average Pooling of gradients over spatial dimensions (height, width)
    # alpha_k = mean(grad_k)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight activation feature maps by channel importance weights
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: keep features that positively contribute to target class score
    heatmap = tf.maximum(heatmap, 0.0)
    
    # Normalize heatmap to [0, 1]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy()
    
    # Resize heatmap to match input image resolution (224, 224)
    heatmap_resized = cv2.resize(heatmap_np, (img_array.shape[2], img_array.shape[1]))
    return heatmap_resized, int(pred_index)

def display_gradcam_overlay(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Blends Grad-CAM heatmap overlay onto original RGB image.
    
    Args:
        original_img (np.ndarray): Original image array (H, W, 3) in uint8 [0, 255] or float [0, 1].
        heatmap (np.ndarray): Normalized 2D heatmap array (H, W) in float [0, 1].
        alpha (float): Overlay transparency factor (0.0 to 1.0).
        colormap (int): OpenCV colormap identifier.
        
    Returns:
        tuple: (heatmap_colored_rgb, overlay_rgb) both as uint8 np.ndarray (H, W, 3).
    """
    if original_img.dtype != np.uint8:
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)

    # Convert 2D heatmap [0, 1] to uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Apply OpenCV colormap (RGB <-> BGR conversion for proper rendering)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend overlay: img * (1 - alpha) + heatmap * alpha
    overlay_bgr = cv2.addWeighted(
        cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 1 - alpha,
        heatmap_colored_bgr, alpha, 0
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_colored_rgb, overlay_rgb
