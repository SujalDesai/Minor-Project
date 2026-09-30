import tensorflow as tf
from tensorflow.keras import layers, models, Model
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import INPUT_SHAPE, GRADCAM_TARGET_LAYER

def build_transfer_learning_model(num_classes, input_shape=INPUT_SHAPE, trainable_backbone=False):
    """
    Builds a MobileNetV2 transfer learning model for crop disease classification.
    
    Args:
        num_classes (int): Number of target classification categories.
        input_shape (tuple): RGB image shape (H, W, C).
        trainable_backbone (bool): Whether to unfreeze MobileNetV2 base layers initially.
        
    Returns:
        tf.keras.Model: Configured classification model.
    """
    # Load MobileNetV2 base model pre-trained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze backbone layers initially
    base_model.trainable = trainable_backbone

    # Functional Architecture
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CropDisease_MobileNetV2")
    return model, base_model

def unfreeze_for_finetuning(model, base_model, fine_tune_at=100):
    """
    Unfreezes base model layers from index fine_tune_at onwards for fine-tuning.
    
    Args:
        model (tf.keras.Model): Top model.
        base_model (tf.keras.Model): MobileNetV2 backbone.
        fine_tune_at (int): Layer index from which to start fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    print(f"Fine-tuning enabled for backbone layers {fine_tune_at} to {len(base_model.layers)}")

def get_gradcam_target_layer(model, layer_name=GRADCAM_TARGET_LAYER):
    """
    Inspects model and returns the target convolutional layer for Grad-CAM.
    Validates layer existence and checks if inside base model or top model.
    
    Returns:
        tuple: (layer_object, is_inside_base, layer_name_str)
    """
    # First check direct layers of the model
    try:
        layer = model.get_layer(layer_name)
        return layer, False, layer_name
    except ValueError:
        pass

    # Check inside nested base model (e.g. MobileNetV2)
    for layer in model.layers:
        if isinstance(layer, Model) or hasattr(layer, 'get_layer'):
            try:
                sub_layer = layer.get_layer(layer_name)
                return sub_layer, True, layer_name
            except ValueError:
                pass

    # Fallback search for last Conv2D or ReLU feature map layer
    print(f"[WARNING] Specified Grad-CAM layer '{layer_name}' not found. Searching for suitable Conv2D layer...")
    
    def search_layers(layer_list):
        for l in reversed(layer_list):
            if isinstance(l, (layers.Conv2D, layers.Activation, layers.ReLU)) or "relu" in l.name.lower() or "conv" in l.name.lower():
                return l
        return None

    # Search in main model or base model
    found_layer = None
    is_base = False
    for l in model.layers:
        if hasattr(l, 'layers'):
            found_layer = search_layers(l.layers)
            if found_layer:
                is_base = True
                break
        elif isinstance(l, layers.Conv2D):
            found_layer = l
            break

    if found_layer is not None:
        print(f"[INFO] Auto-selected target layer: '{found_layer.name}' (inside_base={is_base})")
        return found_layer, is_base, found_layer.name
    else:
        raise RuntimeError(f"Could not find any suitable convolutional layer in model for Grad-CAM visualization.")

if __name__ == "__main__":
    m, base = build_transfer_learning_model(num_classes=4)
    m.summary()
    target_l, is_base, name = get_gradcam_target_layer(m)
    print(f"Verified Grad-CAM Target Layer: {name}, Base: {is_base}")
