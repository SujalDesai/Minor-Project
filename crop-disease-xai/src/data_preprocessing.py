import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, 
    RANDOM_SEED, AUGMENTATION_CONFIG
)

def validate_image(image_path):
    """
    Validates if an image file is readable and convertible to RGB.
    
    Args:
        image_path (str or Path): Path to image file.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False

def preprocess_single_image(image_input, target_size=IMAGE_SIZE):
    """
    Preprocesses a single image for MobileNetV2 inference / Grad-CAM.
    Accepts file path, PIL Image, or numpy array (H, W, C).
    
    Returns:
        np.ndarray: Batch-dimension array of shape (1, H, W, C) scaled to [0, 1].
    """
    if isinstance(image_input, (str, Path)):
        if not validate_image(image_input):
            raise ValueError(f"Invalid or corrupted image: {image_input}")
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = (image_input * 255).astype(np.uint8) if image_input.max() <= 1.0 else image_input.astype(np.uint8)
        img = Image.fromarray(image_input).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def load_dataset_filepaths():
    """
    Discovers classes and filepaths in RAW_DATA_DIR.
    Ensures images are valid and non-corrupted.
    
    Returns:
        tuple: (filepaths, labels, class_names)
    """
    class_dirs = [d for d in os.listdir(RAW_DATA_DIR) if (RAW_DATA_DIR / d).is_dir()]
    class_names = sorted(class_dirs)
    
    if not class_names:
        raise FileNotFoundError(f"No class directories found in {RAW_DATA_DIR}")

    filepaths = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_path = RAW_DATA_DIR / class_name
        valid_count = 0
        for fname in os.listdir(class_path):
            fpath = class_path / fname
            if fpath.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                if validate_image(fpath):
                    filepaths.append(str(fpath))
                    labels.append(class_idx)
                    valid_count += 1

        print(f"Discovered class '{class_name}': {valid_count} valid images.")

    return np.array(filepaths), np.array(labels), class_names

def create_train_val_test_splits(test_size=0.15, val_size=0.15, random_state=RANDOM_SEED):
    """
    Splits dataset into stratified Train, Validation, and Test sets.
    
    Returns:
        dict: Containing train, val, and test filepaths and labels + class_names.
    """
    filepaths, labels, class_names = load_dataset_filepaths()
    
    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        filepaths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Split train and validation set
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
    )

    print(f"\nDataset Split Summary:")
    print(f"  Train samples: {len(X_train)} ({len(X_train)/len(filepaths):.1%})")
    print(f"  Val samples:   {len(X_val)} ({len(X_val)/len(filepaths):.1%})")
    print(f"  Test samples:  {len(X_test)} ({len(X_test)/len(filepaths):.1%})")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,     "y_val": y_val,
        "X_test": X_test,   "y_test": y_test,
        "class_names": class_names
    }

def build_data_generator(augment=False):
    """
    Builds a tf.keras ImageDataGenerator.
    """
    if augment:
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=AUGMENTATION_CONFIG["rotation_range"],
            width_shift_range=AUGMENTATION_CONFIG["width_shift_range"],
            height_shift_range=AUGMENTATION_CONFIG["height_shift_range"],
            zoom_range=AUGMENTATION_CONFIG["zoom_range"],
            horizontal_flip=AUGMENTATION_CONFIG["horizontal_flip"],
            fill_mode=AUGMENTATION_CONFIG["fill_mode"]
        )
    else:
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

if __name__ == "__main__":
    splits = create_train_val_test_splits()
    print("Classes found:", splits["class_names"])
