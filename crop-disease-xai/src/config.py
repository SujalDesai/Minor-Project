import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Output Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Documentation & Examples
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "sample_leaves"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model & Training Hyperparameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}

# Grad-CAM Settings
GRADCAM_TARGET_LAYER = "out_relu"  # MobileNetV2 top feature map layer before GAP

# Class Mapping Definitions (PlantVillage folder names -> Clean Display Names)
TARGET_CLASSES = {
    "Tomato___healthy": "Tomato Healthy",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Potato___Early_blight": "Potato Early Blight"
}

# Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.80,
    "MODERATE": 0.60
}

# UI / UX Styling Parameters (AgriHealth Theme)
COLOR_PALETTE = {
    "primary": "#2E7D32",       # Living Green
    "secondary": "#00838F",     # Clean Teal
    "accent": "#F57F17",        # Harvest Gold
    "background": "#F4F6F4",    # Clinical Off-White
    "text": "#1A251E",          # Dark Slate
    "card_bg": "#FFFFFF"
}

def ensure_directories():
    """Ensure all required project directories exist."""
    directories = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
        FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR,
        EXAMPLES_DIR, DOCS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("Project directories created/verified.")
