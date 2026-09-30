import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PATH, CLASS_NAMES_PATH, FIGURES_DIR, REPORTS_DIR, 
    IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, RAW_DATA_DIR, ensure_directories
)
from src.data_preprocessing import create_train_val_test_splits, build_data_generator, preprocess_single_image

def evaluate_crop_disease_model():
    """
    Evaluates trained MobileNetV2 model on unseen test dataset.
    Generates metrics, reports, confusion matrix, and class distribution plots.
    """
    ensure_directories()
    print("==================================================")
    print("Evaluating Model on Unseen Test Dataset")
    print("==================================================")

    # 1. Load Model & Class Mapping
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    # 2. Load Test Split
    splits = create_train_val_test_splits()
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    print(f"\nEvaluating on {len(X_test)} unseen test samples...")

    # Load and preprocess test images into array
    test_images = []
    for fpath in X_test:
        img_batch = preprocess_single_image(fpath)
        test_images.append(img_batch[0])

    X_test_arr = np.array(test_images, dtype=np.float32)

    # 3. Model Inference on Test Set
    raw_preds = model.predict(X_test_arr, batch_size=BATCH_SIZE)
    y_pred = np.argmax(raw_preds, axis=1)

    # 4. Calculate Empirical Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print(f"\n--- EMPIRICAL TEST PERFORMANCE ---")
    print(f"  Test Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (Macro): {precision_macro:.4f}")
    print(f"  Recall (Macro):    {recall_macro:.4f}")
    print(f"  F1-Score (Macro):  {f1_macro:.4f}")

    # Detailed Classification Report
    clr_report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    clr_report_str = classification_report(
        y_test, y_pred, target_names=class_names
    )
    print("\nClassification Report:\n", clr_report_str)

    # 5. Save Summary & Reports
    summary_metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "test_samples_count": len(y_test),
        "classification_report": clr_report_dict
    }

    with open(REPORTS_DIR / "classification_report.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)

    with open(REPORTS_DIR / "evaluation_summary.txt", "w") as f:
        f.write("==================================================\n")
        f.write("EXPLAINABLE AI CROP DISEASE DETECTOR - TEST EVALUATION\n")
        f.write("==================================================\n\n")
        f.write(f"Test Accuracy:     {accuracy*100:.2f}%\n")
        f.write(f"Precision (Macro): {precision_macro:.4f}\n")
        f.write(f"Recall (Macro):    {recall_macro:.4f}\n")
        f.write(f"F1-Score (Macro):  {f1_macro:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(clr_report_str)

    print(f"Evaluation reports saved to: {REPORTS_DIR}")

    # 6. Generate Figures
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_class_distribution()

    print("==================================================")
    print("Model Evaluation Complete!")
    print("==================================================")
    return summary_metrics

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots and saves confusion matrix heatmap to outputs/figures/confusion_matrix.png."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=True
    )
    plt.title("Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Disease Class", fontsize=11)
    plt.ylabel("Actual Disease Class", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

def plot_class_distribution():
    """Plots and saves class distribution histogram to outputs/figures/class_distribution.png."""
    class_counts = {}
    for d in os.listdir(RAW_DATA_DIR):
        dp = RAW_DATA_DIR / d
        if dp.is_dir():
            count = len([f for f in os.listdir(dp) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[d.replace("_", " ")] = count

    plt.figure(figsize=(9, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=['#2E7D32', '#00838F', '#F57F17', '#43A047'])
    plt.title("Dataset Class Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Plant Disease Category", fontsize=11)
    plt.ylabel("Number of Leaf Images", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    dist_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Class distribution plot saved to: {dist_path}")

if __name__ == "__main__":
    evaluate_crop_disease_model()
