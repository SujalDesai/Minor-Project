import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PATH, CLASS_NAMES_PATH, FIGURES_DIR, BATCH_SIZE, 
    EPOCHS, LEARNING_RATE, IMAGE_SIZE, RANDOM_SEED, ensure_directories
)
from src.data_preprocessing import create_train_val_test_splits, build_data_generator
from src.model import build_transfer_learning_model, unfreeze_for_finetuning

def set_reproducible_seed(seed=RANDOM_SEED):
    """Sets random seeds for numpy and tensorflow for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_dataframe_for_generator(filepaths, labels, class_names):
    """Converts filepaths and numeric labels to pandas DataFrame for flow_from_dataframe."""
    import pandas as pd
    str_labels = [class_names[lbl] for lbl in labels]
    return pd.DataFrame({'filepath': filepaths, 'class_name': str_labels})

def train_crop_disease_model():
    """
    Main training execution pipeline.
    """
    ensure_directories()
    set_reproducible_seed(RANDOM_SEED)

    print("==================================================")
    print("Starting Model Training Pipeline")
    print("==================================================")

    # 1. Discover data & create splits
    splits = create_train_val_test_splits()
    class_names = splits["class_names"]
    num_classes = len(class_names)

    # Save class names mapping to models/class_names.json
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to: {CLASS_NAMES_PATH}")

    # 2. Build Data Generators
    df_train = create_dataframe_for_generator(splits["X_train"], splits["y_train"], class_names)
    df_val = create_dataframe_for_generator(splits["X_val"], splits["y_val"], class_names)

    train_datagen = build_data_generator(augment=True)
    val_datagen = build_data_generator(augment=False)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col="filepath",
        y_col="class_name",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col="filepath",
        y_col="class_name",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # 3. Build Transfer Learning Model
    print("\nBuilding MobileNetV2 Architecture...")
    model, base_model = build_transfer_learning_model(num_classes=num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n--- Phase 1: Training Classification Head ---")
    history_phase1 = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Phase 2: Optional Fine-tuning top layers of backbone
    print("\n--- Phase 2: Fine-Tuning Backbone Top Layers ---")
    unfreeze_for_finetuning(model, base_model, fine_tune_at=120)
    
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10.0)
    model.compile(
        optimizer=fine_tune_optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase2 = model.fit(
        train_generator,
        epochs=EPOCHS + 5,
        initial_epoch=len(history_phase1.history['loss']),
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Combine histories
    full_history = combine_histories(history_phase1.history, history_phase2.history)

    # Save final model state
    model.save(str(MODEL_PATH))
    print(f"\nFinal model saved to: {MODEL_PATH}")

    # Plot Accuracy & Loss Curves
    plot_training_curves(full_history)

    print("==================================================")
    print("Training Pipeline Execution Finished Successfully!")
    print("==================================================")
    return model, full_history

def combine_histories(h1, h2):
    combined = {}
    for key in h1.keys():
        combined[key] = h1[key] + h2[key]
    return combined

def plot_training_curves(history):
    """Plots and saves accuracy and loss curves to outputs/figures/accuracy_loss_curves.png."""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    # Accuracy Plot
    ax1.plot(epochs_range, acc, 'o-', color='#2E7D32', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, 's--', color='#00838F', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Loss Plot
    ax2.plot(epochs_range, loss, 'o-', color='#D32F2F', label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, 's--', color='#F57F17', label='Validation Loss', linewidth=2)
    ax2.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curve_path = FIGURES_DIR / "accuracy_loss_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Training curves figure saved to: {curve_path}")

if __name__ == "__main__":
    train_crop_disease_model()
