import unittest
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import IMAGE_SIZE, INPUT_SHAPE
from src.data_preprocessing import preprocess_single_image, validate_image
from src.model import build_transfer_learning_model, get_gradcam_target_layer
from src.gradcam import make_gradcam_heatmap, display_gradcam_overlay
from src.utils import generate_human_readable_explanation, generate_confidence_assessment

class TestCropDiseaseXAI(unittest.TestCase):

    def setUp(self):
        """Create a temporary synthetic test image in memory."""
        self.test_img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        self.pil_img = Image.fromarray(self.test_img_array)
        self.num_classes = 4

    def test_image_preprocessing_dimensions_and_scaling(self):
        """Test preprocessing produces shape (1, 224, 224, 3) in range [0, 1]."""
        batch_img = preprocess_single_image(self.pil_img, target_size=IMAGE_SIZE)
        self.assertEqual(batch_img.shape, (1, 224, 224, 3))
        self.assertGreaterEqual(batch_img.min(), 0.0)
        self.assertLessEqual(batch_img.max(), 1.0)

    def test_invalid_image_handling(self):
        """Test invalid image path raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_single_image("non_existent_file.png")

    def test_model_build_and_softmax_sum(self):
        """Test model construction and prediction probability sum equals 1.0."""
        model, _ = build_transfer_learning_model(num_classes=self.num_classes)
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        preds = model.predict(dummy_input, verbose=0)
        
        self.assertEqual(preds.shape, (1, self.num_classes))
        self.assertAlmostEqual(float(np.sum(preds[0])), 1.0, places=4)

    def test_gradcam_target_layer_discovery(self):
        """Test Grad-CAM layer inspector locates a valid target convolutional layer."""
        model, _ = build_transfer_learning_model(num_classes=self.num_classes)
        layer_obj, is_base, layer_name = get_gradcam_target_layer(model, "out_relu")
        self.assertIsNotNone(layer_obj)
        self.assertTrue(is_base)
        self.assertEqual(layer_name, "out_relu")

    def test_gradcam_heatmap_dimensions_and_overlay(self):
        """Test Grad-CAM heatmap generates array of shape (224, 224) and valid overlay."""
        model, _ = build_transfer_learning_model(num_classes=self.num_classes)
        dummy_batch = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        heatmap, pred_idx = make_gradcam_heatmap(dummy_batch, model, pred_index=0)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertGreaterEqual(heatmap.min(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)

        orig_uint8 = np.uint8(dummy_batch[0] * 255)
        heatmap_colored, overlay = display_gradcam_overlay(orig_uint8, heatmap)
        self.assertEqual(heatmap_colored.shape, (224, 224, 3))
        self.assertEqual(overlay.shape, (224, 224, 3))

    def test_confidence_threshold_ratings(self):
        """Test qualitative confidence ratings and color assignments."""
        rating_high, color_high, _ = generate_confidence_assessment(0.95)
        self.assertEqual(rating_high, "High Confidence")

        rating_mod, color_mod, _ = generate_confidence_assessment(0.70)
        self.assertEqual(rating_mod, "Moderate Confidence")

        rating_low, color_low, _ = generate_confidence_assessment(0.40)
        self.assertEqual(rating_low, "Low Confidence")

if __name__ == "__main__":
    unittest.main()
