# 🎓 Viva Voce Preparation Guide

> **Project Title:** Explainable AI for Crop Disease Detection  
> **Module:** Milestone 1 (Agricultural Pathology Classification & Grad-CAM XAI)

---

### Q1: What problem is this project solving?
**Answer:** Crop diseases significantly reduce agricultural yield. Traditional diagnosis requires expert agronomists, who may not be accessible in rural regions. Deep learning models can automate diagnosis, but farmers and experts distrust "black-box" predictions. This project provides an **Explainable AI framework** that classifies crop diseases and visually highlights *why* the model made its decision.

### Q2: Why use Artificial Intelligence / Deep Learning?
**Answer:** Plant leaf diseases exhibit subtle visual patterns (e.g., concentric rings in Early Blight, water-soaked lesions in Late Blight). AI can learn these complex, non-linear visual patterns directly from thousands of leaf images much faster and more consistently than manual rule-based programming.

### Q3: Why Computer Vision?
**Answer:** Visual inspection of plant leaves is the primary diagnostic method in agriculture. Computer Vision algorithms process image pixel matrices, allowing automated visual diagnosis from smartphone photographs taken in the field.

### Q4: Why did you choose MobileNetV2?
**Answer:** MobileNetV2 is an efficient, lightweight Convolutional Neural Network (CNN) architecture designed for resource-constrained environments. It uses **Depthwise Separable Convolutions** and **Inverted Residual Blocks**, which drastically reduce parameter count and computational complexity while maintaining high accuracy.

### Q5: What is Transfer Learning?
**Answer:** Transfer learning is a machine learning technique where a pre-trained model (trained on a massive dataset like ImageNet with 1.4 million images) is repurposed for a specific target task (crop disease detection). Instead of training a deep network from scratch, we transfer pre-learned feature extractors (edges, textures, shapes) and fine-tune only the classification layers.

### Q6: What is a Convolutional Neural Network (CNN)?
**Answer:** A CNN is a deep learning architecture specialized for grid-like data like images. It uses sliding filter kernels (convolutions) to automatically extract spatial feature hierarchies—ranging from low-level edges in early layers to high-level disease lesions in deeper layers.

### Q7: What is Data Augmentation and why is it important?
**Answer:** Data augmentation artificially expands the diversity of the training dataset by applying realistic transformations (e.g., horizontal flips, small rotations, zoom, spatial shifts). It prevents overfitting and ensures the model generalizes well to leaves photographed under varying real-world orientations.

### Q8: What is the Softmax function?
**Answer:** Softmax is an activation function applied to the final output layer of a multi-class neural network. It transforms raw logit scores into a normalized probability distribution where each class probability is between $0.0$ and $1.0$, and the sum across all classes equals $1.0$.

### Q9: What is Model Confidence?
**Answer:** Model confidence is the maximum probability output by the Softmax function for the top predicted class. It reflects the model's internal statistical certainty regarding its prediction based on learned training patterns.

### Q10: What is Accuracy?
**Answer:** Accuracy is the percentage of total test samples correctly classified by the model:
$$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$$

### Q11: What is Precision?
**Answer:** Precision measures the proportion of positive predictions that were actually correct:
$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

### Q12: What is Recall?
**Answer:** Recall (Sensitivity) measures the proportion of actual disease cases that were correctly identified by the model:
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

### Q13: What is the F1-Score?
**Answer:** The F1-score is the harmonic mean of Precision and Recall, providing a balanced performance metric especially when dealing with slightly imbalanced datasets:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Q14: What is a Confusion Matrix?
**Answer:** A confusion matrix is a tabular layout that compares actual ground-truth class labels against model-predicted labels. It reveals specific class misclassifications (e.g., how often Early Blight is confused with Late Blight).

### Q15: What is Explainable AI (XAI)?
**Answer:** XAI encompasses techniques and methods that make artificial intelligence outputs understandable, transparent, and interpretable to human users, transforming black-box algorithms into glass-box systems.

### Q16: What is Grad-CAM?
**Answer:** Grad-CAM stands for **Gradient-Weighted Class Activation Mapping**. It is a visual explanation technique for CNNs that uses gradients of a target class score flowing into the final convolutional feature maps to produce a coarse localization map highlighting important image regions.

### Q17: How does Grad-CAM work step-by-step?
**Answer:**
1. Perform forward pass to get score $y^c$ for target class $c$.
2. Compute gradients $\partial y^c / \partial A^k$ w.r.t. feature map layer $A^k$.
3. Compute channel weights $\alpha_k$ via Global Average Pooling of gradients.
4. Calculate weighted sum of feature maps $\sum \alpha_k A^k$.
5. Apply ReLU activation to retain positive correlations.
6. Normalize and upscale heatmap to input image size.

### Q18: What does the Grad-CAM heatmap mean?
**Answer:** High-intensity regions (red/yellow pixels) represent image areas that exerted the strongest positive influence on the network's prediction for that specific disease class.

### Q19: What does Grad-CAM NOT mean?
**Answer:** Grad-CAM does **not** establish biological causation, prove pathogen presence under a microscope, or guarantee 100% real-world diagnostic correctness. It simply interprets internal model feature correlation.

### Q20: What are the current limitations of this system?
**Answer:**
1. Prototype scope covers 4 specific plant disease classes.
2. Background clutter or extreme shadows can affect heatmap clarity.
3. System runs locally on single leaf images rather than full field drone footage.

### Q21: What is the Future Scope for Phase 2, 3, and 4?
**Answer:**
- **Phase 2**: Healthcare disease risk prediction using tabular ML + SHAP explanations.
- **Phase 3**: Food quality image classification + visual XAI.
- **Phase 4**: Integrated Agriculture + Healthcare + Food Quality XAI platform.
