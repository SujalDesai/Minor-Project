# 📊 Mid-Evaluation Presentation Outline (12 Slides)

> **Project Title:** Explainable AI for Agriculture and Healthcare  
> **Current Milestone:** Phase 1 — Explainable AI for Crop Disease Detection

---

### Slide 1: Title Slide
- **Title:** Explainable AI for Crop Disease Detection
- **Subtitle:** Transparent Plant Pathology Classification using Transfer Learning & Grad-CAM
- **Presenter:** CSE Minor Project Student
- **Domain:** Artificial Intelligence / Computer Vision / Explainable AI (XAI)
- **Visual Recommendation:** System Header Banner with Leaf Icon & Grad-CAM Overlay graphic.

---

### Slide 2: Problem Statement
- **Agricultural Pathology Impact:** Crop diseases cause significant yield loss globally.
- **Black-Box AI Limitation:** Standard deep neural networks classify images without explaining *why*.
- **Lack of Trust:** Farmers & agronomists hesitate to adopt uninterpretable ML models for real-world crop management.
- **Visual Recommendation:** Split image showing a black-box model vs. an Explainable AI model.

---

### Slide 3: Motivation
- Bridge the gap between high-accuracy Deep Learning models and human decision trust.
- Provide actionable visual attributions that correlate with leaf lesion regions.
- Build a lightweight local inference system that runs on accessible hardware.
- **Visual Recommendation:** Photograph of an affected crop leaf alongside a smartphone dashboard mockup.

---

### Slide 4: Objectives
- Acquire real PlantVillage agricultural dataset for key crop pathologies.
- Build an end-to-end MobileNetV2 transfer learning pipeline.
- Implement genuine Keras GradientTape **Grad-CAM** feature map visualization.
- Develop an interactive academic web dashboard with Streamlit.
- Validate model performance using empirical metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
- **Visual Recommendation:** Bulleted checklist with green status icons.

---

### Slide 5: Proposed System
- **Input:** Single leaf image (upload or pre-loaded sample).
- **Processing:** Image integrity validation $\rightarrow$ $224 \times 224 \times 3$ RGB resizing $\rightarrow$ MobileNetV2 feature extraction.
- **Output:** Predicted Disease Class + Confidence Score + Top-3 Breakdown + Grad-CAM Heatmap & Overlay + Qualitative Narrative.
- **Visual Recommendation:** End-to-end block diagram showing inputs, model block, and outputs.

---

### Slide 6: System Architecture
- Flow chart of training and inference pipelines.
- Data discovery, stratified splits, data augmentation.
- MobileNetV2 feature extractor $\rightarrow$ GlobalAveragePooling2D $\rightarrow$ Dropout(0.2) $\rightarrow$ Softmax classifier.
- Grad-CAM gradient flow w.r.t. layer `out_relu`.
- **Visual Recommendation:** Mermaid Architecture Diagram from `README.md`.

---

### Slide 7: Dataset & Preprocessing
- **Dataset:** PlantVillage Public Plant Pathology Dataset.
- **Classes (4 Scope):** Tomato Healthy, Tomato Early Blight, Tomato Late Blight, Potato Early Blight.
- **Stratification:** 70% Train, 15% Validation, 15% Test.
- **Augmentation:** Rotation ($\pm 20^\circ$), Horizontal Flip, Zoom, Translation.
- **Visual Recommendation:** Bar graph of dataset class distribution (`outputs/figures/class_distribution.png`).

---

### Slide 8: Model Architecture (MobileNetV2)
- Lightweight Depthwise Separable Convolutions.
- Pre-trained ImageNet weights transferred for low sample requirement.
- Two-Phase Training: Head training ($lr=10^{-4}$) followed by fine-tuning ($lr=10^{-5}$).
- Early stopping and model checkpoint callbacks to prevent overfitting.
- **Visual Recommendation:** Model summary layer box diagram.

---

### Slide 9: Explainable AI & Grad-CAM
- Gradient-Weighted Class Activation Mapping formulation:
  $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
- Genuine computation via TensorFlow `tf.GradientTape`.
- Generates heatmap overlays pinpointing influential visual regions.
- Academic disclaimers detailing what Grad-CAM means vs. what it does not mean.
- **Visual Recommendation:** Side-by-side comparison: Original Leaf | Raw Heatmap | Visual Overlay.

---

### Slide 10: Current Empirical Results
- **Test Accuracy:** Derived directly from unseen test set evaluation.
- **Macro F1-Score / Precision / Recall:** Real evaluation metrics.
- **Confusion Matrix Analysis:** Zero hardcoded values.
- **Visual Recommendation:** Test set confusion matrix (`outputs/figures/confusion_matrix.png`) & training curves (`outputs/figures/accuracy_loss_curves.png`).

---

### Slide 11: Working Streamlit Demo
- Streamlit Academic Dashboard UI featuring 4 Tabs:
  1. 🌱 **Prediction**: Image upload & top-3 confidence bar chart.
  2. 🔍 **Explainability**: Heatmap overlays & narrative explanations.
  3. 📊 **Model Performance**: Confusion matrix & live metrics.
  4. ℹ️ **About Project**: System architecture & limitations.
- **Visual Recommendation:** Screenshot of the live Streamlit dashboard running.

---

### Slide 12: Future Scope & Roadmap
- **Phase 2:** Healthcare disease risk prediction using tabular ML + SHAP explanations.
- **Phase 3:** Food quality image classification + visual XAI.
- **Phase 4:** Unified Agriculture + Healthcare + Food Quality XAI Platform.
- **Visual Recommendation:** Multi-phase project milestone timeline chart.
