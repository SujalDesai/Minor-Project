import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

# Add src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    MODEL_PATH, CLASS_NAMES_PATH, FIGURES_DIR, REPORTS_DIR, 
    COLOR_PALETTE, EXAMPLES_DIR, CONFIDENCE_THRESHOLDS
)
from src.predict import predict_crop_disease, load_inference_model

# Page Configuration
st.set_page_config(
    page_title="AgriHealth XAI - Crop Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Senior Front-End CSS Styling (AgriHealth Living Green & Clean Teal Theme)
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F8FAF8;
        color: #1A251E;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #121A14 !important;
        border-right: 1px solid #1E2D22;
    }
    [data-testid="stSidebar"] * {
        color: #E2EFE5 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] code {
        background-color: #1C3322 !important;
        color: #A5D6A7 !important;
        border: 1px solid #2E7D32 !important;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.85rem;
    }
    
    /* Main Header Banner */
    .main-header {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #00838F 100%);
        padding: 28px 36px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.2);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #E0F2F1 !important;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
        opacity: 0.95;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 4px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px;
        padding: 0 24px;
        background-color: #FFFFFF;
        color: #2E7D32 !important;
        font-weight: 600;
        border: 1px solid #C8E6C9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E8F5E9;
        border-color: #2E7D32;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        border: 1px solid #2E7D32 !important;
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.25) !important;
    }
    .stTabs [aria-selected="true"] * {
        color: #FFFFFF !important;
    }

    /* File Uploader Fix */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    [data-testid="stFileUploaderDropZone"] {
        background-color: #F8FAF8 !important;
        border: 2px dashed #2E7D32 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploaderDropZone"] * {
        color: #1A251E !important;
    }

    /* Infobox / Alert Styling Overrides (Fix white-on-blue illegibility) */
    div[data-testid="stAlert"], div.stAlert {
        background-color: #E0F2FE !important;
        border: 1px solid #BAE6FD !important;
        border-left: 5px solid #0284C7 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.08) !important;
    }
    div[data-testid="stAlert"] *, div.stAlert * {
        color: #034E68 !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
    }

    /* UI Cards */
    .custom-card {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    .placeholder-card {
        background: #FFFFFF;
        border: 2px dashed #CBD5E1;
        border-radius: 14px;
        padding: 32px;
        text-align: center;
        color: #475569;
    }
    .placeholder-card h4 {
        color: #1E293B;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #2E7D32;
        border-top: 1px solid #F1F5F9;
        border-right: 1px solid #F1F5F9;
        border-bottom: 1px solid #F1F5F9;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    .metric-card-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        font-weight: 700;
    }
    .metric-card-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 6px solid #D97706;
        padding: 20px 24px;
        border-radius: 12px;
        margin-top: 20px;
        color: #78350F;
    }
    .disclaimer-box h4 {
        color: #92400E;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 8px;
    }
    .disclaimer-box ul {
        margin-bottom: 0;
        padding-left: 20px;
    }
    .disclaimer-box li {
        color: #78350F;
        margin-bottom: 4px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=64)
        st.title("AgriHealth XAI")
        st.caption("Explainable AI for Plant Pathology")
        st.markdown("---")
        
        st.subheader("⚙️ System Status")
        if MODEL_PATH.exists() and CLASS_NAMES_PATH.exists():
            st.markdown("🟢 **Model Loaded** `MobileNetV2`")
            with open(CLASS_NAMES_PATH, "r") as f:
                c_names = json.load(f)
            st.markdown(f"**Target Pathology Classes ({len(c_names)}):**")
            for cn in c_names:
                st.markdown(f"- `{cn}`")
        else:
            st.error("🔴 Model weights not found!")
            st.info("Run `python src/train.py` to train model.")
            
        st.markdown("---")
        st.subheader("💡 Sample Test Leaves")
        sample_files = list(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
        selected_sample = None
        if sample_files:
            sample_options = ["None (Upload Custom Image)"] + [f.name for f in sample_files]
            chosen_sample_name = st.selectbox("Quick Sample Loader", sample_options)
            if chosen_sample_name != "None (Upload Custom Image)":
                selected_sample = EXAMPLES_DIR / chosen_sample_name
                st.image(str(selected_sample), caption=chosen_sample_name, use_container_width=True)

        st.markdown("---")
        st.caption("College Minor Project | CSE Milestone 1")

    # Main Header Banner
    st.markdown(
        """
        <div class="main-header">
            <h1>🌱 Explainable AI for Crop Disease Detection</h1>
            <p>Transparent plant disease classification using Transfer Learning (MobileNetV2) and Gradient-Weighted Class Activation Mapping (Grad-CAM)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌱 Prediction", 
        "🔍 Explainability (Grad-CAM)", 
        "📊 Model Performance", 
        "ℹ️ About Project"
    ])

    # -------------------------------------------------------------
    # TAB 1: PREDICTION
    # -------------------------------------------------------------
    with tab1:
        st.markdown("### Upload Plant Leaf Image")
        col_up, col_prev = st.columns([1, 1], gap="large")

        image_to_analyze = None
        
        with col_up:
            uploaded_file = st.file_uploader(
                "Select a leaf image (JPG, JPEG, PNG)", 
                type=["jpg", "jpeg", "png"],
                help="Upload a clear close-up image of a crop leaf."
            )
            if uploaded_file is not None:
                image_to_analyze = Image.open(uploaded_file)
            elif selected_sample is not None:
                image_to_analyze = Image.open(selected_sample)

        with col_prev:
            if image_to_analyze is not None:
                st.markdown("#### Input Image Preview")
                st.image(image_to_analyze, caption="Selected Input Leaf Image", use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="placeholder-card">
                        <div style="font-size: 2.5rem; margin-bottom: 8px;">🍃</div>
                        <h4>No Leaf Image Selected</h4>
                        <p>Upload a JPG/PNG file on the left or select a sample image from the sidebar to begin analysis.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if image_to_analyze is not None:
            st.markdown("---")
            analyze_btn = st.button("🚀 Analyze Leaf Image", type="primary", use_container_width=True)
            
            if analyze_btn or "last_result" in st.session_state:
                if analyze_btn:
                    with st.spinner("Running MobileNetV2 Inference & Grad-CAM Gradient Tape..."):
                        try:
                            result = predict_crop_disease(image_to_analyze, compute_gradcam=True)
                            st.session_state["last_result"] = result
                        except Exception as e:
                            st.error(f"Prediction Error: {e}")
                            return

                result = st.session_state.get("last_result")
                if result:
                    st.success("✅ Analysis Completed Successfully!")
                    res_col1, res_col2 = st.columns([1, 1], gap="large")
                    
                    with res_col1:
                        st.markdown(f"### Predicted Diagnosis")
                        st.markdown(f"**Class:** `{result['predicted_class']}`")
                        
                        exp = result["explanation"]
                        st.markdown(
                            f"""
                            <div class="metric-card" style="border-left-color: {exp['confidence_color']};">
                                <div class="metric-card-title">{exp['confidence_badge']}</div>
                                <div class="metric-card-value" style="color: {exp['confidence_color']};">{result['confidence_str']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown(exp["summary"])

                    with res_col2:
                        st.markdown("### Top 3 Probabilities")
                        top3_df = pd.DataFrame(result["top3_predictions"])
                        top3_df["Probability"] = top3_df["probability"]
                        st.bar_chart(
                            data=top3_df,
                            x="class_name",
                            y="Probability",
                            color="#2E7D32",
                            use_container_width=True
                        )

    # -------------------------------------------------------------
    # TAB 2: EXPLAINABILITY (GRAD-CAM)
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### Gradient-Weighted Class Activation Mapping (Grad-CAM)")
        
        result = st.session_state.get("last_result")
        if result is None:
            st.markdown(
                """
                <div class="placeholder-card">
                    <div style="font-size: 2.5rem; margin-bottom: 8px;">🔍</div>
                    <h4>Grad-CAM Visualizations Ready</h4>
                    <p>Please upload and analyze a leaf image in the <strong>'Prediction'</strong> tab first to view visual explanations.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"**Target Predicted Class:** `{result['predicted_class']}` | "
                f"**Feature Map Layer:** `MobileNetV2 (out_relu)`"
            )
            
            c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
            with c1:
                st.markdown("##### 1. Original Leaf Image")
                st.image(result["original_image"], use_container_width=True)
            with c2:
                st.markdown("##### 2. Raw Grad-CAM Heatmap")
                if result["heatmap_rgb"] is not None:
                    st.image(result["heatmap_rgb"], use_container_width=True)
                else:
                    st.warning("Heatmap unavailable.")
            with c3:
                st.markdown("##### 3. Visual Overlay")
                if result["overlay_rgb"] is not None:
                    st.image(result["overlay_rgb"], use_container_width=True)
                else:
                    st.warning("Overlay unavailable.")

            st.markdown("---")
            st.markdown("### 💡 Interpretability Explanation")
            exp = result["explanation"]
            st.write(exp["gradcam_explanation"])
            
            st.markdown(
                f"""
                <div class="disclaimer-box">
                    <h4>Academic & Scientific Disclaimer</h4>
                    <p>{exp['disclaimer']}</p>
                    <strong>What Grad-CAM does NOT mean:</strong>
                    <ul>
                        {"".join([f"<li>{item}</li>" for item in exp['limitations']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------
    # TAB 3: MODEL PERFORMANCE
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### Model Validation & Empirical Performance")
        
        report_path = REPORTS_DIR / "classification_report.json"
        if report_path.exists():
            with open(report_path, "r") as f:
                metrics = json.load(f)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
            with m2:
                st.metric("Macro Precision", f"{metrics['precision_macro']:.4f}")
            with m3:
                st.metric("Macro Recall", f"{metrics['recall_macro']:.4f}")
            with m4:
                st.metric("Macro F1-Score", f"{metrics['f1_macro']:.4f}")

            st.markdown("---")
            col_fig1, col_fig2 = st.columns(2, gap="large")
            
            cm_path = FIGURES_DIR / "confusion_matrix.png"
            curves_path = FIGURES_DIR / "accuracy_loss_curves.png"
            dist_path = FIGURES_DIR / "class_distribution.png"

            with col_fig1:
                st.markdown("#### Test Confusion Matrix")
                if cm_path.exists():
                    st.image(str(cm_path), use_container_width=True)
                else:
                    st.info("Confusion matrix figure will appear after running `evaluate.py`.")

            with col_fig2:
                st.markdown("#### Training & Validation Curves")
                if curves_path.exists():
                    st.image(str(curves_path), use_container_width=True)
                else:
                    st.info("Training curves figure will appear after running `train.py`.")

            st.markdown("---")
            if dist_path.exists():
                st.markdown("#### Raw Dataset Class Distribution")
                st.image(str(dist_path), use_container_width=True)
        else:
            st.info("📊 Model performance metrics will be displayed here once `src/evaluate.py` has executed.")

    # -------------------------------------------------------------
    # TAB 4: ABOUT PROJECT
    # -------------------------------------------------------------
    with tab4:
        st.markdown("### Project Overview & Architecture")
        
        st.markdown(
            """
            ### 📌 Problem Statement & Objective
            Crop diseases pose a major threat to agricultural security and yield. Deep learning visual classifiers 
            can automatically detect leaf diseases, but standard deep neural networks operate as black boxes. 
            This project provides **Explainable AI (XAI)** using **Grad-CAM** to make disease diagnoses transparent, 
            interpretable, and actionable for farmers and agronomists.

            ---

            ### 🏗 System Architecture Diagram
            ```mermaid
            graph TD
                A[Leaf Image Upload] --> B[Data Preprocessing 224x224 RGB]
                B --> C[MobileNetV2 Backbone Transfer Learning]
                C --> D[Softmax Disease Classifier]
                D --> E[Predicted Class & Confidence]
                C --> F[Grad-CAM Gradient Tape Analysis]
                F --> G[Visual Heatmap & Overlay]
                E --> H[Streamlit XAI Dashboard]
                G --> H
            ```

            ---

            ### 🔬 Technical Methodology
            - **Backbone**: MobileNetV2 pre-trained on ImageNet.
            - **Transfer Learning**: Global Average Pooling + Dropout(0.2) + Dense Softmax Classifier.
            - **XAI Technique**: Gradient-Weighted Class Activation Mapping (Grad-CAM) targeting layer `out_relu`.
            - **Dataset**: PlantVillage public plant pathology dataset.
            - **Inference Pipeline**: 100% Local Python execution without third-party API dependencies.

            ---

            ### 🚀 Limitations & Future Scope
            - **Current Milestone Scope**: Focuses on crop disease detection (Tomato & Potato leaf disease subset).
            - **Future Scope (Phase 2 & 3)**:
              1. Healthcare disease-risk prediction using tabular ML + SHAP explanations.
              2. Food quality image classification + visual XAI.
              3. Unified Agriculture + Healthcare + Food Quality XAI Platform.
            """
        )

if __name__ == "__main__":
    main()
