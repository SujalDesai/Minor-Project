import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report():
    pdf_path = PROJECT_ROOT / "docs" / "PROJECT_SUMMARY_FOR_CHATGPT.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#2E7D32"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#00838F"),
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A251E"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2C3E50"),
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2E7D32"),
        backColor=colors.HexColor("#F4F6F4"),
        spaceAfter=6
    )

    story = []

    # Title Banner
    story.append(Paragraph("🌱 Explainable AI for Crop Disease Detection", title_style))
    story.append(Paragraph("College Minor Project Milestone 1 — Implementation Summary Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E7D32"), spaceAfter=12))

    # Executive Overview
    story.append(Paragraph("Executive Overview & Status", heading_style))
    story.append(Paragraph(
        "This document provides a comprehensive technical reference of the Explainable AI (XAI) Crop Disease "
        "Detection module built in the workspace. It summarizes the project architecture, data acquisition, "
        "MobileNetV2 transfer learning pipeline, genuine Grad-CAM interpretability, Streamlit dashboard, "
        "and execution steps so you can copy/paste or reference this state in ChatGPT for further discussions.",
        body_style
    ))

    # Summary Table of Accomplishments
    data = [
        ["Component", "Implementation Status", "Details"],
        ["Project Structure", "Completed", "Modular architecture under crop-disease-xai/"],
        ["Dataset", "Downloaded Real Images", "PlantVillage dataset (Tomato & Potato leaf disease categories)"],
        ["Model Architecture", "MobileNetV2 Transfer Learning", "Pre-trained ImageNet backbone + GlobalAveragePooling + Softmax"],
        ["Explainable AI", "Genuine Grad-CAM", "TensorFlow GradientTape targeting 'out_relu' feature layer"],
        ["Web Application", "Streamlit Dashboard", "4 Tabs: Prediction, Explainability, Performance, About"],
        ["Evaluation", "Empirical Test Set", "Accuracy, Precision, Recall, F1, Confusion Matrix figures"],
        ["Testing & Docs", "Verified", "Pytest basic suite, README, VIVA.md, MID_EVALUATION.md"]
    ]
    
    t = Table(data, colWidths=[120, 130, 280])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E7D32')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F4F6F4')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Detailed Directory & File Breakdown
    story.append(Paragraph("Project Files & Module Responsibilities", heading_style))
    file_info = [
        "<b>app.py</b>: Main Streamlit dashboard implementing 4 interactive tabs and custom CSS styling.",
        "<b>src/config.py</b>: Centralized configuration parameters, image sizes, paths, confidence thresholds, and palette.",
        "<b>src/dataset_downloader.py</b>: Automated fetcher downloading real PlantVillage images from GitHub.",
        "<b>src/data_preprocessing.py</b>: Image integrity validation, stratified train/val/test splits, and augmentation.",
        "<b>src/model.py</b>: MobileNetV2 backbone definition and dynamic Grad-CAM layer inspector.",
        "<b>src/train.py</b>: Two-phase transfer learning execution, ModelCheckpoint, EarlyStopping, and curve generation.",
        "<b>src/evaluate.py</b>: Test set evaluation, empirical metrics calculation, confusion matrix, and distribution plots.",
        "<b>src/gradcam.py</b>: Genuine TensorFlow GradientTape Grad-CAM heatmap computation and OpenCV overlay blending.",
        "<b>src/predict.py</b>: High-level inference pipeline returning structured prediction, top-3 probabilities, and Grad-CAM.",
        "<b>src/utils.py</b>: Qualitative confidence ratings, narrative explanations, and academic disclaimers.",
        "<b>tests/test_basic.py</b>: Unit tests for preprocessing, model output shapes, softmax sum, and Grad-CAM dimensions.",
        "<b>docs/VIVA.md</b>: 21 comprehensive viva voce questions and answers.",
        "<b>docs/MID_EVALUATION.md</b>: 12-slide mid-evaluation presentation outline."
    ]
    for fi in file_info:
        story.append(Paragraph(f"• {fi}", body_style))

    story.append(Spacer(1, 10))

    # Execution Commands
    story.append(Paragraph("Commands to Run the Project", heading_style))
    story.append(Paragraph("1. Install requirements:", body_style))
    story.append(Paragraph("pip install -r requirements.txt", code_style))
    story.append(Paragraph("2. Download PlantVillage dataset images:", body_style))
    story.append(Paragraph("python src/dataset_downloader.py", code_style))
    story.append(Paragraph("3. Run unit test suite:", body_style))
    story.append(Paragraph("pytest tests/test_basic.py", code_style))
    story.append(Paragraph("4. Train MobileNetV2 model:", body_style))
    story.append(Paragraph("python src/train.py", code_style))
    story.append(Paragraph("5. Evaluate model on unseen test set:", body_style))
    story.append(Paragraph("python src/evaluate.py", code_style))
    story.append(Paragraph("6. Launch Streamlit XAI web dashboard:", body_style))
    story.append(Paragraph("streamlit run app.py", code_style))

    story.append(Spacer(1, 10))

    # Future Roadmap
    story.append(Paragraph("Future Scope Roadmap (Phases 2-4)", heading_style))
    story.append(Paragraph(
        "• <b>Phase 2</b>: Healthcare disease risk prediction using tabular ML + SHAP explanations.<br/>"
        "• <b>Phase 3</b>: Food quality image classification + visual XAI.<br/>"
        "• <b>Phase 4</b>: Unified Agriculture + Healthcare + Food Quality XAI platform.",
        body_style
    ))

    doc.build(story)
    print(f"PDF summary report successfully generated at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf_report()
