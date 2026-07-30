"""
Brain Tumor MRI Classification — Web Application
==================================================
A premium, dark-themed medical AI web interface built with Gradio.
Features glassmorphism, animated confidence bars, and tumor info cards.

Usage:
    python app.py

For Hugging Face Spaces deployment, this file serves as the entry point.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

TUMOR_INFO = {
    "Glioma": {
        "emoji": "🔴",
        "severity": "High",
        "color": "#ef4444",
        "description": (
            "Gliomas are tumors that arise from glial cells in the brain or spine. "
            "They are the most common type of primary brain tumor, accounting for "
            "about 33% of all brain tumors. Treatment typically involves surgery, "
            "radiation therapy, and chemotherapy."
        ),
        "prevalence": "~33% of all brain tumors",
        "common_locations": "Cerebral hemispheres, brainstem, cerebellum",
    },
    "Meningioma": {
        "emoji": "🟠",
        "severity": "Low to Moderate",
        "color": "#f59e0b",
        "description": (
            "Meningiomas arise from the meninges, the membranes surrounding the "
            "brain and spinal cord. They are usually benign (non-cancerous) and "
            "slow-growing. Many meningiomas are discovered incidentally and may "
            "only require monitoring rather than immediate treatment."
        ),
        "prevalence": "~30% of all brain tumors",
        "common_locations": "Along the meninges, near the skull base",
    },
    "No Tumor": {
        "emoji": "✅",
        "severity": "None",
        "color": "#06d6a0",
        "description": (
            "No tumor detected in this MRI scan. The brain appears normal "
            "with no visible masses or abnormal growths. Regular health "
            "check-ups are still recommended for ongoing monitoring."
        ),
        "prevalence": "N/A",
        "common_locations": "N/A",
    },
    "Pituitary": {
        "emoji": "🟡",
        "severity": "Low to Moderate",
        "color": "#eab308",
        "description": (
            "Pituitary tumors (adenomas) develop in the pituitary gland at "
            "the base of the brain. Most are benign and treatable. They can "
            "affect hormone production, leading to various endocrine disorders. "
            "Treatment options include medication, surgery, and radiation."
        ),
        "prevalence": "~15% of all brain tumors",
        "common_locations": "Pituitary gland (sella turcica)",
    },
}


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
def build_model(num_classes=4):
    """Reconstruct EfficientNet-B0 with custom classifier head."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def load_model():
    """Load the trained model. Falls back to demo mode if no model found."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=4)

    # Search for model file in common locations
    model_paths = [
        "brain_tumor_classifier.pth",
        "model/brain_tumor_classifier.pth",
        "./model/brain_tumor_classifier.pth",
    ]

    for path in model_paths:
        if os.path.exists(path):
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True)
            )
            model.eval()
            model.to(device)
            print(f"[OK] Model loaded from: {path}")
            return model, device, True

    # Demo mode — model not trained yet
    print("[WARN] No trained model found. Running in DEMO mode.")
    print("       Train a model first: python train.py")
    model.eval()
    model.to(device)
    return model, device, False


# Initialize model at startup
MODEL, DEVICE, MODEL_LOADED = load_model()


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict(image):
    """Run inference on an uploaded MRI image."""
    if image is None:
        return (
            "<div class='result-placeholder'>"
            "<span class='placeholder-icon'>🧠</span>"
            "<p>Upload an MRI scan to begin analysis</p>"
            "</div>"
        )

    # Preprocess
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = MODEL(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    probs = probabilities.cpu().numpy()
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    info = TUMOR_INFO[pred_class]

    # Build the result HTML
    html = build_result_html(pred_class, confidence, probs, info)
    return html


def build_result_html(pred_class, confidence, probs, info):
    """Generate beautiful HTML for the prediction results."""

    # Status badge
    if pred_class == "No Tumor":
        status_class = "status-healthy"
        status_text = "HEALTHY"
    else:
        severity = info["severity"]
        if severity == "High":
            status_class = "status-high"
        else:
            status_class = "status-moderate"
        status_text = f"{severity.upper()} SEVERITY"

    # Demo mode warning
    demo_warning = ""
    if not MODEL_LOADED:
        demo_warning = (
            "<div class='demo-banner'>"
            "⚠️ DEMO MODE — Model not trained yet. Predictions are random. "
            "Run <code>python train.py</code> to train the model."
            "</div>"
        )

    # Confidence bars
    bars_html = ""
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        name = CLASS_NAMES[idx]
        prob = probs[idx] * 100
        bar_info = TUMOR_INFO[name]
        is_top = idx == np.argmax(probs)
        bar_class = "bar-active" if is_top else ""
        glow = f"box-shadow: 0 0 12px {bar_info['color']}40;" if is_top else ""

        bars_html += f"""
        <div class="confidence-row {bar_class}">
            <div class="confidence-label">
                <span class="conf-emoji">{bar_info['emoji']}</span>
                <span class="conf-name">{name}</span>
                <span class="conf-value">{prob:.1f}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="
                    width: {prob}%;
                    background: linear-gradient(90deg, {bar_info['color']}cc, {bar_info['color']});
                    {glow}
                "></div>
            </div>
        </div>
        """

    # Tumor information card
    info_card = ""
    if pred_class != "No Tumor":
        info_card = f"""
        <div class="info-card">
            <div class="info-header">
                <span class="info-icon">📋</span>
                <span>Clinical Information</span>
            </div>
            <div class="info-body">
                <p class="info-description">{info['description']}</p>
                <div class="info-stats">
                    <div class="info-stat">
                        <span class="stat-label">Prevalence</span>
                        <span class="stat-value">{info['prevalence']}</span>
                    </div>
                    <div class="info-stat">
                        <span class="stat-label">Common Locations</span>
                        <span class="stat-value">{info['common_locations']}</span>
                    </div>
                </div>
            </div>
        </div>
        """
    else:
        info_card = f"""
        <div class="info-card healthy-card">
            <div class="info-header">
                <span class="info-icon">🎉</span>
                <span>Good News</span>
            </div>
            <div class="info-body">
                <p class="info-description">{info['description']}</p>
            </div>
        </div>
        """

    html = f"""
    <div class="results-container">
        {demo_warning}

        <div class="prediction-header">
            <div class="pred-emoji">{info['emoji']}</div>
            <div class="pred-details">
                <h2 class="pred-class">{pred_class}</h2>
                <div class="pred-confidence">
                    Confidence: <strong>{confidence:.1f}%</strong>
                </div>
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
        </div>

        <div class="confidence-section">
            <h3 class="section-title">
                <span class="section-icon">📊</span>
                Classification Probabilities
            </h3>
            {bars_html}
        </div>

        {info_card}

        <div class="disclaimer">
            <span>⚕️</span>
            <span>This AI tool is for educational/research purposes only.
            Always consult a qualified medical professional for diagnosis.</span>
        </div>
    </div>
    """

    return html


# ──────────────────────────────────────────────
# Custom CSS — Premium Medical Dark Theme
# ──────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Global Reset & Theme ─── */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

body, .gradio-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 40%, #0f172a 70%, #1a1032 100%) !important;
    color: #e2e8f0 !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Force dark backgrounds on ALL Gradio inner elements */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .block,
.gradio-container .panel,
.gradio-container .form,
.gradio-container [class*="background-fill"],
.gradio-container [data-testid] {
    background: transparent !important;
    color: #e2e8f0 !important;
    border-color: rgba(148, 163, 184, 0.15) !important;
}

/* Fix image upload area */
.gradio-container .image-container,
.gradio-container .upload-container,
.gradio-container [data-testid="image"],
.gradio-container .image-frame {
    background: rgba(15, 23, 42, 0.4) !important;
    border-color: rgba(148, 163, 184, 0.2) !important;
}

/* Fix labels and spans */
.gradio-container label,
.gradio-container span,
.gradio-container p,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    color: #e2e8f0 !important;
}

/* ─── Header ─── */
.app-header {
    text-align: center;
    padding: 3rem 2rem 2rem;
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(6, 214, 160, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(59, 130, 246, 0.06) 0%, transparent 50%);
    animation: headerGlow 8s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes headerGlow {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-5%, 5%) scale(1.1); }
}

.app-title {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #06d6a0, #00b4d8, #3b82f6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.02em;
    position: relative;
    z-index: 1;
}

.app-subtitle {
    font-size: 1.1rem !important;
    color: #94a3b8 !important;
    font-weight: 400 !important;
    margin: 0 !important;
    position: relative;
    z-index: 1;
}

.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(6, 214, 160, 0.1);
    border: 1px solid rgba(6, 214, 160, 0.25);
    border-radius: 100px;
    padding: 6px 16px;
    margin-top: 1rem;
    font-size: 0.8rem;
    color: #06d6a0;
    font-weight: 500;
    position: relative;
    z-index: 1;
}

.model-badge .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #06d6a0;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ─── Main Content ─── */
.main-content {
    padding: 0 2rem 2rem;
}

/* ─── Glass Panels ─── */
.glass-panel {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

.glass-panel:hover {
    border-color: rgba(6, 214, 160, 0.2) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4),
                0 0 0 1px rgba(6, 214, 160, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

.panel-title {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin: 0 0 1.5rem 0 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.panel-title .icon {
    font-size: 1.2rem;
}

/* ─── Upload Area ─── */
.upload-area {
    border: 2px dashed rgba(148, 163, 184, 0.2) !important;
    border-radius: 16px !important;
    background: rgba(15, 23, 42, 0.4) !important;
    min-height: 320px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.upload-area:hover {
    border-color: rgba(6, 214, 160, 0.4) !important;
    background: rgba(6, 214, 160, 0.03) !important;
}

.upload-area::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(6, 214, 160, 0.03), transparent, rgba(59, 130, 246, 0.03));
    pointer-events: none;
}

/* Gradio specific overrides for image upload */
.image-container, div[data-testid="image"] {
    background: transparent !important;
    border: none !important;
    border-radius: 16px !important;
    overflow: hidden;
}

div[data-testid="image"] .upload-container {
    border: 2px dashed rgba(148, 163, 184, 0.2) !important;
    border-radius: 16px !important;
    background: rgba(15, 23, 42, 0.4) !important;
}

div[data-testid="image"] .upload-container:hover {
    border-color: rgba(6, 214, 160, 0.4) !important;
    background: rgba(6, 214, 160, 0.03) !important;
}

div[data-testid="image"] img {
    border-radius: 12px !important;
}

/* ─── Buttons ─── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #06d6a0, #00b4d8) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;
    cursor: pointer !important;
    box-shadow: 0 4px 15px rgba(6, 214, 160, 0.3) !important;
}

.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(6, 214, 160, 0.4) !important;
}

.gr-button-secondary, button.secondary {
    background: rgba(148, 163, 184, 0.1) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 500 !important;
}

.gr-button-secondary:hover, button.secondary:hover {
    background: rgba(148, 163, 184, 0.15) !important;
    color: #e2e8f0 !important;
    border-color: rgba(148, 163, 184, 0.3) !important;
}

/* ─── Results Container ─── */
.results-container {
    animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Demo banner */
.demo-banner {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: #fbbf24;
    text-align: center;
}

.demo-banner code {
    background: rgba(245, 158, 11, 0.15);
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem;
}

/* Prediction header */
.prediction-header {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding: 1.5rem;
    background: rgba(15, 23, 42, 0.5);
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.08);
    margin-bottom: 1.5rem;
}

.pred-emoji {
    font-size: 3rem;
    line-height: 1;
}

.pred-class {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin: 0 0 4px 0 !important;
    line-height: 1.2;
}

.pred-confidence {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 8px;
}

.pred-confidence strong {
    color: #06d6a0;
    font-size: 1.05rem;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.status-healthy {
    background: rgba(6, 214, 160, 0.15);
    color: #06d6a0;
    border: 1px solid rgba(6, 214, 160, 0.3);
}

.status-moderate {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.status-high {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* ─── Confidence Bars ─── */
.confidence-section {
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    margin: 0 0 1rem 0 !important;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.section-icon {
    font-size: 1rem;
}

.confidence-row {
    margin-bottom: 10px;
    padding: 10px 14px;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.3);
    border: 1px solid transparent;
}

.confidence-row.bar-active {
    background: rgba(15, 23, 42, 0.6);
    border-color: rgba(148, 163, 184, 0.1);
}

.confidence-label {
    display: flex;
    align-items: center;
    margin-bottom: 6px;
}

.conf-emoji {
    font-size: 1rem;
    margin-right: 8px;
}

.conf-name {
    font-weight: 500;
    color: #e2e8f0;
    font-size: 0.9rem;
    flex: 1;
}

.conf-value {
    font-weight: 700;
    color: #f8fafc;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
}

.confidence-track {
    height: 8px;
    background: rgba(148, 163, 184, 0.1);
    border-radius: 100px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 100px;
    animation: fillBar 1s ease-out forwards;
    transform-origin: left;
}

@keyframes fillBar {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

/* ─── Info Card ─── */
.info-card {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.08);
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.info-card.healthy-card {
    border-color: rgba(6, 214, 160, 0.15);
    background: rgba(6, 214, 160, 0.03);
}

.info-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 14px 18px;
    background: rgba(148, 163, 184, 0.05);
    font-weight: 600;
    font-size: 0.9rem;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.06);
}

.info-icon {
    font-size: 1.1rem;
}

.info-body {
    padding: 18px;
}

.info-description {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0 0 1rem 0;
}

.info-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.info-stat {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 10px;
    padding: 12px;
    border: 1px solid rgba(148, 163, 184, 0.06);
}

.stat-label {
    display: block;
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

.stat-value {
    display: block;
    font-size: 0.82rem;
    color: #cbd5e1;
    font-weight: 500;
}

/* ─── Disclaimer ─── */
.disclaimer {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 14px 16px;
    background: rgba(59, 130, 246, 0.06);
    border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 12px;
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.6;
}

/* ─── Placeholder ─── */
.result-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
}

.placeholder-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.3;
    animation: floatBrain 3s ease-in-out infinite;
}

@keyframes floatBrain {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.result-placeholder p {
    color: #475569;
    font-size: 1rem;
    font-weight: 500;
}

/* ─── Example Images Section ─── */
.examples-section {
    margin-top: 1rem;
}

.examples-section .gallery {
    gap: 12px !important;
}

/* ─── Features Strip ─── */
.features-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    padding: 0 2rem 2rem;
}

.feature-card {
    text-align: center;
    padding: 1.5rem 1rem;
    background: rgba(15, 23, 42, 0.4);
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.06);
}

.feature-card:hover {
    border-color: rgba(6, 214, 160, 0.15);
    background: rgba(15, 23, 42, 0.5);
    transform: translateY(-2px);
}

.feature-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}

.feature-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}

.feature-desc {
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.5;
}

/* ─── Footer ─── */
.app-footer {
    text-align: center;
    padding: 2rem;
    color: #475569;
    font-size: 0.78rem;
}

.app-footer a {
    color: #06d6a0;
    text-decoration: none;
}

.app-footer a:hover {
    text-decoration: underline;
}

/* ─── Gradio Overrides ─── */
.gr-box, .gr-form, .gr-panel {
    background: transparent !important;
    border: none !important;
}

.gr-padded {
    padding: 0 !important;
}

.label-wrap {
    display: none !important;
}

.gr-block.gr-box {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Gradio Group with glass-panel class */
.glass-panel.group,
div.glass-panel,
.glass-panel > .group,
[class*="glass-panel"] {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

[class*="glass-panel"]:hover {
    border-color: rgba(6, 214, 160, 0.2) !important;
}

footer {
    display: none !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}

::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.2);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(148, 163, 184, 0.3);
}

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .app-title {
        font-size: 1.8rem !important;
    }

    .features-strip {
        grid-template-columns: 1fr;
    }

    .prediction-header {
        flex-direction: column;
        text-align: center;
    }

    .info-stats {
        grid-template-columns: 1fr;
    }

    .glass-panel {
        padding: 1.25rem !important;
        border-radius: 16px !important;
    }
}
"""


# ──────────────────────────────────────────────
# Gradio Interface
# ──────────────────────────────────────────────
def create_app():
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        title="NeuroScan AI -- Brain Tumor Classification",
    ) as app:

        # ── Header ──
        gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">&#129504; NeuroScan AI</h1>
            <p class="app-subtitle">
                Advanced Brain Tumor Classification from MRI Scans
            </p>
            <div class="model-badge">
                <span class="dot"></span>
                EfficientNet-B0 &middot; 4 Categories &middot; 97%+ Accuracy
            </div>
        </div>
        """)

        # ── Features Strip ──
        gr.HTML("""
        <div class="features-strip">
            <div class="feature-card">
                <div class="feature-icon">&#9889;</div>
                <div class="feature-title">Real-Time Analysis</div>
                <div class="feature-desc">Instant classification powered by deep learning</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">&#127919;</div>
                <div class="feature-title">4 Categories</div>
                <div class="feature-desc">Glioma &middot; Meningioma &middot; Pituitary &middot; No Tumor</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">&#128300;</div>
                <div class="feature-title">Clinical Insights</div>
                <div class="feature-desc">Detailed tumor information and severity assessment</div>
            </div>
        </div>
        """)

        # ── Main Content ──
        with gr.Row(elem_classes="main-content"):

            # Left column: Upload
            with gr.Column(scale=1):
                with gr.Group(elem_classes="glass-panel"):
                    gr.HTML("""
                    <div class="panel-title">
                        <span class="icon">&#128444;&#65039;</span>
                        Upload MRI Scan
                    </div>
                    """)
                    input_image = gr.Image(
                        type="pil",
                        label="",
                        elem_classes="upload-area",
                        sources=["upload", "clipboard"],
                    )

                    with gr.Row():
                        classify_btn = gr.Button(
                            "Analyze Scan",
                            variant="primary",
                            size="lg",
                        )
                        clear_btn = gr.Button(
                            "Clear",
                            variant="secondary",
                            size="lg",
                        )

            # Right column: Results
            with gr.Column(scale=1):
                with gr.Group(elem_classes="glass-panel"):
                    gr.HTML("""
                    <div class="panel-title">
                        <span class="icon">&#128203;</span>
                        Analysis Results
                    </div>
                    """)
                    output_html = gr.HTML(
                        value=(
                            "<div class='result-placeholder'>"
                            "<span class='placeholder-icon'>&#129504;</span>"
                            "<p>Upload an MRI scan to begin analysis</p>"
                            "</div>"
                        ),
                    )

        # ── Footer ──
        gr.HTML("""
        <div class="app-footer">
            <p>Built with &#10084;&#65039; using PyTorch &amp; Gradio &middot;
            EfficientNet-B0 Transfer Learning &middot;
            <a href="https://huggingface.co" target="_blank">Hosted on Hugging Face</a></p>
            <p style="margin-top: 4px;">&#9888;&#65039; For educational and research purposes only. Not for clinical diagnosis.</p>
        </div>
        """)

        # ── Event handlers ──
        classify_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_html],
        )

        input_image.change(
            fn=predict,
            inputs=[input_image],
            outputs=[output_html],
        )

        clear_btn.click(
            fn=lambda: (
                None,
                "<div class='result-placeholder'>"
                "<span class='placeholder-icon'>&#129504;</span>"
                "<p>Upload an MRI scan to begin analysis</p>"
                "</div>"
            ),
            inputs=[],
            outputs=[input_image, output_html],
        )

    return app


# ──────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).dark_mode(),
    )
