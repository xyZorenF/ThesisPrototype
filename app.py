"""
Gradio Web Application: Arabica Coffee Leaf Nutrient Deficiency Classification.
Thesis Research Prototype:
"Cross-Anomaly Few-Shot Domain Adaptation for Local Nutrient Deficiency"

Architecture:
- Feature Extractor: MobileNetV2 (ImageNet init -> JMuBEN source-trained -> FROZEN at inference)
- Target Classifier: 3-Way K-Shot Nearest-Centroid Prototypical Classifier (CoLeaf support set: N, P, K)
- Explainability: Target-score Prototype Grad-CAM (-squared distance to c_k from 'out_relu')
- Preprocessing: Optional OpenCV Leaf Isolation (applied identically to support and query)
"""

import os
import sys
import argparse
import numpy as np
import gradio as gr
from PIL import Image

from core.model import load_proposed_encoder
from core.roi import preprocess_leaf_roi
from core.dataset import load_coleaf_support_dataset, generate_synthetic_coleaf, TARGET_CLASSES
from core.prototypes import PrototypeClassifier
from core.gradcam import compute_prototype_gradcam, overlay_gradcam

# Global application state
ENCODER_MODEL = None
CONV_MODEL = None
CLASSIFIER = None
ENCODER_STATUS = ""
IS_FALLBACK = False


def initialize_system(demo_mode: bool = False, weights_path: str = "weights/jmuben_encoder.weights.h5"):
    """Initializes the proposed model, datasets, and prototypical classifier."""
    global ENCODER_MODEL, CONV_MODEL, CLASSIFIER, ENCODER_STATUS, IS_FALLBACK

    print("=" * 70)
    print("INITIALIZING THESIS PROTOTYPE: COFFEE NUTRIENT DEFICIENCY (N-P-K)")
    print("=" * 70)

    # Check for CoLeaf dataset
    dataset_map, found_root = load_coleaf_support_dataset()
    has_data = all(len(dataset_map.get(cls, [])) > 0 for cls in TARGET_CLASSES)

    if not has_data or demo_mode:
        print("[*] CoLeaf dataset not fully present. Generating synthetic sample dataset for prototype demo...")
        generate_synthetic_coleaf(output_dir="dataset/coleaf", samples_per_class=12)
        dataset_map, found_root = load_coleaf_support_dataset()

    print(f"[*] CoLeaf Support Dataset Root: '{found_root}'")
    for cls in TARGET_CLASSES:
        print(f"    - Class {cls}: {len(dataset_map.get(cls, []))} available images")

    # Load proposed MobileNetV2 encoder
    ENCODER_MODEL, CONV_MODEL, ENCODER_STATUS, IS_FALLBACK = load_proposed_encoder(weights_path=weights_path)
    print(ENCODER_STATUS)

    # Initialize prototypical classifier with caching
    CLASSIFIER = PrototypeClassifier(
        feature_extractor=ENCODER_MODEL,
        conv_model=CONV_MODEL,
        base_dirs=["dataset/coleaf", "support", "data/coleaf"],
        seed=42,
    )
    print("[*] Prototype Classifier initialized with fixed random seed (42). Ready.\n")


def predict_nutrient_deficiency(input_image, roi_preprocessing):
    """
    Main prediction pipeline called by the Gradio UI.
    Few-shot learning support set (10-shot) is handled strictly in the backend.
    """
    global CLASSIFIER, ENCODER_STATUS, IS_FALLBACK

    if input_image is None:
        return (
            "⚠️ Please upload or capture a coffee leaf image.",
            {},
            None,
            None,
            "No input image provided.",
        )

    # Handled strictly in the backend: 10-shot support set per deficiency class
    k_shots = 10

    # Ensure RGB numpy array
    if isinstance(input_image, Image.Image):
        query_rgb = np.array(input_image.convert("RGB"))
    else:
        query_rgb = np.array(input_image)
        if query_rgb.shape[-1] == 4:
            query_rgb = query_rgb[:, :, :3]  # Drop alpha channel if present

    # Execute classification
    try:
        result = CLASSIFIER.classify_image(
            rgb_image=query_rgb,
            k_shots=k_shots,
            apply_roi=bool(roi_preprocessing),
        )
    except Exception as e:
        err_msg = f"Classification Error: {str(e)}"
        return err_msg, {}, None, None, err_msg

    predicted_label = result["predicted_label"]
    predicted_class = result["predicted_class"]
    confidences = result["confidences"]
    distances = result["distances"]
    preprocessed_img = result["preprocessed_image"]
    target_proto = result["prototype_tensor"]
    warnings = result["warnings"]

    # Compute target-score prototype Grad-CAM (using 'out_relu')
    heatmap_224 = compute_prototype_gradcam(
        conv_model=CONV_MODEL,
        preprocessed_rgb=preprocessed_img,
        target_prototype=target_proto,
    )
    gradcam_overlay = overlay_gradcam(preprocessed_img, heatmap_224, alpha=0.45)

    # Formulate distance and diagnosis summary text
    status_notes = []
    if IS_FALLBACK:
        status_notes.append("⚠️ **Notice:** Encoder using ImageNet fallback weights (`weights/jmuben_encoder.weights.h5` not found).")
    if warnings:
        for w in warnings:
            status_notes.append(f"⚠️ {w}")

    dist_summary = (
        f"### **Prediction:** {predicted_label} *(10-Shot Prototypical Nearest-Centroid)*\n"
        f"- **Smallest Squared Euclidean Distance:** `{distances[predicted_class]:.4f}`\n\n"
        f"**Distances to CoLeaf Prototypes:**\n"
        f"- **N (Nitrogen):** `{distances['N']:.4f}`\n"
        f"- **P (Phosphorus):** `{distances['P']:.4f}`\n"
        f"- **K (Potassium):** `{distances['K']:.4f}`\n\n"
        f"ℹ️ *Note: Confidence bars display softmax relative-closeness scores derived from negative Euclidean distances. "
        f"They represent relative similarity to support prototypes and are NOT calibrated diagnostic probabilities.*"
    )

    if status_notes:
        dist_summary = "\n\n".join(status_notes) + "\n\n---\n" + dist_summary

    return (
        f"## {predicted_label}",
        confidences,
        gradcam_overlay,
        preprocessed_img,
        dist_summary,
    )


def create_gradio_app():
    """Builds the clean Gradio Blocks interface."""
    custom_css = """
    .disclaimer-box {
        background-color: #fdf2e9;
        border-left: 4px solid #e67e22;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    }
    .disclaimer-box .disclaimer-text {
        color: #000000 !important;
        font-weight: 500;
        font-size: 0.95rem;
        margin: 0;
    }
    .header-box {
        text-align: center;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(title="Coffee Nutrient Deficiency Few-Shot Prototype") as demo:
        # Compatible styling injection for Gradio 4, 5, and 6+
        gr.HTML(f"<style>{custom_css}</style>")

        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # ☕ Arabica Coffee Leaf Nutrient Deficiency Classification
                ### Cross-Anomaly Few-Shot Domain Adaptation Prototype (Benguet, Philippines)
                """
            )

        with gr.Row():
            gr.Markdown(
                """
                <div class="disclaimer-box">
                    <p class="disclaimer-text">
                        ⚠️ <b>Research prototype. Not a diagnostic tool. Supports only N, P, K classes.
                        Based on expert visual labels, not lab tests.</b>
                    </p>
                </div>
                """
            )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=1):
                gr.Markdown("### 📸 1. Input Leaf Image")
                input_image = gr.Image(
                    label="Upload Coffee Leaf or Capture from Camera",
                    type="numpy",
                    sources=["upload", "webcam"],
                )

                roi_checkbox = gr.Checkbox(
                    value=True,
                    label="Enable Leaf ROI Preprocessing",
                    info="Isolate the coffee leaf blade from background clutter using OpenCV HSV thresholding & morphology.",
                )

                predict_btn = gr.Button("🔍 Classify Leaf Deficiency", variant="primary", size="lg")

                # Sample test images for 1-click evaluation
                examples_dir = "examples"
                if os.path.isdir(examples_dir):
                    ex_files = [
                        os.path.join(examples_dir, f)
                        for f in os.listdir(examples_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                    curated_examples = []
                    for prefix in ["N_", "P_", "K_"]:
                        matched = [f for f in ex_files if os.path.basename(f).startswith(prefix)]
                        curated_examples.extend(matched[:2])
                    if curated_examples:
                        gr.Markdown("#### 🍃 Real CoLeaf Test Examples:")
                        gr.Examples(
                            examples=[[ex, True] for ex in curated_examples],
                            inputs=[input_image, roi_checkbox],
                            label="Click any test leaf below to load it:",
                        )

            # Right Column: Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 2. Classification & Relative Closeness")
                prediction_header = gr.Markdown("## Awaiting Image...")
                confidence_bar = gr.Label(
                    label="Relative Closeness to CoLeaf Prototypes",
                    num_top_classes=3,
                )

                gr.Markdown("### 🎯 3. Target Prototype-Score Grad-CAM")
                with gr.Row():
                    gradcam_output = gr.Image(
                        label="Grad-CAM Overlay ('out_relu' w.r.t Target Score)",
                        type="numpy",
                    )
                    preprocessed_preview = gr.Image(
                        label="Preprocessed / Isolated Leaf (224x224)",
                        type="numpy",
                    )

                details_text = gr.Markdown("")

        # Event binding
        predict_btn.click(
            fn=predict_nutrient_deficiency,
            inputs=[input_image, roi_checkbox],
            outputs=[
                prediction_header,
                confidence_bar,
                gradcam_output,
                preprocessed_preview,
                details_text,
            ],
        )

    return demo


def find_available_port(preferred_port: int = 7860, max_attempts: int = 25) -> int:
    """Finds the next available port if preferred_port is occupied."""
    import socket
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred_port


def main():
    parser = argparse.ArgumentParser(description="Run the Coffee Leaf Nutrient Deficiency Few-Shot Prototype.")
    parser.add_argument("--demo", action="store_true", help="Launch in demo mode with synthetic CoLeaf images.")
    parser.add_argument("--port", type=int, default=7860, help="Local server port (auto-increments if occupied).")
    parser.add_argument("--weights", type=str, default="weights/jmuben_encoder.weights.h5", help="Path to weights file.")
    parser.add_argument("--share", action="store_true", default=True, help="Create a public Gradio share link (for Colab).")
    args = parser.parse_args()

    # Initialize model and prototypes
    initialize_system(demo_mode=args.demo, weights_path=args.weights)

    # Determine available port
    target_port = find_available_port(args.port)
    if target_port != args.port:
        print(f"[*] Port {args.port} is already in use by another process. Automatically using port {target_port} instead.")

    # Launch Gradio app (use 127.0.0.1 on Windows so terminal links open directly)
    app = create_gradio_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=target_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
