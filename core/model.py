"""
Feature Extractor Model: MobileNetV2 proposed encoder.
- Initialized with ImageNet weights.
- Fine-tuned on JMuBEN coffee biotic stress (Cercospora, rust, phoma, leaf miner, healthy).
- Output: 1280-dimensional global average pooled feature vector f(x).
- Feature layer for Grad-CAM: 'out_relu'.
- FROZEN at inference.
- If weights/jmuben_encoder.weights.h5 is missing, falls back to ImageNet with a visible warning.
"""

import os
import logging

# Ensure local .keras directory inside workspace is checked first if present
_local_keras = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".keras"))
if os.path.isdir(_local_keras):
    os.environ.setdefault("KERAS_HOME", _local_keras)

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

logger = logging.getLogger("thesis_prototype.model")


def build_feature_extractor(input_shape=(224, 224, 3)):
    """
    Constructs the MobileNetV2 feature extractor and conv-feature model.

    Returns:
        tuple (feature_extractor, conv_model):
            - feature_extractor: Model mapping (224, 224, 3) -> 1280-d feature vector f(x).
            - conv_model: Model mapping (224, 224, 3) -> (7, 7, 1280) activations from 'out_relu'.
    """
    # Base MobileNetV2 without top classifier
    try:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling=None,
        )
    except Exception as e:
        logger.warning(f"Could not load ImageNet weights directly ({e}). Attempting local weight load or unweighted init.")
        local_weights = os.path.join(_local_keras, "models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
        if os.path.exists(local_weights):
            base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=local_weights, pooling=None)
        else:
            base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None, pooling=None)


    # Final feature map layer requested by thesis: 'out_relu'
    out_relu_layer = base_model.get_layer("out_relu")

    # Inputs and outputs
    img_input = base_model.input
    conv_features = out_relu_layer.output
    # Global average pooling produces the 1280-dimensional embedding
    embedding = tf.keras.layers.GlobalAveragePooling2D(name="prototype_gap")(conv_features)

    # Build keras models
    feature_extractor = tf.keras.Model(inputs=img_input, outputs=embedding, name="mobilenetv2_embedding")
    conv_model = tf.keras.Model(inputs=img_input, outputs=conv_features, name="mobilenetv2_out_relu")

    # Freeze all layers for target-domain inference
    feature_extractor.trainable = False
    conv_model.trainable = False

    return feature_extractor, conv_model


def load_proposed_encoder(weights_path="weights/jmuben_encoder.weights.h5", input_shape=(224, 224, 3)):
    """
    Loads the proposed architecture:
    MobileNetV2 initialized with ImageNet, then fine-tuned on JMuBEN source domain.
    Falls back gracefully with an alert if weights_path does not exist.

    Returns:
        tuple (feature_extractor, conv_model, weights_status_msg, is_fallback)
    """
    feature_extractor, conv_model = build_feature_extractor(input_shape=input_shape)

    if os.path.exists(weights_path):
        try:
            # Load fine-tuned weights
            feature_extractor.load_weights(weights_path)
            status_msg = f"[SUCCESS] Loaded proposed JMuBEN fine-tuned weights from '{weights_path}'."
            is_fallback = False
            logger.info(status_msg)
        except Exception as e:
            status_msg = (
                f"[WARNING] Found '{weights_path}' but encountered error while loading ({str(e)}). "
                "Falling back to ImageNet baseline initialization."
            )
            is_fallback = True
            logger.warning(status_msg)
    else:
        status_msg = (
            f"[INFO / FALLBACK] Proposed JMuBEN weights file '{weights_path}' not found. "
            "Falling back to ImageNet baseline feature extractor. "
            "(To train source weights, execute: python train_source.py)"
        )
        is_fallback = True
        logger.info(status_msg)

    # Re-verify frozen state
    feature_extractor.trainable = False
    conv_model.trainable = False

    return feature_extractor, conv_model, status_msg, is_fallback
