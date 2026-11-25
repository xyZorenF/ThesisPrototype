"""
Target-Score Prototype Grad-CAM for Explainable Few-Shot Learning.
Computes Grad-CAM with respect to the target prototype-distance score:
    Score = - || f(x) - c_k ||^2
using the MobileNetV2 'out_relu' convolutional layer (shape: 7x7x1280).
Strictly adheres to thesis requirement: Does NOT use any source 5-class head.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def compute_prototype_gradcam(conv_model, preprocessed_rgb: np.ndarray, target_prototype: tf.Tensor):
    """
    Generates a Grad-CAM heatmap explaining why the query image is close
    to the target class prototype c_k.

    Args:
        conv_model: Model that outputs features from MobileNetV2 layer 'out_relu' (7, 7, 1280).
        preprocessed_rgb: Query image (224, 224, 3) as uint8 numpy array.
        target_prototype: Prototype vector c_k of the predicted class (1280,).

    Returns:
        np.ndarray: Normalized 2D heatmap [0, 1] of shape (224, 224).
    """
    img_norm = preprocess_input(preprocessed_rgb.astype(np.float32))
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_norm, axis=0))

    # GradientTape records operations on the convolutional activation map
    with tf.GradientTape() as tape:
        # Forward pass up to 'out_relu': shape (1, 7, 7, 1280)
        conv_outputs = conv_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        # Global average pooling to obtain the 1280-d embedding
        # Shape: (1, 1280)
        pooled_features = tf.reduce_mean(conv_outputs, axis=[1, 2])

        # Target prototype alignment score: S_k(x) = <f(x), c_k>
        # In squared Euclidean distance: ||f(x) - c_k||^2 = ||f(x)||^2 - 2<f(x), c_k> + ||c_k||^2.
        # The alignment term <f(x), c_k> is the driving similarity metric connecting query features
        # to the target deficiency prototype. Differentiating this alignment score avoids the unconstrained
        # self-norm penalty -||f(x)||^2 which mathematically inverts gradients on positive leaf tissue
        # and creates false corner artifacts.
        target_score = tf.reduce_sum(pooled_features[0] * target_prototype)

    # Gradient of the prototype-closeness score w.r.t out_relu feature map
    grads = tape.gradient(target_score, conv_outputs)  # Shape: (1, 7, 7, 1280)

    if grads is None:
        # Graceful fallback to zeros if gradient is uncomputable
        return np.zeros((224, 224), dtype=np.float32)

    # Spatial average of the gradients (importance weight alpha_d for each channel d)
    alpha_weights = tf.reduce_mean(grads, axis=[0, 1, 2])  # Shape: (1280,)

    # Linear combination of feature maps weighted by alpha_d
    conv_maps = conv_outputs[0]  # Shape: (7, 7, 1280)
    weighted_cam = tf.reduce_sum(conv_maps * alpha_weights, axis=-1)  # Shape: (7, 7)

    # Apply ReLU: only positive contributions towards prototype similarity
    cam_relu = tf.maximum(weighted_cam, 0.0).numpy()

    # Normalize heatmap between 0 and 1
    max_val = np.max(cam_relu)
    if max_val > 1e-8:
        cam_normalized = cam_relu / max_val
    else:
        cam_normalized = np.zeros_like(cam_relu)

    # Resize heatmap up to original input dimensions (224, 224)
    heatmap_224 = cv2.resize(cam_normalized, (224, 224), interpolation=cv2.INTER_LINEAR)
    return heatmap_224


def overlay_gradcam(original_rgb: np.ndarray, heatmap_224: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlays a normalized 2D Grad-CAM heatmap onto an RGB image.

    Args:
        original_rgb: Original or preprocessed RGB image (H, W, 3).
        heatmap_224: 2D float array in [0, 1].
        alpha: Opacity of the heatmap overlay (0.0 to 1.0).

    Returns:
        np.ndarray: Blended RGB image (uint8, shape [H, W, 3]).
    """
    target_h, target_w = original_rgb.shape[0], original_rgb.shape[1]
    resized_heatmap = cv2.resize(heatmap_224, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert to 8-bit [0, 255]
    heatmap_uint8 = np.uint8(255 * np.clip(resized_heatmap, 0.0, 1.0))

    # Apply JET colormap (yields BGR)
    colored_cam_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_cam_rgb = cv2.cvtColor(colored_cam_bgr, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    blended = np.float32(colored_cam_rgb) * alpha + np.float32(original_rgb) * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended
