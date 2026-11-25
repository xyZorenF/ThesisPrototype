"""
Few-Shot Prototypical Nearest-Centroid Classifier.
Computes prototype vectors c_k for N, P, and K from support embeddings,
calculates squared Euclidean distances, softmax relative-closeness scores,
and caches prototypes in-memory to prevent recomputation on every upload.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .roi import preprocess_leaf_roi
from .dataset import load_coleaf_support_dataset, sample_k_shots, TARGET_CLASSES


class PrototypeClassifier:
    """
    Nearest-Centroid Prototypical Classifier for N, P, and K nutrient deficiencies.
    Features are extracted using frozen MobileNetV2 representations.
    """

    def __init__(self, feature_extractor, conv_model=None, base_dirs=None, seed: int = 42):
        self.feature_extractor = feature_extractor
        self.conv_model = conv_model
        self.base_dirs = base_dirs
        self.seed = seed

        # In-memory prototype cache:
        # Key: (k_shots: int, roi_enabled: bool)
        # Value: dict containing:
        #   "prototypes": { "N": tensor, "P": tensor, "K": tensor },
        #   "sample_paths": { "N": [...], "P": [...], "K": [...] }
        self._cache = {}

    def clear_cache(self):
        """Clears prototype cache."""
        self._cache.clear()

    def _load_and_preprocess_image(self, img_path: str, apply_roi: bool) -> np.ndarray:
        """Reads image from disk and runs through leaf ROI isolation and MobileNetV2 normalization."""
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Could not read support image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        processed = preprocess_leaf_roi(rgb, target_size=(224, 224), apply_isolation=apply_roi)
        # MobileNetV2 standard normalization [-1, 1]
        norm = preprocess_input(processed.astype(np.float32))
        return norm

    def get_or_compute_prototypes(self, k_shots: int, apply_roi: bool):
        """
        Retrieves cached prototypes or computes them from the CoLeaf support set.

        Returns:
            tuple (prototypes_dict, sample_paths_dict, warnings_list)
        """
        cache_key = (k_shots, apply_roi)
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            return entry["prototypes"], entry["sample_paths"], []

        # Load dataset files
        dataset_map, found_root = load_coleaf_support_dataset(self.base_dirs)
        sampled_files, warnings = sample_k_shots(dataset_map, k_shots=k_shots, seed=self.seed)

        prototypes = {}
        for cls in TARGET_CLASSES:
            file_list = sampled_files.get(cls, [])
            if not file_list:
                raise ValueError(
                    f"Cannot compute prototype for class '{cls}': no support images found in {found_root}."
                )

            # Batch process support images
            batch_tensors = []
            for path in file_list:
                img_norm = self._load_and_preprocess_image(path, apply_roi=apply_roi)
                batch_tensors.append(img_norm)

            batch_np = np.stack(batch_tensors, axis=0)  # Shape: (K, 224, 224, 3)
            # Feature extraction with frozen backbone
            embeddings = self.feature_extractor(batch_np, training=False)  # Shape: (K, 1280)
            # Normalize individual embeddings before computing centroid to prevent variance skew
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=-1)
            # Class centroid: mean of support feature vectors, L2-normalized
            mean_vector = tf.reduce_mean(norm_embeddings, axis=0)  # Shape: (1280,)
            centroid = tf.math.l2_normalize(mean_vector, axis=-1)
            prototypes[cls] = centroid

        # Store in cache
        self._cache[cache_key] = {
            "prototypes": prototypes,
            "sample_paths": sampled_files,
        }

        return prototypes, sampled_files, warnings

    def classify_image(self, rgb_image: np.ndarray, k_shots: int = 10, apply_roi: bool = True):
        """
        Classifies an input RGB leaf image against CoLeaf prototypes.

        Args:
            rgb_image: Query leaf image as RGB numpy array.
            k_shots: Number of support shots per class (1, 5, or 10).
            apply_roi: Whether to apply OpenCV leaf isolation.

        Returns:
            dict containing:
                "predicted_class": 'N', 'P', or 'K'
                "predicted_label": full name (e.g. 'Nitrogen (N) Deficiency')
                "distances": {'N': float, 'P': float, 'K': float}
                "confidences": {'Nitrogen (N)': float, 'Phosphorus (P)': float, 'Potassium (K)': float}
                "preprocessed_image": RGB preprocessed query image (224, 224, 3)
                "prototype_tensor": Target prototype tensor c_k for the predicted class
                "warnings": list of string warnings
        """
        # Ensure prototypes are ready
        prototypes, sample_paths, warnings = self.get_or_compute_prototypes(
            k_shots=k_shots, apply_roi=apply_roi
        )

        # Preprocess query image identically to support images
        preprocessed_query = preprocess_leaf_roi(
            rgb_image, target_size=(224, 224), apply_isolation=apply_roi
        )
        query_norm = preprocess_input(preprocessed_query.astype(np.float32))
        query_batch = np.expand_dims(query_norm, axis=0)  # (1, 224, 224, 3)

        # Extract frozen query embedding and normalize
        query_raw = self.feature_extractor(query_batch, training=False)[0]  # (1280,)
        query_embedding = tf.math.l2_normalize(query_raw, axis=-1)

        # Compute squared Euclidean distance on unit hypersphere
        # On L2-normalized vectors: ||u - v||^2 = 2 - 2 * cos_sim(u, v)
        # This completely eliminates centroid magnitude bias.
        distances = {}
        for cls in TARGET_CLASSES:
            proto = prototypes[cls]
            sq_dist = tf.reduce_sum(tf.square(query_embedding - proto)).numpy().item()
            distances[cls] = float(sq_dist)

        # Classify by smallest distance
        predicted_class = min(distances, key=distances.get)

        # Softmax over negative distances for relative closeness.
        # Since squared distances on unit sphere are bounded in [0, 4],
        # a standard temperature scale (e.g. tau = 0.1) produces crisp, calibrated confidences.
        dist_values = np.array([distances[c] for c in TARGET_CLASSES], dtype=np.float32)
        tau = 0.2
        neg_dist_scaled = -dist_values / tau
        # Shift for numerical stability
        exp_neg = np.exp(neg_dist_scaled - np.max(neg_dist_scaled))
        softmax_scores = exp_neg / np.sum(exp_neg)

        full_names = {
            "N": "Nitrogen (N) Deficiency",
            "P": "Phosphorus (P) Deficiency",
            "K": "Potassium (K) Deficiency",
        }

        confidences = {
            full_names[cls]: float(softmax_scores[idx])
            for idx, cls in enumerate(TARGET_CLASSES)
        }

        return {
            "predicted_class": predicted_class,
            "predicted_label": full_names[predicted_class],
            "distances": distances,
            "confidences": confidences,
            "preprocessed_image": preprocessed_query,
            "prototype_tensor": prototypes[predicted_class],
            "warnings": warnings,
        }
