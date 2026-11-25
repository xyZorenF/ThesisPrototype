"""
Core package for Benguet Arabica Coffee Few-Shot Nutrient Deficiency Classification.
"""
from .model import build_feature_extractor, load_proposed_encoder
from .roi import preprocess_leaf_roi
from .dataset import load_coleaf_support_dataset, generate_synthetic_coleaf
from .prototypes import PrototypeClassifier
from .gradcam import compute_prototype_gradcam, overlay_gradcam

__all__ = [
    "build_feature_extractor",
    "load_proposed_encoder",
    "preprocess_leaf_roi",
    "load_coleaf_support_dataset",
    "generate_synthetic_coleaf",
    "PrototypeClassifier",
    "compute_prototype_gradcam",
    "overlay_gradcam",
]
