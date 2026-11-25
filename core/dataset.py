"""
CoLeaf Target Dataset Loader & Deterministic Support Sampler.
Handles scanning CoLeaf-DB directories for Nitrogen (N), Phosphorus (P), and Potassium (K)
nutrient deficiency classes, filtering out non-target categories (e.g. Boron, Iron, Healthy).
Provides synthetic data generator for instant out-of-the-box demo testing.
"""

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image

TARGET_CLASSES = ["N", "P", "K"]
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith(VALID_EXTENSIONS)


def resolve_coleaf_class_folder(base_dir: str, target_class: str) -> str:
    """
    Finds the directory matching N, P, or K within base_dir.
    Supports names like 'nitrogen (N)', 'nitrogen', 'N', etc.
    """
    if not os.path.isdir(base_dir):
        return None

    entries = os.listdir(base_dir)
    pattern_map = {
        "N": ["nitrogen", "n_deficiency", "class_n", "^n$"],
        "P": ["phosphorus", "phosphorous", "p_deficiency", "class_p", "^p$"],
        "K": ["potassium", "k_deficiency", "class_k", "^k$"],
    }

    patterns = pattern_map.get(target_class, [])

    # Exact or subfolder check
    for entry in entries:
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path):
            continue
        entry_lower = entry.lower().strip()

        # Direct exact match
        if entry_lower == target_class.lower():
            return full_path

        # Substring / keyword match
        for pat in patterns:
            if pat in entry_lower:
                return full_path

    return None


def load_coleaf_support_dataset(base_dirs=None):
    """
    Scans candidate directories for CoLeaf N, P, K leaf images.

    Returns:
        dict: {
            "N": [list of file paths],
            "P": [list of file paths],
            "K": [list of file paths]
        }
    """
    if base_dirs is None:
        base_dirs = [
            "dataset/coleaf",
            "coleaf",
            "data/coleaf",
            "support",
        ]

    dataset_map = {"N": [], "P": [], "K": []}
    found_root = None

    for root in base_dirs:
        if os.path.isdir(root):
            # Check if classes exist inside this root
            found_any = False
            for cls in TARGET_CLASSES:
                cls_dir = resolve_coleaf_class_folder(root, cls)
                if cls_dir and os.path.isdir(cls_dir):
                    files = [
                        os.path.join(cls_dir, f)
                        for f in os.listdir(cls_dir)
                        if _is_image_file(f)
                    ]
                    if files:
                        dataset_map[cls] = sorted(files)
                        found_any = True
            if found_any:
                found_root = root
                break

    return dataset_map, found_root


def sample_k_shots(dataset_map, k_shots: int, seed: int = 42):
    """
    Deterministically samples k_shots images per class using a fixed seed.

    Args:
        dataset_map: dict of {"N": [...], "P": [...], "K": [...]}
        k_shots: Number of support shots per class (1, 5, 10)
        seed: Random seed for exact reproducibility

    Returns:
        tuple (sampled_dict, warning_msgs)
    """
    rng = random.Random(seed)
    sampled = {}
    warnings = []

    for cls in TARGET_CLASSES:
        available_files = dataset_map.get(cls, [])
        num_avail = len(available_files)

        if num_avail == 0:
            warnings.append(f"No support images found for deficiency class '{cls}'.")
            sampled[cls] = []
        elif num_avail < k_shots:
            warnings.append(
                f"Class '{cls}' has only {num_avail} images available, which is fewer than the requested {k_shots} shots. Using all {num_avail} images."
            )
            sampled[cls] = sorted(available_files)
        else:
            # Deterministic selection
            cls_pool = list(available_files)
            cls_pool.sort()  # Ensure invariant initial ordering across operating systems
            sampled[cls] = rng.sample(cls_pool, k_shots)

    return sampled, warnings


def generate_synthetic_coleaf(output_dir="dataset/coleaf", samples_per_class=12):
    """
    Generates synthetic sample images for N, P, and K deficiency with realistic
    foliage color profiles for instant testing in demo mode.
    """
    os.makedirs(output_dir, exist_ok=True)
    class_dirs = {
        "N": os.path.join(output_dir, "nitrogen (N)"),
        "P": os.path.join(output_dir, "phosphorus (P)"),
        "K": os.path.join(output_dir, "potassium (K)"),
    }

    for cls, path in class_dirs.items():
        os.makedirs(path, exist_ok=True)

    rng = np.random.RandomState(42)

    for cls, path in class_dirs.items():
        for i in range(1, samples_per_class + 1):
            filename = os.path.join(path, f"sample_{cls}_{i:03d}.jpg")
            if os.path.exists(filename):
                continue

            # Create an ellipse leaf shape on neutral white/light background
            canvas = np.ones((256, 256, 3), dtype=np.uint8) * 235
            center = (128, 128)
            axes = (80 + rng.randint(-5, 5), 50 + rng.randint(-5, 5))
            angle = rng.randint(-15, 15)

            # Mask for leaf
            leaf_mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.ellipse(leaf_mask, center, axes, angle, 0, 360, 255, -1)

            # Generate leaf texture
            leaf_img = np.zeros((256, 256, 3), dtype=np.uint8)

            if cls == "N":
                # Nitrogen: Uniform chlorosis (pale yellow-green with pale apex/base)
                base_color = np.array([190, 215, 80], dtype=np.float32)  # Pale yellow-green
                noise = rng.normal(0, 10, (256, 256, 3))
                leaf_img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
                # Slightly greener main vein
                cv2.line(leaf_img, (60, 128), (200, 128), (140, 180, 50), thickness=3)

            elif cls == "P":
                # Phosphorus: Interveinal discoloration with purplish/dark-tan bronze spots
                base_color = np.array([70, 120, 50], dtype=np.float32)  # Darker olive green
                noise = rng.normal(0, 12, (256, 256, 3))
                leaf_img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
                # Irregular purplish / bronze patches
                for _ in range(4):
                    spot_center = (rng.randint(80, 180), rng.randint(90, 160))
                    cv2.circle(leaf_img, spot_center, rng.randint(10, 20), (130, 70, 90), -1)

            elif cls == "K":
                # Potassium: Yellow-brown marginal scorching and necrotic tip band
                base_color = np.array([85, 155, 60], dtype=np.float32)  # Green leaf body
                noise = rng.normal(0, 10, (256, 256, 3))
                leaf_img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
                # Outer contour scorch (brown necrotic margin)
                margin_mask = cv2.dilate(leaf_mask, np.ones((5, 5), np.uint8), iterations=1) - cv2.erode(leaf_mask, np.ones((9, 9), np.uint8), iterations=1)
                leaf_img[margin_mask > 0] = [160, 100, 40]  # Brown scorched edge

            # Composite leaf over background
            final_img = np.where(leaf_mask[:, :, None] > 0, leaf_img, canvas)
            cv2.imwrite(filename, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    return class_dirs
