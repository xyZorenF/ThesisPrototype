"""
ROI Preprocessing module: Whole-leaf segmentation using OpenCV.
Applies color thresholding in HSV color space, morphological refinement,
and largest-contour masking to isolate the coffee leaf blade from background clutter.
Applied identically to both support images and uploaded query images.
"""

import cv2
import numpy as np


def preprocess_leaf_roi(image: np.ndarray, target_size=(224, 224), apply_isolation: bool = True) -> np.ndarray:
    """
    Preprocess an input coffee leaf image.

    Args:
        image: Input RGB image as numpy array (uint8, shape [H, W, 3]).
        target_size: Output tuple (height, width) for model input (default 224x224).
        apply_isolation: If True, isolates the leaf from the background;
                         if False, resizes the image directly.

    Returns:
        Processed RGB image as numpy array (uint8, shape [target_size[0], target_size[1], 3]).
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image provided for ROI preprocessing.")

    # Ensure RGB uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if not apply_isolation:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # 1. Convert RGB to HSV to segment green, chlorotic yellow, and necrotic brown leaf tissue
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Coffee leaves span green (chlorophyll), yellow (nitrogen chlorosis),
    # reddish/tan (phosphorus deficiency), and brown (potassium marginal necrosis).
    # Backgrounds in field/bench photos are typically white paper, soil, sky, or neutral desk.
    # We define a broad mask capturing plant foliage tones:
    # Lower: Hue 10 (reddish/brown), Saturation 25 (exclude pure whites/grays), Value 20 (exclude deep shadow)
    # Upper: Hue 95 (yellow-greens and deep greens), Saturation 255, Value 255
    lower_foliage1 = np.array([10, 25, 20], dtype=np.uint8)
    upper_foliage1 = np.array([95, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_foliage1, upper_foliage1)

    # Also capture wrapping hues for purplish/reddish dark discoloration (often seen in P deficiency)
    lower_foliage2 = np.array([160, 25, 20], dtype=np.uint8)
    upper_foliage2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_foliage2, upper_foliage2)

    combined_mask = cv2.bitwise_or(mask1, mask2)

    # 2. Morphological operations: remove salt-and-pepper noise and close intra-leaf gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Find largest connected contour (the main leaf blade)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select contour with largest area
        largest_contour = max(contours, key=cv2.contourArea)
        total_image_area = image.shape[0] * image.shape[1]

        # Check if the contour is sufficiently large (> 3% of frame area)
        if cv2.contourArea(largest_contour) > 0.03 * total_image_area:
            leaf_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
            cv2.drawContours(leaf_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Smooth mask edge slightly to avoid harsh pixelated boundaries
            leaf_mask = cv2.GaussianBlur(leaf_mask, (5, 5), 0)
            leaf_mask_3c = cv2.merge([leaf_mask, leaf_mask, leaf_mask])

            # Apply mask to image (background replaced with black)
            isolated = np.where(leaf_mask_3c > 127, image, np.zeros_like(image))
            return cv2.resize(isolated, target_size, interpolation=cv2.INTER_AREA)

    # Fallback if no prominent leaf contour identified (e.g., image already closely cropped)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
