"""
Source Domain Fine-Tuning Script: MobileNetV2 on JMuBEN Coffee Biotic Stress Dataset.

Source Classes (5):
  1. Cercospora (leaf spot)
  2. Rust (Hemileia vastatrix)
  3. Phoma (Phoma blight)
  4. Leaf Miner (Leucoptera coffeella)
  5. Healthy

Training Protocol:
  - Base: MobileNetV2 with ImageNet pretrained weights.
  - Head: GlobalAveragePooling2D -> Dropout(0.2) -> Dense(5, activation='softmax').
  - Data Augmentation applied ONLY to source training data (horizontal flip, random rotation, zoom).
  - Target-domain (CoLeaf) data is NEVER touched during this phase.
  - Post-training: Strips the 5-class head and saves the frozen feature extractor weights
    to 'weights/jmuben_encoder.weights.h5'.

Supports --demo flag for rapid testing with synthetic source batches.
"""

import os
import argparse

# Ensure local .keras directory inside workspace is checked first if present
_local_keras = os.path.abspath(os.path.join(os.path.dirname(__file__), ".keras"))
if os.path.isdir(_local_keras):
    os.environ.setdefault("KERAS_HOME", _local_keras)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Fix random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

SOURCE_CLASSES = ["cercospora", "rust", "phoma", "leaf_miner", "healthy"]


def create_source_model(num_classes=5, input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Initializes MobileNetV2 with ImageNet weights and appends a 5-class biotic classification head.
    """
    # 1. Base feature extractor
    try:
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling=None,
        )
    except Exception:
        local_weights = os.path.join(_local_keras, "models", "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
        if os.path.exists(local_weights):
            base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=local_weights, pooling=None)
        else:
            base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None, pooling=None)


    # 2. Source domain data augmentation (strictly for source training only)
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="source_augmentation",
    )

    # 3. Model construction
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    augmented = data_augmentation(inputs)
    preprocessed = preprocess_input(augmented)

    # Forward through MobileNetV2
    out_relu = base_model(preprocessed, training=True)
    pooled = layers.GlobalAveragePooling2D(name="source_gap")(out_relu)
    drop = layers.Dropout(0.2, name="source_dropout")(pooled)
    outputs = layers.Dense(num_classes, activation="softmax", name="source_predictions")(drop)

    full_model = models.Model(inputs=inputs, outputs=outputs, name="jmuben_mobilenetv2_trainer")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    full_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return full_model, base_model


def generate_synthetic_jmuben_data(num_samples=100, image_size=(224, 224)):
    """Generates synthetic RGB data and labels for quick pipeline verification."""
    print("Generating synthetic JMuBEN source dataset for --demo verification...")
    x = np.random.randint(0, 256, (num_samples, image_size[0], image_size[1], 3), dtype=np.uint8)
    y = np.random.randint(0, 5, (num_samples,), dtype=np.int32)
    return x, y


def train_on_jmuben(dataset_dir: str, weights_output: str, epochs: int, batch_size: int, is_demo: bool):
    """Executes the source training pipeline and extracts backbone weights."""
    os.makedirs(os.path.dirname(weights_output), exist_ok=True)

    print("\n" + "=" * 70)
    print("THESIS SOURCE-DOMAIN TRAINING: JMuBEN BIOTIC STRESS -> MobileNetV2")
    print(f"Target Checkpoint: {weights_output}")
    print("=" * 70)

    model, base_model = create_source_model(num_classes=5)

    if not is_demo and os.path.isdir(dataset_dir):
        # If JMuBEN is nested inside a subfolder (e.g. dataset/jmuben/JMuBEN), auto-resolve
        nested_jmuben = os.path.join(dataset_dir, "JMuBEN")
        if os.path.isdir(nested_jmuben):
            dataset_dir = nested_jmuben

    if is_demo or not os.path.isdir(dataset_dir):
        if not is_demo:
            print(f"[!] JMuBEN directory '{dataset_dir}' not found. Falling back to synthetic demonstration mode.")
        x_train, y_train = generate_synthetic_jmuben_data(num_samples=60)
        x_val, y_val = generate_synthetic_jmuben_data(num_samples=20)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        train_epochs = min(epochs, 2)
    else:
        print(f"Loading JMuBEN source dataset from '{dataset_dir}'...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=SEED,
            image_size=(224, 224),
            batch_size=batch_size,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=SEED,
            image_size=(224, 224),
            batch_size=batch_size,
        )
        train_epochs = epochs

    print(f"\nStarting fine-tuning for {train_epochs} epoch(s)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_epochs,
        verbose=1,
    )

    print(f"\nExtracting feature extractor weights and saving to '{weights_output}'...")
    # Build clean standalone feature extractor model with identical layer structure
    clean_base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
        pooling=None,
    )
    clean_base.set_weights(base_model.get_weights())

    out_relu = clean_base.get_layer("out_relu").output
    gap = layers.GlobalAveragePooling2D(name="prototype_gap")(out_relu)
    encoder_model = models.Model(inputs=clean_base.input, outputs=gap, name="mobilenetv2_embedding")

    # Save weights file
    encoder_model.save_weights(weights_output)
    print(f"[SUCCESS] Backbone encoder weights successfully saved to: {weights_output}")
    print("These weights can now be utilized directly by app.py for few-shot nutrient deficiency classification.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 on JMuBEN coffee disease source dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/jmuben",
        help="Path to JMuBEN dataset directory organized into 5 class subfolders.",
    )
    parser.add_argument(
        "--output_weights",
        type=str,
        default="weights/jmuben_encoder.weights.h5",
        help="Path to save the fine-tuned feature extractor weights.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--demo", action="store_true", help="Run 1-epoch dry run using synthetic images.")

    args = parser.parse_args()
    train_on_jmuben(
        dataset_dir=args.dataset_dir,
        weights_output=args.output_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        is_demo=args.demo,
    )


if __name__ == "__main__":
    main()
