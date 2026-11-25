# Cross-Anomaly Few-Shot Domain Adaptation for Local Nutrient Deficiency in Arabica Coffee Leaves (Benguet, Philippines)

This repository contains the research prototype web application and training pipeline for evaluating **Cross-Anomaly Few-Shot Domain Adaptation** to classify primary macronutrient deficiencies in Arabica coffee (*Coffea arabica*) leaves:
- **Nitrogen (N) Deficiency**
- **Phosphorus (P) Deficiency**
- **Potassium (K) Deficiency**

---

## 🏛️ System Architecture & Scientific Rationale

### 1. Cross-Anomaly Representation Transfer
- **Source Domain**: Biotic stress and foliar diseases from the public **JMuBEN / JMuBEN2** datasets (5 classes: Cercospora, Rust, Phoma, Leaf Miner, Healthy).
- **Target Domain**: Abiotic macronutrient deficiencies (**Nitrogen, Phosphorus, Potassium**) drawn from the **CoLeaf** dataset.
- **Why Cross-Anomaly?** Training on foliar anomalies imparts intermediate representations tailored to coffee leaf structures, color shifts (chlorosis, necrosis, bronzing), and cellular lesions, providing a richer domain prior than generic ImageNet weights.

### 2. Feature Extractor Backbone
- **Architecture**: `MobileNetV2` (ImageNet initialization $\to$ JMuBEN supervised fine-tuning).
- **Output Representation**: 1,280-dimensional global average pooled feature vector $\mathbf{f}(x)$ derived from the terminal ReLU layer (`out_relu`).
- **Inference Mode**: Backbone weights are **completely frozen** during few-shot classification.

### 3. Prototypical Nearest-Centroid Classifier
- **Support Formulation**: For each target deficiency $k \in \{N, P, K\}$, the class prototype $\mathbf{c}_k$ is computed as the centroid (mean feature vector) of $K$ support images ($K \in \{1, 5, 10\}$):
  $$\mathbf{c}_k = \frac{1}{K} \sum_{i=1}^K \mathbf{f}(x_i^{(k)})$$
- **Distance Metric**: Squared Euclidean distance:
  $$d_k(x) = \|\mathbf{f}(x) - \mathbf{c}_k\|_2^2$$
- **Decision Rule**:
  $$\hat{y} = \arg\min_{k \in \{N, P, K\}} d_k(x)$$
- **Relative Closeness Softmax**:
  $$p_k(x) = \frac{\exp(-d_k(x))}{\sum_{j \in \{N, P, K\}} \exp(-d_j(x))}$$
  *(Explanatory Note: These represent relative similarity scores to reference prototypes, not calibrated diagnostic probabilities).*
- **Prototype Caching**: Prototypes are computed once and stored in memory for each `(k_shots, roi_enabled)` configuration.

### 4. Target-Score Prototype Grad-CAM
- Grad-CAM heatmaps are computed using `tf.GradientTape` against the actual **target prototype-distance score**:
  $$S_k(x) = - \|\mathbf{f}(x) - \mathbf{c}_k\|_2^2$$
  with respect to the `out_relu` activation tensor ($7 \times 7 \times 1280$).
- **No Source Head Dependency**: Ensures the visual explanation directly reflects the features that drove the nearest-centroid few-shot decision.

### 5. Region of Interest (ROI) Leaf Isolation
- Optional OpenCV module applying HSV foliage color segmentation, morphological smoothing, and contour hierarchy masking to eliminate bench and soil artifacts.
- Applied symmetrically to both support images and uploaded query images.

---

## 📂 Directory Structure

```text
thesisPROTOTYPE/
├── app.py                            # Gradio interactive web application
├── train_source.py                   # Source-domain fine-tuning on JMuBEN
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation and usage guide
├── coffee_deficiency_prototype.ipynb # Google Colab executable notebook
│
├── core/
│   ├── __init__.py
│   ├── model.py                      # MobileNetV2 encoder & weights loader
│   ├── dataset.py                    # CoLeaf parser & deterministic K-shot sampler
│   ├── prototypes.py                 # Nearest-centroid classifier & caching
│   ├── roi.py                        # OpenCV whole-leaf isolation
│   └── gradcam.py                    # Target-score Grad-CAM on 'out_relu'
│
├── dataset/
│   ├── coleaf/                       # Place CoLeaf target dataset folders here
│   │   ├── nitrogen/                 # (or 'nitrogen (N)', 'N')
│   │   ├── phosphorus/               # (or 'phosphorus (P)', 'P')
│   │   └── potassium/                # (or 'potassium (K)', 'K')
│   └── jmuben/                       # (Optional) JMuBEN source dataset for training
│
└── weights/
    └── jmuben_encoder.weights.h5     # Checkpoint created by train_source.py
```

---

## 🚀 Setup & Execution

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
git clone <your-repo>
cd thesisPROTOTYPE
pip install -r requirements.txt
```

### 2. Immediate Testing via Synthetic Demo Mode
You can test the entire application immediately without manually downloading datasets:
```bash
python app.py --demo
```
This automatically populates `dataset/coleaf/` with synthetic coffee leaves exhibiting characteristic N, P, and K foliar deficiency patterns and launches the Gradio interface at `http://localhost:7860`.

### 3. Using Real CoLeaf Dataset
Place the CoLeaf dataset folders into `dataset/coleaf/` or `support/`:
```text
dataset/coleaf/
  ├── nitrogen (N)/
  ├── phosphorus (P)/
  └── potassium (K)/
```
Then launch the app:
```bash
python app.py
```

### 4. Fine-Tuning on JMuBEN Source Dataset
To train the proposed feature extractor on JMuBEN coffee disease images:
```bash
python train_source.py --dataset_dir dataset/jmuben --epochs 15 --batch_size 32
```
To run a fast 1-epoch dry run using synthetic source data:
```bash
python train_source.py --demo
```
The fine-tuned backbone weights will be saved to `weights/jmuben_encoder.weights.h5`.

---

## ☁️ Running on Google Colab

1. Open `coffee_deficiency_prototype.ipynb` in Google Colab.
2. Select **Runtime $\to$ Change runtime type $\to$ T4 GPU** (or CPU).
3. Execute the cells sequentially. The final cell launches `app.py` with `share=True`, creating a public `https://xxxx.gradio.live` link accessible from mobile phones and web browsers.

---

## ⚠️ Research Disclaimer
> **Research prototype. Not a diagnostic tool. Supports only N, P, K classes. Based on expert visual labels, not lab tests.**
