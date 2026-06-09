# MobileNet-CIFAR10-lightweight-image-classification
Comparative study of MobileNetV1 and MobileNetV2 on the CIFAR-10 dataset using transfer learning, fine-tuning, and a custom margin Loss for efficient edge-device image classification.

## Team
1) Varad Kishor Wani - CS25B1044
2) Hrishikesh Manojkumar Dhanlobhe - CS25B1014
3) Anuj Yogesh Chavan - AD25B1005
   
## Project Overview

This project builds an image classification system using MobileNet (V1 and V2) architectures on the CIFAR-10 dataset. The goal is to correctly identify objects across 10 categories using transfer learning on pretrained ImageNet weights, followed by fine-tuning for improved accuracy. Along the way, we explored a custom loss function, two-phase training, and a comparative study between MobileNetV1 and MobileNetV2 under both standard and custom loss settings, this was done to check which of the combination stands out and make it more accurate and efficient for edge devices. 

---

## Dataset

**Dataset:** CIFAR-10

* Total Classes: 10
* Total Images: 60,000
* Image Type: RGB Images (resized to 96 × 96 × 3 for MobileNet input)
* Train / Test Split: 50,000 / 10,000

### Classes

| Label | Class      |
| ----- | ---------- |
| 0     | Airplane   |
| 1     | Automobile |
| 2     | Bird       |
| 3     | Cat        |
| 4     | Deer       |
| 5     | Dog        |
| 6     | Frog       |
| 7     | Horse      |
| 8     | Ship       |
| 9     | Truck      |

---

# Team Contribution and Workflow

We split the work into three main areas so each person could go deep on their part while keeping everything connected.

---

## Anuj Yogesh Chavan — Custom Loss Function

This member designed and implemented the novel **Edge-Aware Margin Loss**, the defining innovation of this project. Standard Cross-Entropy only penalises wrong predictions, but a prediction like Class A = 0.51, Class B = 0.49 is technically correct yet dangerously unconfident. The proposed loss adds a margin penalty that actively widens the gap between the top two predicted probabilities.

### Mathematical Formulation

```
Total Loss = CE + λ × (1 − (p1 − p2))
```

Where:
- **CE** = Categorical Cross-Entropy
- **p1** = Highest predicted probability
- **p2** = Second highest predicted probability
- **λ** = Margin penalty weight (default: 0.1)

### Why the Proposed Loss Works

- Maintains classification accuracy through Cross-Entropy
- Encourages higher prediction confidence
- Increases separation between competing classes
- Reduces ambiguity in predictions
- Improves robustness during inference

### Implementation (TensorFlow/Keras)

```python
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, lam=0.1, name="edge_aware_margin_loss"):
        super().__init__(name=name)
        self.lam = lam

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        top2 = tf.math.top_k(y_pred, k=2).values
        p1, p2 = top2[:, 0], top2[:, 1]
        margin_penalty = 1.0 - (p1 - p2)
        return tf.reduce_mean(ce + self.lam * margin_penalty)
```

**Key outputs:** Custom loss implementation, loss function analysis

---

## Varad Kishor Wani — MobileNetV2 Implementation 

This member implemented the **full MobileNetV2 pipeline end-to-end**, covering both experimental configurations: **MobileNetV2 with standard Categorical Cross-Entropy** and **MobileNetV2 with the Margin Loss**. The work included data preprocessing, building the MobileNetV2 transfer-learning model, running the two-phase training strategy, fine-tuning the backbone, and evaluating both variants on the CIFAR-10 test set.

On the data side, CIFAR-10 was loaded via `tf.keras.datasets.cifar10`, pixel values were normalised to `[0, 1]`, labels were one-hot encoded, and images were resized from the native 32×32 to 96×96 to satisfy MobileNet's minimum input requirement. A class-distribution analysis was performed to confirm the dataset is perfectly balanced (5,000 samples per class).

### MobileNetV2 Architecture

```
Input Image (96 × 96 × 3)
        ↓
MobileNetV2 Backbone (pretrained on ImageNet, initially frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (10, Softmax)
        ↓
Output Class
```

Total Params: **2,270,794** (~8.66 MB) · Trainable head params (Phase 1): **12,810** · Non-trainable (Phase 1): **2,257,984**

### Training Configuration (both V2 experiments)

**Phase 1 — Head Training**
- Freeze entire MobileNetV2 backbone
- Train only `GlobalAveragePooling → Dropout → Dense` head
- Optimizer: Adam (lr = 1e-3), Epochs: 5, Batch: 128

**Phase 2 — Full Fine-Tuning**
- Unfreeze all layers
- Optimizer: Adam (lr = 1e-5), Epochs: 10, Batch: 128

The exact same pipeline was run twice — once compiled with `categorical_crossentropy` and once compiled with `CustomLoss(lam=0.1)` — so the two V2 results are directly comparable.

**Key outputs:** Preprocessing pipeline, class-distribution visualisation, full MobileNetV2 model (CE), full MobileNetV2 model (Custom Loss), two-phase training scripts, fine-tuning curves, evaluation metrics for both V2 variants

---

## Hrishikesh ManojKumar Dhanlobhe — MobileNetV1 Implementation

This member implemented the **full MobileNetV1 pipeline end-to-end** for both experimental configurations: **MobileNetV1 with standard Categorical Cross-Entropy** and **MobileNetV1 with the Edge-Aware Margin Loss**. The work covered building the V1 transfer-learning model, running the single-phase fine-tuning strategy, and evaluating both variants on the CIFAR-10 test set. Comparative-study figures, confusion matrices, and per-class reports for V1 were also produced here.

### MobileNetV1 Model Architecture

```
Input Image (96 × 96 × 3)
        ↓
MobileNetV1 Backbone (pretrained on ImageNet)
   ├── conv1 (Conv2D 3×3, stride 2) → BatchNorm → ReLU6
   ├── 13 × Depthwise-Separable Conv Blocks
   │       (DepthwiseConv2D 3×3 → BN → ReLU6 →
   │        Pointwise Conv2D 1×1 → BN → ReLU6)
   └── Output feature map (3 × 3 × 1024)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (10, Softmax)
        ↓
Output Class
```

Total Params: **3,239,114** (~12.36 MB) · Trainable head params (initial): **10,250** · Non-trainable: **3,228,864**

MobileNetV1 relies on **depthwise separable convolutions**, which factorise a standard convolution into a depthwise convolution (per-channel spatial filter) followed by a pointwise 1×1 convolution. This drastically reduces FLOPs and parameter count compared to a standard convolutional stack while preserving representational power — making it well-suited to edge deployment.

### Training Configuration (both V1 experiments)

**Single-Phase Fine-Tuning**
- Unfreeze all backbone layers from the start
- Optimizer: Adam (lr = 1e-5), Epochs: 10, Batch: 128

The same pipeline was run twice — once compiled with `categorical_crossentropy` and once compiled with `CustomLoss(lam=0.1)` — to isolate the effect of the Edge-Aware Margin Loss on the V1 backbone.

### Evaluation Methodology

For evaluation, the team looked at accuracy, precision, recall, F1 (all macro-averaged), a full per-class classification report, and a confusion matrix heatmap. Per-class breakdowns were especially useful for spotting which visual categories were harder to separate (e.g., cat vs. dog, automobile vs. truck).

**Key outputs:** Full MobileNetV1 model (CE), full MobileNetV1 model (Custom Loss), single-phase fine-tuning scripts, training curves, confusion matrix visualisations, comparative study, per-class classification reports, model size benchmarking (`.h5` export)

---

# Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV1 (MobileNet)
* MobileNetV2
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

# Methodology

## Data Preprocessing
1. Load CIFAR-10 via `tf.keras.datasets.cifar10`
2. Normalise pixel values: `x / 255.0`
3. One-hot encode labels with `to_categorical`
4. Resize images to 96 × 96 using `tf.image.resize`

## Two-Phase Training (MobileNetV2)

**Phase 1 — Head Training**
- Freeze all MobileNetV2 backbone layers
- Train only GlobalAveragePooling → Dropout → Dense head
- Optimizer: Adam (lr = 1e-3), Epochs: 5, Batch: 128

**Phase 2 — Full Fine-Tuning**
- Unfreeze all layers
- Continue training with lower learning rate
- Optimizer: Adam (lr = 1e-5), Epochs: 10, Batch: 128

## Single-Phase Training (MobileNetV1)
- Unfreeze all layers from the start
- Optimizer: Adam (lr = 1e-5), Epochs: 10, Batch: 128

---

## Configuration Summary

| Dimension | MobileNetV1 + CE | MobileNetV1 + Custom Loss | MobileNetV2 + CE | MobileNetV2 + Custom Loss |
|---|---|---|---|---|
| **Loss Function** | Cross-Entropy | Edge-Aware Margin (λ=0.1) | Cross-Entropy | Edge-Aware Margin (λ=0.1) |
| **Training Strategy** | Single-phase | Single-phase | Two-phase | Two-phase |
| **Total Epochs** | 10 | 10 | 5 + 10 | 5 + 10 |
| **Learning Rate** | 1e-5 | 1e-5 | 1e-3 → 1e-5 | 1e-3 → 1e-5 |
| **Batch Size** | 128 | 128 | 128 | 128 |
| **Input Resolution** | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 |
| **Pretrained Weights** | ImageNet | ImageNet | ImageNet | ImageNet |
| **Total Parameters** | 3,239,114 | 3,239,114 | 2,270,794 | 2,270,794 |
| **Architecture** | Depthwise separable convs | Depthwise separable convs | Inverted residuals + linear bottlenecks | Inverted residuals + linear bottlenecks |
| **Confidence Optimization** | No | Yes (maximises p1−p2) | No | Yes (maximises p1−p2) |


## Performance Results

| Metric | MobileNetV1 + CE | MobileNetV1 + Custom Loss | MobileNetV2 + CE | MobileNetV2 + Custom Loss |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 0.8732 | 0.8760 | 0.9057 | **0.9083** |
| **Precision (macro)** | 0.8730 | 0.8762 | 0.9066 | **0.9086** |
| **Recall (macro)** | 0.8732 | 0.8760 | 0.9057 | **0.9083** |
| **F1 Score (macro)** | 0.8729 | 0.8757 | 0.9059 | **0.9083** |
| **Model Size (.h5)** | 37.25 MB | 37.25 MB | 26.39 MB | 26.39 MB |

## Best Result

| Model | Accuracy | F1 Score |
|--------|----------|----------|
| MobileNetV2 + Edge-Aware Margin Loss | 90.83% | 0.9083 |

Highest Accuracy: 90.83%
Smallest Model Size: 26.39 MB
Suitable for Edge Deployment



# Comparative Study

The project evaluated four configurations across two backbone architectures and two loss functions on the CIFAR-10 test set (10,000 images). The table below summarises the measured performance for each model.

| Metric                        | MobileNetV1 + CE | MobileNetV1 + Custom Loss | MobileNetV2 + CE | MobileNetV2 + Custom Loss |
| ----------------------------- | ---------------- | ------------------------- | ---------------- | ------------------------- |
| **Model Size (.h5 on disk)**  | 37.25 MB         | 37.25 MB                  | 26.39 MB         | 26.39 MB                  |
| **Test Accuracy**             | 0.8732           | 0.8760                    | 0.9057           | **0.9083**                |
| **Precision (macro)**         | 0.8730           | 0.8762                    | 0.9066           | **0.9086**                |
| **Recall (macro)**            | 0.8732           | 0.8760                    | 0.9057           | **0.9083**                |
| **F1 Score (macro)**          | 0.8729           | 0.8757                    | 0.9059           | **0.9083**                |


### Key Observations

**MobileNetV1 vs MobileNetV2:** MobileNetV2 outperforms MobileNetV1 by roughly **3 percentage points** in accuracy on CIFAR-10 (90.57% vs 87.32% with CE) while also producing a **smaller .h5 file** (26.39 MB vs 37.25 MB). The inverted residual blocks and linear bottlenecks in V2 enable better gradient flow and richer feature representations, and the two-phase training strategy used with V2 also promotes more stable fine-tuning compared to the full-unfreeze approach used with V1.

**Cross-Entropy vs Edge-Aware Margin Loss:** Switching from CE to the custom Edge-Aware Margin Loss gives a consistent small lift on both backbones — **+0.28%** accuracy on V1 (0.8732 → 0.8760) and **+0.26%** accuracy on V2 (0.9057 → 0.9083). Precision, recall, and F1 all move in the same direction, indicating the gain is genuine rather than a recall/precision trade-off. The custom loss adds an explicit penalty when the model's top-2 probabilities are close together, encouraging the network to produce well-separated class scores — particularly valuable for confusable pairs like Cat/Dog or Automobile/Truck.

**Combined Effect:** **MobileNetV2 + Edge-Aware Margin Loss is the strongest configuration overall**, achieving 90.83% accuracy and 0.9083 macro-F1 while keeping the smallest deployable model size. Even on the lighter V1 backbone, the margin penalty produces a measurable improvement, confirming that the loss design generalises across architectures.

---

# Evaluation Metrics

All four models were evaluated on the held-out CIFAR-10 test set (10,000 images) using:

* **Accuracy** — overall fraction of correctly classified images
* **Precision (macro)** — average precision across all 10 classes
* **Recall (macro)** — average recall across all 10 classes
* **F1 Score (macro)** — harmonic mean of precision and recall
* **Confusion Matrix** — 10×10 heatmap highlighting inter-class confusions
* **Per-Class Classification Report** — detailed precision/recall/F1 breakdown per category
* **Model Size** — disk size of saved `.h5` model file

---

# Conclusion

This project demonstrates an efficient image classification framework using lightweight MobileNet architectures, Transfer Learning, and Fine-Tuning on the CIFAR-10 dataset. A novel **Edge-Aware Margin Loss** is introduced to improve prediction confidence by explicitly maximising the separation between the top two predicted class probabilities. Experimental results across four configurations — MobileNetV1/V2 × CE/Custom Loss — provide a thorough comparative study of how backbone choice and loss design interact. The findings confirm that combining a stronger backbone (V2) with confidence-aware training (custom loss) yields the most robust classifier (**90.83% accuracy, 0.9083 macro-F1**), and that even lighter models (V1) benefit meaningfully from the margin penalty. If extended further, adaptive λ scheduling, data augmentation, and ensembling would be natural next steps.
