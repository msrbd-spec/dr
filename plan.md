# MASTER PROJECT SPECIFICATION: CODA-Net
**Title:** Confident Ordinal Domain-Agnostic Network (CODA-Net) for Robust Diabetic Retinopathy Grading
**Target Venues:** IEEE TMI, Medical Image Analysis (MedIA), Scientific Reports
**Objective:** Achieve ~90%+ internal accuracy (5-class), 97%+ binary accuracy, and 85%+ external accuracy by explicitly solving Label Noise, Ordinal Class Overlap, and Domain Shift.

---

## 1. DATASET UTILIZATION STRATEGY
1.  **EyePACS (~35,000 images):** 
    *   *Role:* Supervised/Self-Supervised Pretraining. Used to initialize the CNN backbone with rich retinal features, preventing overfitting on the smaller APTOS dataset.
2.  **APTOS-2019 (3,662 images):** 
    *   *Role:* Primary training and internal 5-Fold CV evaluation. Undergoes Confident Learning to clean label noise.
3.  **Messidor-2:** 
    *   *Role:* Strictly held-out external validation. A small, unlabelled subset (e.g., 200 images) is used *only* as a style-pool for Feature-Space MixStyle.

---

## 2. CORE NOVELTIES & MATHEMATICAL FORMULATIONS

### Novelty A: Confident Learning for Label Noise (The 90%+ Enabler)
*   **The Flaw in Current Literature:** Models are trained on APTOS assuming 100% label accuracy. However, inter-rater reliability is ~85%. Forcing a model to learn mislabeled images flattens the decision boundary, capping accuracy.
*   **The Novelty:** We use Confident Learning (CL) to estimate the joint distribution of noisy labels and true labels. We prune or re-weight training samples that are mathematically identified as human errors.
*   **Mathematical Formulation:**
    1. Obtain out-of-fold predicted probabilities $\hat{P}$ for all training samples via cross-validation.
    2. Compute the confident joint probability matrix $C_{\tilde{y}, y^*}$, where a sample is counted if $\hat{P}(\text{class } j) \ge \text{threshold}_j$.
    3. Filter the dataset: Remove sample $x_i$ if its given label $\tilde{y}_i$ contradicts the confident true label $y^*_i$.
    *Result:* The model trains on a "clean" manifold, allowing it to confidently reach 90%+ on the test set.

### Novelty B: Ordinal Contrastive Loss (Solving Grade Overlap)
*   **The Flaw in Current Literature:** Cross-Entropy (Softmax) treats Grade 0 and Grade 4 as equally distinct as Grade 1 and Grade 2. This destroys the continuous severity spectrum of DR.
*   **The Novelty:** We replace the standard classification head with a projection head trained via an **Ordinal Contrastive Loss**. It pulls features of the same grade together, but pushes different grades apart by a margin strictly proportional to their clinical severity difference.
*   **Mathematical Formulation:**
    Let $f_i, f_j$ be feature vectors for samples with labels $y_i, y_j$.
    Distance: $D_{i,j} = ||f_i - f_j||_2$
    Ordinal Margin: $m_{i,j} = \alpha \cdot |y_i - y_j|$ (where $\alpha$ is a base margin).
    Loss: $\mathcal{L}_{ordinal} = \sum_{i,j} \max(0, D_{i,j} - m_{i,j})^2$ (if $y_i \neq y_j$) 
    $+ \sum_{i,j} D_{i,j}^2$ (if $y_i = y_j$)

### Novelty C: Feature-Space Style Mixing (MixStyle) (The 85%+ External Enabler)
*   **The Flaw in Current Literature:** Pixel-space FDA is destroyed by high-pass morphological filters (Ben Graham).
*   **The Novelty:** We perform domain adaptation in the *feature space* inside the CNN. We compute the channel-wise mean and variance of an APTOS feature map and mix it with the mean and variance of a Messidor-2 feature map.
*   **Mathematical Formulation:**
    Let $x_{aptos}$ and $x_{messidor}$ be feature maps at Stage 2 of the CNN.
    $\mu_A, \sigma_A = \text{mean}(x_{aptos}), \text{std}(x_{aptos})$
    $\mu_M, \sigma_M = \text{mean}(x_{messidor}), \text{std}(x_{messidor})$
    Interpolation weight $\lambda \sim \text{Beta}(\alpha, \alpha)$
    $\gamma_{mix} = \lambda \sigma_A + (1-\lambda) \sigma_M$
    $\beta_{mix} = \lambda \mu_A + (1-\lambda) \mu_M$
    $x_{adapted} = \gamma_{mix} \left( \frac{x_{aptos} - \mu_A}{\sigma_A} \right) + \beta_{mix}$

---

## 3. ALGORITHMIC PSEUDOCODE

**Algorithm 1: Confident Learning Pruning**
```python
def clean_training_data(X_train, y_noisy, baseline_model):
    # 1. Get out-of-fold probabilities
    probs = cross_val_predict_proba(baseline_model, X_train, y_noisy, cv=5)
    
    # 2. Find class-specific confidence thresholds
    thresholds = [mean(probs[y_noisy == c, c]) for c in range(5)]
    
    # 3. Identify label errors
    label_errors = []
    for i in range(len(y_noisy)):
        given_label = y_noisy[i]
        pred_label = argmax(probs[i])
        if pred_label != given_label and probs[i, pred_label] >= thresholds[pred_label]:
            label_errors.append(i)
            
    # 4. Return cleaned dataset
    X_clean = delete(X_train, label_errors)
    y_clean = delete(y_noisy, label_errors)
    return X_clean, y_clean
```

**Algorithm 2: MixStyle Forward Pass**
```python
def mixstyle_forward(x_aptos, x_messidor, alpha=0.1):
    if not training:
        return x_aptos
        
    # Calculate statistics
    mu_a, sig_a = calc_mean_std(x_aptos)
    mu_m, sig_m = calc_mean_std(x_messidor)
    
    # Sample mixing weight
    lam = np.random.beta(alpha, alpha)
    
    # Mix statistics
    mu_mix = lam * mu_a + (1 - lam) * mu_m
    sig_mix = lam * sig_a + (1 - lam) * sig_m
    
    # Apply to normalized APTOS features
    x_norm = (x_aptos - mu_a) / sig_a
    return x_norm * sig_mix + mu_mix
```

---

## 4. COMPREHENSIVE ABLATION STRATEGY (The 5 Tables)

**Table 1: The 10-Model Baseline Plateau**
*   *Models:* SwinV2-L, ConvNeXtV2-L, ResNet50, EfficientNetV2-M, MobileNetV3-L, MaxViT-L, DenseNet121, RegNetY-120, VGG16_BN, SqueezeNet1_1.
*   *Setup:* APTOS only, Softmax, No MixStyle, No Confident Learning.
*   *Purpose:* Proves the 85% internal / 65% external plateau is a fundamental data/domain issue.

**Table 2: Label Noise Resolution (Confident Learning)**
*   *Setup:* Best Baseline Model (e.g., ConvNeXtV2-L).
*   *Rows:* Standard Training $\rightarrow$ Class-Balanced Loss $\rightarrow$ **Confident Learning (Cleaned Data)**.
*   *Purpose:* Proves that removing human label noise is the mathematical key to breaking the 85% barrier and reaching 90%+.

**Table 3: Grade Boundary Resolution (Ordinal Loss)**
*   *Setup:* CL-Cleaned Model.
*   *Rows:* Standard Cross-Entropy $\rightarrow$ Focal Loss $\rightarrow$ **Ordinal Contrastive Loss**.
*   *Purpose:* Proves that enforcing ordinal geometry improves QWK and adjacent-grade accuracy.

**Table 4: Domain Generalization (MixStyle)**
*   *Setup:* CL + Ordinal Model tested on Messidor-2.
*   *Rows:* No Domain Adaptation $\rightarrow$ Pixel-Space FDA (Baseline Bug) $\rightarrow$ **Feature-Space MixStyle**.
*   *Purpose:* Proves that feature-level style mixing bypasses the Ben Graham high-pass filter, jumping external accuracy to 85%+.

**Table 5: Final SOTA Comparison**
*   *Rows:* Compare CODA-Net against the 3 SOTA papers (EffNet-SVM, KD-MobileNet, Q-SqueezeNet) on both 5-Class and Binary (Referable DR) metrics.

---

## 5. CODEBASE DIRECTORY TREE

```text
CODA-Net/
├── main.py                     # Master execution script (CLI arguments for ablations)
├── requirements.txt
├── data/
│   ├── aptos/                  # APTOS-2019 images and CSVs
│   ├── eyepacs/                # EyePACS images
│   └── messidor/               # Messidor-2 images and CSVs
├── src/
│   ├── data/
│   │   ├── dataset.py          # DRDataset (handles CLAHE, Ben Graham)
│   │   └── transforms.py       # Albumentations pipelines
│   ├── models/
│   │   ├── baselines.py        # 10 timm baseline loaders
│   │   ├── coda_net.py         # Backbone + MixStyle Layer + Ordinal Head
│   │   └── mixstyle.py         # Feature-space statistics mixing logic
│   ├── training/
│   │   ├── trainer.py          # Standard training loop
│   │   ├── confident_learning.py # Out-of-fold prob generation & pruning
│   │   └── losses.py           # Ordinal Contrastive Loss
│   └── utils/
│       ├── metrics.py          # Acc, QWK, Prec, Rec, F1 (strictly 2dp)
│       └── visualizer.py       # Matplotlib generators (Blue/Red curves, CM, t-SNE)
└── results/                    # AUTO-GENERATED
    ├── baseline_swinv2_l/
    │   ├── curves/             # train_val_acc.png, auroc.png, tsne.png
    │   ├── matrices/           # cm.png
    │   └── metrics.csv         # 2dp formatted results
    ├── baseline_convnextv2_l/
    └── proposed_CODA_Net/
```
