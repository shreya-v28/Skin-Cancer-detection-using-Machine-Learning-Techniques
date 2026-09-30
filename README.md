# Skin-Cancer-detection-using-Machine-Learning-Techniques
Computer vision and deep learning framework for 4-class skin cancer stage classification (ISIC dataset) using image preprocessing, Otsu segmentation, GLCM texture extraction, and ResNet50.

# Multiclass Skin Cancer Detection & Melanoma Staging

An end-to-end computer vision and deep learning workflow for multi-class skin lesion classification. The system categorizes dermoscopic skin images into four distinct classes: Benign, Melanoma In-situ (Stage 0), Melanoma Invasive (Stage 1), and Melanoma Later Stages.

---

## 📌 Pipeline Overview

1. Preprocessing & Artifact Removal:**
   - Dull Razor Algorithm:** Removes hair artifacts via morphological operations[cite: 2].
   - Filtering: Median and Conservative smoothing to reduce salt-and-pepper/impulse noise while retaining sharp edge boundaries[cite: 2].
2. Segmentation & Feature Extraction:**
   - Otsu’s Thresholding: Isolates lesion foreground from surrounding skin tissue[cite: 2].
   - GLCM (Gray-Level Co-occurrence Matrix):** Extracts statistical texture features (Entropy, Energy, Correlation, Homogeneity, Contrast, Dissimilarity).
3. Classification:**
   - Multi-class classification evaluated across Custom CNN, VGG16, and ResNet50 architectures[cite: 2].

---

## 📊 Experimental Results

| Classifier Model | Accuracy Obtained |
| :--- | :--- |
| GLCM + CNN | 70% |
| Custom CNN | 80% |
| ResNet50 | 85% |
| VGG16 | 74.8% |

Data source: ISIC Archive (~2,400 dataset images + augmented training set of ~3,200 images via horizontal/vertical flips).

---

## 🛠️ Tech Stack & Requirements

- Language:** Python 3.x[cite: 2]
- Libraries:** OpenCV (`cv2`), NumPy, Scikit-Image (`skimage`), Scikit-Learn, Keras / TensorFlow, PyTorch, Matplotlib, Seaborn

```bash
pip install opencv-python numpy scikit-image scikit-learn tensorflow matplotlib seaborn
