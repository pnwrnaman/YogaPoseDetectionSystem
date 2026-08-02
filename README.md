# 🧘‍♀️ Intelligent Yoga Pose Classification & Real-Time Feedback System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow/Keras](https://img.shields.io/badge/TensorFlow-DNN-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Real%20Time-red)

> **A 3D Biomechanical and Tabular Ensemble Approach to Active Fitness Coaching**

Traditional 2D Convolutional Neural Networks (CNNs) struggle with severe limb occlusion and self-overlapping in complex yoga poses. This project bypasses those limitations by transforming monocular video feeds into 3D geometric data. By engineering a hybrid voting ensemble (DNN + XGBoost + LightGBM), this system classifies 82 distinct yoga postures with exceptional accuracy and provides real-time, on-screen physical corrections.

---

## ✨ Key Features

*   **Robust 82-Class Recognition:** Trained on the challenging Yoga-82 dataset, capable of distinguishing highly similar postures.
*   **3D Biomechanical Feature Engineering:** Extracts 33 spatial keypoints using MediaPipe and mathematically derives 18 specific joint angles, ignoring background noise and clothing textures.
*   **Real-Time Active Feedback:** Uses an OpenCV-based HUD to calculate joint angle deltas against a pre-computed statistical reference, rendering immediate textual corrections for user safety.
*   **Hybrid Voting Ensemble:** Combines the rigid mathematical boundaries of Gradient Boosted Trees (XGBoost, LightGBM) with a deeply regularized Deep Neural Network (DNN) utilizing Swish activation.
*   **Synthetic Data Balancing:** Utilizes SMOTE to artificially synthesize mathematically valid 3D keypoint variations for minority poses, solving severe class imbalance.

---

## 📐 How It Works: The Biomechanical Pipeline

Instead of analyzing pixels, we treat posture classification as a **Tabular Data** problem. 

### 1. Spatial Extraction & Angle Calculation
MediaPipe Pose infers the Z-axis (depth). We use 3D vector algebra to calculate angles across 18 critical joints. For a joint $B$ (e.g., elbow) connecting $A$ (shoulder) and $C$ (wrist), the angle $\theta$ is derived using the vector dot product:

$$ \theta = \arccos \left( \frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| \times |\vec{BC}|} \right) $$

### 2. Torso Normalization
To classify the *posture* rather than the user's height, all coordinates are normalized using the Euclidean distance between the shoulders and hips. The final input is a 150-dimensional feature vector.

### 3. Inference & Correction Loop
Live angles are subtracted from a pre-calculated statistical dictionary (`pose_reference.json`). If the delta exceeds our safety threshold (e.g., 15°), the system triggers a localized on-screen warning.
<img width="720" height="1280" alt="yogagif" src="https://github.com/user-attachments/assets/bd8007cc-b0da-471a-9e03-f89ce253f404" />


## 📊 Performance & Results

Evaluated against an unseen test set of 1,599 images, the 3D tabular approach significantly outperformed baseline 2D hierarchical object detectors (like YOLOv8) which suffer from the "Level-1 Bottleneck" during occlusion.

| Architecture | Top-1 Accuracy | Top-5 Accuracy | Key Advantage |
| :--- | :---: | :---: | :--- |
| EfficientNet-B0 (2D CNN) | ~72.0% | ~88.5% | Fails on dense occlusion |
| YOLOv8 (Hierarchical) | ~75.5% | ~91.0% | Struggles with Z-axis inference |
| **Our Proposed 3D Ensemble** | **86.62%** | **95.81%** | **Mathematically enforces joint boundaries** |

---

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* Web camera for live inference

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pnwrnaman/YogaPoseDetectionSystem.git](github.com/pnwrnaman/YogaPoseDetectionSystem.git)
