# 3D-CNN-NERF-Monocular-Video


A PyTorch implementation for generating Neural Radiance Fields (NeRF) from a single monocular video source. This repository provides a complete pipeline, ranging from extracting frames from raw video footage to estimating camera poses and training the final NeRF model for 3D scene reconstruction.

## 📝 Project Overview

This project aims to reconstruct 3D scenes using only a standard video input. It leverages deep learning techniques to synthesize novel views by learning the scene's geometry and view-dependent appearance.

### Anim_NERF & Visualization
* `EXP04-01_simplenerf.py`: A simplified/baseline NeRF implementation.
* `EXP04-02_NeRF Rendering and Visualization.py`: Rendering tools for the simplified model.


### Simple NERF & Visualization
* `EXP04-01_simplenerf.py`: A simplified/baseline NeRF implementation.
* `EXP04-02_NeRF Rendering and Visualization.py`: Rendering tools for the simplified model.

---

## 📂 Repository Structure

The project is organized into sequential experiments (`EXP`). The core pipeline is contained within the `EXP05` series.

### Core Pipeline
| File Name | Description |
| :--- | :--- |
| `EXP05-01_extract_frames.py` | **Step 1:** Extracts individual frames from the input video file. |
| `EXP05-02_preprocessing_nerf.py` | **Step 2:** Preprocesses frames (resizing, normalization) for the model. |
| `EXP05-03_camera generation.py` | **Step 3:** Generates camera parameters (extrinsics/intrinsics) for each frame. |
| `EXP05-04_Inspection for nerf.py` | **Step 4:** Visualization tool to inspect rays and camera poses before training. |
| `EXP05-05_nerf_model.py` | **Step 5:** The main training script for the Neural Radiance Field. |
### Key Features
* **Frame Extraction:** Automated tools to convert video into image sequences.
* **Preprocessing:** Normalization and resizing pipelines optimized for NeRF.
* **Pose Estimation:** Generation of camera intrinsics and extrinsics from monocular data.
* **Ray Inspection:** Visualization tools to debug ray marching and camera alignment.
* **NeRF Training:** Custom implementation of the NeRF training loop.


---

## 🛠️ Installation & Prerequisites

Ensure you have Python 3.x installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/Sribidya/3D-CNN-NERF-Monocular-Video.git](https://github.com/Sribidya/3D-CNN-NERF-Monocular-Video.git)
cd 3D-CNN-NERF-Monocular-Video

# Install dependencies (Example)
pip install torch torchvision numpy opencv-python matplotlib


