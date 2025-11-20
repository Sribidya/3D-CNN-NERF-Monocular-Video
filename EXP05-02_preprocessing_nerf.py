import os
import glob
import cv2
import h5py
import numpy as np
from pathlib import Path

# -------------------------------
# SETTINGS
# -------------------------------
DATASET_ROOT = "dataset/people_snapshot_public/female-1-casual"
OUTPUT_ROOT = "output/female-1-casual"
EXTRACTED_FRAMES = os.path.join(OUTPUT_ROOT, "images")
MASKS_FOLDER = os.path.join(OUTPUT_ROOT, "masks")
CAMERA_FILE = os.path.join(OUTPUT_ROOT, "cameras.npz")

os.makedirs(EXTRACTED_FRAMES, exist_ok=True)
os.makedirs(MASKS_FOLDER, exist_ok=True)

# -------------------------------
# 1. EXTRACT FRAMES
# -------------------------------
video_files = glob.glob(os.path.join(DATASET_ROOT, "*.mp4"))
if not video_files:
    raise FileNotFoundError("No MP4 video found in dataset folder.")
video_file = video_files[0]

cap = cv2.VideoCapture(video_file)
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(EXTRACTED_FRAMES, f"frame_{idx:04d}.png"), frame)
    idx += 1
cap.release()
print(f"[INFO] Extracted {idx} frames to {EXTRACTED_FRAMES}")

num_frames = idx

# -------------------------------
# 2. SAVE MASKS
# -------------------------------
mask_h5_path = os.path.join(DATASET_ROOT, "masks.hdf5")
if not os.path.exists(mask_h5_path):
    raise FileNotFoundError(f"Masks file not found: {mask_h5_path}")

with h5py.File(mask_h5_path, "r") as f:
    mask_key = list(f.keys())[0]
    all_masks_dataset = f[mask_key]
    num_masks = min(num_frames, all_masks_dataset.shape[0])
    print(f"[INFO] Saving {num_masks} masks...")
    for i in range(num_masks):
        mask_i = np.array(all_masks_dataset[i])
        if mask_i.ndim > 2:
            mask_i = mask_i.squeeze()
        np.save(os.path.join(MASKS_FOLDER, f"frame_{i:04d}.npy"), mask_i)

print(f"[OK] Masks saved to {MASKS_FOLDER}")

# -------------------------------
# 3. CREATE CAMERA MATRICES
# -------------------------------
# Example intrinsic K from your previous post
K = np.array([
    [-0.11669534, 0.0, -0.00090632],
    [0.0, 0.2515035, -0.00095365],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Extrinsics: translation only, assume identity rotation
t = np.array([0.0, 0.0, 0.0]).reshape(3,1)
R = np.eye(3, dtype=np.float32)
w2c_matrix = np.hstack([R, t])  # shape (3,4)

# Repeat for all frames
w2c_all = np.tile(w2c_matrix[np.newaxis, :, :], (num_frames, 1, 1))  # shape (num_frames, 3,4)

np.savez(CAMERA_FILE, intrinsic=K, extrinsic=w2c_all)
print(f"[OK] Cameras saved to {CAMERA_FILE}")

