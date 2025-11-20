import numpy as np
import os

# -------------------------------
# SETTINGS
# -------------------------------
OUTPUT_ROOT = "output/female-1-casual"  # must match your preprocessing output
num_frames = 757                         # number of extracted frames
H, W = 512, 512                          # image height and width (adjust if needed)
focal = 500.0                             # approximate focal length

# -------------------------------
# CAMERA INTRINSICS
# -------------------------------
K = np.array([[focal, 0, W/2],
              [0, focal, H/2],
              [0, 0, 1]], dtype=np.float32)

# -------------------------------
# CAMERA EXTRINSICS (circular trajectory)
# -------------------------------
w2c_list = []
radius = 2.0
for i in range(num_frames):
    theta = 2 * np.pi * i / num_frames
    cam_pos = np.array([radius * np.sin(theta), 0.0, radius * np.cos(theta)])
    
    # Look at origin
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0,1,0])
    right = np.cross(up, forward)
    up = np.cross(forward, right)
    
    R = np.stack([right, up, forward], axis=1)
    t = -R.T @ cam_pos
    
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3,:3] = R
    w2c[:3,3] = t
    w2c_list.append(w2c)

w2c_all = np.stack(w2c_list, axis=0)

# -------------------------------
# SAVE TO NPZ
# -------------------------------
os.makedirs(OUTPUT_ROOT, exist_ok=True)
np.savez(os.path.join(OUTPUT_ROOT, "cameras.npz"), intrinsic=K, extrinsic=w2c_all)
print(f"[INFO] Saved cameras.npz with {num_frames} extrinsics to {OUTPUT_ROOT}")

import glob
images = glob.glob("output/female-1-casual/images/*.*")
print(len(images))
