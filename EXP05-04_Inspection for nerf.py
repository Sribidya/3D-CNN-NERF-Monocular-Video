
import os
import pickle
import glob
import numpy as np
from PIL import Image
import h5py
import pickle


def check_smpl_model(pkl_path):
    """
    Checks if a .pkl file is a full SMPL model or just a canonical mesh.
    """
    with open(pkl_path, "rb") as f:
        data = data = pickle.load(f, encoding='latin1')

    # Check type
    if isinstance(data, dict):
        keys = data.keys()
    else:
        keys = dir(data)

    has_joints = 'J_regressor' in keys
    has_weights = 'weights' in keys
    has_template = 'v_template' in keys

    if has_joints and has_weights and has_template:
        print(f"[FULL SMPL MODEL] {pkl_path}")
    elif 'vertices' in keys or 'faces' in keys:
        print(f"[CANONICAL/FITTED MESH] {pkl_path}")
    else:
        print(f"[UNKNOWN TYPE]{pkl_path}keys/attributes{keys}")


dataset_folder = "dataset/people_snapshot_public/female-1-casual" 
for file in os.listdir(dataset_folder):
    if file.endswith(".pkl"):
        check_smpl_model(os.path.join(dataset_folder, file))


h5_path = "dataset/people_snapshot_public/female-1-casual/reconstructed_poses.hdf5"
with h5py.File(h5_path, "r") as f:
    print("Datasets in HDF5 file:")
    def print_name(name):
        print(name)
    f.visit(print_name)

cam = np.load("output/female-1-casual/cameras.npz")
print("Keys:", cam.files)  
data = np.load("output/female-1-casual/cameras.npz")
print(data.files)            
print(data['extrinsic'].shape)  


# Load images
img_files = sorted(glob.glob("output/female-1-casual/images/*.*"))
images = [np.array(Image.open(f).convert("RGB")) for f in img_files]

# Load masks
mask_files = sorted(glob.glob("output/female-1-casual/masks/*.npy"))
masks = [np.load(f) for f in mask_files]

# Load cameras
cam = np.load("output/female-1-casual/cameras.npz")
K = cam["intrinsic"]
w2c_all = cam["extrinsic"]

# Check lengths
print("Images:", len(images))
print("Masks:", len(masks))
print("Camera extrinsics:", w2c_all.shape)

# Assert consistency
assert len(images) == len(masks) == w2c_all.shape[0]
print("[OK] All lengths match!")



