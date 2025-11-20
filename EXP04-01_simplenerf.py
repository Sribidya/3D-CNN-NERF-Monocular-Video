import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
# ADDED: Metric imports
from skimage.metrics import structural_similarity as ssim 
import lpips 
from pytorch3d.renderer import RayBundle
from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points
from scipy.spatial.transform import Rotation as R
import math
import cv2
from tqdm import tqdm # Added tqdm for progress visualization
from datetime import datetime # ADDED: For time-based output folders

# -------------------------------
# LPIPS Configuration and Setup
# -------------------------------
# The weight for LPIPS loss component. Must be tuned (e.g., 0.001 to 0.01).
LAMBDA_LPIPS = 0.005 

# Placeholder for LPIPS Model (initialized in train_nerf)
LPIPS_MODEL = None 

# -------------------------------
# 1. HarmonicEmbedding (Positional Encoding)
# -------------------------------
class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        super().__init__()
        # 2^0, 2^1, 2^2, ...
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
    def forward(self, x):
        """
        x: [..., D] (e.g., 3D coordinates or 3D direction vector)
        return: [..., D * n_harmonic * 2] (concatenated sine and cosine features)
        """
        # (x * 2^i)
        embed = (x[..., None] * self.frequencies) # [..., D, n_harm]
        # Flatten D and n_harm dimensions
        embed = embed.reshape(*x.shape[:-1], -1) # [..., D*n_harm]
        # Concatenate sin and cos components
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

# -------------------------------
# 2. Basic NeRF
# -------------------------------
class NeuralRadianceField(nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        # 3 coordinates (x,y,z) * 2 (sin/cos) * n_harmonic_functions
        embedding_dim = n_harmonic_functions * 2 * 3
        
        # Core NeRF MLP for density (sigma) and feature extraction
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, n_hidden_neurons),
            nn.Softplus(beta=10.0),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.Softplus(beta=10.0),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.Softplus(beta=10.0),
            nn.Linear(n_hidden_neurons, n_hidden_neurons),
            nn.Softplus(beta=10.0),
        )

        # Color MLP (View-Dependent)
        # Input is feature vector + embedded ray direction
        self.color_layer = nn.Sequential(
            nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            nn.Softplus(beta=10.0),
            nn.Linear(n_hidden_neurons, 3), # RGB output
            nn.Sigmoid(), # Clamp output between 0 and 1
        )
        
        # Density (Sigma) Layer
        self.density_layer = nn.Sequential(
            nn.Linear(n_hidden_neurons, 1), # Sigma output
            nn.Softplus(beta=10.0), # Ensure sigma >= 0
        )
        
        # Set initial densities small to encourage sparsity
        self.density_layer[0].bias.data[0] = -1.5
        
    def _get_densities(self, features):
        # Return sigma >= 0 (not alpha)
        sigmas = self.density_layer(features)
        return sigmas
    
    def _get_colors(self, features, rays_directions):
        """
        features: [..., N_samples, C] (Output of core MLP)
        rays_directions: [..., 3] or [..., N_samples, 3] (Original direction vector)
        """
        spatial_size = features.shape[:-1] # e.g. [N_rays, N_samples]
        rays_directions_normed = F.normalize(rays_directions, dim=-1)
        # Embed the ray direction
        rays_embedding = self.harmonic_embedding(rays_directions_normed)
        
        # Expand direction embedding to match the feature dimensions
        if len(spatial_size) > len(rays_embedding.shape[:-1]):
            expand_shape = spatial_size + (rays_embedding.shape[-1],)
            rays_embedding_expand = rays_embedding[..., None, :].expand(*expand_shape)
        else:
            rays_embedding_expand = rays_embedding
            
        color_input = torch.cat((features, rays_embedding_expand), dim=-1)
        return self.color_layer(color_input)
    
    def forward(self, ray_bundle: RayBundle, **kwargs):
        # 1. Map samples on ray to world coordinates
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # 2. Encode coordinates
        embeds = self.harmonic_embedding(rays_points_world)
        # 3. Pass through core MLP
        features = self.mlp(embeds)
        # 4. Predict density (sigma)
        sigmas = self._get_densities(features)
        # 5. Predict color (using features + view direction)
        colors = self._get_colors(features, ray_bundle.directions)
        return sigmas, colors

# -------------------------------
# 3. Volume rendering
# -------------------------------
def volume_render(sigmas, colors, t_vals, white_bg=True):
    """
    sigmas: [N_rays, N_samples, 1]
    colors: [N_rays, N_samples, 3]
    t_vals: [N_rays, N_samples] (sample depths)
    """
    # 1. Calculate distances (deltas) between samples
    deltas = t_vals[..., 1:] - t_vals[..., :-1] # [N_rays, N_samples-1]
    # Append a large distance for the last sample (reaching infinity)
    delta_inf = 1e10 * torch.ones_like(deltas[..., :1])
    deltas = torch.cat([deltas, delta_inf], dim=-1) # [N_rays, N_samples]
    deltas = deltas.unsqueeze(-1) # [N_rays, N_samples, 1]
    
    # 2. Calculate Alpha (opacity)
    # alpha = 1 - exp(-sigma * delta)
    alphas = 1.0 - torch.exp(-sigmas * deltas)
    eps = 1e-10
    
    # 3. Calculate Transmittance (T_i)
    # T_i = product(1 - alpha_j) for j < i. Transmittance is the probability light reaches sample i.
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alphas[:, :1]), 1.0 - alphas + eps], dim=1),
        dim=1,
    )
    trans = trans[:, :-1]
    
    # 4. Calculate Weights (W_i)
    # Weight = T_i * alpha_i (how much this sample contributes to the final pixel)
    weights = alphas * trans # [N_rays, N_samples, 1]
    
    # 5. Integrate RGB and calculate accumulated opacity (acc)
    rgb = (weights * colors).sum(dim=1) # [N_rays, 3]
    acc = weights.sum(dim=1) # [N_rays, 1]
    
    # 6. Apply white background (if accumulated opacity is < 1, fill remaining with white)
    if white_bg:
        rgb = rgb + (1.0 - acc) * 1.0
        
    return rgb, acc

# -------------------------------
# 4. Dataset for single subject (video + camera.pkl)
# -------------------------------
class PeopleSnapshotDataset(Dataset):
    def __init__(
        self,
        subject_dir,
        video_name="female-1-casual.mp4",
        n_rays=1024,
        n_samples=64,
        near=0.5,
        far=3.0,
        device="cpu",
    ):
        super().__init__()
        self.subject_dir = subject_dir
        self.n_rays = n_rays
        self.n_samples = n_samples
        self.near = near
        self.far = far
        self.device = device
        
        # ---------- video ----------
        video_path = os.path.join(subject_dir, video_name)
        print("Loading video from:", video_path)
        # torchvision.io.read_video returns (video_data, audio_data, info)
        try:
            video_data, _, _ = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")
            # Convert to (F, H, W, 3) and normalize to [0, 1]
            self.frames = video_data.permute(0, 2, 3, 1).float() / 255.0 # [F,H,W,3]
        except Exception as e:
            # Added more robust error handling for video loading
            print(f"Error loading video with torchvision.io.read_video: {e}. Trying opencv fallback.")
            cap = cv2.VideoCapture(video_path)
            frames_list = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB, normalize, and convert to tensor
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(torch.from_numpy(frame_rgb).float() / 255.0)
            cap.release()
            self.frames = torch.stack(frames_list) if frames_list else torch.empty(0)


        self.num_frames, self.H, self.W, _ = self.frames.shape
        print("Video shape:", self.frames.shape)
        
        # ---------- camera ----------
        cam_path = os.path.join(subject_dir, "camera.pkl")
        print("Loading camera from:", cam_path)
        with open(cam_path, "rb") as f:
            self.cam = pickle.load(f, encoding="latin1")
            
        print("camera.pkl keys:", self.cam.keys())
        
        # ---------- INTRINSICS (K matrix) ----------
        # Logic to extract fx, fy, cx, cy from various possible keys
        if 'camera_f' in self.cam and 'camera_c' in self.cam:
            camera_f = np.array(self.cam['camera_f'])
            camera_c = np.array(self.cam['camera_c'])
            if camera_f.ndim == 0 or camera_f.size == 1:
                self.fx = self.fy = float(camera_f)
            elif camera_f.size == 2:
                self.fx = float(camera_f[0])
                self.fy = float(camera_f[1])
            else:
                raise ValueError(f"Unexpected camera_f shape: {camera_f.shape}")
            self.cx = float(camera_c[0])
            self.cy = float(camera_c[1])
        elif 'camera_k' in self.cam: # New robust check
            K_all = np.array(self.cam["camera_k"])
            # K matrix is 3x3
            if K_all.ndim == 3:
                K = K_all[0] # [F,3,3] -> take first frame
            elif K_all.ndim == 2:
                K = K_all # [3,3] directly
            elif K_all.ndim == 1 and K_all.size == 9:
                K = K_all.reshape(3, 3)
            elif K_all.ndim == 1 and K_all.size == 4:
                 # Assume [fx, fy, cx, cy] format
                self.fx = float(K_all[0])
                self.fy = float(K_all[1])
                self.cx = float(K_all[2])
                self.cy = float(K_all[3])
                K = None # Bypass 3x3 parsing
            else:
                raise ValueError(f"Unexpected camera_k format: ndim={K_all.ndim}, shape={K_all.shape}")
            
            if K is not None:
                self.fx = float(K[0, 0])
                self.fy = float(K[1, 1])
                self.cx = float(K[0, 2])
                self.cy = float(K[1, 2])
        else:
            raise KeyError(f"Missing Intrinsic Keys (expected 'camera_f'/'camera_c' or 'camera_k'). Available keys: {list(self.cam.keys())}")


        print(f"Intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        
        # ---------- EXTRINSICS (R and T matrices) ----------
        # This is the segment that fixes the original 'R_all' KeyError by using 'camera_rt'
        
        # Check for the correct key, preferring 'camera_rt' which is common for People Snapshot
        if 'camera_rt' not in self.cam:
             # Fallback check in case the dataset uses simple R/T keys
             if 'R_all' in self.cam and 'T_all' in self.cam:
                 R_all = np.array(self.cam["R_all"])
                 t_all = np.array(self.cam["T_all"])
             else:
                 raise KeyError(f"Missing Extrinsic Keys (expected 'camera_rt' or 'R_all'/'T_all'). Available keys: {list(self.cam.keys())}")
        else:
            rt_all = np.array(self.cam["camera_rt"])
            if rt_all.shape == (3,):
                # rt_all is a rotation vector (rodrigues), assumes constant T is also available
                R_mat = R.from_rotvec(rt_all).as_matrix()
                R_all = np.stack([R_mat] * self.num_frames)
                t_all_np = np.array(self.cam["camera_t"])
                if t_all_np.shape != (3,):
                    raise ValueError(f"Unexpected camera_t shape: {t_all_np.shape}")
                t_all = np.stack([t_all_np] * self.num_frames)
            elif rt_all.ndim == 3 and rt_all.shape[-2:] == (3, 4):
                # [F, 3, 4] format
                R_all = rt_all[..., :3]
                t_all = rt_all[..., 3]
            elif rt_all.ndim == 3 and rt_all.shape[-2:] == (4, 4):
                # [F, 4, 4] format (Homogeneous matrix)
                R_all = rt_all[..., :3, :3]
                t_all = rt_all[..., :3, 3]
            else:
                raise ValueError(f"Unexpected camera_rt shape: {rt_all.shape}")

        # Store R_all and T_all on the specified device
        self.R_all = torch.from_numpy(R_all).float().to(self.device) # [F,3,3]
        self.T_all = torch.from_numpy(t_all).float().to(self.device) # [F,3]
        print("R_all shape:", self.R_all.shape)
        print("T_all shape:", self.T_all.shape)
        
    def __len__(self):
        return self.num_frames
    
    def _sample_pixels(self, frame_idx):
        """ Sample random pixels across the frame. """
        # Sampling is done on CPU, then moved to device in __getitem__
        ys = torch.randint(0, self.H, (self.n_rays,))
        xs = torch.randint(0, self.W, (self.n_rays,))
        return xs, ys
        
    def __getitem__(self, idx):
        frame_idx = idx
        frame = self.frames[frame_idx] # [H,W,3]
        
        # R and T are already on device (cuda), so just get them
        R = self.R_all[frame_idx] # [3,3] (cuda)
        T = self.T_all[frame_idx] # [3] (cuda)
        
        # Sample pixels on CPU
        xs_cpu, ys_cpu = self._sample_pixels(frame_idx) # [N_rays] (cpu)
        
        # Move sampled coordinates and target RGB to device
        xs = xs_cpu.to(self.device) # (cuda)
        ys = ys_cpu.to(self.device) # (cuda)
        target_rgb = frame[ys_cpu, xs_cpu].to(self.device) # [N_rays,3] (cuda)

        # 1. Project pixel coordinates to camera space directions
        x_cam = (xs.float() - self.cx) / self.fx
        y_cam = (ys.float() - self.cy) / self.fy
        ones = torch.ones_like(x_cam)
        dirs_cam = torch.stack([x_cam, y_cam, ones], dim=-1) # [N_rays,3] (cuda)
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        
        # 2. Transform directions and origin to World Space
        R_t = R.t() # [3,3] (cuda)
        
        # --- FIX: Ensure all operands are on the same device (cuda) ---
        # dirs_cam.T is implicitly moved to the device of the operation or R_t
        dirs_world = (R_t @ dirs_cam.T).T # [N_rays,3] (cuda)
        
        cam_center = -R_t @ T # [3] (cuda)
        origins = cam_center[None, :].expand_as(dirs_world) # [N_rays,3] (cuda)
        
        # 3. Define sample depths (t_vals)
        t_vals = torch.linspace(self.near, self.far, self.n_samples, device=self.device)
        t_vals = t_vals.view(1, -1).expand(self.n_rays, -1) # [N_rays,N_samples] (cuda)
        
        # 4. Create RayBundle (PyTorch3D structure for NeRF input)
        # All components are now on the correct device (cuda)
        ray_bundle = RayBundle(
            origins=origins,
            directions=dirs_world,
            lengths=t_vals,
            xys=None,
        )
        return ray_bundle, target_rgb

# -------------------------------
# 5. Training loop
# -------------------------------
def train_nerf(
    subject_dir,
    video_name="female-1-casual.mp4",
    device="cuda",
    n_epochs=5,
    batch_size=1,
    n_rays=1024,
    n_samples=64,
    lr=1e-4,
):
    # Ensure device is correctly set
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Initialize Dataset and DataLoader
    try:
        dataset = PeopleSnapshotDataset(
            subject_dir=subject_dir,
            video_name=video_name,
            n_rays=n_rays,
            n_samples=n_samples,
            device=device, # Pass the device for R_all/T_all storage
        )
    except KeyError as e:
        print("\n--- CRITICAL DATA LOADING ERROR ---")
        print(f"A required key was missing from camera.pkl: {e}")
        print("Please verify the path and contents of your camera.pkl file.")
        return None, None, [] 
        
    # Use collate_fn=lambda x: x[0] because each item returned by __getitem__ is already a full batch of rays
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x[0])
    
    # 2. Initialize Model, Optimizer, and LPIPS Model
    model = NeuralRadianceField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # LPIPS Model Initialization (for later evaluation)
    global LPIPS_MODEL
    if LPIPS_MODEL is None and device.type == 'cuda':
        try:
            LPIPS_MODEL = lpips.LPIPS(net='vgg').to(device)
            LPIPS_MODEL.eval()
            print("LPIPS VGG model initialized successfully for evaluation.")
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS model: {e}. LPIPS evaluation will be skipped.")
        
    # NEW: List to store loss history for plotting
    loss_history = []
    
    # 3. Training Loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        # Wrap the inner loop with tqdm for a progress bar
        for ray_bundle, target_rgb in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            # ray_bundle and target_rgb are already on device (cuda) from __getitem__
            optimizer.zero_grad()
            
            # Forward pass: NeRF prediction
            sigmas, colors = model(ray_bundle)
            t_vals = ray_bundle.lengths
            
            # Volume rendering
            pred_rgb, _ = volume_render(sigmas, colors, t_vals)
            
            # Loss calculation
            # We use MSE on the ray batch for efficiency (standard NeRF training).
            mse_loss = F.mse_loss(pred_rgb, target_rgb)

            # NOTE ON LPIPS LOSS INTEGRATION:
            # To use LPIPS as a loss function, you must reconstruct the full image 
            # from the rays (or use large enough ray batches) every N steps, as LPIPS 
            # requires spatial context (H, W). Here we use standard MSE on rays:
            # loss = mse_loss + LAMBDA_LPIPS * lpips_loss (if full image was available)
            loss = mse_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{n_epochs} | loss = {avg_loss:.6f}")
        
        # NEW: Store average loss for the epoch
        loss_history.append(avg_loss)
        
    # NEW: Save the training history for visualization
    history_file = "training_history.pkl"
    try:
        with open(history_file, "wb") as f:
            pickle.dump({"loss_history": loss_history, "epochs": n_epochs}, f)
        print(f"Training history successfully saved to {history_file}")
    except Exception as e:
        print(f"Warning: Could not save training history to {history_file}. Error: {e}")
        
    # Update: Return loss history as well
    return model, dataset, loss_history

# -------------------------------
# 6. Evaluation
# -------------------------------
def evaluate_nerf(model, dataset, frame_idx, device, batch_size=4096):
    model.eval()
    with torch.no_grad():
        # Get ground truth image and parameters
        gt_image = dataset.frames[frame_idx].to(device) # [H, W, 3]
        H, W = dataset.H, dataset.W
        
        # R and T are already on the correct device (stored in dataset)
        R = dataset.R_all[frame_idx] 
        T = dataset.T_all[frame_idx] 
        fx, fy, cx, cy = dataset.fx, dataset.fy, dataset.cx, dataset.cy
        
        # Generate all pixel coordinates for evaluation (on device)
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        xs = xs.flatten()
        ys = ys.flatten()
        num_rays = H * W
        
        pred_image = torch.zeros(num_rays, 3, device=device)
        
        # Iterate over batches for inference
        for start in tqdm(range(0, num_rays, batch_size), desc="Rendering Frame"):
            end = min(start + batch_size, num_rays)
            xs_batch = xs[start:end]
            ys_batch = ys[start:end]
            
            # Generate RayBundle (same logic as in __getitem__)
            x_cam = (xs_batch.float() - cx) / fx
            y_cam = (ys_batch.float() - cy) / fy
            ones = torch.ones_like(x_cam)
            dirs_cam = torch.stack([x_cam, y_cam, ones], dim=-1) #[batch, 3]
            dirs_cam = F.normalize(dirs_cam, dim=-1)
            
            R_t = R.t()
            dirs_world = (R_t @ dirs_cam.T).T #[batch, 3]
            cam_center = -R_t @ T #[3]
            origins = cam_center[None, :].expand_as(dirs_world)  #[batch, 3]
            
            t_vals = torch.linspace(dataset.near, dataset.far, dataset.n_samples, device=device)
            t_vals = t_vals.view(1, -1).expand(len(xs_batch), -1)  #[batch, N_samples]
            
            ray_bundle = RayBundle(
                origins=origins,
                directions=dirs_world,
                lengths=t_vals,
                xys=None,
            )
            
            # Predict and render
            sigmas, colors = model(ray_bundle)
            pred_rgb, _ = volume_render(sigmas, colors, t_vals)
            pred_image[start:end] = pred_rgb
            
        pred_image = pred_image.reshape(H, W, 3)
        
        # ---------------------------------------------
        # Compute Comprehensive Metrics (PSNR, SSIM, LPIPS)
        # ---------------------------------------------
        
        # 1. PSNR (Pixel-level fidelity)
        mse = F.mse_loss(pred_image, gt_image)
        psnr = -10.0 * torch.log10(mse)
        
        # Prepare images for CPU-based metrics (SSIM, LPIPS)
        pred_np = pred_image.cpu().numpy()
        gt_np = gt_image.cpu().numpy()
        
        # 2. SSIM (Structural Similarity)
        # SSIM requires H, W, 3 format (NumPy)
        ssim_val = ssim(gt_np, pred_np, data_range=1.0, channel_axis=-1)

        # 3. LPIPS (Learned Perceptual Similarity)
        lpips_val = None
        global LPIPS_MODEL
        if LPIPS_MODEL is not None:
            try:
                # LPIPS requires N, C, H, W and normalization to [-1, 1]
                pred_lpips = pred_image.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                gt_lpips = gt_image.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                
                with torch.no_grad():
                    # Calculate perceptual distance
                    lpips_tensor = LPIPS_MODEL(pred_lpips.to(device), gt_lpips.to(device))
                    lpips_val = lpips_tensor.mean().item()
            except Exception as e:
                print(f"Warning: LPIPS calculation failed during evaluation: {e}")
        
        # Save images
        timestamp_folder = datetime.now().strftime("%H%M")
        output_dir = os.path.join("output_renders", timestamp_folder)
        
        try:
            # 1. Attempt to create the specific HHMM directory
            os.makedirs(output_dir, exist_ok=True)
            print(f"INFO: Images will be saved to time-stamped directory: {output_dir}")
            
        except Exception as e:
            # 2. If directory creation fails for any reason (like permission or path error)
            fallback_dir = os.path.join("output_renders", "fallback_eval")
            os.makedirs(fallback_dir, exist_ok=True)
            output_dir = fallback_dir
            print(f"WARNING: Directory creation failed ({e}). Falling back to: {output_dir}")
                    
        
        torchvision.utils.save_image(pred_image.permute(2,0,1), os.path.join(output_dir, f"rendered_frame_{frame_idx}.png"))
        torchvision.utils.save_image(gt_image.permute(2,0,1), os.path.join(output_dir, f"gt_frame_{frame_idx}.png"))
        
        print(f"Images saved to: {output_dir}")
        
        print(f"\n--- Evaluation Metrics for Frame {frame_idx} ---")
        print(f"PSNR (Higher is better): {psnr.item():.2f} dB")
        print(f"SSIM (Closer to 1 is better): {ssim_val:.4f}")
        if lpips_val is not None:
            print(f"LPIPS (Closer to 0 is better): {lpips_val:.4f}")
        else:
            print("LPIPS: Skipped (Model not initialized or failed to load weights)")
        
        return psnr.item() # Retaining original return value for simplicity

# -------------------------------
# 7. Entry point
# -------------------------------
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Please ADJUST this path to match your local setup:
    subject_dir = "dataset/people_snapshot_public/female-1-casual" 
    video_name = "female-1-casual.mp4"
    
    # Check if the directory exists before proceeding
    if not os.path.exists(subject_dir):
        print(f"Error: Subject directory not found at '{subject_dir}'")
        print("Please update the 'subject_dir' variable in the '__main__' block to point to your data.")
    else:
        # UPDATED: We capture the loss_history list
        model, dataset, loss_history = train_nerf(
            subject_dir=subject_dir,
            video_name=video_name,
            device="cuda",
            n_epochs=500,
            batch_size=1,
            n_rays=2048,
            n_samples=128,
            lr=1e-4,
        )
        
        if model is not None and dataset is not None:
            # Evaluate on a sample frame, e.g., frame 0
            psnr = evaluate_nerf(model, dataset, frame_idx=0, device="cuda")