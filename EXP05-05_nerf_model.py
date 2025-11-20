import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# Dataset
# ==============================
class SimpleNeRFDataset(Dataset):
    def __init__(self, root="./output/female-1-casual"):
        # Cameras
        data = np.load(os.path.join(root, "cameras.npz"))
        self.K = data["intrinsic"]
        self.w2c_all = data["extrinsic"]

        # Images
        img_files = sorted(glob.glob(os.path.join(root, "images", "*.*")))
        self.images = [np.array(Image.open(f).convert("RGB")) for f in img_files]
        self.images = np.stack(self.images, axis=0)

        # Masks
        mask_files = sorted(glob.glob(os.path.join(root, "masks", "*.npy")))
        self.masks = [np.load(f) for f in mask_files]
        self.masks = np.stack(self.masks, axis=0)

        self.H = self.images.shape[1]
        self.W = self.images.shape[2]

        assert len(self.images) == len(self.w2c_all) == len(self.masks), \
            f"Mismatch: {len(self.images)} images, {len(self.w2c_all)} cameras, {len(self.masks)} masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        w2c = torch.tensor(self.w2c_all[idx], dtype=torch.float32)
        return {
            "img": img.permute(2,0,1),  # C,H,W
            "mask": mask,               # H,W
            "w2c": w2c
        }

# ==============================
# NeRF Model
# ==============================
class NeRF(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 4)  # RGB + Density

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

# ==============================
# Ray generation
# ==============================
def get_rays(H, W, K, w2c, device):
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    dirs = torch.stack([(i - K[0,2]) / K[0,0],
                        (j - K[1,2]) / K[1,1],
                        torch.ones_like(i)], -1)

    R = w2c[:3,:3]
    t = w2c[:3,3]

    dirs = dirs @ R.T
    origins = t.expand_as(dirs)
    return origins.reshape(-1,3), dirs.reshape(-1,3)

# ==============================
# Volume rendering
# ==============================
def render_rays(model, origins, dirs, chunk_size=1024*32):
    rgb_list = []
    for i in range(0, origins.shape[0], chunk_size):
        o_chunk = origins[i:i+chunk_size]
        d_chunk = dirs[i:i+chunk_size]
        t_vals = torch.linspace(0, 1, 32, device=origins.device)

        # Fix broadcasting: [num_rays, 32, 3]
        pts = o_chunk[:, None, :] + d_chunk[:, None, :] * t_vals[None, :, None]

        rgb_sigma = model(pts.reshape(-1,3)).reshape(pts.shape[0], 32, 4)
        rgb = torch.sigmoid(rgb_sigma[..., :3])
        sigma = F.relu(rgb_sigma[..., 3])

        deltas = torch.ones_like(sigma) * (1.0 / 32)
        alpha = 1 - torch.exp(-sigma * deltas)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0],1), device=alpha.device),
                       1 - alpha + 1e-10], -1), -1
        )[:, :-1]

        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        rgb_list.append(rgb_map)
    return torch.cat(rgb_list, dim=0)

# ==============================
# Metrics
# ==============================
def compute_psnr(x, y):
    return -10 * np.log10(np.mean((x - y)**2) + 1e-8)

def compute_ssim(x, y):
    H, W = x.shape[:2]
    min_side = min(H, W)
    win = 7 if min_side >= 7 else max(3, (min_side // 2) * 2 + 1)
    return ssim(x, y, channel_axis=2, win_size=win, data_range=1.0)

# ==============================
# Training loop
# ==============================
def train(dataset, batch_size=1, ray_batch=1024*32, lr=1e-4, epochs=1, print_every=10):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = NeRF().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lpips_fn = LPIPS(net='vgg').to(DEVICE)

    H, W = dataset.H, dataset.W
    K = torch.tensor(dataset.K, dtype=torch.float32, device=DEVICE)

    print(f"[INFO] Training for {epochs} epoch(s)")
    for epoch in range(epochs):
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        running_lpips = 0.0

        for batch_idx, batch in enumerate(tqdm(loader)):
            img_gt = batch["img"].to(DEVICE)[0].permute(1,2,0)  # HWC
            mask = batch["mask"].to(DEVICE)[0]
            w2c = batch["w2c"].to(DEVICE)[0]

            origins, dirs = get_rays(H, W, K, w2c, DEVICE)

            # Masked rays
            mask_flat = mask.reshape(-1) > 0.5
            origins_fg = origins[mask_flat]
            dirs_fg = dirs[mask_flat]

            # Render rays
            rgb_pred = render_rays(model, origins_fg, dirs_fg, chunk_size=ray_batch)

            # Fill full image for metrics
            rgb_full = torch.zeros((H*W,3), device=DEVICE)
            rgb_full[mask_flat] = rgb_pred
            rgb_full = rgb_full.reshape(H,W,3)

            # Loss
            loss = F.mse_loss(rgb_full, img_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            rgb_np = rgb_full.detach().cpu().numpy()
            gt_np = img_gt.detach().cpu().numpy()
            psnr_v = compute_psnr(rgb_np, gt_np)
            ssim_v = compute_ssim(rgb_np, gt_np)
            lp = lpips_fn(
                rgb_full.permute(2,0,1).unsqueeze(0),
                img_gt.permute(2,0,1).unsqueeze(0)
            ).item()

            running_loss += loss.item()
            running_psnr += psnr_v
            running_ssim += ssim_v
            running_lpips += lp

            if batch_idx % print_every == 0:
                print(f"[Epoch {epoch+1} Batch {batch_idx}] "
                      f"Loss: {loss.item():.4f} | PSNR: {psnr_v:.2f} | "
                      f"SSIM: {ssim_v:.3f} | LPIPS: {lp:.4f}")

        num_batches = len(loader)
        print(f"[Epoch {epoch+1}] Avg Loss: {running_loss/num_batches:.4f} | "
              f"Avg PSNR: {running_psnr/num_batches:.2f} | "
              f"Avg SSIM: {running_ssim/num_batches:.3f} | "
              f"Avg LPIPS: {running_lpips/num_batches:.4f}")

    # -----------------------------
    # Show final reconstructed image
    # -----------------------------
    plt.figure(figsize=(8,8))
    plt.imshow(np.clip(rgb_np, 0, 1))
    plt.title("Last batch reconstructed image")
    plt.axis("off")
    #plt.savefig("reconstructed_image.png", bbox_inches='tight')
    plt.show()

    print("[INFO] Training complete.")
    return model

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    dataset = SimpleNeRFDataset(root="output/female-1-casual")
    model = train(dataset, batch_size=1, ray_batch=1024*32, lr=1e-4, epochs=10, print_every=50)
