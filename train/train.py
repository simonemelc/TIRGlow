import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import matplotlib.pyplot as plt
from PIL import Image as PILImage
from collections import namedtuple
import torchvision.models as models
import kornia.color as Kcolor  

# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToPILImage

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "dataset"))
sys.path.append(str(project_root / "model"))

from dataset.HDRT_dataset import HDRT_Dataset
from model.ThermalIntrinsicNet_attention import ThermalIntrinsicNet


# === ARGUMENTS ===
parser = argparse.ArgumentParser(description="Train Thermal Intrinsic Net on HDRT")
parser.add_argument("--device", type=int, default=3, help="GPU ID")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data for validation")


# Loss Weights FIXED FOR THE FOLLOWING
# parser.add_argument("--w_R", type=float, default=1.0, help="Weight for Reflectance Loss")
# parser.add_argument("--w_S", type=float, default=1.0, help="Weight for Shading Loss")
# parser.add_argument("--w_phy", type=float, default=0.5, help="Weight for Physical Consistency Loss")
# parser.add_argument("--w_refinement", type=float, default=1.0, help="Weight for Final Refinement Loss")

parser.add_argument("--w_lab", type=float, default=1.0, help="Weight for Lab Loss")
parser.add_argument("--w_edge", type=float, default=0.8, help="Weight for Edge Loss")
parser.add_argument("--w_perc", type=float, default=0.2, help="Weight for Perceptual Loss")

# Paths 
parser.add_argument("--root_rgb", type=str, default='/medias/db/ImagingSecurity_misc/HDRT/registered_RGB_reduced_size/registered_RGB_reduced_size/')
parser.add_argument("--root_thermal", type=str, default='/medias/db/ImagingSecurity_misc/HDRT/infrared/infrared')
parser.add_argument("--root_targets_R", type=str, default='/medias/db/ImagingSecurity_misc/HDRT/registered_RGB_reduced_size/registered_RGB_reduced_size/reflectance')
parser.add_argument("--root_targets_S", type=str, default='/medias/db/ImagingSecurity_misc/HDRT/registered_RGB_reduced_size/registered_RGB_reduced_size/shading')


parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel on all GPUs")
parser.add_argument("--has_thermal", action="store_true")

args = parser.parse_args()

# === CONFIG ENV ===
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
# device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

use_ddp = args.ddp
has_thermal = args.has_thermal

if use_ddp:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    def is_main():
        return dist.get_rank() == 0
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))

else:
    # single GPU
    local_rank = args.device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    def is_main():
        return True

    world_size = 1

if is_main():
    print(f"Running with use_ddp={use_ddp} | world_size={world_size} | local_rank={local_rank} | device={device}")


# === OUTPUT DIRS ===

run_tag = f"attention/thermal_{args.has_thermal}_lr{args.lr}_wLab{args.w_lab}_wEdge{args.w_edge}_wPerc{args.w_perc}"

save_dir = str(project_root / f"output_models/train/{run_tag}")
ckpt_dir = str(project_root / f"checkpoints/{run_tag}")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)


if is_main():

    print(f"""
    === CONFIGURATION ===
    • DEVICE       : {device}
    • BATCH SIZE   : {args.batch_size}
    • HAS THERMAL  : {args.has_thermal}
    • EPOCHS       : {args.epochs}
    • VAL SPLIT    : {args.val_split*100}%
    • WEIGHTS      :  Lab = {args.w_lab} | Edge = {args.w_edge} | Perceptual = {args.w_perc}
    • SAVE DIR     : {save_dir}
    • USE DDP      : {args.ddp}
    """)


def view(img):
    """
    As in Careaga et al.
    Gamma for visualization of reflectance and shading
    """
    img_gamma = np.power(img, 1/2.2)
    
    max_val = np.percentile(img_gamma, 100)
    if max_val > 1e-6:
        img_norm = img_gamma / max_val
    else:
        img_norm = img_gamma
        
    return np.clip(img_norm, 0, 1)


# === LOSS FUNCTION ===


class EdgeLoss(nn.Module):
    """
    Solve blur
    """
    def __init__(self):
        super().__init__()
        k_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        k_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', k_x)
        self.register_buffer('kernel_y', k_y)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        px = F.conv2d(pred_gray, self.kernel_x, padding=1)
        py = F.conv2d(pred_gray, self.kernel_y, padding=1)
        tx = F.conv2d(target_gray, self.kernel_x, padding=1)
        ty = F.conv2d(target_gray, self.kernel_y, padding=1)
        
        # Magnitude
        pmag = torch.sqrt(px**2 + py**2 + 1e-6)
        tmag = torch.sqrt(tx**2 + ty**2 + 1e-6)
        
        # Direction (Cosine Similarity)
        dot_prod = (px * tx) + (py * ty)
        cos_sim = dot_prod / (pmag * tmag + 1e-6)
        
        # Loss: Magnitude L1 + Direction 
        loss_mag = self.l1(pmag, tmag)
        mask = (tmag > 0.05).float().detach() # Ignore noisy in flat zones
        loss_dir = (1.0 - cos_sim) * mask
        
        return loss_mag + 0.5 * loss_dir.mean()

class ColorLossLab(nn.Module):
    """
    Lab Loss Normalized.
    Decouples most luminance and chrominance.
    """
    def __init__(self, w_ab=1.0):
        super().__init__()
        self.w_ab = w_ab
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        x_lab = Kcolor.rgb_to_lab(x)
        y_lab = Kcolor.rgb_to_lab(y)
        
        x_lab[:, 0, :, :] = x_lab[:, 0, :, :] / 100.0   # L
        x_lab[:, 1:, :, :] = x_lab[:, 1:, :, :] / 128.0 # a, b
        y_lab[:, 0, :, :] = y_lab[:, 0, :, :] / 100.0
        y_lab[:, 1:, :, :] = y_lab[:, 1:, :, :] / 128.0
        
        # Loss only on a, b (chrominance)
        return self.w_ab * self.l1(x_lab[:, 1:, :, :], y_lab[:, 1:, :, :])


class Vgg16Extractor(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16Extractor, self).__init__()
        # a VGG pre-trained
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        # Slice 1: relu1_2 (index 4)
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # Slice 2: relu2_2 (index 9)
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # Slice 3: relu3_3 (index 16)
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # Slice 4: relu4_3 (index 23)
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)


class PerceptualLoss(nn.Module):
    def __init__(self, weights=(1.0/32, 1.0/16, 1.0/8, 1.0/4)):
        """
        from https://github.com/pytorch/examples/blob/main/fast_neural_style/neural_style/vgg.py
        weights:  [relu1_2, relu2_2, relu3_3, relu4_3]
        Deep layers --> more semantic feature maps --> more weights
        """
        super().__init__()
        self.vgg = Vgg16Extractor(requires_grad=False)
        self.weights = weights
        self.l1 = nn.L1Loss()
        
        # ImageNet Norm
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def normalize(self, x):
        # x in range [0, 1]
        return (x - self.mean) / self.std

    def forward(self, x, y):
        # x: Prediction, y: Target. Both [B, 3, H, W] in range [0, 1]
        
        x_vgg = self.normalize(x)
        y_vgg = self.normalize(y)
        
        x_features = self.vgg(x_vgg)
        y_features = self.vgg(y_vgg)
        
        loss = 0.0
        # 3. loss 
        # relu1_2
        loss += self.weights[0] * self.l1(x_features.relu1_2, y_features.relu1_2)
        # relu2_2
        loss += self.weights[1] * self.l1(x_features.relu2_2, y_features.relu2_2)
        # relu3_3
        loss += self.weights[2] * self.l1(x_features.relu3_3, y_features.relu3_3)
        # relu4_3
        loss += self.weights[3] * self.l1(x_features.relu4_3, y_features.relu4_3)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for Intrinsic Decomposition and Final Refinement.
    """
    def __init__(self, w_lab, w_edge, w_perc):
        super().__init__()
        self.w_lab = w_lab
        self.w_edge = w_edge
        self.w_perc = w_perc

        self.smoothl1 = nn.SmoothL1Loss()
        self.edge = EdgeLoss()
        self.lab = ColorLossLab()
        self.perceptual = PerceptualLoss()

    def forward(self, pred_R, pred_D, pred_I_final, target_R, target_S_raw, target_I_lin, target_I_srgb):
        """
        pred_R: Predicted Reflectance [0,1]
        pred_D: Predicted Inverse Shading [0,1]
        pred_I_final: Refined Output Image
        target_R: GT Reflectance [0,1]
        target_S_raw: GT Shading (HDR, Unbounded)
        target_I_lin: GT Linear Image (Normal Light) [0,1]
        target_I_srgb: GT Gamma Image (Normal Light) [0,1]
        """
        
        # 1. Intrinsic Loss (linear)
        loss_R_pix = self.smoothl1(pred_R, target_R)
        loss_R_edge = self.edge(pred_R, target_R)
        loss_R_lab = self.lab(pred_R, target_R)

        loss_R = loss_R_pix +  self.w_edge * loss_R_edge + self.w_lab * loss_R_lab #2.0

        target_D = 1.0 / (target_S_raw + 1.0)  # shading loss in the Inverse Domain D = (1 / (S+1)) as in Dille et al. paper
        loss_S_pix = self.smoothl1(pred_D, target_D)
        loss_S_edge = self.edge(pred_D, target_D)

        loss_S = loss_S_pix + self.w_edge * loss_S_edge
        
        # 2. Physical Consistency Loss (Coarse Reconstruction)
        # Reconstruct S from D: S = (1-D)/(D+eps)
        rec_S = (1.0 - pred_D) / (pred_D + 1e-8)
        pred_I_coarse_lin = pred_R * rec_S             # IID LAW:  I_lin = pred_R * rec_S  or (I_lin)^1/2.2 = (pred_R * rec_S)^1/2.2  
        
        
        # Use Inverse domain representation also here
        pred_I_coarse_lin_inv = 1.0 / (1.0 + pred_I_coarse_lin)
        target_I_lin_inv = 1.0 / (1.0 + target_I_lin)
        loss_phy = self.smoothl1(pred_I_coarse_lin_inv, target_I_lin_inv) 

        # 4. Final Refinement Loss
        # Apply simple gamma for perceptual alignment
        gamma = 1.0 / 2.2
        pred_I_final_srgb = torch.pow(torch.clamp(pred_I_final, 0, 1) + 1e-6, gamma)
        #pred_I_final_srgb = torch.clamp(pred_I_final, 0, 1)

        #Pixel loss
        loss_refinement_pix = self.smoothl1(pred_I_final_srgb, target_I_srgb)
        #VGG loss
        loss_refinement_perc = self.perceptual(pred_I_final_srgb, target_I_srgb)
        #Color Lab loss
        loss_refinement_lab = self.lab(pred_I_final_srgb, target_I_srgb)

        #Edge loss
        loss_refinement_edge = self.edge(pred_I_final_srgb, target_I_srgb)

        loss_refinement = loss_refinement_pix + self.w_perc * loss_refinement_perc + self.w_lab * loss_refinement_lab + self.w_edge * loss_refinement_edge 

        # Weighted Sum
        total = (1.0 * loss_R) + (1.0 * loss_S) + \
                (0.5 * loss_phy) + (1.0 * loss_refinement)
                
        return total, loss_R, loss_S, loss_refinement

# === EMA UTILS ===
def update_ema(model, ema_dict, decay=0.999):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            
            if v.is_floating_point():
                ema_dict[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                ema_dict[k].copy_(v)

def load_ema(model, ema_dict):
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_dict, strict=False)
    return backup


# === DATASET SETUP ===

train_dataset = HDRT_Dataset(
    root_rgb=args.root_rgb,
    root_thermal=args.root_thermal,
    root_targets_R=args.root_targets_R,
    root_targets_S=args.root_targets_S,
    augment=True
)

val_dataset = HDRT_Dataset(
    root_rgb=args.root_rgb,
    root_thermal=args.root_thermal,
    root_targets_R=args.root_targets_R,
    root_targets_S=args.root_targets_S,
    augment=False
)

# Manual Split Train/Val
indices = list(range(len(train_dataset)))
random.seed(42)
random.shuffle(indices)

val_size = int(len(train_dataset) * args.val_split)
train_indices = indices[val_size:]
val_indices = indices[:val_size]

train_ds = Subset(train_dataset, train_indices)
val_ds = Subset(val_dataset, val_indices)


if use_ddp:
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True
    )

else:
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )


if is_main():
    print(f"Data Split: Train={len(train_ds)} | Val={len(val_ds)}")

# === MODEL & OPTIMIZER ===
model = ThermalIntrinsicNet(base_channels=16).to(device) #base_channels=16
print("initializing ThermalIntrinsicNet_attention model")

if use_ddp:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)#, find_unused_parameters=True)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

criterion = CombinedLoss(w_lab=args.w_lab, w_edge=args.w_edge, w_perc=args.w_perc).to(device)

# Initialize EMA
ema_decay = 0.999
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

# === HELPER FOR VISUALIZATION ===

def tensor_to_pil(tensor):
    """
    Convert tensor [C,H,W] to PIL Image for saving.
    Handles Grayscale [1,H,W] -> [H,W] squeezing automatically.
    """

    img = tensor.detach().cpu().float().numpy() 
    img = img.transpose(1, 2, 0)
    
    #NaN/Inf 
    if np.isnan(img).any() or np.isinf(img).any():
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    if img.shape[2] == 1:
        img = img[:, :, 0]
        
    return PILImage.fromarray(img)

# === TRAINING LOOP ===
best_psnr = 0.0
best_val = 100.0
train_hist, val_hist = [], []

if is_main():
    print("\n=== STARTING TRAINING ===")

for epoch in range(args.epochs):

    if use_ddp:
        train_sampler.set_epoch(epoch)

    # --- TRAIN ---
    model.train()
    running_loss = 0.0

    if is_main():
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    else:
        pbar = train_loader

    for batch in pbar:
        # Move data
        rgb_in = batch['input_rgb_low_synth'].to(device)
        therm_in = batch['input_therm'].to(device)
        
        target_R = batch['target_R'].to(device)
        target_S = batch['target_S'].to(device)
        target_I_srgb = batch['target_rgb'].to(device)
        target_I_lin = torch.pow(target_I_srgb + 1e-8, 2.2)
        
        # Forward
        optimizer.zero_grad()

        if has_thermal:
            pred_R, pred_D, pred_Coarse, pred_Final, _, _ = model(rgb_in, therm_in)
        else:
            pred_R, pred_D, pred_Coarse, pred_Final, _, _ = model(rgb_in, x_therm=None)
        
        # Loss
        loss, l_R, l_S, l_refinement = criterion(pred_R, pred_D, pred_Final, target_R, target_S, target_I_lin, target_I_srgb)
        
        # Backward
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        
        optimizer.step()

        update_ema(model, ema, ema_decay)
        
        running_loss += loss.item()
        # if is_main():
        #     pbar.set_postfix({"Total Loss": f"{loss.item():.4f}", "Reflectance": f"{l_R.item():.3f}", "Shading": f"{l_S.item():.3f}", "Refinement": f"{l_refinement.item():.3f}"})
        
    avg_train_loss = running_loss / len(train_loader)
    train_hist.append(avg_train_loss)
    
    # --- VALIDATION ---
    model.eval()
    # Load EMA weights for validation
    backup_weights = load_ema(model, ema)
    
    val_running_loss = 0.0
    psnr_list, ssim_list = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            rgb_in = batch['input_rgb_low_synth'].to(device) #input_rgb_low
            therm_in = batch['input_therm'].to(device)
            target_R = batch['target_R'].to(device)
            target_S = batch['target_S'].to(device)
            target_I_srgb = batch['target_rgb'].to(device)
            target_I_lin = torch.pow(target_I_srgb + 1e-8, 2.2)

            if model.training:
                if torch.rand(1).item() < 0.2: 
                    # "Zero-Lux"
                    rgb_in = torch.zeros_like(rgb_in)
            
            # Forward
            if has_thermal:
                pred_R, pred_D, pred_Coarse, pred_Final, pred_res, _ = model(rgb_in, therm_in)
            else:
                pred_R, pred_D, pred_Coarse, pred_Final, pred_res, _ = model(rgb_in, x_therm=None)
            
            # Loss
            v_loss, _, _, _ = criterion(pred_R, pred_D, pred_Final, target_R, target_S, target_I_lin, target_I_srgb)
            val_running_loss += v_loss.item()
            
            # Metrics
            res_lin = pred_Final.cpu().numpy().transpose(0, 2, 3, 1)
            res_srgb = np.clip(res_lin, 0, 1) ** (1/2.2)
            tgt_srgb = target_I_srgb.cpu().numpy().transpose(0, 2, 3, 1)
            
            for p, t in zip(res_srgb, tgt_srgb):
                psnr_list.append(psnr(t, p, data_range=1.0))
                ssim_list.append(ssim(t, p, channel_axis=-1, data_range=1.0))


            # Save sample images (First batch only)
            if i == 0 and (epoch % 20 == 0) and is_main():
                vis_dir = os.path.join(save_dir, f"epoch{epoch+1:03d}_samples")
                os.makedirs(vis_dir, exist_ok=True)

                r_cpu = pred_R[0].detach().cpu().numpy().transpose(1, 2, 0)
                d_cpu = pred_D[0].detach().cpu().numpy().transpose(1, 2, 0)
                coarse_cpu = pred_Coarse[0].detach().cpu().numpy().transpose(1, 2, 0)
                res_cpu = pred_res[0].detach().cpu().numpy().transpose(1, 2, 0)
                final_cpu = pred_Final[0].detach().cpu().numpy().transpose(1, 2, 0)
                gt_cpu = target_I_srgb[0].detach().cpu().numpy().transpose(1, 2, 0)
                in_cpu = rgb_in[0].detach().cpu().numpy().transpose(1, 2, 0)
                th_cpu = therm_in[0].detach().cpu().numpy().transpose(1, 2, 0)

                # Reflectance (lin [0,1])
                pred_r_vis = view(r_cpu)
                tensor_to_pil(torch.from_numpy(pred_r_vis.transpose(2, 0, 1))).save(os.path.join(vis_dir, "pred_R_view.png"))

                # D (Inverse Shading) 
                tensor_to_pil(torch.from_numpy(d_cpu.transpose(2, 0, 1))).save(os.path.join(vis_dir, "pred_D_inverse.png"))
                
                # Normal Shading is S = (1 - D) / D
                rec_S_cpu = (1.0 - d_cpu) / (d_cpu + 1e-6)
                rec_S_vis = view(rec_S_cpu)
                tensor_to_pil(torch.from_numpy(rec_S_vis.transpose(2, 0, 1))).save(os.path.join(vis_dir, "pred_S_view.png"))

                # Coarse prediction
                coarse_cpu = np.clip(coarse_cpu, 0.0, None)
                coarse_vis = (coarse_cpu) ** (1/2.2)
                tensor_to_pil(torch.from_numpy(coarse_vis.transpose(2, 0, 1))).save(os.path.join(vis_dir, "pred_coarse.png"))

                # Residual prediction
                res_vis = (res_cpu / 2.0) + 0.5 
                res_vis = np.clip(res_vis, 0.0, 1.0)
                tensor_to_pil(torch.from_numpy(res_vis.transpose(2, 0, 1))).save(os.path.join(vis_dir, "pred_residual.png"))

                # Final prediction
                final_cpu = np.clip(final_cpu, 0.0, None)
                final_vis = (final_cpu) ** (1/2.2)
                #tensor_to_pil(torch.from_numpy(final_cpu.transpose(2, 0, 1))).save(os.path.join(vis_dir, "pred_final_nogamma.png"))
                tensor_to_pil(torch.from_numpy(final_vis.transpose(2, 0, 1))).save(os.path.join(vis_dir, "pred_final.png"))
                
                # GT e Input
                tensor_to_pil(torch.from_numpy(gt_cpu.transpose(2, 0, 1))).save(os.path.join(vis_dir, "gt.png"))
                #tensor_to_pil(torch.from_numpy(in_cpu.transpose(2, 0, 1))).save(os.path.join(vis_dir, "input_low.png"))
                #tensor_to_pil(torch.from_numpy(th_cpu.transpose(2, 0, 1))).save(os.path.join(vis_dir, "input_thermal.png"))
            # ----------------------------------------------------


        if use_ddp:
            val_tensor = torch.tensor([val_running_loss, np.sum(psnr_list), np.sum(ssim_list), len(psnr_list)], device=device)
            dist.all_reduce(val_tensor) # Sum all GPUs
            
            total_loss = val_tensor[0].item()
            total_psnr = val_tensor[1].item()
            total_ssim = val_tensor[2].item()
            total_count = val_tensor[3].item()
            
            avg_val_loss = total_loss / (len(val_loader) * world_size)
            avg_psnr = total_psnr / total_count
            avg_ssim = total_ssim / total_count
        else:
            avg_val_loss = val_running_loss / len(val_loader)
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)

 
    val_hist.append(avg_val_loss)
    scheduler.step()


    model.load_state_dict(backup_weights)

    if use_ddp:
        model_to_save = model.module
    else:
        model_to_save = model

    if is_main():
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_psnr = avg_psnr
            
            ema_state_dict = {k: v.cpu() for k, v in ema.items()} 
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "ema_state_dict": ema_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_psnr": best_psnr
            }, os.path.join(ckpt_dir, f"best_model.pth"))
            
            print(f"✓ Best model saved (PSNR: {best_psnr:.2f})")

# === FINAL PLOTS ===
if is_main():
    actual_epochs = min(len(train_hist), len(val_hist))
    log_df = pd.DataFrame({"epoch": list(range(1, actual_epochs+1)),
                           "train_loss": train_hist[:actual_epochs], "val_loss": val_hist[:actual_epochs]})
    log_df.to_csv(os.path.join(save_dir, "loss_curves.csv"), index=False)

    plt.figure(figsize=(10,5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    print("Training Complete.")


#python train.py --device 3 --has_thermal
#torchrun --nproc_per_node=4 train.py --ddp --has_thermal