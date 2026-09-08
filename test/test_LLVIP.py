# python test_LLVIP.py --ckpt /path/to/ckpt.pth --has_thermal --self_ensemble

import os
import sys
import argparse
from pathlib import Path
import json
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Metric Imports
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color as skcolor
from skimage.color import deltaE_ciede2000
from skimage import img_as_ubyte 
import matplotlib

# LPIPS Import
try:
    import lpips
    have_lpips = True
except ImportError:
    print("LPIPS library not found. LPIPS metric will be skipped.")
    have_lpips = False

# -------------------------
# Path Setup
# -------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dataset.LLVIP_dataset import LLVIP_Dataset
from model.ThermalIntrinsicNet_attention import ThermalIntrinsicNet

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser(description="Unified Evaluation Script (RT-X Net Compliant)")

# Model and Data Paths
parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint (.pth)")
parser.add_argument("--root_rgb", type=str, default='/medias/db/ImagingSecurity_misc/melcarne/sota_lie/dataset/test_data/LLVIP', help="Path to input RGB images")
parser.add_argument("--root_thermal", type=str, default='/medias/db/ImagingSecurity_misc/melcarne/sota_lie/dataset/thermal/LLVIP', help="Path to thermal images")
parser.add_argument("--root_target", type=str, default='/medias/db/ImagingSecurity_misc/melcarne/sota_lie/dataset/target/LLVIP', help="Path to ground truth images")

# Settings
parser.add_argument("--device", type=int, default=3, help="GPU Device ID")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

# Model Configuration Flags
parser.add_argument("--has_thermal", action="store_true", help="Set if the model expects thermal input")
parser.add_argument("--self_ensemble", action="store_true", help="Enable Geometric Self-Ensemble (8x inference)")

args = parser.parse_args()

# Device Setup
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on device: {device}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f" Seed locked to {seed}. Deterministic mode ON.")

seed_everything(42)

# -------------------------
# Dataset & DataLoader
# -------------------------
print("Loading LLVIP dataset...")
val_dataset = LLVIP_Dataset(
    root_rgb=args.root_rgb,
    root_thermal=args.root_thermal,
    root_target=args.root_target,
    is_test=True
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers,
    worker_init_fn=seed_worker
)

# -------------------------
# Output Directory
# -------------------------
run_tag = "attention"
if args.self_ensemble:
    run_tag += "_ensemble"

out_dir = project_root / f"output_models/eval_LLVIP/{run_tag}"
pred_dir = out_dir / "predictions"

pred_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# LPIPS
# -------------------------
lpips_fn = None
if have_lpips:
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

# -------------------------
# Load Model
# -------------------------
print("Architecture: Attention + Refinement")
model = ThermalIntrinsicNet(base_channels=16).to(device)
print("initializing ThermalIntrinsicNet_attention model")

ckpt = torch.load(args.ckpt, map_location="cpu")

def clean_state_dict(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}

if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
    state_dict = clean_state_dict(ckpt["ema_state_dict"])
else:
    state_dict = clean_state_dict(ckpt["model_state_dict"])

model.load_state_dict(state_dict, strict=False)
model.eval()

# -------------------------
# Save Prediction Helper
# -------------------------
def save_prediction(pred_lin_or_srgb, filename, is_srgb=False):
    """
    Saves the predicted image.
    If is_srgb=True, pred is assumed to be numpy HWC sRGB [0,1].
    If is_srgb=False, pred is assumed to be Tensor CHW linear [0,1].
    """
    if not is_srgb:
        # Convert tensor linear to numpy sRGB
        pred_lin = pred_lin_or_srgb.detach().cpu().numpy().transpose(1, 2, 0)
        pred_srgb = np.power(np.clip(pred_lin, 0, 1), 1.0 / 2.2)
    else:
        # Already numpy sRGB
        pred_srgb = pred_lin_or_srgb

    pred_uint8 = (pred_srgb * 255).astype(np.uint8)
    Image.fromarray(pred_uint8).save(pred_dir / filename)

# -------------------------
# Helper Function: Self Ensemble
# -------------------------
def run_self_ensemble(model, rgb_in, therm_in, has_thermal):
    outputs = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                
                # 1. Transform Inputs
                t_rgb = rgb_in.clone()
                t_therm = therm_in.clone() if has_thermal else None

                if hflip:
                    t_rgb = torch.flip(t_rgb, (-1,)) 
                    if has_thermal: t_therm = torch.flip(t_therm, (-1,))
                if vflip:
                    t_rgb = torch.flip(t_rgb, (-2,))
                    if has_thermal: t_therm = torch.flip(t_therm, (-2,))
                if rot:
                    t_rgb = torch.rot90(t_rgb, dims=(-2, -1))
                    if has_thermal: t_therm = torch.rot90(t_therm, dims=(-2, -1))

                # 2. Inference
                if has_thermal:
                    _, _, _, pred_lin, _, att_maps = model(t_rgb, t_therm)
                else:
                    _, _, _, pred_lin, _, att_maps = model(t_rgb, x_therm=None)

                # 3. Inverse Transform Output
                if rot:
                    pred_lin = torch.rot90(pred_lin, dims=(-2, -1), k=3)
                if vflip:
                    pred_lin = torch.flip(pred_lin, (-2,))
                if hflip:
                    pred_lin = torch.flip(pred_lin, (-1,))
                
                outputs.append(pred_lin)

    return torch.stack(outputs).mean(dim=0)

# -------------------------
# Inference Loop
# -------------------------
psnr_vals, ssim_vals, lpips_vals, de_vals = [], [], [], []
csv_rows = []

print(f"Starting inference... (Self Ensemble: {args.self_ensemble})")

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(val_loader)):

        rgb_in = batch["input_rgb"].to(device)
        therm_in = batch["input_therm"].to(device)
        target_srgb = batch["target_rgb"].to(device)
        filenames = batch["filename"]

        # Inference
        if args.self_ensemble:
            pred_Final_lin = run_self_ensemble(model, rgb_in, therm_in, args.has_thermal)
        else:
            if args.has_thermal:
                _, _, _, pred_Final_lin, _, att_maps = model(rgb_in, therm_in)
            else:
                _, _, _, pred_Final_lin, _, att_maps = model(rgb_in, x_therm=None)

        pred_lin_np = pred_Final_lin.cpu().numpy().transpose(0, 2, 3, 1)
        pred_srgb_np = np.power(np.clip(pred_lin_np, 0, 1), 1.0 / 2.2)
        tgt_srgb_np = target_srgb.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(rgb_in.size(0)):

            p_img = pred_srgb_np[i] # H, W, C [0,1]
            t_img = tgt_srgb_np[i]  # H, W, C [0,1]

            p_u8 = img_as_ubyte(p_img)
            t_u8 = img_as_ubyte(t_img)

            psnr_vals.append(psnr(t_u8, p_u8, data_range=255))
            ssim_vals.append(ssim(t_u8, p_u8, channel_axis=2, data_range=255))

            try:
                de = deltaE_ciede2000(
                    skcolor.rgb2lab(t_img),
                    skcolor.rgb2lab(p_img)
                ).mean()
            except:
                de = np.nan
            de_vals.append(de)

            lp = np.nan
            if have_lpips:
                # Re-convert numpy back to torch for LPIPS to ensure we use the RECTIFIED p_img
                p_t = torch.from_numpy(p_img).permute(2,0,1).unsqueeze(0).float().to(device)
                t_t = torch.from_numpy(t_img).permute(2,0,1).unsqueeze(0).float().to(device)
                
                # LPIPS expects input in [-1, 1]
                lp = lpips_fn(p_t * 2 - 1, t_t * 2 - 1).item()
                lpips_vals.append(lp)

            # Save rectified image
            save_prediction(p_img, filenames[i], is_srgb=True)

            csv_rows.append({
                "filename": filenames[i],
                "psnr": psnr_vals[-1],
                "ssim": ssim_vals[-1],
                "deltaE": de,
                "lpips": lp
            })

# -------------------------
# Save Metrics
# -------------------------
out_dir.mkdir(exist_ok=True)
pd.DataFrame(csv_rows).to_csv(out_dir / "metrics.csv", index=False)

summary = {
    "psnr_avg": float(np.mean(psnr_vals)),
    "ssim_avg": float(np.mean(ssim_vals)),
    "deltaE_avg": float(np.nanmean(de_vals)),
    "lpips_avg": float(np.mean(lpips_vals)) if have_lpips else "N/A"
}

with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("Evaluation complete.")
print(f"Results saved to: {out_dir}")