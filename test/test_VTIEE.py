import os
import sys
import argparse
from pathlib import Path
import math
import random
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color as skcolor
from skimage.color import deltaE_ciede2000
from skimage import img_as_ubyte 

from torch.utils.data import DataLoader

# Attempt lpips
have_lpips = False
try:
    import lpips
    have_lpips = True
except Exception:
    pass

# -------------------------
# Path setup
# -------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "dataset"))
sys.path.append(str(project_root / "model"))

from dataset.VTIEE_dataset import V_TIEE_Dataset
from model.ThermalIntrinsicNet_attention import ThermalIntrinsicNet

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--root_rgb", type=str, default='/medias/db/ImagingSecurity_misc/melcarne/sota_lie/dataset/test_data/VTIEE')
parser.add_argument("--root_thermal", type=str, default='/medias/db/ImagingSecurity_misc/melcarne/sota_lie/dataset/thermal/VTIEE')
parser.add_argument("--root_target", type=str, default='/medias/db/ImagingSecurity_misc/melcarne/sota_lie/dataset/target/VTIEE')
parser.add_argument("--has_thermal", action="store_true")

# --- NEW ARGUMENTS ---
parser.add_argument("--self_ensemble", action="store_true", help="Enable Geometric Self-Ensemble (8x inference)")

args = parser.parse_args()


device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# -------------------------
# Dataset & Loader
# -------------------------
print("Loading dataset...")

v_tiee_dataset = V_TIEE_Dataset(
    root_rgb=args.root_rgb,
    root_thermal=args.root_thermal,
    root_target=args.root_target,
    is_test=True
)

val_loader = DataLoader(
    v_tiee_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
)

# -------------------------
# Output directories
# -------------------------
run_tag = "attention"
if args.self_ensemble:
    run_tag += "_ensemble"

out_dir = project_root / f"output_models/eval_VTIEE/{run_tag}"
pred_dir = out_dir / "predictions"

pred_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Model loading
# -------------------------
print(f"Loading checkpoint from {args.ckpt}...")
model = ThermalIntrinsicNet(base_channels=16).to(device) #base_channels=16
print("initializing ThermalIntrinsicNet_attention model")

ckpt = torch.load(args.ckpt, map_location="cpu")

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
    loaded_weights = clean_state_dict(ckpt["ema_state_dict"])
else:
    loaded_weights = clean_state_dict(ckpt["model_state_dict"])

model.load_state_dict(loaded_weights, strict=False)
model.eval()

# -------------------------
# LPIPS
# -------------------------
lpips_fn = None
if have_lpips:
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    print("LPIPS initialized.")

# -------------------------
# Helper Function: Self Ensemble
# -------------------------
def run_self_ensemble(model, rgb_in, therm_in, has_thermal):
    """
    Performs Geometric Self-Ensemble (8 variations: Flip H/V and Rot90 combinations).
    Returns the averaged output tensor.
    """
    outputs = []
    
    # Iterate over 2x2x2 = 8 geometric transformations
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                
                # 1. Transform Inputs
                t_rgb = rgb_in.clone()
                t_therm = therm_in.clone() if has_thermal else None

                if hflip:
                    t_rgb = torch.flip(t_rgb, (-1,)) # Flip width
                    if has_thermal: t_therm = torch.flip(t_therm, (-1,))
                if vflip:
                    t_rgb = torch.flip(t_rgb, (-2,)) # Flip height
                    if has_thermal: t_therm = torch.flip(t_therm, (-2,))
                if rot:
                    t_rgb = torch.rot90(t_rgb, dims=(-2, -1)) # Rotate 90
                    if has_thermal: t_therm = torch.rot90(t_therm, dims=(-2, -1))

                # 2. Inference
                if has_thermal:
                    _, _, _, pred_lin, _, att_maps = model(t_rgb, t_therm)
                else:
                    _, _, _, pred_lin, _, att_maps = model(t_rgb, x_therm=None)

                # 3. Inverse Transform Output
                if rot:
                    pred_lin = torch.rot90(pred_lin, dims=(-2, -1), k=3) # Rotate back 270
                if vflip:
                    pred_lin = torch.flip(pred_lin, (-2,))
                if hflip:
                    pred_lin = torch.flip(pred_lin, (-1,))
                
                outputs.append(pred_lin)

    # 4. Average results
    return torch.stack(outputs).mean(dim=0)

# -------------------------
# Save prediction helper
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


def save_attention_maps(attention_list, img_name, output_folder, original_size):
    """
    attention_list: [B, 1, H, W] 
    img_name: ID 
    original_size:  (H, W) 
    """
    import matplotlib.cm as cm
    

    map_dir = os.path.join(output_folder, "attention_maps")
    os.makedirs(map_dir, exist_ok=True)

    for idx, att_tensor in enumerate(attention_list):

        att_up = F.interpolate(att_tensor, size=original_size, mode='bilinear', align_corners=False)
        

        att_map = att_up[0, 0].detach().cpu().numpy()
        

        att_map = np.clip(att_map, 0, 1)


        heatmap = matplotlib.colormaps['inferno'](att_map)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # 5. Salva
        im = Image.fromarray(heatmap)
        im.save(os.path.join(map_dir, f"{img_name}_level_{idx}.png"))

# -------------------------
# Inference loop
# -------------------------
psnr_vals, ssim_vals, lpips_vals, de_vals = [], [], [], []
csv_rows = []

print("\nStarting Inference Loop...")
print(f"Self Ensemble: {args.self_ensemble}")

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(val_loader)):

        rgb_in = batch["input_rgb"].to(device)
        therm_in = batch["input_therm"].to(device)
        target_srgb = batch["target_rgb"].to(device)

        # IMPORTANT:
        # If your dataset uses a different key name, change ONLY this line
        filenames = batch["filename"]

        # --- MODIFIED INFERENCE BLOCK ---
        if args.self_ensemble:
            pred_Final_lin = run_self_ensemble(model, rgb_in, therm_in, args.has_thermal)
            # Note: att_maps are not easily averaged in self-ensemble, so we might skip saving them or use the last one
            att_maps = [] 
        else:
            if args.has_thermal:
                _, _, _, pred_Final_lin, _, att_maps = model(rgb_in, therm_in)
            else:
                _, _, _, pred_Final_lin, _, att_maps = model(rgb_in, x_therm=None)
        # -------------------------------

        pred_lin_np = pred_Final_lin.cpu().numpy().transpose(0, 2, 3, 1)
        pred_srgb_np = np.power(np.clip(pred_lin_np, 0, 1), 1.0 / 2.2)
        tgt_srgb_np = target_srgb.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(rgb_in.size(0)):

            p_img = pred_srgb_np[i]
            t_img = tgt_srgb_np[i]

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

            lp_val = np.nan
            if have_lpips:
                # p_img is (numpy), so we must convert it back to tensor for LPIPS
                p_t = torch.from_numpy(p_img).permute(2,0,1).unsqueeze(0).float().to(device)
                t_t = torch.from_numpy(t_img).permute(2,0,1).unsqueeze(0).float().to(device)
                
                # LPIPS expects inputs in [-1, 1]
                lp_val = lpips_fn(p_t * 2 - 1, t_t * 2 - 1).item()
                lpips_vals.append(lp_val)

            # Save prediction
            save_prediction(p_img, filenames["rgb_name"][i], is_srgb=True)

            current_count = len(csv_rows)
            
            # Save attention maps only if available (self-ensemble might not produce them easily)
            if att_maps:
                orig_H, orig_W = rgb_in.shape[2], rgb_in.shape[3]
                current_maps = [m[i:i+1] for m in att_maps]
                save_attention_maps(current_maps, f"sample_{current_count:04d}", out_dir, (orig_H, orig_W))


            csv_rows.append({
                "filename": filenames["rgb_name"][i],
                "psnr": psnr_vals[-1],
                "ssim": ssim_vals[-1],
                "deltaE": de,
                "lpips": lp_val
            })

# -------------------------
# Save metrics
# -------------------------
out_dir.mkdir(exist_ok=True)

metrics_df = pd.DataFrame(csv_rows)
metrics_df.to_csv(out_dir / "metrics.csv", index=False)

summary = {
    "psnr_avg": float(np.mean(psnr_vals)),
    "ssim_avg": float(np.mean(ssim_vals)),
    "deltaE_avg": float(np.nanmean(de_vals)),
    "lpips_avg": float(np.mean(lpips_vals)) if have_lpips else "N/A"
}

with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 30)
print("      TEST COMPLETE")
print("=" * 30)
print(f"Metrics saved to: {out_dir / 'metrics.csv'}")
print(f"PSNR (uint8):   {summary['psnr_avg']:.2f} dB")
print(f"SSIM (uint8):   {summary['ssim_avg']:.4f}")
print(f"LPIPS (VGG):    {summary['lpips_avg']}")
print(f"DeltaE (CIEDE): {summary['deltaE_avg']:.4f}")
print("=" * 30)