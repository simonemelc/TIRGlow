import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms

class V_TIEE_Dataset(Dataset):
    """
    Dataset loader for the reorganized V-TIEE dataset.
    It expects the following structure: 
    - root_rgb (Input): .../RGB/Input/N_zzzzz.png
    - root_target: .../RGB/Target/N_zzzzz.png
    - root_thermal: .../Thermal/N.png

    The N_zzzzz.png file names are used to look up the corresponding N.png thermal file.
    """
    def __init__(self, root_rgb, root_thermal, root_target, is_test=False):

        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.root_target = root_target
        self.is_test = is_test
        
        # Determine if low-light simulation should be applied (based on original LLVIP logic)
        self.same_root = (self.root_rgb == self.root_target)
        
        if self.same_root:
            print(f"[INFO] Input and Target paths are the same ('{root_rgb}').")

        # 1. List RGB Input files (N_zzzzz.png)
        rgb_files = sorted([
            f for f in os.listdir(root_rgb) 
            if f.lower().endswith(('.png'))
        ])
        
        self.file_list = []

        # 2. Verify the existence of the triplet (Input, Thermal, Target) for each RGB file
        for rgb_filename in rgb_files:
            # Extract the N ID (e.g., from "1_8340.png" extract "1")
            try:
                # The file name format is N_zzzzz.png
                n_id = rgb_filename.split('_')[0]
                thermal_filename = f"{n_id}.png"
            except IndexError:
                # Skip files with unexpected naming format
                print(f"[Warning] Unexpected RGB file name format: {rgb_filename}. Skipped.")
                continue

            path_in = os.path.join(self.root_rgb, rgb_filename)
            path_th = os.path.join(self.root_thermal, thermal_filename)
            path_gt = os.path.join(self.root_target, rgb_filename) # Target has the same name as Input

            # Verify the existence of the triplet
            if os.path.exists(path_in) and os.path.exists(path_th) and os.path.exists(path_gt):
                # Store the RGB file name (for Input and Target) and the Thermal name
                self.file_list.append({
                    'rgb_name': rgb_filename,
                    'thermal_name': thermal_filename
                })
            else:
                if not os.path.exists(path_th):
                    print(f"[Warning] Thermal ({thermal_filename}) missing for triplet {rgb_filename}")
                if not os.path.exists(path_gt):
                    print(f"[Warning] Target ({rgb_filename}) missing for triplet {rgb_filename}")

        print(f"Dataset V-TIEE loaded. {len(self.file_list)} triplets (RGB-Thermal-Target).")



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_info = self.file_list[idx]
        rgb_filename = file_info['rgb_name']
        thermal_filename = file_info['thermal_name']
        
        # --- PATHS ---
        rgb_path = os.path.join(self.root_rgb, rgb_filename)
        therm_path = os.path.join(self.root_thermal, thermal_filename)
        target_path = os.path.join(self.root_target, rgb_filename)

        # --- 1. LOAD INPUT RGB (Visible) ---
        # Read image (BGR by default), convert to RGB, Normalize [0,1]
        input_rgb = cv2.imread(rgb_path).astype(np.float32) / 255.0
        input_rgb = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2RGB)

        # --- 2. LOAD THERMAL (Infrared) ---
        # Thermal is single-channel (grayscale)
        input_therm_raw = cv2.imread(therm_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Normalize Thermal [0,1] (Min-Max per image)
        t_min, t_max = input_therm_raw.min(), input_therm_raw.max()
        if t_max - t_min > 0:
            input_therm = (input_therm_raw - t_min) / (t_max - t_min)
        else:
            # Handle flat image case (t_max == t_min)
            input_therm = input_therm_raw / 255.0

        # --- 3. LOAD TARGET RGB (Ground Truth) ---
        # Read image (BGR by default), convert to RGB, Normalize [0,1] 
        target_rgb = cv2.imread(target_path).astype(np.float32) / 255.0
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)

        # --- TO TENSOR (HWC -> CHW) ---
        input_rgb = torch.from_numpy(input_rgb.transpose(2, 0, 1))      # 3, H, W
        input_therm = torch.from_numpy(input_therm[np.newaxis, ...])    # 1, H, W (add channel dimension)
        target_rgb = torch.from_numpy(target_rgb.transpose(2, 0, 1))    # 3, H, W

        # --- RESIZE ---
        if not self.is_test:
            # TRAINING: 384x384 
            resize_t = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC)
            input_rgb = resize_t(input_rgb)
            input_therm = resize_t(input_therm)
            target_rgb = resize_t(target_rgb)
        else:

            _, h, w = input_rgb.shape
            
            new_h = (h // 16) * 16
            new_w = (w // 16) * 16
            
            if new_h != h or new_w != w:
                resize_t = transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
                input_rgb = resize_t(input_rgb)
                input_therm = resize_t(input_therm)
                target_rgb = resize_t(target_rgb)

        return {
            'input_rgb': input_rgb,       
            'input_therm': input_therm,   
            'target_rgb': target_rgb,   
            'filename': file_info
        }