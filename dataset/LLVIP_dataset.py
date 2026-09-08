import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import torchvision.transforms.functional as TF

class LLVIP_Dataset(Dataset):
    def __init__(self, root_rgb, root_thermal, root_target, is_test=False):

        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.root_target = root_target
        self.is_test = is_test

        self.simulate_low_light = (self.root_rgb == self.root_target)
        
        if self.simulate_low_light:
            print(f"[INFO] Input and Target paths are the same ('{root_rgb}'). Physics-Based Low Light Simulation.")


        raw_files = sorted([
            f for f in os.listdir(root_rgb) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        self.file_list = []

        for f in raw_files:
            path_in = os.path.join(self.root_rgb, f)
            path_th = os.path.join(self.root_thermal, f)
            path_gt = os.path.join(self.root_target, f)
            
   
            if os.path.exists(path_in) and os.path.exists(path_th) and os.path.exists(path_gt):
                self.file_list.append(f)
            else:
                if not os.path.exists(path_th):
                    print(f"[Warning] Thermal missing for: {f}")
                if not os.path.exists(path_gt):
                    print(f"[Warning] Target missing for: {f}")

        print(f"Dataset LLVIP loaded. {len(self.file_list)} triplets (RGB-Thermal-Target). Test Mode: {self.is_test}")

    def physics_based_low_light_simulation(self, img):
        """
        Robust Low-light simulation with Poisson-Gaussian Noise.
        Args:
            img (np.array): Normal Light Input [H, W, C] in range [0, 1] float32
        """
        # 1. INVERSE ISP: sRGB -> Linear Domain
        img_linear = np.power(img, 2.2)
        
        # 2. PHYSICAL DEGRADATION (Linear Domain)
        if np.random.rand() < 0.60:
            gain = np.random.uniform(0.005, 0.05) 
        else:
            gain = np.random.uniform(0.05, 0.7) 
        img_dark_linear = img_linear * gain
        
        # B. Noise Injection (Poisson-Gaussian Model)
        if gain < 0.05:
            shot_noise_level = np.random.uniform(0.005, 0.02)
            read_noise_level = np.random.uniform(0.0005, 0.001)
        else:
            shot_noise_level = np.random.uniform(0.001, 0.005)
            read_noise_level = np.random.uniform(0.0001, 0.0005)
        
        # Shot Noise
        noise_source = np.random.standard_normal(img_dark_linear.shape)
        shot_noise = noise_source * np.sqrt(img_dark_linear + 1e-6) * shot_noise_level

        # Read Noise
        noise_source_read = np.random.standard_normal(img_dark_linear.shape)
        read_noise = noise_source_read * read_noise_level
        
        noisy_linear = img_dark_linear + shot_noise + read_noise
        noisy_linear = np.clip(noisy_linear, 0.0, 1.0)
        
        # 3. FORWARD ISP: Linear -> sRGB
        noisy_srgb = np.power(noisy_linear + 1e-6, 1/2.2)
        
        # 4. QUANTIZATION 
        noisy_quantized = np.round(noisy_srgb * 255.0) / 255.0
        
        return noisy_quantized.astype(np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # --- PATHS ---
        rgb_path = os.path.join(self.root_rgb, filename)
        therm_path = os.path.join(self.root_thermal, filename)
        target_path = os.path.join(self.root_target, filename)

        # --- 1. LOAD INPUT RGB (Visible) ---
        input_rgb = cv2.imread(rgb_path).astype(np.float32) / 255.0
        input_rgb = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2RGB)

        # --- APPLICAZIONE SIMULAZIONE (Se attiva) ---
        if self.simulate_low_light:
            input_rgb = self.physics_based_low_light_simulation(input_rgb)

        # --- 2. LOAD THERMAL (Infrared) ---
        input_therm_raw = cv2.imread(therm_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Normalize Thermal [0,1]
        t_min, t_max = input_therm_raw.min(), input_therm_raw.max()
        if t_max - t_min > 0:
            input_therm = (input_therm_raw - t_min) / (t_max - t_min)
        else:
            input_therm = input_therm_raw / 255.0

        # --- 3. LOAD TARGET RGB (Ground Truth) ---
        target_rgb = cv2.imread(target_path).astype(np.float32) / 255.0
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)

        # --- TO TENSOR (HWC -> CHW) ---
        input_rgb = torch.from_numpy(input_rgb.transpose(2, 0, 1))      # 3, H, W
        input_therm = torch.from_numpy(input_therm[np.newaxis, ...])    # 1, H, W
        target_rgb = torch.from_numpy(target_rgb.transpose(2, 0, 1))    # 3, H, W

        # --- RESIZE LOGIC ---
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
            'filename': filename
        }