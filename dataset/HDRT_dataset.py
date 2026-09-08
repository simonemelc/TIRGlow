import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import re
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class HDRT_Dataset(Dataset):
    def __init__(self, root_rgb, root_thermal, root_targets_R, root_targets_S, augment=False):
        """
        root_rgb: external RGB root, with normal (odd) and low (even) JPG files
        root_thermal:folder with all the thermal .TIFF files
        root_targets_R: folder with GT Reflectance from Teacher (PNG 16-bit)
        root_targets_S: folder with GT Reflectance from Teacher (NPY Float16)
        augment: Data augmentation. Defualt is false
        """
        self.root_rgb = root_rgb
        self.root_thermal = root_thermal
        self.root_targets_R = root_targets_R
        self.root_targets_S = root_targets_S
        self.augment = augment
        self.fixed_gain = None


        def extract_num(fname):
            m = re.search(r'(\d+)', fname)
            return int(m.group(1)) if m else None

        all_files = sorted(
            [f for f in os.listdir(root_rgb) if f.endswith('.JPG')],
            key=lambda x: extract_num(x)
        )

        # Normal/low pairs based on consecutive numbers
        self.normal_files = []
        self.low_files = []

        i = 0
        while i < len(all_files) - 1:
            f1 = all_files[i]
            f2 = all_files[i+1]

            n1 = extract_num(f1)
            n2 = extract_num(f2)

            if n2 == n1 + 1:
                # NORMAL = f1, LOW = f2
                self.normal_files.append(f1) 
                self.low_files.append(f1) #f2, experiment with normal light also for input and then low light sim.
                i += 2
            else:
                i += 1


        self.resize = transforms.Resize((384, 384))
        self.to_tensor = transforms.ToTensor() 

    def __len__(self):
        return len(self.low_files)


    def set_fixed_gain(self, gain):
        """
        Fix gain value for testint.
        dataset.set_fixed_gain(0.005) 'Hard'.
        None standard mode.
        """
        self.fixed_gain = gain

    def physics_based_low_light_simulation(self, img, is_training=True):
            """
            Calibrated Adaptive Low-Light Simulation.
            Produces visually dark images (sRGB) with realistic, subtle noise.
            """
            
            # 1. Analyze Input Brightness
            input_mean = np.mean(img) + 1e-6
            
            # 2. Define "Target Darkness" (Linear Domain)
            # NOTE: Because of Gamma Correction (1/2.2), Linear values must be TINY to look dark.
            # Linear 0.001 -> sRGB ~0.04 (Pixel val ~11/255) -> Dark
            # Linear 0.0001 -> sRGB ~0.015 (Pixel val ~4/255) -> Pitch Black
            
            target_mean = None
            gain = None

            if is_training:
                if np.random.rand() < 0.60:
                    # HARD MODE: Pitch Black / Deep Shadows
                    # Target Linear: 0.0005 - 0.001
                    target_mean = np.random.uniform(0.0005, 0.001)
                else:
                    # MEDIUM MODE: Dim Street Light
                    # Target Linear: 0.001 - 0.005
                    target_mean = np.random.uniform(0.001, 0.005)
            else:
                # VALIDATION: Stable Hard Mode
                if hasattr(self, 'fixed_gain') and self.fixed_gain is not None:
                    gain = self.fixed_gain
                    target_mean = None
                else:
                    # Validation Target: Very Dark but structure exists
                    target_mean = 0.001 #0.0005

            # 3. Calculate Adaptive Gain
            if target_mean is not None:
                gain = target_mean / input_mean
                # Allow even smaller gains for extremely bright inputs
                gain = np.clip(gain, 1e-5, 1.0)

            # --- PHYSICS PIPELINE ---

            # 4. INVERSE ISP
            img_linear = np.power(img, 2.2)
            
            # 5. DEGRADATION
            img_dark_linear = img_linear * gain
            
            # 6. NOISE INJECTION
            
            # Check actual darkness achieved
            current_darkness = np.mean(img_dark_linear)
            
            if current_darkness < 0.0005: 
                # Extreme Low Light
                shot_noise_level = np.random.uniform(0.005, 0.01) 
                read_noise_level = np.random.uniform(0.0005, 0.0001) 
            else:
                # Standard Low Light
                shot_noise_level = np.random.uniform(0.001, 0.005) 
                read_noise_level = np.random.uniform(0.00005, 0.0002)
                
            # A. Shot Noise
            noise_source = np.random.standard_normal(img_dark_linear.shape)
            shot_noise = noise_source * np.sqrt(img_dark_linear + 1e-6) * shot_noise_level

            # B. Read Noise
            noise_source_read = np.random.standard_normal(img_dark_linear.shape)
            read_noise = noise_source_read * read_noise_level
            
            noisy_linear = img_dark_linear + shot_noise + read_noise
            noisy_linear = np.clip(noisy_linear, 0.0, 1.0)
            
            # 7. FORWARD ISP
            noisy_srgb = np.power(noisy_linear + 1e-6, 1/2.2)
            
            # 8. QUANTIZATION 
            noisy_quantized = np.round(noisy_srgb * 255.0) / 255.0
            
            return noisy_quantized.astype(np.float32)

   

    def __getitem__(self, idx):
        
        low_light_filename = self.low_files[idx]
        normal_filename = self.normal_files[idx]

        match = re.search(r'(\d+)', normal_filename)
        normal_id = int(match.group(1))

        prefix = normal_filename.split(f"{normal_id:05d}")[0]
        normal_filename_base = f"{prefix}{normal_id:05d}"

        # --- INPUT (Student) ---
        
        # A) RGB Low Light
        rgb_low_path = os.path.join(self.root_rgb, low_light_filename)

        # Load, BGR->RGB, [0,1] normalization
        input_rgb_low = cv2.imread(rgb_low_path).astype(np.float32) / 255.0
        input_rgb_low = cv2.cvtColor(input_rgb_low, cv2.COLOR_BGR2RGB)

        input_rgb_low_synth = self.physics_based_low_light_simulation(input_rgb_low, is_training=self.augment)

        # B) Thermal
        # Same ID of Normal Light file: "RGB00261.JPG" <--> "T00261.tiff" 

        therm_filename = f"T{normal_id:05d}.tiff"
        therm_path = os.path.join(self.root_thermal, therm_filename)
        
        # 16-bit (Flag -1)
        if os.path.exists(therm_path):
            input_therm_raw = cv2.imread(therm_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            if input_therm_raw.ndim == 3:
                input_therm_raw = input_therm_raw[:, :, 0]

            # Min-Max norm
            t_min, t_max = input_therm_raw.min(), input_therm_raw.max()
            if t_max - t_min > 0:
                input_therm = (input_therm_raw - t_min) / (t_max - t_min)
            else:
                input_therm = np.zeros_like(input_therm_raw)
        else:
            # Fallback
            print(f"Warning: Missing thermal {therm_path}")
            input_therm = np.zeros((input_rgb_low.shape[0], input_rgb_low.shape[1]), dtype=np.float32)

        # --- TARGETS (Teacher) ---
        
        # A) Reflectance Target (PNG 16-bit)
        # "RGB00261.JPG" <--> "RGB00261_alb.png"
        alb_target_name = f"{normal_filename_base}_alb.png"
        alb_path = os.path.join(self.root_targets_R, alb_target_name)
        
        target_R = cv2.imread(alb_path, -1) # uint16
        if target_R is None:
            raise FileNotFoundError(f"Target R not found: {alb_path}")
        
        target_R = target_R.astype(np.float32) / 65535.0  #  [0-1]
        target_R = cv2.cvtColor(target_R, cv2.COLOR_BGR2RGB) # BGR to RGB

        # B) Shading Target (NPY Float16)
        # "RGB00261.JPG" <--> "RGB00261_shd.npy"
        shd_target_name = f"{normal_filename_base}_shd.npy"
        shd_path = os.path.join(self.root_targets_S, shd_target_name)
        
        target_S = np.load(shd_path).astype(np.float32) # float raw (HDR)
        
        # Grayscale shading HxW -> HxWx1
        if target_S.ndim == 2:
            target_S = target_S[:, :, np.newaxis]


        # C) RGB Target (JPG)
        rgb_normal_path = os.path.join(self.root_rgb, normal_filename_base + ".JPG")
        # Load, BGR->RGB, [0,1] normalization
        target_rgb = cv2.imread(rgb_normal_path).astype(np.float32) / 255.0
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)

        # --- TO TENSOR (HWC -> CHW) ---
        input_rgb_low = torch.from_numpy(input_rgb_low.transpose(2, 0, 1))               # 3, H, W
        input_rgb_low_synth = torch.from_numpy(input_rgb_low_synth.transpose(2, 0, 1))   # 3, H, W
        input_therm = torch.from_numpy(input_therm[np.newaxis, ...])                     # 1, H, W
        target_R = torch.from_numpy(target_R.transpose(2, 0, 1))                         # 3, H, W
        target_S = torch.from_numpy(target_S.transpose(2, 0, 1))                         # 1, H, W
        target_rgb = torch.from_numpy(target_rgb.transpose(2, 0, 1))                     # 3, H, W

        # --- RESIZE A 384×384 ---

        input_rgb_low = self.resize(input_rgb_low)
        input_rgb_low_synth = self.resize(input_rgb_low_synth)
        input_therm   = self.resize(input_therm)
        target_R      = self.resize(target_R)
        target_S      = self.resize(target_S)
        target_rgb    = self.resize(target_rgb)

        # --- DATA AUGMENTATION ---
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                input_rgb_low = TF.hflip(input_rgb_low)
                input_rgb_low_synth = TF.hflip(input_rgb_low_synth)
                input_therm   = TF.hflip(input_therm)
                target_R      = TF.hflip(target_R)
                target_S      = TF.hflip(target_S)
                target_rgb    = TF.hflip(target_rgb)

            # Random Vertical Flip
            if random.random() > 0.5:
                input_rgb_low = TF.vflip(input_rgb_low)
                input_rgb_low_synth = TF.vflip(input_rgb_low_synth)
                input_therm   = TF.vflip(input_therm)
                target_R      = TF.vflip(target_R)
                target_S      = TF.vflip(target_S)
                target_rgb    = TF.vflip(target_rgb)
                

        return {
            'input_rgb_low': input_rgb_low,
            'input_rgb_low_synth': input_rgb_low_synth,
            'input_therm': input_therm,
            'target_R': target_R,
            'target_S': target_S,
            'target_rgb' : target_rgb,
            'filename': low_light_filename
        }