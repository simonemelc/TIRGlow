import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard Conv Block: Conv -> Norm -> Activation
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        return self.relu(out + residual)

class AttentionGate(nn.Module):
    """
    Inspired by Han et al. NeurImg-HDR+ Attention Gate. Here we apply it to every resolution level
    Filters the features from the Encoder (x) using the gating signal from the Decoder (g).
    Logic: Focus on informative regions, suppress noise/bad exposure.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        scale = self.psi(psi) # [B, 1, H, W]
        return x * scale, scale

class IdentityGate(nn.Module):
    """
    Dummy Gate for Ablation Study (No Attention).
    It accepts the same arguments but does NOTHING.
    Returns:
        - The original 'x' (Skip connection unmodified)
        - A mask of all 1s (so visualization shows 'everything passed')
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # We don't need layers here

    def forward(self, g, x):
        # Create a dummy mask of 1s with shape [B, 1, H, W]
        # x is [B, C, H, W], we want mask to be 1 channel
        mask = torch.ones_like(x[:, 0:1, :, :]) 
        
        # Return x unmodified (Identity) + dummy mask
        return x, mask

class ResidualRefinementBlock(nn.Module):
    def __init__(self, in_ch, mid_ch=32, num_blocks=4):
        super().__init__()
        self.entry = ConvBlock(in_ch, mid_ch)
        res_layers = []
        for _ in range(num_blocks):
            res_layers.append(ResBlock(mid_ch))
        self.body = nn.Sequential(*res_layers)
        self.exit = nn.Conv2d(mid_ch, 3, kernel_size=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        return self.exit(x)

class ThermalIntrinsicNet(nn.Module):
    def __init__(self, base_channels=16, gating_config="all"):
        """
        gating_config (str): 
            - "all": Attention on ALL levels (Proposed Method)
            - "none": Attention on NO levels (Full Ablation)
            - "shallow_only": Attention only on Level 2 (Han et al. style)
        """
        super().__init__()
        
        # --- 1. DUAL ENCODER ---
        self.rgb_enc1 = ConvBlock(3, base_channels)
        self.rgb_enc2 = ConvBlock(base_channels, base_channels*2, stride=2)
        self.rgb_enc3 = ConvBlock(base_channels*2, base_channels*4, stride=2)
        self.rgb_enc4 = ConvBlock(base_channels*4, base_channels*8, stride=2)

        self.thm_enc1 = ConvBlock(1, base_channels)
        self.thm_enc2 = ConvBlock(base_channels, base_channels*2, stride=2)
        self.thm_enc3 = ConvBlock(base_channels*2, base_channels*4, stride=2)
        self.thm_enc4 = ConvBlock(base_channels*4, base_channels*8, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels*8 * 2, base_channels*16)

        # --- 2. DECODER WITH CONFIGURABLE GATES ---
        
        # Determine which gates are active based on config
        # default: [Level4(Deep), Level3(Mid), Level2(Shallow)]
        if gating_config == "all":
            active_gates = [True, True, True]  # Full
        elif gating_config == "none":
            active_gates = [False, False, False] # Full Ablation
        elif gating_config == "shallow_only":
            active_gates = [False, False, True] # Han et al. style (Only highest res)
        else:
            raise ValueError(f"Unknown gating_config: {gating_config}")

        print(f">> Model Initialized with Gating Config: {gating_config} -> Active Gates [Deep, Mid, Shallow]: {active_gates}")

        # Helper to select class
        def get_gate(is_active):
            return AttentionGate if is_active else IdentityGate

        # === Block 4 (Deep / Low Res) ===
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse_skip4 = nn.Conv2d(base_channels*4 * 2, base_channels*4, kernel_size=1) 
        Gate4 = get_gate(active_gates[0])
        self.att4 = Gate4(F_g=base_channels*16, F_l=base_channels*4, F_int=base_channels*4)
        self.dec4 = ConvBlock(base_channels*16 + base_channels*4, base_channels*8)

        # === Block 3 (Mid Res) ===
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse_skip3 = nn.Conv2d(base_channels*2 * 2, base_channels*2, kernel_size=1)
        Gate3 = get_gate(active_gates[1])
        self.att3 = Gate3(F_g=base_channels*8, F_l=base_channels*2, F_int=base_channels*2)
        self.dec3 = ConvBlock(base_channels*8 + base_channels*2, base_channels*4)

        # === Block 2 (Shallow / High Res) ===
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse_skip2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        Gate2 = get_gate(active_gates[2])
        self.att2 = Gate2(F_g=base_channels*4, F_l=base_channels, F_int=base_channels)
        self.dec2 = ConvBlock(base_channels*4 + base_channels, base_channels)

        # --- 3. HEADS ---
        self.head_R = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        self.head_D = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # --- 4. REFINEMENT ---
        self.refine_net = ResidualRefinementBlock(in_ch=4, mid_ch=32, num_blocks=4)

    def forward(self, x_rgb, x_therm=None):
        
        if x_therm is None:
            x_therm = torch.zeros_like(x_rgb[:, 0:1, :, :])
        
        # --- Encoding ---
        r1 = self.rgb_enc1(x_rgb)   
        r2 = self.rgb_enc2(r1)      
        r3 = self.rgb_enc3(r2)      
        r4 = self.rgb_enc4(r3)      
        
        t1 = self.thm_enc1(x_therm)
        t2 = self.thm_enc2(t1)
        t3 = self.thm_enc3(t2)
        t4 = self.thm_enc4(t3)
        
        # Bottleneck
        neck_in = torch.cat([r4, t4], dim=1)
        neck = self.bottleneck(neck_in)
        
        # --- Decoding ---
        
        # Block 4
        d4 = self.up4(neck) 
        skip4 = self.fuse_skip4(torch.cat([r3, t3], dim=1)) 
        skip4_att, att4_map = self.att4(g=d4, x=skip4)
        d4 = self.dec4(torch.cat([d4, skip4_att], dim=1))
        
        # Block 3
        d3 = self.up3(d4) 
        skip3 = self.fuse_skip3(torch.cat([r2, t2], dim=1)) 
        skip3_att, att3_map = self.att3(g=d3, x=skip3)
        d3 = self.dec3(torch.cat([d3, skip3_att], dim=1))
        
        # Block 2 
        d2 = self.up2(d3) 
        skip2 = self.fuse_skip2(torch.cat([r1, t1], dim=1)) 
        skip2_att, att2_map = self.att2(g=d2, x=skip2)
        d2 = self.dec2(torch.cat([d2, skip2_att], dim=1))

        # --- Intrinsic Decomposition ---
        pred_R = self.head_R(d2)
        pred_D = self.head_D(d2)
        
        pred_D_safe = torch.clamp(pred_D, min=0.01, max=1.0)
        pred_S = (1.0 - pred_D_safe) / pred_D_safe
        coarse_image = pred_R * pred_S
        
        # --- Refinement ---
        coarse_norm = coarse_image / (coarse_image + 1.0)
        refine_in = torch.cat([coarse_norm, x_therm], dim=1)
        residual = self.refine_net(refine_in)
        
        final_image = torch.sigmoid(coarse_norm + residual)
        
        attention_maps = [att4_map, att3_map, att2_map]
        
        return pred_R, pred_D, coarse_image, final_image, residual, attention_maps