# TIRGlow: A Multimodal Intrinsics-Guided Thermal-Aware Framework for RGB Low-Light Image Enhancement

This is the official implementation of the paper accepted for publication at the **33rd IEEE International Conference on Image Processing (ICIP 2026)**.




[![Conference](https://img.shields.io/badge/Conference-ICIP_2026-blue)](https://2026.ieeeicip.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)

---

<p align="center">
  <img src="teaser_cr.jpg" alt="A teaser example on a very low-light image fusing RGB and TIR data" width="100%">
</p>

**Authors:** Simone Melcarne and Jean-Luc Dugelay || Eurecom Research Center, Digital Security Department, Biot, France

---

## The Framework

<p align="center">
  <img src="figures/pipeline_tirglow.png" alt="Framework Overview" width="100%">
</p>

Low-light image enhancement (LLIE) is challenging when the visible signal is severely degraded by noise and information loss. We propose a **Multimodal Intrinsics-Guided Framework** that fuses low-light RGB and thermal infrared (TIR) data to reconstruct well-lit images, explicitly leveraging the physical structure of image formation:

$$I = R_{RGB} \odot S_{gray}$$

where $R \in \mathbb{R}^{3 \times H \times W}$ is the illumination-invariant reflectance and $S \in \mathbb{R}^{1 \times H \times W}$ is the shading component that captures light distribution and geometry.

### Key Components

* **Teacher–Student Distillation:** Since no ground-truth reflectance/shading exists for LLIE, a frozen, pretrained intrinsic decomposition network ([Careaga & Aksoy](https://github.com/compphoto/Intrinsic)) acts as a **teacher**. The teacher is run **offline, once, on the normal-exposure images** to pre-compute target reflectance ($\*\_alb.png$) and shading ($\*\_shd.npy$) maps, which are then loaded from disk during training to supervise the student.
* **Dual-Encoder Fusion (FB):** Independent RGB and TIR encoders extract multi-scale features, fused at every resolution level through **Fusion Blocks**.
* **Multi-Scale Attention Gating (AG):** Inspired by Attention U-Net, gated skip connections selectively inject thermal structural details only where the visible signal is degraded, suppressing thermal noise elsewhere:

$$\hat{X}^{l} = X^{l} \odot M^{l}, \qquad M^{l} = \sigma\left(W_\psi \, \phi\left(W_x X^{l} + W_g \hat{G}^{l}\right)\right)$$

* **Coarse Reconstruction:** The decoder predicts reflectance $\hat{R}$ and (inverse) shading $\hat{D}$, which are combined via the intrinsic law to obtain a coarse linear reconstruction $\hat{I}_{coarse} = \hat{R} \odot \hat{S}$.
* **Residual Refinement Net:** A stack of Residual Blocks, conditioned on the thermal input, predicts a residual correction map to recover fine details and produce the final output $\hat{I}_{out}$.
* **Composite Objective:** End-to-end training combines reflectance/shading supervision, a physical-consistency loss against the coarse reconstruction, and a refinement loss (pixel + edge + CIELAB color + VGG perceptual) on the final sRGB output.

---

## Dataset Preparation

Training requires triplets of low-light RGB $I_{low}$, thermal $T$, and well-lit ground truth $I_{gt}$. The model is trained on **HDRT** and evaluated zero-shot on **LLVIP** and **V-TIEE**.

> **Note:** although HDRT provides a real low-light (under-exposed) frame for each scene, the model is **not** trained on it directly. Its exposure reduction is not aggressive enough to represent severe darkness, so instead the well-exposed frame is used as ground truth and a physics-based low-light simulation pipeline (adaptive exposure gain + shot/read noise injection, computed **on-the-fly** for every training sample) synthesizes the low-light input from it. See `HDRT_Dataset.physics_based_low_light_simulation` and Eq. 6–7 of the paper.

### 1. Download Data

1. **HDRT Dataset (Thermal & HDR/SDR RGB) — training:**
   Official dataset page: [https://huggingface.co/datasets/jingchao-peng/HDRTDataset](https://huggingface.co/datasets/jingchao-peng/HDRTDataset)
   Paper / project details: [HDRT: A Large-Scale Dataset for Infrared-Guided HDR Imaging](https://arxiv.org/abs/2406.05475)

2. **LLVIP Dataset (Thermal & RGB) — zero-shot evaluation:**
   Official project page: [https://bupt-ai-cz.github.io/LLVIP/](https://bupt-ai-cz.github.io/LLVIP/)

3. **V-TIEE Dataset (Thermal & RGB) — zero-shot evaluation:**
   Released alongside RT-X Net: [https://github.com/jhakrraman/rt-xnet](https://github.com/jhakrraman/rt-xnet)
   <!-- TODO: per la valutazione riportata nel paper, il test set LLVIP usa i 70 campioni ufficiali di Jha et al. e V-TIEE una selezione manuale di 28 immagini pulite; vedi Supplementary. Se pubblichi le liste esatte, linkale qui. -->

4. **Teacher network (Careaga et al., intrinsic decomposition):**
   Used offline to pre-compute reflectance/shading supervision — official repository: [https://github.com/compphoto/Intrinsic](https://github.com/compphoto/Intrinsic)

### 2. Folder Structure

`HDRT_Dataset` expects the following layout (paths are passed as CLI arguments to `train.py`, see below):

```text
/path/to/HDRT/
├── RGB/
│   ├── RGB00260.JPG           # normal-exposure frame  (odd index)
│   ├── RGB00261.JPG           # paired low-light frame (even index, consecutive number)
│   ├── ...
│   ├── reflectance/
│   │   ├── RGB00260_alb.png   # 16-bit teacher reflectance target
│   │   └── ...
│   └── shading/
│       ├── RGB00260_shd.npy   # float32 teacher shading target
│       └── ...
└── infrared/
    ├── T00260.tiff            # thermal frame, same numeric ID as the normal RGB frame
    └── ...
```

For evaluation, `LLVIP_Dataset` expects three separate roots (visible/RGB input, thermal input, and RGB ground truth), passed as `--root_rgb`, `--root_thermal`, `--root_target` to `test_LLVIP.py`:

```text
/path/to/LLVIP/
├── test_data/LLVIP/     # low-light visible input
├── thermal/LLVIP/       # paired thermal input
└── target/LLVIP/        # well-lit RGB ground truth
```

For V-TIEE, `V_TIEE_Dataset` expects a different layout: input and target share the same filename (`N_zzzzz.png`) in two separate folders, while the thermal frame is looked up by stripping the suffix (`N_zzzzz.png` → `N.png`):

```text
/path/to/VTIEE/
├── RGB/
│   ├── Input/
│   │   ├── 1_08340.png    # low-light input
│   │   └── ...
│   └── Target/
│       ├── 1_08340.png    # well-lit ground truth (same filename as Input)
│       └── ...
└── Thermal/
    ├── 1.png               # thermal frame, matched via the prefix before "_"
    └── ...
```

Pass these as `--root_rgb .../RGB/Input`, `--root_target .../RGB/Target`, `--root_thermal .../Thermal` to `test_VTIEE.py`.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/simonemelc/TIRGlow.git
cd TIRGlow
```

2. **Create and activate the environment**

```bash
conda create -n tirglow python=3.9 -y
conda activate tirglow
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download our pretrained student model**

```bash
└── checkpoints/
  └── best_model.pth
```

[Google Drive](https://drive.google.com/file/d/1V7ykuz0DFxaDjtvbwHXJ7uIkrb6sg_Gb/view?usp=drive_link)

5. **Teacher network weights**
   The pretrained intrinsic decomposition network ([Careaga & Aksoy](https://github.com/compphoto/Intrinsic)) is needed if you want to (re-)generate the `reflectance/` and `shading/` target folders yourself. Follow the instructions in their official repository.

---

## Usage

### Training

The model is trained end-to-end on HDRT. At every iteration, a fresh low-light version of the normal-exposure RGB frame is synthesized on-the-fly (adaptive exposure gain + shot/read noise, Eq. 6–7 of the paper), while thermal input and teacher-derived reflectance/shading targets are loaded from disk.

```bash
python train.py \
  --root_rgb ./data/HDRT/registered_RGB_reduced_size \
  --root_thermal ./data/HDRT/infrared \
  --root_targets_R ./data/HDRT/RGB/reflectance \
  --root_targets_S ./data/HDRT/RGB/shading \
  --has_thermal \
  --batch_size 8 \
  --epochs 150 \
  --lr 1e-4 \
  --val_split 0.2 \
  --w_lab 1.0 --w_edge 0.8 --w_perc 0.2 \
  --device 0
```

Multi-GPU training via `DistributedDataParallel` is also supported:

```bash
torchrun --nproc_per_node=4 train.py --ddp --has_thermal \
  --root_rgb ./data/HDRT/RGB \
  --root_thermal ./data/HDRT/infrared \
  --root_targets_R ./data/HDRT/RGB/reflectance \
  --root_targets_S ./data/HDRT/RGB/shading
```

Checkpoints are saved to `checkpoints/attention/thermal_<has_thermal>_lr<lr>_wLab<w_lab>_wEdge<w_edge>_wPerc<w_perc>/best_model.pth`.

### Evaluation

Zero-shot evaluation on LLVIP, computing PSNR, SSIM, LPIPS and CIE ΔE₀₀:

```bash
python test_LLVIP.py \
  --ckpt ./checkpoints/attention/thermal_True_lr0.0001_wLab1.0_wEdge0.8_wPerc0.2/best_model.pth \
  --root_rgb ./data/LLVIP/test_data/LLVIP \
  --root_thermal ./data/LLVIP/thermal/LLVIP \
  --root_target ./data/LLVIP/target/LLVIP \
  --has_thermal \
  --device 0
```

Add `--self_ensemble` to enable 8x geometric self-ensembling (horizontal/vertical flips + rotation) at test time. Per-image metrics and predicted images are saved to `output_models/eval_LLVIP/<run_tag>/`, with an aggregated `summary.json`.

Zero-shot evaluation on V-TIEE follows the same pattern (note: LPIPS here uses the VGG backbone, vs. AlexNet in `test_LLVIP.py`):

```bash
python test_VTIEE.py \
  --ckpt ./checkpoints/attention/thermal_True_lr0.0001_wLab1.0_wEdge0.8_wPerc0.2/best_model.pth \
  --root_rgb ./data/VTIEE/RGB/Input \
  --root_thermal ./data/VTIEE/Thermal \
  --root_target ./data/VTIEE/RGB/Target \
  --has_thermal \
  --device 0
```

Per-image metrics, predicted images and attention-map visualizations are saved to `output_models/eval_VTIEE/<run_tag>/`, with an aggregated `summary.json`.

---

## Citation


If you find this work useful, please cite:

```bibtex
@INPROCEEDINGS{11630490,
  author={Melcarne, Simone and Dugelay, Jean-Luc},
  booktitle={2026 IEEE International Conference on Image Processing (ICIP)}, 
  title={A Multimodal Intrinsics-Guided Thermal-Aware Framework for RGB Low-Light Image Enhancement}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Lighting;Color;Modeling;Printing;Image enhancement;Poles and zeros;Equations;Reflectivity;Training;Noise;Low-light Image Enhancement;Multimodal Fusion;Thermal Infrared;Intrinsic Decomposition},
  doi={10.1109/ICIP61757.2026.11630490}}
```

