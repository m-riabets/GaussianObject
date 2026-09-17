# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GaussianObject** is a framework for high-quality 3D object reconstruction from four views using Gaussian Splatting (3DGS). The method achieves state-of-the-art results even under COLMAP-free conditions (without precise camera pose estimates).

**Paper**: GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting (SIGGRAPH Asia 2024 / ACM TOG)

**Key Innovation**: Multi-stage pipeline combining visual hull initialization, coarse 3D Gaussian optimization, leave-one-out analysis for data generation, Gaussian repair via diffusion-based LoRA fine-tuning, and iterative refinement.

## Architecture & Pipeline

The project follows a **4-stage reconstruction pipeline**:

1. **Visual Hull** (`visual_hull.py`): Constructs initial 3D point cloud using visual hull from multi-view masks and camera parameters
2. **Coarse 3DGS Training** (`train_gs.py`): Optimizes Gaussian splatting representation with L1+SSIM loss on masked input views
3. **Leave-One-Out (LOO) Analysis** (`leave_one_out_stage1.py`, `leave_one_out_stage2.py`): Generates training pairs by removing one view and adding 3D noise, creating corrupted/reference image pairs for repair model training
4. **Gaussian Repair** (`train_repair.py`): Fine-tunes ControlNet/Diffusion model (via LoRA) to correct corrupted Gaussians, then applies it for iterative refinement

### Key Modules

- **`scene/`**: Core data structures
  - `gaussian_model.py`: 3D Gaussian representation with parameters (position, covariance, SH coefficients, opacity)
  - `cameras.py`: Camera calibration and view management
  - `dataset_readers.py`: Loaders for Colmap, DUSt3R/MASt3R, NeRF360 formats
  - `colmap_loader.py`: COLMAP MVS data parsing

- **`gaussian_renderer/`**: Rendering engine
  - `__init__.py`: Differentiable Gaussian rasterization via CUDA kernels
  - `network_gui.py`: Real-time viewer (WebSocket-based visualization)
  - `render_w_pose.py`: Variant with pose optimization (from `diff-gaussian-rasterization-w-pose`)

- **`utils/`**: Shared utilities
  - `camera_utils.py`: Camera matrix operations, SfM camera loading
  - `loss_utils.py`: L1, SSIM, LPIPS, TV losses; monocular depth regularization
  - `sh_utils.py`: Spherical harmonics evaluation for view-dependent color
  - `pose_utils.py`: Camera pose refinement (for dust3r initialization)
  - `image_utils.py`: Metrics (PSNR)

- **`threestudio/`** & **`cldm/`**: Diffusion/ControlNet wrapper for repair model
- **`dust3r/`** & **`mast3r/`**: Submodule integrations for camera pose/depth estimation
- **`preprocess/`**: Data preprocessing scripts (monodepth, masking, downsampling)

### Submodules (in `submodules/`)

- **`diff-gaussian-rasterization`**: Core CUDA rasterizer (ashawkey fork)
- **`diff-gaussian-rasterization-w-pose`**: Rasterizer with pose gradient flow (rmurai0610)
- **`simple-knn`**: Fast k-NN for 3D point operations
- **`pytorch3d`**: 3D vision ops (e.g., 3D transformations)
- **`minLoRA`**: Lightweight LoRA implementation
- **`CLIP`**: Vision-language model for semantic guidance
- **`segment-anything`** (SAM): Automatic foreground masking
- **`croco`**: DUSt3R dependency for pose estimation

## Common Development Tasks

### Environment Setup

```bash
# Clone with submodules (critical—these are C++ CUDA extensions)
git clone https://github.com/GaussianObject/GaussianObject.git --recursive
cd GaussianObject

# Install PyTorch for CUDA 11.8 (or adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Python packages
pip install -r requirements.txt

# Optional: build croco for DUSt3R pose estimation
cd submodules/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..
```

### Download Pretrained Models

```bash
# Download ControlNet v1.1 Tile + Stable Diffusion v1.5 weights
cd models
python download_hf_models.py
cd ..

# For COLMAP-free (CF) mode: download SAM + DUSt3R/MASt3R
cd models
sh download_preprocess_models.sh
cd ..
```

### Full Reconstruction Pipeline

```bash
# 1. Visual Hull (coarse 3D initialization from masks + cameras)
python visual_hull.py \
    --sparse_id 4 \
    --data_dir data/mip360/kitchen \
    --reso 2 --not_vis

# 2. Coarse Gaussian Training (4-view optimization)
python train_gs.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background

# 3. Leave-One-Out: generate training pairs for repair model
python leave_one_out_stage1.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background

python leave_one_out_stage2.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background

# 4. Fine-tune ControlNet with LoRA on LOO pairs
python train_lora.py --exp_name controlnet_finetune/kitchen \
    --prompt xxy5syt00 --sh_degree 2 --resolution 4 --sparse_num 4 \
    --data_dir data/mip360/kitchen \
    --gs_dir output/gs_init/kitchen \
    --loo_dir output/gs_init/kitchen_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora

# 5. Train Gaussian Repair (iterative refinement with diffusion)
python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu 0 \
    tag="kitchen" \
    system.init_dreamer="output/gs_init/kitchen" \
    system.exp_name="output/controlnet_finetune/kitchen" \
    data.data_dir="data/mip360/kitchen" \
    data.resolution=4 \
    data.sparse_num=4 \
    data.prompt="a photo of a xxy5syt00" \
    system.sh_degree=2
```

### Rendering & Visualization

```bash
# Render test views (coarse model)
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --skip_all --skip_train

# Render novel view path (final repaired model)
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --render_path \
    --load_ply output/gaussian_object/kitchen/save/last.ply

# Real-time interactive viewer (if network GUI is enabled in train_gs.py)
# Connects via WebSocket on port 6006 during training
```

### COLMAP-Free (CF) Mode

For arbitrary captured images without COLMAP:

```bash
# Prepare data: images in ./data/<dataset>/images/*.png
# Create sparse_4.txt, sparse_test.txt with image indices (0-indexed)

# 1. Generate masks with SAM
# Run segment_anything.ipynb manually

# 2. Estimate coarse poses with DUSt3R
python pred_poses.py -s data/realcap/rabbit --sparse_num 4
# or MASt3R: python pred_poses_mast3r.py -s data/realcap/rabbit --sparse_num 4

# 3. Generate depth maps
python preprocess/pred_monodepth.py -s data/realcap/rabbit

# 4-5. Run standard pipeline with --use_dust3r flag
python train_gs.py -s data/realcap/rabbit -m output/gs_init/rabbit \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --random_background --use_dust3r
```

### Configuration

- **`configs/gaussian-object.yaml`**: Standard COLMAP-based pipeline config (loss weights, optimizer, densification schedule, LoRA ranks)
- **`configs/gaussian-object-colmap-free.yaml`**: CF variant with adjusted parameters
- **`arguments/__init__.py`**: Command-line argument parser definitions (ModelParams, OptimizationParams, PipelineParams)

## Code Style & Conventions

- **Imports**: Follow standard Python + PyTorch patterns. Custom modules (`scene`, `gaussian_renderer`, `utils`, `arguments`) are imported directly.
- **Tensor device placement**: Always explicit `.cuda()` or `.to(device)` calls; GPU assumed (CUDA 11.8).
- **Rendering loop**: Iterative in `train_gs.py` with Tensorboard logging; modular `render()` function in `gaussian_renderer/__init__.py`.
- **Loss computation**: Vectorized in `utils/loss_utils.py`; composed via weighted sum in training loop.
- **Pose optimization**: Optional `pose_utils.py` refinement when `--use_dust3r` flag is set.

## Key Design Patterns

1. **Lazy initialization**: Scene and Gaussians created separately, combined via `Scene(args, gaussians)`.
2. **Composable rendering**: `render(camera, gaussians, pipe, bg)` returns dict with 'render', 'opacity', 'depth' keys.
3. **Multi-resolution training**: `--resolution` flag controls downsampling; all views processed identically.
4. **SH coefficient annealing**: `gaussians.oneupSHdegree()` increases spherical harmonics degree every 1000 iterations.
5. **Densification/pruning**: Adaptive Gaussian splitting/merging based on gradient magnitude and opacity.

## Common Pitfalls & Notes

- **Submodules critical**: CUDA rasterizers must be built during pip install. Missing submodules cause import failures.
- **COLMAP vs. DUSt3R**: Colmap needs `sparse/` directory structure; DUSt3R outputs `.json` pose files. Flag `--use_dust3r` switches loader logic.
- **GPU memory**: 4-view training typically requires ~20–24 GB VRAM; LoRA fine-tuning needs ~40+ GB (ControlNet + UNet + Gaussian rendering).
- **Prompt understanding**: ControlNet LoRA trains on specific object names (e.g., "xxy5syt00" for Mip-NeRF360 objects); prompt must match for repair to work well.
- **Data directory structure**: Masks (`.png` in `masks/`), depth maps (`.npy` in `zoe_depth/`), train/test splits (`.txt` files with image IDs, 0-indexed) are expected.

## References

- **Original 3DGS**: https://github.com/graphdeco-inria/gaussian-splatting
- **ThreeStudio**: https://github.com/threestudio-project/threestudio (diffusion integration)
- **ControlNet**: https://github.com/lllyasviel/ControlNet (tile-based inpainting)
- **DUSt3R/MASt3R**: Pose/depth estimation from image pairs (NAVER Research)
