#!/bin/bash

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gaussian-car

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo $PYTORCH_CUDA_ALLOC_CONF

# Generate Coarse Poses
python pred_poses.py -s data/realcap/rabbit --sparse_num 4

# Gaussian repair
python train_gs.py -s data/realcap/rabbit -m output/gs_init/rabbit \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --white_background --random_background --use_dust3r

python render.py \
    -m output/gs_init/rabbit \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/rabbit/refined_cams.json \
    --white_background --render_path --use_dust3r

python leave_one_out_stage1.py -s data/realcap/rabbit \
    -m output/gs_init/rabbit_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/rabbit/refined_cams.json \
    --white_background --random_background --use_dust3r

python leave_one_out_stage2.py -s data/realcap/rabbit \
    -m output/gs_init/rabbit_loo \
    -r 8 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name dust3r_4 \
    --dust3r_json output/gs_init/rabbit/refined_cams.json \
    --white_background --random_background --use_dust3r

python train_lora.py --exp_name controlnet_finetune/rabbit \
    --prompt xxy5syt00 --sh_degree 2 --resolution 8 --sparse_num 4 \
    --data_dir data/realcap/rabbit \
    --gs_dir output/gs_init/rabbit \
    --loo_dir output/gs_init/rabbit_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora --use_dust3r

python train_repair.py \
    --config configs/gaussian-object-colmap-free.yaml \
    --train --gpu 0 \
    tag="rabbit" \
    system.init_dreamer="output/gs_init/rabbit" \
    system.exp_name="output/controlnet_finetune/rabbit" \
    system.refresh_size=8 \
    data.data_dir="data/realcap/rabbit" \
    data.resolution=8 \
    data.sparse_num=4 \
    data.prompt="a photo of a xxy5syt00" \
    data.json_path="output/gs_init/rabbit/refined_cams.json" \
    data.refresh_size=8 \
    system.sh_degree=2
