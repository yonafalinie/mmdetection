#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu:ampere:2
#SBATCH --job-name=mmdetmultiple2
#SBATCH -o log/mmdetmultiple2.out

module load cuda/11.8



python3 mmdet/utils/collect_env.py


# python3 ./tools/train.py /home3/qljx17/MMOln-ssos/mmdetection/projects/DINOv2/configs/dino-5scale_dinov2-b_8xb2-12e_coco.py 

# python3 tools/train.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/DINOv2/configs/oln_box_dinov2_fpn_1x_coco_batch2.py

bash tools/dist_train.sh \
    /home3/qljx17/MMOln-ssos/mmdetection/projects/DINOv2/configs/oln_box_dinov2_fpn_1x_coco_batch2_.py \
    2
