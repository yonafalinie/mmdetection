#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=28g
#SBATCH -w gpu-toby0
#SBATCH -p gpu-private
#SBATCH --qos=long-high-prio
#SBATCH --job-name=olnboxdinov2
#SBATCH -o log/olnboxdinov2.out
#SBATCH -t 07-00:00:00

module load cuda/11.8



# python3 mmdet/utils/collect_env.py


python3 tools/train.py /home3/qljx17/MMOln-ssos/mmdetection/projects/DINOv2/configs/oln_box_dinov2_fpn_1x_dbf6.py

# bash tools/dist_train.sh \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/OLN_convnext/oln_mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py \
#     2 \
#     --resume /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco/epoch_10.pth




