#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name=olnssosfpn
#SBATCH -o log/olnssosboxcocofpnoodsampling1startepoch8_k_100.out 
#SBATCH -t 07-00:00:00

module load cuda/11.8


# python3 mmdet/utils/collect_env.py


# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch8_k10.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch8_k15.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch8_k20.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch8_k25.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch8_k30.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch3_k40.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch8_k50.py

python3 tools/train.py \
   /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/fpn/coco/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch8_k100.py

# python3 ./tools/test.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10/epoch_12.pth 

# python3 ./tools/test.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10/epoch_12.pth \
#     --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.00 test_evaluator.anomaly_score_threshold=0.82 \
#     test_dataloader.dataset.data_root=/home2/projects/datasets/dbf6 \
#     test_dataloader.dataset.data_prefix.img=images \
#     test_dataloader.dataset.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_evaluator.eval_class=nonvoc \
#     test_evaluator.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_dataloader.dataset.eval_class=nonvoc