#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name=olnclipsimplefpnbdd
#SBATCH -o log/olnclipsimplefpnbdd.out

module load cuda/11.8


# bash tools/dist_train.sh \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/OLN_sam/oln_sam_vitb_coco.py \
#     2 

python3 tools/train.py \
   /home3/qljx17/MMOln-ssos/mmdetection/projects/OLN_clip/oln_clip_simplefpn_bdd.py

# python3 ./tools/test.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10/epoch_12.pth 

# python3 ./tools/test.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/convnext/dbf6/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_convnext-t-p4-w7_fpn_amp-ms-crop-3x_dbf6_oodsampling1_startepoch3_k10/epoch_12.pth \
#     --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.00 test_evaluator.anomaly_score_threshold=0.82 \
#     test_dataloader.dataset.data_root=/home2/projects/datasets/dbf6 \
#     test_dataloader.dataset.data_prefix.img=images \
#     test_dataloader.dataset.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_evaluator.eval_class=nonvoc \
#     test_evaluator.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_dataloader.dataset.eval_class=nonvoc