#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name=olnssosvitdetltdimagingoodsampling1startepoch3_k10
#SBATCH -o log/olnssosvitdetltdimagingoodsampling1startepoch3_k10.out
#SBATCH -t 07-00:00:00

module load cuda/11.8


# python3 mmdet/utils/collect_env.py


python3 ./tools/train.py \
    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/vitdet/ltdimaging/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_ltdimaging_oodsampling1_startepoch3_k10.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/vitdet/sixray10/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_sixray10_oodsampling1_startepoch3_k10.py

# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/vitdet/sixray10/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_sixray10_oodsampling1_startepoch3_k15.py

# python3 tools/train.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/vitdet/sixray10/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_sixray10_oodsampling1_startepoch3_k20.py

# CONFIG=/home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/vitdet/coco/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_coco_oodsampling1_startepoch3_k10.py
# CHECKPOINT_DIR=/home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_coco_oodsampling1_startepoch3_k10

# for CKPT in "$CHECKPOINT_DIR"/*.pth; do
#     echo "Evaluating $CKPT"
#     python3 tools/test.py "$CONFIG" "$CKPT"
# done







# ###For dbf6 dataset
# python3 ./tools/test.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/vitdet/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_dbf6_oodsampling1_startepoch3_k10.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_coco_oodsampling1_startepoch3_k10/iter_168188.pth \
#     --cfg-options test_evaluator.mode=ood \
#     test_evaluator.optimal_score_threshold=0.50 \
#     test_evaluator.anomaly_score_threshold=0.48 \
#     test_dataloader.dataset.data_root=/home2/projects/datasets/dbf6 \
#     test_dataloader.dataset.data_prefix.img=images \
#     test_dataloader.dataset.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_evaluator.eval_class=nonvoc \
#     test_evaluator.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_dataloader.dataset.eval_class=nonvoc



#  ###For coco dataset
# python3 ./tools/test.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/vitdet/coco/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_coco_oodsampling1_startepoch3_k10.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_vitdet_box_vit-b-mae_lsj-100e_coco_oodsampling1_startepoch3_k10/iter_168188.pth \
#     --cfg-options test_evaluator.mode=ood \
#     test_evaluator.optimal_score_threshold=0.56 \
#     test_evaluator.anomaly_score_threshold=0.48 \
#     test_dataloader.dataset.data_root=/home2/projects/datasets/coco \
#     test_dataloader.dataset.data_prefix.img=val2017 \
#     test_dataloader.dataset.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
#     test_evaluator.eval_class=nonvoc \
#     test_evaluator.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
#     test_dataloader.dataset.eval_class=nonvoc