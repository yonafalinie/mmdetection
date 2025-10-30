#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=28g
#SBATCH -w gpu-toby0
#SBATCH -p gpu-private
#SBATCH --qos=long-high-prio
#SBATCH --job-name=empattn
#SBATCH -o log/olnssosboxdbf6oodempattnoodsampling1startepoch3_k_25_30_40_50_100.out
#SBATCH -t 07-00:00:00

module load cuda/11.8



# python3 mmdet/utils/collect_env.py

# export PYTHONPATH=/home3/qljx17/MMOln-ssos/mmdetection/projects/OLN_vitadapter:$PYTHONPATH

# python3 tools/train.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/OLN/configs/oln_box/oln_box_r50_fpn_1x_coco.py


# python3 tools/train.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k10.py
# python3 tools/train.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k15.py
# python3 tools/train.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k20.py

python3 tools/train.py \
    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k25.py
python3 tools/train.py \
    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k30.py
python3 tools/train.py \
    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k40.py
python3 tools/train.py \
    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k50.py
python3 tools/train.py \
    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/empiricalattention/oln_ssos_box_empiricalattention_dbf6_oodsampling1_startepoch3_k100.py






# python3 ./tools/test.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch12.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_r50_fpn_1x_coco_oodsampling1_startepoch12/epoch_18.pth     \
#     --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.46 test_evaluator.anomaly_score_threshold=0.76 \
#     test_dataloader.dataset.data_root=/home2/projects/datasets/coco \
#     test_dataloader.dataset.data_prefix.img=val2017 \
#     test_dataloader.dataset.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
#     test_evaluator.eval_class=nonvoc \
#     test_evaluator.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
#     test_dataloader.dataset.eval_class=nonvoc



# python3 ./tools/test.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/oln_ssos_box_r50_fpn_1x_dbf6_oodsampling1_startepoch12.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_r50_fpn_1x_dbf6_oodsampling1_startepoch12/epoch_18.pth \
#     --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.10 test_evaluator.anomaly_score_threshold=0.83 \
#     test_dataloader.dataset.data_root=/home2/projects/datasets/dbf6 \
#     test_dataloader.dataset.data_prefix.img=images \
#     test_dataloader.dataset.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_evaluator.eval_class=nonvoc \
#     test_evaluator.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#     test_dataloader.dataset.eval_class=nonvoc