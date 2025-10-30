#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu:pascal:1
#SBATCH --job-name=id
#SBATCH -o log/id.out

module load cuda/11.8

# python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"


python3 ./tools/test.py \
  /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/clip/coco/oln_ssos_clip_simplefpn_coco_oodsampling1_startepoch3_k25.py \
  /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_clip_simplefpn_coco_oodsampling1_startepoch3_k25/epoch_12.pth


# python3 ./tools/test.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/swin/coco/oln_ssos_box_swin-t-p4-w7_fpn_1x_coco_oodsampling1_startepoch3_k50.py \
#     /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_swin-t-p4-w7_fpn_1x_coco_oodsampling1_startepoch3_k50/epoch_4.pth \
#     --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.0 test_evaluator.anomaly_score_threshold=0.8 \
#     test_dataloader.dataset.data_root=/home2/projects/datasets/coco \
#     test_dataloader.dataset.data_prefix.img=val2017 \
#     test_dataloader.dataset.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
#     test_evaluator.eval_class=nonvoc \
#     test_evaluator.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
#     test_dataloader.dataset.eval_class=nonvoc


# python3 ./tools/test.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/swin/dbf6/oln_ssos_box_swin-t-p4-w7_fpn_1x_dbf6_oodsampling1_startepoch3_k15.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_swin-t-p4-w7_fpn_1x_dbf6_oodsampling1_startepoch3_k15/epoch_6.pth \
#    --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.0 test_evaluator.anomaly_score_threshold=0.7617862820625305 \
#    test_dataloader.dataset.data_root=/home2/projects/datasets/dbf6 \
#    test_dataloader.dataset.data_prefix.img=images \
#    test_dataloader.dataset.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#    test_evaluator.eval_class=nonvoc \
#    test_evaluator.ann_file=/home2/projects/datasets/dbf6/annotations/test_db6_only_firearms.json \
#    test_dataloader.dataset.eval_class=nonvoc    


# python3 ./tools/test.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/dinov2/sixray10/oln_ssos_box_DINOv2_sixray10_oodsampling1_startepoch3_k5.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_DINOv2_sixray10_oodsampling1_startepoch3_k5/epoch_6.pth \
#    --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.0 test_evaluator.anomaly_score_threshold=0.967205822467804 \
#    test_dataloader.dataset.data_root=/home2/projects/datasets/SIXRay10 \
#    test_dataloader.dataset.data_prefix.img=image/test \
#    test_dataloader.dataset.ann_file=/home2/projects/datasets/SIXRay10/annotation/SIXRay10_test_only_firearms.json \
#    test_evaluator.eval_class=nonvoc \
#    test_evaluator.ann_file=/home2/projects/datasets/SIXRay10/annotation/SIXRay10_test_only_firearms.json \
#    test_dataloader.dataset.eval_class=nonvoc


# python3 ./tools/test.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/swin/ltdimaging/oln_ssos_box_swin-t-p4-w7_fpn_1x_ltdimaging_oodsampling1_startepoch3_k50.py \
#    /home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_ssos_box_swin-t-p4-w7_fpn_1x_ltdimaging_oodsampling1_startepoch3_k50/epoch_18.pth \
#    --cfg-options test_evaluator.mode=ood test_evaluator.optimal_score_threshold=0.0 test_evaluator.anomaly_score_threshold=0.5008193254470825 \
#    test_dataloader.dataset.data_root=/home2/projects/datasets/CHALearn_LTDImaging \
#    test_dataloader.dataset.data_prefix.img=data/Day/images/ \
#    test_dataloader.dataset.ann_file=/home2/projects/datasets/CHALearn_LTDImaging/data/test_day_only_vehicles.json \
#    test_evaluator.eval_class=nonvoc \
#    test_evaluator.ann_file=/home2/projects/datasets/CHALearn_LTDImaging/data/test_day_only_vehicles.json \
#    test_dataloader.dataset.eval_class=nonvoc