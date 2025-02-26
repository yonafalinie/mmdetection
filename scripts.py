
python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/DINOv2/configs/dino-5scale_dinov2-b_8xb2-12e_coco.py > log/train.log 2>&1


python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/OLN/configs/oln_mask/oln_mask_r50_fpn_1x_coco.py > log/trainmask.log 2>&1

python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/OLN_convnext/oln_mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py > log/trainconvnext.log 2>&1

python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/OLN_swin/oln_mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py > log/trainswin.log 2>&1

python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/oln_ssos_box_r50_fpn_1x_coco.py > log/trainbox.log 2>&1

python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/DINOv2/configs/oln_box_dinov2_fpn_1x_dbf6_ftlayer11.py > log/traindinov2dbf6_ftlayer11.log 2>&1

python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/DINOv2/configs/oln_box_dinov2_fpn_1x_dbf6_fullfreeze.py > log/traindinov2dbf6_fullfreeze.log 2>&1

python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/DINOv2/configs/oln_box_dinov2_fpn_1x_dbf6.py > log/traindinov2dbf6.log 2>&1

python3 ./tools/train.py /home/neel/data/Code/MMOln-ssos/mmdetection/projects/OLN/configs/oln_box/oln_box_r50_fpn_1x_dbf6.py > log/trainboxdbf6.log 2>&1
