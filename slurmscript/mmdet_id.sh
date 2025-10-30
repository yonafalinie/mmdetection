#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu:pascal:1
#SBATCH --job-name=olnssosclipsimplefpncocooodsampling1startepoch3_k5_10_20_25_30_40_50_100_id.out
#SBATCH -o log/olnssosclipsimplefpncocooodsampling1startepoch3_k5_10_20_25_30_40_50_100_id.out

module load cuda/11.8

BASE_CONFIG_DIR=/home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/clip/coco
BASE_WORK_DIR=/home3/qljx17/MMOln-ssos/mmdetection/work_dirs

K_VALUES=(5 10 20 25 30 40 50 100)  # Can be expanded to (5 10 20 25 30 40 50 100) 

for x in "${K_VALUES[@]}"; do
    CONFIG=${BASE_CONFIG_DIR}/oln_ssos_clip_simplefpn_coco_oodsampling1_startepoch3_k${x}.py
    WORK_DIR=${BASE_WORK_DIR}/oln_ssos_clip_simplefpn_coco_oodsampling1_startepoch3_k${x}

    for CKPT_PATH in $(ls "${WORK_DIR}"/epoch_*.pth | sort -V); do 
        [ -e "$CKPT_PATH" ] || continue

        CKPT_FILENAME=$(basename "$CKPT_PATH")
        EPOCH_NUM=${CKPT_FILENAME#epoch_}
        EPOCH_NUM=${EPOCH_NUM%.pth}

        echo "Testing k=${x}, epoch=${EPOCH_NUM}..."
        python3 ./tools/test.py "$CONFIG" "$CKPT_PATH"
    done

done