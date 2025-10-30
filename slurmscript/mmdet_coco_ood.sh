#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu:pascal:1
#SBATCH --job-name=olnssosclipsimplefpncocooodsampling1startepoch3_k100_ood
#SBATCH -o log/olnssosclipsimplefpncocooodsampling1startepoch3_k100_ood.out
#SBATCH -t 07-00:00:00

module load cuda/11.8


# python3 mmdet/utils/collect_env.py    

# ====== Define Variables Here ======
K=100
XXXX="clip"
AAAA="olnssosclipsimplefpncocooodsampling1startepoch3_k5_10_20_25_30_40_50_100_threshold_results"  
YYYYY="coco"
ZZZZZ="oln_ssos_clip_simplefpn_coco_oodsampling1_startepoch3"

# ====== Construct Paths Dynamically ======
CSV_FILE="/home3/qljx17/MMOln-ssos/mmdetection/csv/${XXXX}/${YYYYY}/${AAAA}.csv"
CONFIG_FILE="/home3/qljx17/MMOln-ssos/mmdetection/projects/oln-ssos/configs/oln_box/${XXXX}/${YYYYY}/${ZZZZZ}_k${K}.py"
WORK_DIR="/home3/qljx17/MMOln-ssos/mmdetection/work_dirs/${ZZZZZ}_k${K}"

# Get all unique epochs for the current K from the CSV (skip header)
EPOCHS=$(awk -F',' -v k="$K" 'NR>1 && $1==k {print $2}' "$CSV_FILE" | sort -nu)

# Loop over each epoch number found in the CSV
for EPOCH in $EPOCHS; do
    # Extract threshold values from CSV using awk (skip header)
    THRESHOLDS=$(awk -F',' -v k="$K" -v e="$EPOCH" 'NR>1 && $1==k && $2==e {print $3, $7}' "$CSV_FILE")

    # If no matching row, skip
    if [ -z "$THRESHOLDS" ]; then
        echo "No threshold data for k=$K epoch=$EPOCH"
        continue
    fi

    # Parse optimal and all thresholds
    read -r OPTIMAL ALL <<< "$THRESHOLDS"

    # Build checkpoint path
    CKPT="${WORK_DIR}/epoch_${EPOCH}.pth"

    echo "=== Running epoch $EPOCH (k=$K): optimal=$OPTIMAL, anomaly=$ALL ==="

    python3 ./tools/test.py "$CONFIG_FILE" "$CKPT" \
        --cfg-options \
        test_evaluator.mode=ood \
        test_evaluator.optimal_score_threshold=0.0 \
        test_evaluator.anomaly_score_threshold="$ALL" \
        test_dataloader.dataset.data_root=/home2/projects/datasets/coco \
        test_dataloader.dataset.data_prefix.img=val2017 \
        test_dataloader.dataset.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
        test_evaluator.eval_class=nonvoc \
        test_evaluator.ann_file=/home2/projects/datasets/coco/annotations/instances_val2017_ood_rm_overlap.json \
        test_dataloader.dataset.eval_class=nonvoc
done


# test_evaluator.optimal_score_threshold="$OPTIMAL" \