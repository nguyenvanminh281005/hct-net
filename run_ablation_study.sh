#!/bin/bash

# Script để chạy Ablation Study cho HCT-Net
# Khảo sát sự cần thiết của 3 thành phần: Transformer, Complexity, Entropy
# (Dice luôn có sẵn trong tất cả các trường hợp)

# Thiết lập cấu hình chung
DATASET="cvc"
DATASET_ROOT="/mnt/data/KHTN2023/research25/hct-netm/datasets/cvc"
EPOCHS=10
TRAIN_BATCH=2
VAL_BATCH=2
LAYERS=7

# ========== 8 TRƯỜNG HỢP ABLATION ==========

echo "======================================"
echo "ABLATION STUDY: 8 Configurations"
echo "======================================"
echo ""

# 1. ALL: dice + transformer + complexity + entropy (FULL MODEL)
# echo ">>> Running (1/8): ALL - Full model with all components"
# python hct_net/train_CVCDataset.py \
#     --ablation_mode all \
#     --dataset ${DATASET} \
#     --dataset_root ${DATASET_ROOT} \
#     --epochs ${EPOCHS} \
#     --train_batch ${TRAIN_BATCH} \
#     --val_batch ${VAL_BATCH} \
#     --layers ${LAYERS} \
#     --note "ablation_all"

# echo ""
# echo ">>> Completed (1/8)"
# echo ""

# 2. NO_TRANSFORMER: dice + complexity + entropy
echo ">>> Running (2/8): NO_TRANSFORMER - Without transformer loss"
python hct_net/train_CVCDataset.py \
    --ablation_mode no_transformer \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --note "ablation_no_transformer"

echo ""
echo ">>> Completed (2/8)"
echo ""

# 3. NO_COMPLEXITY: dice + transformer + entropy
echo ">>> Running (3/8): NO_COMPLEXITY - Without complexity loss"
python hct_net/train_CVCDataset.py \
    --ablation_mode no_complexity \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --note "ablation_no_complexity"

echo ""
echo ">>> Completed (3/8)"
echo ""

# 4. NO_ENTROPY: dice + transformer + complexity
echo ">>> Running (4/8): NO_ENTROPY - Without entropy loss"
python hct_net/train_CVCDataset.py \
    --ablation_mode no_entropy \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --note "ablation_no_entropy"

echo ""
echo ">>> Completed (4/8)"
echo ""

# 5. ONLY_TRANSFORMER: dice + transformer
echo ">>> Running (5/8): ONLY_TRANSFORMER - Dice + Transformer only"
python hct_net/train_CVCDataset.py \
    --ablation_mode only_transformer \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --note "ablation_only_transformer"

echo ""
echo ">>> Completed (5/8)"
echo ""

# 6. ONLY_COMPLEXITY: dice + complexity
echo ">>> Running (6/8): ONLY_COMPLEXITY - Dice + Complexity only"
python hct_net/train_CVCDataset.py \
    --ablation_mode only_complexity \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --note "ablation_only_complexity"

echo ""
echo ">>> Completed (6/8)"
echo ""

# 7. ONLY_ENTROPY: dice + entropy
echo ">>> Running (7/8): ONLY_ENTROPY - Dice + Entropy only"
python hct_net/train_CVCDataset.py \
    --ablation_mode only_entropy \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --note "ablation_only_entropy"

echo ""
echo ">>> Completed (7/8)"
echo ""

# 8. NONE: chỉ dice (BASELINE)
echo ">>> Running (8/8): NONE - Dice only (baseline)"
python hct_net/train_CVCDataset.py \
    --ablation_mode none \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --epochs ${EPOCHS} \
    --layers ${LAYERS} \
    --note "ablation_baseline"

echo ""
echo ">>> Completed (8/8)"
echo ""

echo "======================================"
echo "ABLATION STUDY COMPLETED!"
echo "======================================"
echo ""
echo "All 8 configurations have been run."
echo "Check results in ./search_exp/ directory"
echo "Compare metrics in wandb project: hct-net-ablation"
