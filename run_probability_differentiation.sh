#!/bin/bash

# Script để chạy training với các phương pháp phân hóa probability khác nhau
# Sử dụng 4 cách được đề xuất để tránh probability hội tụ về cùng giá trị

echo "=========================================="
echo "PROBABILITY DIFFERENTIATION EXPERIMENTS"
echo "=========================================="
echo ""

# Base command
BASE_CMD="python ./hct_net/train_CVCDataset_pareto_v2.py \
    --layers 9 \
    --epochs 50 \
    --train_batch 2 \
    --val_batch 2 \
    --transformer_init_bias 0.3"

# # ==========================================
# # EXPERIMENT 1: Baseline (Original - indecision penalty)
# # ==========================================
# echo "Experiment 1: Baseline (indecision penalty only)"
# echo "------------------------------------------"
# $BASE_CMD \
#     --diversity_loss_type indecision \
#     --positional_init_scale 0.0 \
#     --positional_bias_factor 0.0 \
#     --note "exp1_baseline_indecision"

# echo ""
# echo "✅ Completed Experiment 1"
# echo ""

# # ==========================================
# # EXPERIMENT 2: Cách 1 + Cách 3 (Per-connection complexity + Positional bias)
# # ==========================================
# echo "Experiment 2: Per-connection complexity + Positional bias"
# echo "------------------------------------------"
# $BASE_CMD \
#     --diversity_loss_type indecision \
#     --positional_init_scale 0.15 \
#     --positional_bias_factor 0.1 \
#     --note "exp2_positional_bias"

# echo ""
# echo "✅ Completed Experiment 2"
# echo ""

# # ==========================================
# # EXPERIMENT 3: Cách 1 + Cách 2 (Per-connection + Variance loss)
# # ==========================================
# echo "Experiment 3: Per-connection complexity + Variance loss"
# echo "------------------------------------------"
# $BASE_CMD \
#     --diversity_loss_type variance \
#     --positional_init_scale 0.0 \
#     --positional_bias_factor 0.05 \
#     --note "exp3_variance_loss"

# echo ""
# echo "✅ Completed Experiment 3"
# echo ""

# # ==========================================
# # EXPERIMENT 4: Cách 1 + Cách 2 (Per-connection + Repulsion loss)
# # ==========================================
# echo "Experiment 4: Per-connection complexity + Repulsion loss"
# echo "------------------------------------------"
# $BASE_CMD \
#     --diversity_loss_type repulsion \
#     --positional_init_scale 0.0 \
#     --positional_bias_factor 0.05 \
#     --note "exp4_repulsion_loss"

# echo ""
# echo "✅ Completed Experiment 4"
# echo ""

# # ==========================================
# # EXPERIMENT 5: Cách 1 + Cách 4 (Per-connection + Gumbel-Softmax)
# # ==========================================
# echo "Experiment 5: Per-connection complexity + Gumbel-Softmax"
# echo "------------------------------------------"
# $BASE_CMD \
#     --diversity_loss_type indecision \
#     --positional_init_scale 0.0 \
#     --positional_bias_factor 0.05 \
#     --use_gumbel_softmax \
#     --gumbel_temperature 1.0 \
#     --gumbel_anneal \
#     --gumbel_temp_min 0.5 \
#     --note "exp5_gumbel_softmax"

# echo ""
# echo "✅ Completed Experiment 5"
# echo ""

# ==========================================
# EXPERIMENT 6: ALL METHODS COMBINED (Recommended)
# ==========================================
echo "Experiment 6: ALL METHODS COMBINED (RECOMMENDED)"
echo "------------------------------------------"
$BASE_CMD \
    --diversity_loss_type variance \
    --positional_init_scale 0.15 \
    --positional_bias_factor 0.1 \
    --use_gumbel_softmax \
    --gumbel_temperature 1.0 \
    --gumbel_anneal \
    --gumbel_temp_min 0.5 \
    --pareto_weight_dice 0.3 \
    --pareto_weight_complexity 0.3 \
    --pareto_weight_connection 0.4 \
    --note "exp6_all_combined_run1"

echo ""
echo "✅ Completed Experiment 6"
echo ""

echo "Experiment 6: ALL METHODS COMBINED (RECOMMENDED)"
echo "------------------------------------------"
$BASE_CMD \
    --diversity_loss_type variance \
    --positional_init_scale 0.15 \
    --positional_bias_factor 0.1 \
    --use_gumbel_softmax \
    --gumbel_temperature 1.0 \
    --gumbel_anneal \
    --gumbel_temp_min 0.5 \
    --pareto_weight_dice 0.3 \
    --pareto_weight_complexity 0.3 \
    --pareto_weight_connection 0.4 \
    --note "exp6_all_combined_run2"

echo ""
echo "✅ Completed Experiment 6"
echo ""


echo "Experiment 6: ALL METHODS COMBINED (RECOMMENDED)"
echo "------------------------------------------"
$BASE_CMD \
    --diversity_loss_type variance \
    --positional_init_scale 0.15 \
    --positional_bias_factor 0.1 \
    --use_gumbel_softmax \
    --gumbel_temperature 1.0 \
    --gumbel_anneal \
    --gumbel_temp_min 0.5 \
    --pareto_weight_dice 0.3 \
    --pareto_weight_complexity 0.3 \
    --pareto_weight_connection 0.4 \
    --note "exp6_all_combined_run3"

echo ""
echo "✅ Completed Experiment 6"
echo ""

# echo "=========================================="
# echo "ALL EXPERIMENTS COMPLETED!"
# echo "=========================================="
# echo ""
# echo "Summary of experiments:"
# echo "  1. Baseline (indecision only)"
# echo "  2. Positional bias (Cách 3)"
# echo "  3. Variance loss (Cách 2a)"
# echo "  4. Repulsion loss (Cách 2b)"
# echo "  5. Gumbel-Softmax (Cách 4)"
# echo "  6. All methods combined (RECOMMENDED)"
# echo ""
# echo "Check results in ./search_exp/ directory"
