#!/bin/bash

# ========================================================
# Workspace, dataset, and evaluation tools
# ========================================================
# Modify the following paths to your own workspace
WORKSPACE="/import/c4dm-05/ll307/workspace/PM2S-draft"

# Modify the following paths to your own dataset directory
ASAP="/import/c4dm-05/ll307/datasets/asap-dataset-master"
A_MAPS="/import/c4dm-05/ll307/datasets/A-MAPS_1.1"
CPM="/import/c4dm-05/ll307/datasets/CPM"
ACPAS="/import/c4dm-05/ll307/datasets/ACPAS-dataset"

# Modify the following paths to your own evaluation tool directory
MV2H="/import/c4dm-05/ll307/tools/MV2H/bin"

# # ========================================================
# # Feature preparation 
# # ========================================================
# python3 feature_preparation.py \
#     --dataset_folder $ASAP $A_MAPS $CPM $ACPAS \
#     --feature_folder $WORKSPACE/features \
#     --workers 4 \

# ========================================================
# Model training
# ========================================================
# feature can be 'beat', 'quant', 'time_sig', 'key_sig', 'hand'
python3 train.py \
    --workspace $WORKSPACE \
    --ASAP $ASAP \
    --A_MAPS $A_MAPS \
    --CPM $CPM \
    --feature 'beat'