#!/bin/bash
# cd to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_ROOT="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/custom"
OUTPUT_LOC="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/custom"

# Stage 1: Preprocessing (all 8 GPUs working together on single sequence)
echo "=== Stage 1: Preprocessing ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/custom/preprocess.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}
echo "Preprocessing complete."

# Stage 2: SfM (all 8 GPUs working together on single sequence)
echo "=== Stage 2: Structure from Motion ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/custom/sfm.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}
echo "SfM complete."

# Stage 3: Visualization
echo "=== Stage 3: Visualization ==="
python experiments/custom/visualization.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

echo "All done."