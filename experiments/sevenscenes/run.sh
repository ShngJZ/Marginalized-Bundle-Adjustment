#!/bin/bash
# cd to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_ROOT="/fsx/users/shngjz/Marginalized-Bundle-Adjustment-Datasets/7scenes"
OUTPUT_LOC="/fsx/users/shngjz/Marginalized-Bundle-Adjustment/release/7scenes"

# Stage 1: Preprocessing (8 GPUs in parallel, wait for all to finish)
echo "=== Stage 1: Preprocessing ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/sevenscenes/preprocess.py \
        --data-root ${DATA_ROOT} \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "Preprocessing complete."

# Stage 2: SfM (all 8 GPUs working together)
echo "=== Stage 2: Structure from Motion ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/sevenscenes/sfm.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}
echo "SfM complete."

# Stage 3: Visualization
echo "=== Stage 3: Visualization ==="
python experiments/sevenscenes/visualization.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

# Stage 4: Evaluation
echo "=== Stage 4: Evaluation ==="
python experiments/sevenscenes/summarize_result.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

echo "All done."