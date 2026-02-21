#!/bin/bash
# cd to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_ROOT="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/eth3d"
OUTPUT_LOC="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/eth3d"

# Stage 1: Preprocessing (8 GPUs in parallel, wait for all to finish)
echo "=== Stage 1: Preprocessing ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/eth3d/preprocess.py \
        --data-root ${DATA_ROOT} \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "Preprocessing complete."

# Stage 2: SfM (8 GPUs in parallel, wait for all to finish)
echo "=== Stage 2: Structure from Motion ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/eth3d/sfm.py \
        --data-root ${DATA_ROOT} \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "SfM complete."

# Stage 3: Visualization
echo "=== Stage 3: Visualization ==="
python experiments/eth3d/visualization.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

# Stage 4: Evaluation
echo "=== Stage 4: Evaluation ==="
python experiments/eth3d/summarize_result.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

echo "All done."