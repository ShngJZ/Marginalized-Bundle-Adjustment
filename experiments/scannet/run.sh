#!/bin/bash
# cd to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_ROOT="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet"
OUTPUT_LOC="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet"

# Stage 1: Preprocessing (8 GPUs in parallel, wait for all to finish)
echo "=== Stage 1: Preprocessing ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/scannet/preprocess.py \
        --data-root ${DATA_ROOT}/scans_test \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "Preprocessing complete."

# Stage 2: SfM (8 GPUs in parallel, 1 GPU per process)
echo "=== Stage 2a: Structure from Motion (uncalibrated) ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/scannet/sfm.py \
        --data-root ${DATA_ROOT} \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "SfM (uncalibrated) complete."

echo "=== Stage 2b: Structure from Motion (calibrated) ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/scannet/sfm.py \
        --data-root ${DATA_ROOT} \
        --output-location ${OUTPUT_LOC} \
        --calibrated &
done
wait
echo "SfM (calibrated) complete."

# Stage 3: COLMAP Baseline (8 GPUs in parallel)
echo "=== Stage 3: COLMAP Baseline ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/scannet/colmap.py \
        --data-root ${DATA_ROOT}/scans_test \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "COLMAP baseline complete."

# Stage 4: Visualization
echo "=== Stage 4: Visualization ==="
python experiments/scannet/visualization.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

# Stage 5: Evaluation
echo "=== Stage 5a: Evaluation (uncalibrated) ==="
python experiments/scannet/summarize_result.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

echo "=== Stage 5b: Evaluation (calibrated) ==="
python experiments/scannet/summarize_result.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC} \
    --calibrated

echo "All done."