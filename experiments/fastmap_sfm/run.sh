#!/bin/bash
# cd to project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_ROOT="/fsx/users/shngjz/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm"
TNT_SAIL_ROOT="/fsx/users/shngjz/Marginalized-Bundle-Adjustment-Datasets/tnt_sail"
OUTPUT_LOC="/fsx/users/shngjz/Marginalized-Bundle-Adjustment/release/fastmap_sfm"

# Stage 1: Preprocessing (8 GPUs in parallel)
echo "=== Stage 1: Preprocessing ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/fastmap_sfm/preprocess.py \
        --data-root ${DATA_ROOT} \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "Preprocessing complete."

# Stage 2: Pose Initialization (8 GPUs in parallel, 1 GPU per process)
echo "=== Stage 2: Pose Initialization ==="
for GPU_ID in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/fastmap_sfm/sfm_twoview_init.py \
        --data-root ${DATA_ROOT} \
        --output-location ${OUTPUT_LOC} &
done
wait
echo "Pose initialization complete."

# Stage 3: SfM (all 8 GPUs working together)
echo "=== Stage 3: Structure from Motion ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/fastmap_sfm/sfm.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}
echo "SfM complete."

# Stage 4: Visualization
echo "=== Stage 4: Visualization ==="
python experiments/fastmap_sfm/visualization.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

# Stage 5: Evaluation
echo "=== Stage 5a: FastMap SfM Evaluation ==="
python experiments/fastmap_sfm/evaluation.py \
    --data-root ${DATA_ROOT} \
    --output-location ${OUTPUT_LOC}

echo "=== Stage 5b: TNT-SAIL Evaluation ==="
python experiments/fastmap_sfm/evaluation_tnt_sail.py \
    --data-root ${TNT_SAIL_ROOT} \
    --output-location ${OUTPUT_LOC}

echo "All done."