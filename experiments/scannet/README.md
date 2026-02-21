# ScanNet Dataset Benchmarking

This guide describes how to process and evaluate the ScanNet dataset on a server equipped with 8 GPUs.

> **⏱️ Time & Disk Space Notice:** Running the full pipeline on this dataset may take **several days** (depending on hardware) and require **1–2 TB** of disk space for intermediate and final outputs. Please ensure sufficient storage and plan accordingly.

## Preprocessing

Process the raw ScanNet data using 8 GPUs. Run each command in a separate tmux window:

```bash
# Replace {GPU_ID} with values 0-7
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/scannet/preprocess.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet
```

## Structure from Motion

Run SfM reconstruction after preprocessing is complete:

```bash
# Uncalibrated mode (default)
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/scannet/sfm.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet

# Calibrated mode
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/scannet/sfm.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet \
    --calibrated
```
Note: For parallel GPU processing, run each command with a different GPU_ID (0-7) in separate tmux windows.

## COLMAP Baseline

Run COLMAP reconstruction in both uncalibrated (default) and calibrated modes. Required for evaluation.

```bash
# Uncalibrated mode (default)
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/scannet/colmap.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet

# Calibrated mode
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/scannet/colmap.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet \
    --calibrated
```

## Visualization

Generate visual results for qualitative analysis. The script creates point clouds and videos for each scene:

```bash
python experiments/scannet/visualization.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet
```

## Evaluation

After SfM completion, evaluate results for both modes:

```bash
# Evaluate uncalibrated results
python experiments/scannet/summarize_result.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet

# Evaluate calibrated results
python experiments/scannet/summarize_result.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet \
    --calibrated
```

## Run All Steps

Alternatively, run the full pipeline (preprocessing → SfM → COLMAP baseline → visualization → evaluation) in one go:

```bash
bash experiments/scannet/run.sh
```

