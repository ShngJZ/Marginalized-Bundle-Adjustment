# FastMap SfM & TNT-SAIL Dataset Processing Guide

This guide describes how to process and evaluate the FastMap SfM dataset (which includes TNT scenes as a subset) on a server equipped with 8 GPUs.

> **⏱️ Time & Disk Space Notice:** Running the full pipeline on this dataset may take **several days** (depending on hardware) and require **1–2 TB** of disk space for intermediate and final outputs. Please ensure sufficient storage and plan accordingly.

The pipeline runs one shared inference (preprocessing → pose init → SfM) and two separate evaluations:
1. **FastMap SfM evaluation** — evaluates all scenes
2. **TNT-SAIL evaluation** — evaluates the TNT subset against SAIL-VOS ground truth

## Preprocessing

Process all scenes using 8 GPUs in parallel. Run each command in a separate tmux window:

```bash
# Replace {GPU_ID} with values 0-7
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/fastmap_sfm/preprocess.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm
```

## Pose Initialization

Run two-view SfM initialization using 8 GPUs in parallel:

```bash
# Replace {GPU_ID} with values 0-7
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/fastmap_sfm/sfm_twoview_init.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm
```

## Structure from Motion

Run SfM reconstruction after pose initialization is complete:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/fastmap_sfm/sfm.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm
```

## Visualization

Generate visual results for qualitative analysis:

```bash
python experiments/fastmap_sfm/visualization.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm
```

## Evaluation

### FastMap SfM Evaluation

Evaluate all scenes against FastMap ground truth:

```bash
python experiments/fastmap_sfm/evaluation.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm
```

### TNT-SAIL Evaluation

Evaluate the TNT subset against SAIL-VOS ground truth:

```bash
python experiments/fastmap_sfm/evaluation_tnt_sail.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/tnt_sail \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm
```

## Run All Steps

Run the full pipeline in one go:

```bash
bash experiments/fastmap_sfm/run.sh