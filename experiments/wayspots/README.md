# WaySpots Dataset Processing Guide

This guide describes how to process and evaluate the WaySpots dataset on a server equipped with 8 GPUs.

> **⏱️ Time & Disk Space Notice:** Running the full pipeline on this dataset may take **several days** (depending on hardware) and require **1–2 TB** of disk space for intermediate and final outputs. Please ensure sufficient storage and plan accordingly.

## Preprocessing

Run each command in a separate tmux window:

```bash
# Replace {GPU_ID} with values 0-7
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/wayspots/preprocess.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots
```

## Structure from Motion

Run SfM reconstruction after preprocessing is complete:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/wayspots/sfm.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots
```
Experiment will automatically proceed to the next sequence after completing a scene, continuing until all sequences are processed.

## Visualization

Generate visual results for qualitative analysis. The script creates point clouds and videos for each scene:

```bash
python experiments/wayspots/visualization.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots
```

## Evaluation

After SfM completion, evaluate the results:

```bash
python experiments/wayspots/summarize_result.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots
```

## Run All Steps

Alternatively, run the full pipeline (preprocessing → SfM → visualization → evaluation) in one go:

```bash
bash experiments/wayspots/run.sh