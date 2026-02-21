# Custom Dataset Processing Guide

This guide describes how to process and evaluate a custom dataset on a server equipped with 8 GPUs.

## Preprocessing

Process the custom data using 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/custom/preprocess.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/custom \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/custom
```

## Structure from Motion

Run SfM reconstruction after preprocessing is complete. We recommend parallelizing across GPUs for large scenes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/custom/sfm.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/custom \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/custom
```

## Visualization

Generate visual results for qualitative analysis:

```bash
python experiments/custom/visualization.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/custom \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/custom
```

## Run All Steps

Run the full pipeline (preprocessing → SfM → visualization) in one go:

```bash
bash experiments/custom/run.sh