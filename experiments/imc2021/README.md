# IMC2021 Dataset Processing Guide

This guide describes how to process and evaluate the IMC2021 dataset on a server equipped with 8 GPUs.

## Preprocessing

Run each command in a separate tmux window:

```bash
# Replace {GPU_ID} with values 0-7
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/imc2021/preprocess.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/imc2021 \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/imc2021
```

## Structure from Motion

Run SfM reconstruction after preprocessing is complete:

```bash
# Replace {GPU_ID} with values 0-7
CUDA_VISIBLE_DEVICES={GPU_ID} python experiments/imc2021/sfm.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/imc2021 \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/imc2021
```
Note: run each command with a different GPU_ID (0-7) in separate tmux windows. Each experiment will automatically proceed to the next sequence after completing a scene, continuing until all sequences are processed.

## Visualization

Generate visual results for qualitative analysis. The script creates point clouds and videos for each scene:

```bash
python experiments/imc2021/visualization.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/imc2021 \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/imc2021
```

## Evaluation

After SfM completion, evaluate the results:

```bash
python experiments/imc2021/summarize_result.py \
    --data-root /home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/imc2021 \
    --output-location /home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/imc2021
```

## Run All Steps

Alternatively, run the full pipeline (preprocessing → SfM → visualization → evaluation) in one go:

```bash
bash experiments/imc2021/run.sh