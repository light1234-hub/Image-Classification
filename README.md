# DINOv2-L Multi-Resolution Image Classification

An image classification project using a frozen pretrained **DINOv2 ViT-L/14** encoder and a trained linear classifier for each input resolution.

**Achieved Accuracy: 0.94653 (94.653%)**.

## Approach

- Twenty animal/object classes and a stratified 85/15 training/validation split.
- Normalize train/test visual detail by first resizing each image to **64×64**, then resizing to ViT input size.
- Train four randomly augmented views of each training image.
- Extract a **2,048-dimensional** feature vector: DINOv2-L CLS token (1,024) concatenated with mean patch token (1,024).
- Freeze the backbone and train a Dropout(0.4) → Linear(2048, 20) head using AdamW, label-smoothed cross entropy, and a cosine learning-rate schedule.
- Run at 238×238, 252×252, and 266×266. Select each classifier checkpoint by highest validation accuracy (lowest validation loss to break ties).
- Blend probabilities: **90% 238 + 9.75% 252 + 0.25% 266**.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract the competition datasets so the files appear in the following structure:


Then run:

```bash
python dinov2_kaggle_ensemble.py
```

Set `DATA_ROOT` to use another root directory, e.g.:

```bash
DATA_ROOT=/path/to/data python dinov2_kaggle_ensemble.py
```

The first run downloads DINOv2 weights from PyTorch Hub. A CUDA GPU is strongly recommended. The output is `submission_238_90p0_PREV_10p0.csv`, along with generated feature caches, linear classifier checkpoints, and test logits.

## Reference

Oquab et al. (2023), [*DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193). Pretrained backbone loaded from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
