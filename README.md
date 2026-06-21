# TAGE-Depth
# TAGE-Depth

**TAGE-Depth: A Transparent-Aware Guided Enhancement Framework With Multi-Prior Adaptation for Transparent Object Depth Completion**

This repository contains a clean PyTorch implementation scaffold for **TAGE-Depth**, a multi-prior-guided and raw-preserving framework for transparent object depth completion.

TAGE-Depth uses three complementary priors:

- **Relative-depth prior** for coarse global depth layout.
- **Mask-boundary prior** for contour-aware transparent-object refinement.
- **Raw-reliability prior** for selective correction of unreliable raw depth.

These cues are encoded as multi-scale guidance features and injected into the feature hierarchy through a **Scale-wise Prior Adapter (SPA)**. The network predicts completed depth inside transparent regions while preserving raw depth outside the transparent mask.

## Repository structure

```text
TAGE-Depth/
├── assets/                  # figures, examples, and qualitative results
├── configs/                 # yaml configuration files
├── datasets/                # dataset loading utilities
├── models/                  # TAGE-Depth model and modules
├── scripts/                 # evaluation and visualization scripts
├── utils/                   # losses, metrics, and IO utilities
├── inference.py             # single-sample inference entry
├── sample_inference.py      # minimal inference example
├── test.py                  # quick forward-pass sanity check
├── train.py                 # training entry
├── requirements.txt
├── LICENSE.txt
└── README.md
```

## Installation

```bash
conda create -n tage-depth python=3.10 -y
conda activate tage-depth
pip install -r requirements.txt
```

## Dataset preparation

A recommended dataset layout is:

```text
data/TransCG/
├── train.txt
├── val.txt
├── test.txt
├── rgb/
├── raw_depth/
├── mask/
├── gt_depth/
├── rel_depth/
└── raw_reliability/         # optional; can be generated on the fly
```

Each line in `train.txt`, `val.txt`, and `test.txt` should contain one sample id, for example:

```text
000001
000002
000003
```

For each sample id, the loader searches for files such as:

```text
rgb/000001.png
raw_depth/000001.npy
mask/000001.png
gt_depth/000001.npy
rel_depth/000001.npy
```

Depth maps can be stored as `.npy`, `.png`, `.tif`, or `.tiff`. For best reproducibility, metric depth maps are recommended to be saved as `.npy` arrays in meters.

## Training

```bash
python train.py --config configs/tage_depth.yaml
```

Important default settings in `configs/tage_depth.yaml`:

```yaml
image_size: [240, 320]
batch_size: 2
epochs: 3
lr_adapter: 2.0e-5
lr_head: 5.0e-6
weight_decay: 1.0e-4
seed: 6248
```

## Inference

```bash
python inference.py \
  --checkpoint outputs/checkpoints/best.pth \
  --rgb assets/example_rgb.png \
  --raw-depth assets/example_raw_depth.npy \
  --mask assets/example_mask.png \
  --relative-depth assets/example_relative_depth.npy \
  --output outputs/example_final_depth.npy
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/tage_depth.yaml \
  --checkpoint outputs/checkpoints/best.pth \
  --split test
```

The evaluation reports MAE, RMSE, REL, and threshold accuracies inside valid transparent-object regions.

## Notes

This package is organized for public release. Please replace the placeholder dataset paths and add pretrained weights before final publication.

## Citation

```bibtex
@article{TAGEDepth,
  title={TAGE-Depth: A Transparent-Aware Guided Enhancement Framework With Multi-Prior Adaptation for Transparent Object Depth Completion},
  author={Liu, Zhimin and Sun, Ruhao and Wang, Dongmin and Li, Dong},
  journal={To be added},
  year={2026}
}
```
