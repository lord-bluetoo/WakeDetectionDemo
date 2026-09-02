# Wake Geometry OBB

This project adds landmark-supervised geometry refinement to YOLOv8-OBB for ship-wake detection.

## Model

The P3 feature feeds a geometry head with six outputs:

- dense wake-structure probability;
- wake-tip heatmap;
- tip offsets `dx, dy`;
- two 16-bin Kelvin-arm direction distributions.

The predicted geometry guides two residual paths before the YOLO neck:

- background feature denoising;
- directional feature extraction along the two wake arms.

Both residual scales start at zero, so pretrained YOLO behavior is preserved at initialization.

## Prepare SWIM

The original dataset must contain:

```text
JPEGImages/
Annotations/
Landmarks/
ImageSets/
```

Convert OBB and landmark XML files together:

```bash
python prepare_swim.py --source /path/to/SWIM --output /path/to/swim_yolo_geometry
```

The converted dataset contains matching `labels/` and `landmarks/` sidecars.

## Train

Baseline:

```bash
python train_baseline.py --data /path/to/swim.yaml --fraction 0.2 --seed 42
```

Geometry model:

```bash
python train_geometry.py --data /path/to/swim.yaml --fraction 0.2 --seed 42
```

The first experiment should compare 10% and 20% SWIM with the same split, seed, image size, epochs, and augmentations. Use `configs/geometry.yaml` to disable refinement for the auxiliary-only ablation.

## Kaggle

```bash
python kaggle_run_geometry.py --mode both --fraction 0.2 --epochs 100 --device 0 --name-prefix pilot20_s42
```

## Tests

```bash
pytest
```
