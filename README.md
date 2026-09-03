# Wake Geometry OBB

This project adds landmark-supervised geometry refinement to YOLOv8-OBB for ship-wake detection.

## Model

The P3 feature feeds a geometry head with six outputs:

- dense wake-structure probability;
- wake-tip heatmap;
- tip offsets `dx, dy`;
- two 16-bin Kelvin-arm direction distributions.

The structure branch is trained with landmark-constrained MIL. Each Kelvin arm is divided into segments, and each segment searches for its strongest image-supported linear response inside a band around the annotated direction. A wider band is ignored and distant locations are treated as background.

The predicted geometry guides two residual paths before the YOLO neck:

- background feature suppression gated by `1 - P_structure`;
- multi-distance directional feature extraction gated by `P_structure` and direction confidence.

The default residual scales start small so both paths receive detection gradients from the beginning.

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

Use `--refinement-mode aux|denoise|extract|full` for the four geometry ablations without editing code.

Visualize the learned geometry and refinement gates:

```bash
python visualize_geometry.py \
  --weights runs/wake_geometry/yolov8n_obb_geometry/weights/best.pt \
  --data /path/to/swim.yaml \
  --split val \
  --output runs/geometry_visualization
```

Each image contains landmark/predicted arms, `P_structure`, the tip heatmap, denoising and enhancement gates, and the actual P3 feature change. `geometry_metrics.csv` reports tip, arm-angle, and opening-angle errors.

The first experiment should compare 10% and 20% SWIM with the same split, seed, image size, epochs, and augmentations. Use `configs/geometry.yaml` to disable refinement for the auxiliary-only ablation.

## Kaggle

```bash
python kaggle_run_geometry.py --mode both --fraction 0.2 --epochs 100 --device 0 --name-prefix pilot20_s42
```

## Tests

```bash
pytest
```
