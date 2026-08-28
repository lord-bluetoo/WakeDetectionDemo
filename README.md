# Wake Structure Head v1

这是对“先验证显式尾迹结构学习能否改善 OBB 检测表征”这一假设的最小实现：

```text
image -> YOLOv8 backbone -> P3/8
                         |-> 原 YOLOv8n-OBB 检测分支
                         `-> Structure Head -> P + q_theta
```

第一版没有去噪模块、方向卷积或检测特征门控。Structure Head 只是训练期辅助任务，因此 baseline 和实验组之间唯一新增变量是结构学习约束。

## 实现内容

- `P`：1 个 wake-presence logit，经 sigmoid 得到概率；
- `q_theta`：8 个方向 logits，经 softmax 得到 `[0°, 180°)` 上的分布；
- `theta`：对 `q_theta` 做 180° 周期圆统计得到的连续方向；
- `C`：双角向量合成后的模长，表示方向分布集中度；
- `P*C`：预留给后续方向特征门控，本版不回灌检测分支。

现有 OBB 不被当作精确结构真值。每个 OBB 只是一个正 MIL bag：框内至少有少量位置应当响应，但不会把整个框标成 wake。框外提供背景约束；OBB 长轴只提供一个低精度、soft-bin 的方向先验；稀疏项限制整框全亮；90°/180° 旋转一致性提供无额外标注的等变约束。

总目标为：

```text
L = L_obb
  + lambda_mil * L_mil
  + lambda_bg * L_background
  + lambda_dir * L_orientation
  + lambda_sparse * L_sparse
  + lambda_equiv * L_equivariance
```

## 安装

```bash
python -m pip install -e ".[dev]"
```

数据必须先转换成 Ultralytics OBB 格式。每行标签为：

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

八个坐标均归一化到 `[0, 1]`。可复制 `configs/dataset_example.yaml` 后修改路径。

官方 Kaggle SWIM release 可直接转换。在 Kaggle 中先定位挂载目录，然后运行：

```bash
python prepare_swim.py \
  --source /kaggle/input/datasets/lilitopia/swimship-wake-imagery-mass/SWIM_Dataset_1.0.0 \
  --output /kaggle/working/swim_yolo_obb
```

脚本严格沿用 `ImageSets/train.txt`、`val.txt`、`test.txt` 的 6960/2320/2320 正样本划分，解析 XML 中的 `cx/cy/w/h/angle` 并生成四角点标签。Kaggle 上默认使用符号链接，不会把 11,600 张图片重复复制到 working；独立的 3,010 张 `Negative` 图片暂不混入第一版 benchmark。

转换器默认保留 SWIM XML 中的原始旋转矩形，不裁剪落在图像外的角点。因此少量归一化坐标可能超出 `[0, 1]`，当前 Ultralytics 可能将超出容差的图片/标签标为 corrupt 并忽略；第一轮先用此设置检查训练流程。若之后需要裁剪对照，可在转换命令末尾添加 `--clip-boxes`。

## 最小实验

先用完全相同的数据划分、seed、训练轮数和图像尺寸跑 baseline：

```bash
python train_baseline.py --data /path/to/swim.yaml --epochs 50 --fraction 0.2 --device 0
```

再跑 Structure Head 组：

```bash
python train_structure.py --data /path/to/swim.yaml --epochs 50 --fraction 0.2 --device 0
```

Kaggle 的交互 Session 可能失效。可给训练命令添加 `--archive`，训练成功后自动把当前 run（包括 `best.pt`、`last.pt`、`results.csv` 和图表）压缩到 `/kaggle/working`：

```bash
python train_structure.py \
  --data /kaggle/working/swim_yolo_obb_raw/swim.yaml \
  --epochs 50 --fraction 0.2 --imgsz 640 --batch 16 --device 0 --seed 42 \
  --name pilot20_structure_s42 \
  --archive /kaggle/working/pilot20_structure_s42.zip
```

Baseline 入口同样支持 `--archive`。

## Structure Head 诊断

在决定是否把 `P/C` 接回检测分支之前，先对 Structure 权重运行固定样本诊断：

```bash
python diagnose_structure.py \
  --weights /kaggle/working/WakeDetectionDemo/runs/obb/runs/wake_ablation/pilot20_structure_s42/weights/best.pt \
  --data /kaggle/working/swim_yolo_obb_raw/swim.yaml \
  --split val --num-images 12 --imgsz 640 --device 0 --seed 42 \
  --output /kaggle/working/structure_diagnostics_s42 \
  --archive /kaggle/working/structure_diagnostics_s42.zip
```

输出包括：

- `figures/*.png`：输入与 OBB、`P`、`C`、`P*C`、角度色相和高置信方向场；
- `diagnostics.csv`：逐图的框内/框外响应、方向集中度、归一化熵和相对 OBB 长轴的方向误差；
- `summary.json`：均值与显式标为 heuristic 的坍缩筛查标志；
- `training_curves.png`：Structure losses 和检测 mAP 曲线；
- ZIP：完整训练 run 与上述诊断，便于立即下载。

`summary.json` 中的阈值只用于快速筛查，不是显著性检验。定性图仍是判断该弱监督是否真的定位尾迹结构的主要证据。

完整 SWIM 实验去掉 `--fraction 0.2`。建议先比较 `mAP50`、`mAP75` 和 `mAP50-95`，并至少重复 3 个 seed。Kaggle 单 GPU 可直接使用；本训练器的 v1 目标是单进程/单 GPU，不支持 DDP。

结构超参数在 `configs/structure_v1.yaml`。若要做最干净的损失消融，可把某一项权重设为 0；若显存紧张，可先将 `enable_equivariance` 设为 `false`，它会省掉旋转图像到 P3 的第二次局部前向。

## 代码入口

- `wake_structure/head.py`：9-channel Structure Head；
- `wake_structure/geometry.py`：`q_theta -> theta, C`；
- `wake_structure/targets.py`：OBB 到 MIL/soft-direction 弱目标；
- `wake_structure/losses.py`：全部辅助损失；
- `wake_structure/model.py`：YOLOv8n-OBB 接入与训练器；
- `tests/`：圆统计、目标、损失和真实 YOLO 最小集成测试。

运行测试：

```bash
pytest
```

一个重要限制：OBB 长轴不是 wake 局部方向真值。本版只把它当宽松先验，用来回答“辅助结构学习是否值得继续”这个问题；若 A/B 没有稳定提升，应先查看 `P/theta/C` 是否退化，再决定是否加入 Radon 先验或更可靠的局部方向监督，而不是立刻叠加去噪和 DCN。
