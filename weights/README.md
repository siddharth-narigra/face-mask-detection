# Model Weights

This directory should contain the trained YOLOv5 weights file: `best.pt`

## Model Specification

| Property     | Value         |
| ------------ | ------------- |
| Architecture | YOLOv5        |
| Input Size   | 896×896      |
| Classes      | 3             |
| Format       | PyTorch (.pt) |

## Classes

| ID | Class Name                | Description                              |
| -- | ------------------------- | ---------------------------------------- |
| 0  | `mask_weared_incorrect` | Mask worn improperly (nose/chin exposed) |
| 1  | `with_mask`             | Mask worn correctly                      |
| 2  | `without_mask`          | No mask detected                         |

## Training Details

The model was trained on a custom combined dataset:

- **Total Images**: 8,065
- **Total Annotations**: 25,337
- **Epochs**: 15
- **Batch Size**: 8
- **Optimizer**: SGD (lr=0.01, momentum=0.937)
- **Augmentations**: Mosaic, MixUp, Copy-Paste, HSV shifts

### Performance Metrics

| Metric       | Value |
| ------------ | ----- |
| mAP@0.5      | 89.4% |
| mAP@0.5:0.95 | 60.1% |
| Precision    | 84.2% |
| Recall       | 84.1% |

## Obtaining Weights

The weights file is not included in the repository due to its size (~14MB).

**Option 1**: Contact the repository author for the pre-trained weights.

**Option 2**: Train your own model using the datasets listed below.

## Training Your Own Model

### Dataset Sources

1. [Roboflow Real-time Face Mask Detection](https://universe.roboflow.com/group-tbd/real-time-face-mask-detection-and-validation-system-dataset)
2. [Kaggle Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
3. [Kaggle Face Mask YOLO Format](https://www.kaggle.com/datasets/aditya276/face-mask-dataset-yolo-format)
4. [Kaggle Labeled Mask YOLO_darknet](https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-yolo-darknet)
5. [GitHub MINED30 Face Mask](https://github.com/MINED30/Face_Mask_Detection_YOLO)

### Training Command

```bash
python train.py --img 896 --batch 8 --epochs 15 \
    --data data.yaml --weights yolov5s.pt \
    --hyp hyp.yaml --freeze 10
```

### After Training

Copy the best weights to this directory:

```bash
cp runs/train/exp/weights/best.pt weights/best.pt
```
