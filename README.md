# Haze-Resistant Multi-Scale Fusion Object Detection Network for Road Scenes (DFI-YOLO)

## Introduction
Hazy weather conditions, caused by light scattering and particulate occlusion, significantly degrade image quality and pose substantial challenges to object detection in real-world road scenes.  
To tackle this, we propose DFI-YOLO a haze-resilient object detection framework optimized for degraded visual environments.

## Dataset
We use a hybrid dataset consisting of:
- RTTS: Real-world hazy images with object-level annotations (from RESIDE dataset)
- VOC2012: Standard annotated dataset used to supplement training, including both clear and synthetic hazy images 

## Environment & Training Settings
- Python 3.9.7 
- CUDA 11.8 
- Ultralytics YOLO base
- Training configuration:
- Input size: 640 × 640 
- Batch size: 64
- Epochs: 150
- Initial LR: 1e-3
- Weight decay: 5e-4
- Learning rate scheduler: Cosine annealing

## Training Example
python train.py --img 640 --batch 64 --epochs 130 --data 'your datasets.yaml' --cfg 'yolo11.yaml' --weights '' --device 0
## Evaluation Example
python val.py --weights runs/train/exp/weights/best.pt --data 'your datasets.yaml' --img 640
