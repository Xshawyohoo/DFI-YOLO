import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'datasets/Haze_data/data.yaml',
                # cache=False,
                imgsz=640,
                epochs=130,
                lr0=0.001,              # Initial learning rate
                iou=0.5,
                weight_decay=0.0005,   # Weight decay prevents overfitting
                cos_lr=True,
                warmup_bias_lr=0.01, 
                # single_cls=False,  # Is it single - class detection?
                batch=32,
                # close_mosaic=0,
                # workers=0,
                momentum=0.9,           # The momentum term that controls the gradient update is generally used to accelerate convergence and reduce oscillation.
                patience=50,
                device='0',
                optimizer='Adam', 
                # resume='runs/train/exp21/weights/last.pt', 
                # amp=True, 
                project='runs/HazeData',
                name='train',
                )