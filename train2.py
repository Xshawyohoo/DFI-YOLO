import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml') #exp6
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'datasets/Haze_data/data.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                # cache=False,
                imgsz=640,
                epochs=130,
                lr0=0.001,              # 初始学习率
                iou=0.5,
                weight_decay=0.0005,   # 权重衰减防止过拟合
                cos_lr=True,
                warmup_bias_lr=0.01,  # 通常设置为 0.01 到 0.1 之间
                # single_cls=False,  # 是否是单类别检测
                batch=32,
                # close_mosaic=0,
                # workers=0,
                momentum=0.9,           # 控制梯度更新的动量项，一般用于加速收敛，减少振荡
                patience=50,            # 如果在10个epochs内验证集mAP没有提升，则停止训练
                device='0',
                optimizer='Adam', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                # amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/HazeData',
                name='train',
                )