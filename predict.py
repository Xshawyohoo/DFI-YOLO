from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # 预训练模型 yolov8n.pt 或自定义训练好的模型路径
# 或者加载自定义训练好的模型
model = YOLO('runs/detect/train32/weights/best.pt')

# 进行预测，输入可以是图像、视频路径，甚至是目录或摄像头流
results = model.predict(source='datasets/Hybrid_data/test/images',  # 输入图像、视频路径或目录
                        imgsz=640,
                        save=True,                        # 保存预测结果
                        # show=True,                        # 是否实时显示预测结果
                        conf=0.5)                        # 置信度阈值
# results = model.predict(source='../assets/test3.mp4', save=True, show=True)
# results = model.predict(source='path/to/image_folder/', save=True) #目录中的所有图像进行预测
# results = model.predict(source=0, show=True)  # 0 表示使用默认摄像头进行预测

# 预测结果会自动保存在 runs/predict/exp 目录下
