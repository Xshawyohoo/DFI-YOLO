from ultralytics import YOLO

# model = YOLO('yolo11n.pt')  
model = YOLO('runs/train/weights/best.pt')

# Perform predictions. The input can be an image, a video path, or even a directory or a camera stream.
results = model.predict(source='datasets/Hybrid_data/test/images',  # Input the path or directory of the image or video
                        imgsz=640,
                        save=True,                      
                        # show=True,                       
                        conf=0.5)                        
# results = model.predict(source='../assets/test3.mp4', save=True, show=True)
# results = model.predict(source='path/to/image_folder/', save=True) 
# results = model.predict(source=0, show=True)  # 0 