import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/HazeData/train132/weights/best.pt')
    # 验证模型
    results = model.val(data='datasets/Haze_data/data.yaml',
                       split='val',
                       imgsz=640,
                       batch=32,
                       # rect=False,
                       # save_json=True, # 这个保存coco精度指标的开关
                       project='runs/HazeData',
                       name='val',
                       )  # 进行验证，调整 conf 和 iou 根据需要
    # print(results.box.map)  # map50-95
    # print(results.box.maps)  # a list contains map50-95 of each category

    # # 打印结果
    # print("验证结果:")
    # print(f"mAP@0.5: {results.maps[0]:.4f}")  # mAP@0.5
    # print(f"mAP@0.5:0.95: {results.maps.mean():.4f}")  # mAP@0.5:0.95

    # 额外打印其他指标
    print("所有指标:")
    for key, value in results.results_dict.items():
        print(f"{key}: {value:.4f}")